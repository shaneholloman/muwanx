"""Adapter for converting mjlab types to mjswan internal representations.

mjlab is a **soft dependency** — this module never fails at import time.
When mjlab is not installed the ``adapt_*`` functions simply return their
inputs unchanged (they are assumed to already be mjswan types).

The adapter detects mjlab types by checking the module path of the class
(``type(obj).__module__``) rather than ``isinstance``, so mjlab does not
need to be importable for mjswan to function.

Mapping strategy
----------------
* **Observation / termination / event functions**: mjlab's own function
  object is traced directly (ADR 0005) against the scene's live env at build
  time — there is no mjswan-side reimplementation to look up by name. An
  author can still override what gets traced for a given mjlab function or
  term name via ``register_observation`` / ``register_termination`` /
  ``register_event`` (e.g. to supply a trace-friendly rewrite, or a
  hand-written TS class for a term tracing cannot express).
* **Commands**: ``type(cfg).__name__`` is looked up in the command registry;
  a registered entry either builds+traces the term via ``cfg.build(env)``
  (ADR 0005 §3) or maps to a permanently-native TS class (``TrackingCommand``).
* **Action configs**: ``type(cfg).__name__`` is looked up on
  ``mjswan.envs.mdp.actions``, and dataclass fields are copied
  automatically. Actions stay a fixed, native, non-traced closed set.
"""

from __future__ import annotations

import dataclasses
import re
import warnings
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

from ..command import CommandTermConfig as MjswanCommandTermConfig
from ..command import PendingCommandTrace, PendingResetTrace
from ..command import _custom_registry as _custom_command_registry
from ..envs.mdp import actions as _actions_module
from ..envs.mdp.actions.actions import (
    ActionTermCfg as MjswanActionTermCfg,
)
from ..envs.mdp.events import EventBinding
from ..envs.mdp.events import _custom_registry as _custom_event_registry
from ..envs.mdp.observations import ObservationBinding, _custom_registry
from ..envs.mdp.terminations import TerminationBinding
from ..envs.mdp.terminations import _custom_registry as _custom_term_registry
from ..managers.event_manager import EventTermCfg as MjswanEventTermCfg
from ..managers.observation_manager import (
    ObservationGroupCfg as MjswanObservationGroupCfg,
)
from ..managers.observation_manager import (
    ObservationTermCfg as MjswanObservationTermCfg,
)
from ..managers.termination_manager import (
    TerminationTermCfg as MjswanTerminationTermCfg,
)


def _is_from_mjlab(obj: Any) -> bool:
    """Check whether *obj*'s class originates from the ``mjlab`` package."""
    module = getattr(type(obj), "__module__", "") or ""
    return module.startswith("mjlab")


# ---------------------------------------------------------------------------
# Observation adaptation
# ---------------------------------------------------------------------------


def _adapt_obs_func(
    func: Any, term_name: str | None = None
) -> ObservationBinding | Callable[..., Any]:
    """Resolve the function an observation term's ONNX graph is traced from.

    If *func* is already an mjswan ``ObservationBinding`` (a hand-written TS class)
    it is returned as-is.

    An author can register an override for a given mjlab function or term name
    via ``register_observation`` — either an ``ObservationBinding`` or a
    trace-friendly replacement callable (ADR 0005 §3a's "examples-side
    trace-friendly override" pattern, generalized beyond commands).

    Otherwise, *func* — mjlab's own function object — is returned unchanged; the
    build traces it directly against the scene's live env (ADR 0005). There is
    no mjswan-side reimplementation to look up by name.
    """
    if isinstance(func, ObservationBinding):
        return func
    name = getattr(func, "__name__", None)
    if name and name in _custom_registry:
        return _custom_registry[name]
    if term_name and term_name in _custom_registry:
        return _custom_registry[term_name]
    return func


def _sanitize_obs_params(params: dict[str, Any]) -> dict[str, Any]:
    """Strip mjlab-specific params that are not JSON-serializable.

    ``asset_cfg`` (a ``SceneEntityCfg``) is removed.  When it carries
    entity-scoping information, it is promoted into JSON-friendly fields so
    browser-side observation classes can resolve the correct MuJoCo entities
    at runtime.
    """
    if "asset_cfg" not in params:
        return params
    result = {k: v for k, v in params.items() if k != "asset_cfg"}
    asset_cfg = params["asset_cfg"]
    if _is_from_mjlab(asset_cfg):
        entity_name = getattr(asset_cfg, "name", None)
        if entity_name:
            result["entity_name"] = entity_name
        joint_names = getattr(asset_cfg, "joint_names", None)
        if joint_names:
            names = (
                list(joint_names)
                if isinstance(joint_names, (list, tuple))
                else [joint_names]
            )
            result["joint_names"] = names
            if len(names) == 1:
                name = names[0]
                result["joint_name"] = f"{entity_name}/{name}" if entity_name else name
        site_names = getattr(asset_cfg, "site_names", None)
        if site_names:
            name = (
                site_names[0] if isinstance(site_names, (list, tuple)) else site_names
            )
            # mjlab namespaces entity sites as "{entity_name}/{site_name}"
            result["site_name"] = f"{entity_name}/{name}" if entity_name else name
    return result


def _adapt_obs_term(
    term: Any, term_name: str | None = None
) -> MjswanObservationTermCfg:
    """Convert a single mjlab ``ObservationTermCfg`` to mjswan.

    Params are only sanitized (``asset_cfg`` flattened to JSON-safe strings)
    when *func* resolved to a legacy ``ObservationBinding`` — those params are
    written verbatim into the browser JSON for a named TS class to consume. A
    func destined for ONNX tracing keeps its params as-is: ``trace_term`` calls
    ``func(env, **params)`` and needs the real ``SceneEntityCfg`` mjlab's own
    function expects, not a flattened stand-in.
    """
    raw_params = dict(getattr(term, "params", None) or {})
    func = _adapt_obs_func(term.func, term_name=term_name)
    params = (
        _sanitize_obs_params(raw_params)
        if isinstance(func, ObservationBinding)
        else raw_params
    )
    return MjswanObservationTermCfg(
        func=func,
        params=params,
        scale=getattr(term, "scale", None),
        clip=getattr(term, "clip", None),
        history_length=getattr(term, "history_length", 0) or 0,
    )


def _adapt_obs_group(group: Any) -> MjswanObservationGroupCfg:
    """Convert a single mjlab ``ObservationGroupCfg`` to mjswan."""
    raw_terms = getattr(group, "terms", None) or {}
    terms = {
        name: _adapt_obs_term(cfg, term_name=name) for name, cfg in raw_terms.items()
    }
    return MjswanObservationGroupCfg(
        terms=terms,
        concatenate_terms=getattr(group, "concatenate_terms", True),
        enable_corruption=getattr(group, "enable_corruption", False),
        history_length=getattr(group, "history_length", None),
    )


#: Group names mjlab uses for networks that never leave training. Only the actor is
#: exported to ONNX (``runner.load(load_cfg={"actor": True})`` in ``mjswan.wandb_io``),
#: so a group named like this has no input to feed and is dropped rather than traced,
#: bundled, and evaluated every control step for a value nothing reads.
_TRAINING_ONLY_OBS_GROUPS = frozenset({"critic"})

#: The key a single observation group lands under. It is the ONNX input name the
#: runtime feeds, not a free-form label: ``OnnxModule`` defaults ``in_keys`` to
#: ``['policy']``, and an input it cannot find is a console warning and a policy that
#: never acts. mjlab's own name for the same group is ``"actor"``.
DEFAULT_OBS_GROUP_KEY = "policy"

#: mjlab's own name for the group its actor network reads, and the fallback used when
#: no runner config is available to say otherwise (a hand-built ``env_cfg`` has none).
#: It is mjlab's default for ``RslRlBaseRunnerCfg.obs_groups``: ``{"actor": ("actor",)}``.
_MJLAB_ACTOR_GROUP = "actor"


def _is_obs_group(value: Any) -> bool:
    """Whether *value* is a single observation group rather than a dict of them."""
    if isinstance(value, MjswanObservationGroupCfg):
        return True
    # An mjlab group, or any duck-typed stand-in: it carries `terms`, a dict of them does not.
    return not isinstance(value, Mapping) and hasattr(value, "terms")


def _select_policy_group(
    observations: Mapping[str, Any],
    policy_groups: tuple[str, ...] | None,
) -> Mapping[str, Any]:
    """Reduce an mjlab-shaped group dict to the one group the policy's ONNX input reads.

    An mjlab ``env_cfg.observations`` is keyed by *network* name (``"actor"``,
    ``"critic"``); mjswan's keys are *ONNX input* names. Those namespaces only look
    alike, so a dict that is plainly the former gets remapped onto the latter, and a
    dict that is not is returned untouched — a policy whose input really is named
    ``"observation"`` or ``"obs_history"``, or a genuinely multi-input one, must keep
    the keys its config declares.

    *policy_groups* is the task's own answer, from ``rl_cfg.obs_groups["actor"]``, and
    wins over the ``"actor"`` name. It is consulted only when it actually names one of
    the keys present: a dict that shares no key with the task's group names is not the
    task's dict, whatever the task happens to call its groups, and remapping it on the
    strength of a task id would break a policy whose input is named something else.
    """
    if not observations:
        # `{}` says the policy has no observations. Nothing to select, and nothing wrong.
        return observations

    if policy_groups and not set(policy_groups).isdisjoint(observations):
        if len(policy_groups) != 1:
            # rsl-rl lets one network read several groups concatenated. mjswan feeds one
            # vector per ONNX input and cannot join them, so taking the first would hand
            # the policy a short observation — the one case that has to be loud.
            raise ValueError(
                "The task's runner config feeds its actor network "
                f"{len(policy_groups)} concatenated observation groups "
                f"({', '.join(map(repr, policy_groups))}). mjswan feeds one group per "
                "ONNX input and cannot concatenate them, so pass the single group the "
                "exported policy actually takes: "
                "`observations=env_cfg.observations[<name>]`."
            )
        return {DEFAULT_OBS_GROUP_KEY: observations[policy_groups[0]]}

    if _MJLAB_ACTOR_GROUP in observations:
        return {DEFAULT_OBS_GROUP_KEY: observations[_MJLAB_ACTOR_GROUP]}
    return observations


def adapt_observations(
    observations: Mapping[str, Any] | Any | None,
    *,
    policy_groups: tuple[str, ...] | None = None,
) -> dict[str, MjswanObservationGroupCfg] | None:
    """Adapt observation groups, converting mjlab types if detected.

    Accepts three shapes, because the caller should not have to know which key the
    runtime will look for:

    * a **single** group — mjlab's ``env_cfg.observations["actor"]`` — which lands
      under :data:`DEFAULT_OBS_GROUP_KEY`;
    * mjlab's whole ``env_cfg.observations`` dict, from which the policy's group is
      selected (see :func:`_select_policy_group`) and the rest dropped;
    * a dict already keyed by ONNX input name, passed through as-is.

    That key is not a label the caller is free to choose, so spelling
    ``{"policy": env_cfg.observations["actor"]}`` at every call site put a
    silent-failure mode in the caller's hands for no gain.

    If the values are already ``mjswan.ObservationGroupCfg`` instances they
    are returned as-is.  mjlab ``ObservationGroupCfg`` instances are
    converted transparently. Groups named for a training-only mjlab network
    (:data:`_TRAINING_ONLY_OBS_GROUPS`) are dropped with a warning.

    Args:
        observations: One of the three shapes above.
        policy_groups: The task's ``rl_cfg.obs_groups["actor"]``, when known. Only
            consulted for the dict form.
    """
    if observations is None:
        return None
    if _is_obs_group(observations):
        observations = {DEFAULT_OBS_GROUP_KEY: observations}
    else:
        observations = _select_policy_group(observations, policy_groups)

    # `Any`-valued while filling: the final branch below passes a group through untouched,
    # duck-typed rather than either known class, and narrowing to the return type happens
    # at the boundary as it does in the other `adapt_*`.
    adapted: dict[str, Any] = {}
    for key, group in observations.items():
        if key in _TRAINING_ONLY_OBS_GROUPS:
            warnings.warn(
                f"Dropping observation group {key!r}: mjlab exports only the actor "
                "network, so no ONNX input consumes it. Pass just the policy's own "
                'group — `observations=env_cfg.observations["actor"]`.',
                category=RuntimeWarning,
                stacklevel=3,
            )
            continue
        if isinstance(group, MjswanObservationGroupCfg):
            adapted[key] = group
        elif _is_from_mjlab(group):
            adapted[key] = _adapt_obs_group(group)
        else:
            adapted[key] = group
    return adapted


class MjlabRunnerDefaults(NamedTuple):
    """What an mjlab task's *runner* config contributes to browser playback.

    Everything else on it is training-only, or already inside the exported ONNX — the
    network shape and, when enabled, the observation normalizer, which rsl-rl bakes into
    the graph ahead of the MLP.
    """

    policy_obs_groups: tuple[str, ...] | None
    """``obs_groups["actor"]``: which observation group(s) the actor network reads."""

    clip_actions: float | None
    """The symmetric bound rsl-rl clamps the policy's raw output to."""


_NO_RUNNER_DEFAULTS = MjlabRunnerDefaults(policy_obs_groups=None, clip_actions=None)


def resolve_runner_defaults(task_id: str | None) -> MjlabRunnerDefaults:
    """Read an mjlab task's runner config for the two fields playback needs.

    All-``None`` when the task is unknown or mjlab is absent; the caller then falls back
    on the ``"actor"`` group name and leaves the action unclamped. Cheap enough to call
    per policy — ``load_rl_cfg`` is a registry lookup and a deepcopy.
    """
    if task_id is None:
        return _NO_RUNNER_DEFAULTS
    try:
        from mjlab.tasks.registry import load_rl_cfg
    except ImportError:
        return _NO_RUNNER_DEFAULTS
    try:
        rl_cfg = load_rl_cfg(task_id)
    except (KeyError, AttributeError):
        return _NO_RUNNER_DEFAULTS

    obs_groups = getattr(rl_cfg, "obs_groups", None)
    groups = obs_groups.get("actor") if isinstance(obs_groups, Mapping) else None
    clip = getattr(rl_cfg, "clip_actions", None)
    return MjlabRunnerDefaults(
        policy_obs_groups=tuple(groups) if groups else None,
        clip_actions=float(clip) if clip is not None else None,
    )


# ---------------------------------------------------------------------------
# Termination adaptation
# ---------------------------------------------------------------------------


def _adapt_term_func(
    func: Any, term_name: str | None = None
) -> TerminationBinding | Callable[..., Any]:
    """Resolve the function a termination term's ONNX graph is traced from.

    See :func:`_adapt_obs_func` — same resolution order (mjswan sentinel
    pass-through, then a ``register_termination`` override by function or term
    name, then mjlab's own function traced directly). *term_name* also covers
    closures (e.g. a term configured with a nameless ``_fn``), which can only be
    registered by their dict key.
    """
    if isinstance(func, TerminationBinding):
        return func
    name = getattr(func, "__name__", None)
    if name and name in _custom_term_registry:
        return _custom_term_registry[name]
    if term_name and term_name in _custom_term_registry:
        return _custom_term_registry[term_name]
    return func


def _sanitize_termination_params(params: dict[str, Any]) -> dict[str, Any]:
    """Strip mjlab-only termination params while keeping useful scope data."""
    if not params:
        return params

    result = {
        k: v for k, v in params.items() if k != "asset_cfg" and not _is_from_mjlab(v)
    }
    asset_cfg = params.get("asset_cfg")
    if not _is_from_mjlab(asset_cfg):
        return result

    entity_name = getattr(asset_cfg, "name", None)
    if entity_name:
        result["entity_name"] = entity_name

    body_names = getattr(asset_cfg, "body_names", None)
    if isinstance(body_names, (list, tuple)):
        result["body_names"] = [str(name) for name in body_names]
    elif isinstance(body_names, str):
        result["body_names"] = [body_names]

    return result


def _adapt_term_cfg(
    term: Any, term_name: str | None = None
) -> MjswanTerminationTermCfg:
    """Convert a single mjlab ``TerminationTermCfg`` to mjswan.

    See :func:`_adapt_obs_term` — params are only sanitized for the legacy
    ``TerminationBinding`` path; a func destined for ONNX tracing keeps its
    real params (e.g. ``asset_cfg``) unchanged.
    """
    raw_params = dict(getattr(term, "params", None) or {})
    func = _adapt_term_func(term.func, term_name=term_name)
    params = (
        _sanitize_termination_params(raw_params)
        if isinstance(func, TerminationBinding)
        else raw_params
    )
    return MjswanTerminationTermCfg(
        func=func,
        params=params,
        time_out=getattr(term, "time_out", False),
    )


def adapt_terminations(
    terminations: dict[str, Any] | None,
) -> dict[str, MjswanTerminationTermCfg] | None:
    """Adapt termination configs, converting mjlab types if detected."""
    if terminations is None:
        return None
    return {
        key: term
        if isinstance(term, MjswanTerminationTermCfg)
        else _adapt_term_cfg(term, term_name=key)
        if _is_from_mjlab(term)
        else term
        for key, term in terminations.items()
    }


# ---------------------------------------------------------------------------
# Command adaptation
# ---------------------------------------------------------------------------


def _adapt_command_cfg(term: Any) -> MjswanCommandTermConfig:
    """Convert a single mjlab ``CommandTermCfg`` to mjswan."""

    if isinstance(term, MjswanCommandTermConfig):
        return term

    class_name = type(term).__name__
    spec = _custom_command_registry.get(class_name)
    if spec is None:
        raise ValueError(
            f"No mjswan mapping for mjlab command config '{class_name}'. "
            f"Register one with mjswan.register_command()."
        )

    if spec.is_onnx_traced:
        assert spec.state_fields is not None and spec.command_field is not None
        ui = spec.ui(term) if callable(spec.ui) else spec.ui
        viz = spec.viz(term) if callable(spec.viz) else spec.viz
        return MjswanCommandTermConfig(
            term_name="OnnxCommand",
            pending_trace=PendingCommandTrace(
                mjlab_cfg=term,
                state_fields=spec.state_fields,
                command_field=spec.command_field,
                trace_override=spec.trace_override,
                ui=ui,
                viz=viz,
            ),
        )

    assert spec.serializer is not None
    serialized = dict(spec.serializer(term))
    # A native term may still own a reset-time graph for its randomization.
    reset_trace = spec.reset_trace(term) if spec.reset_trace is not None else None
    return MjswanCommandTermConfig(
        term_name=spec.ts_name,
        params=serialized,
        pending_reset_trace=(
            PendingResetTrace(func=reset_trace[0], params=reset_trace[1])
            if reset_trace is not None
            else None
        ),
    )


def adapt_commands(
    commands: Mapping[str, Any] | None,
) -> dict[str, MjswanCommandTermConfig] | None:
    """Adapt command configs, converting mjlab types if detected."""

    if commands is None:
        return None

    adapted: dict[str, MjswanCommandTermConfig] = {}
    for key, term in commands.items():
        if isinstance(term, MjswanCommandTermConfig):
            adapted[key] = term
            continue
        if _is_from_mjlab(term):
            try:
                adapted[key] = _adapt_command_cfg(term)
            except ValueError as exc:
                warnings.warn(
                    f"Skipping command term '{key}': {exc}",
                    category=RuntimeWarning,
                    stacklevel=2,
                )
            continue
        adapted[key] = term
    return adapted


# ---------------------------------------------------------------------------
# Action adaptation
# ---------------------------------------------------------------------------


_ACTION_CLASS_ALIASES: dict[str, str] = {
    # myosuite's muscle action cfg sits outside mjlab's hierarchy; translate it.
    "MyoMuscleActivationActionCfg": "MuscleActivationActionCfg",
}


def _adapt_action_cfg(term: Any) -> MjswanActionTermCfg | None:
    """Convert a single mjlab ``ActionTermCfg`` to mjswan.

    Looks up ``type(term).__name__`` on ``mjswan.envs.mdp.actions`` to
    find the corresponding mjswan class, then copies all matching
    dataclass fields automatically.

    Returns ``None`` if no mjswan equivalent exists; the caller is
    responsible for dropping the entry.
    """
    class_name = _ACTION_CLASS_ALIASES.get(type(term).__name__, type(term).__name__)
    mjswan_cls = getattr(_actions_module, class_name, None)

    if mjswan_cls is None or not (
        isinstance(mjswan_cls, type) and issubclass(mjswan_cls, MjswanActionTermCfg)
    ):
        warnings.warn(
            f"mjlab action type '{class_name}' has no mjswan equivalent. "
            f"It will be skipped.",
            category=RuntimeWarning,
            stacklevel=3,
        )
        return None

    # Copy all matching dataclass fields from the mjlab instance
    kwargs: dict[str, Any] = {}
    entity_name = getattr(term, "entity_name", None)
    for f in dataclasses.fields(mjswan_cls):
        if f.name == "unsupported_reason":
            continue
        val = getattr(term, f.name, dataclasses.MISSING)
        if val is not dataclasses.MISSING:
            kwargs[f.name] = val

    # Prefix with the entity name, as mjlab does, to match policy_joint_names at runtime.
    if entity_name and "actuator_names" in kwargs:
        raw = kwargs["actuator_names"]
        if isinstance(raw, (list, tuple)):
            kwargs["actuator_names"] = tuple(f"{entity_name}/{n}" for n in raw)

    return mjswan_cls(**kwargs)


def adapt_actions(
    actions: Mapping[str, Any] | None,
) -> Mapping[str, MjswanActionTermCfg] | None:
    """Adapt action configs, converting mjlab types if detected."""
    if actions is None:
        return None
    result: dict[str, MjswanActionTermCfg] = {}
    for key, term in actions.items():
        if isinstance(term, MjswanActionTermCfg):
            result[key] = term
        elif _is_from_mjlab(term) or type(term).__name__ in _ACTION_CLASS_ALIASES:
            adapted = _adapt_action_cfg(term)
            if adapted is not None:
                result[key] = adapted
        else:
            result[key] = term
    return result


def resolve_action_scales(
    actions: Mapping[str, MjswanActionTermCfg] | None,
    joint_names: list[str],
) -> None:
    """Resolve regex-pattern scale/offset dicts in action configs to literal joint names.

    mjlab stores per-joint scale as ``{".*_hip_joint": 0.37, ...}`` using regex
    patterns.  The browser runtime does exact string lookups, so patterns are
    expanded here against *joint_names* (the ordered list of joints the policy
    controls, prefixed with the entity name, e.g. ``"robot/left_hip_joint"``).

    Mutates the ``scale`` and ``offset`` fields of each action term in-place.
    """
    if not actions or not joint_names:
        return

    def _resolve(value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        resolved: dict[str, float] = {}
        for pattern, val in value.items():
            try:
                regex = re.compile(pattern)
            except re.error:
                resolved[pattern] = val
                continue
            for joint_name in joint_names:
                bare = joint_name.split("/")[-1] if "/" in joint_name else joint_name
                if regex.fullmatch(bare) or regex.fullmatch(joint_name):
                    resolved[joint_name] = val
        return resolved if resolved else value

    for term in actions.values():
        scale = getattr(term, "scale", None)
        if isinstance(scale, dict):
            setattr(term, "scale", _resolve(scale))
        offset = getattr(term, "offset", None)
        if isinstance(offset, dict):
            setattr(term, "offset", _resolve(offset))


# ---------------------------------------------------------------------------
# Event adaptation
# ---------------------------------------------------------------------------


def _adapt_event_func(
    func: Any, term_name: str | None = None
) -> EventBinding | Callable[..., Any]:
    """Resolve the function an event term's ONNX graph is traced from.

    See :func:`_adapt_obs_func` — same resolution order.
    """
    if isinstance(func, EventBinding):
        return func
    name = getattr(func, "__name__", None)
    if name and name in _custom_event_registry:
        return _custom_event_registry[name]
    if term_name and term_name in _custom_event_registry:
        return _custom_event_registry[term_name]
    return func


def _sanitize_event_params(params: dict[str, Any]) -> dict[str, Any]:
    """Strip mjlab-only event params while keeping joint scoping data."""
    if not params:
        return params

    result = {
        k: v for k, v in params.items() if k != "asset_cfg" and not _is_from_mjlab(v)
    }
    asset_cfg = params.get("asset_cfg")
    if not _is_from_mjlab(asset_cfg):
        return result

    entity_name = getattr(asset_cfg, "name", None)
    if entity_name:
        result["entity_name"] = entity_name

    joint_names = getattr(asset_cfg, "joint_names", None)
    if isinstance(joint_names, (list, tuple)):
        result["joint_names"] = [str(name) for name in joint_names]
    elif isinstance(joint_names, str):
        result["joint_names"] = [joint_names]

    joint_ids = getattr(asset_cfg, "joint_ids", None)
    if isinstance(joint_ids, (list, tuple)):
        result["joint_ids"] = [int(idx) for idx in joint_ids]

    return result


def _adapt_event_cfg(term: Any, term_name: str | None = None) -> MjswanEventTermCfg:
    """Convert a single mjlab ``EventTermCfg`` to mjswan.

    Covers ``reset``, ``interval``, and ``startup`` modes (ADR 0005 §4) — unlike
    the pre-ONNX design, non-reset modes are no longer dropped here. As in
    :func:`_adapt_obs_term`, params are only sanitized for the legacy
    ``EventBinding`` path; a func destined for ONNX tracing keeps its real
    params unchanged.
    """
    func = _adapt_event_func(term.func, term_name=term_name)
    raw_params = dict(getattr(term, "params", None) or {})
    params = (
        _sanitize_event_params(raw_params)
        if isinstance(func, EventBinding)
        else raw_params
    )
    return MjswanEventTermCfg(
        func=func,
        mode=getattr(term, "mode", "reset"),
        params=params,
        interval_range_s=getattr(term, "interval_range_s", None),
        is_global_time=getattr(term, "is_global_time", False),
        min_step_count_between_reset=getattr(
            term, "min_step_count_between_reset", None
        ),
    )


def adapt_events(
    events: Mapping[str, Any] | None,
) -> dict[str, MjswanEventTermCfg] | None:
    """Adapt event configs, converting mjlab types if detected.

    Returns a dict of mjswan ``EventTermCfg`` objects — serialization (including
    ONNX tracing) happens lazily at build time, once the scene's live env and
    output directory are known (same timing as observations/terminations).
    """
    if not events:
        return None
    result: dict[str, MjswanEventTermCfg] = {}
    for key, term in events.items():
        if isinstance(term, MjswanEventTermCfg):
            result[key] = term
        elif _is_from_mjlab(term):
            result[key] = _adapt_event_cfg(term, term_name=key)
        # non-mjlab, non-mjswan entries are skipped
    return result or None


__all__ = [
    "DEFAULT_OBS_GROUP_KEY",
    "adapt_events",
    "adapt_observations",
    "adapt_actions",
    "adapt_terminations",
    "resolve_action_scales",
    "MjlabRunnerDefaults",
    "resolve_runner_defaults",
]
