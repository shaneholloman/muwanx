"""Build a minimal live env for ONNX tracing (ADR 0005) of non-mjlab scenes.

``add_scene_mjlab`` gets its tracing env for free (mjlab already builds a full
task env). A plain ``add_scene()`` scene (a robot XML with no mjlab task) has
no env at all — but ONNX tracing (:mod:`mjswan.compile`) only ever needs
``env.scene[name].data.<field>`` (and, for write-side terms, entity write
methods), not a fully-configured RL task. This module builds exactly that
much env by reusing mjlab's own ``Entity``/``Scene`` machinery directly — the
same real math ``Entity.data`` uses for a full mjlab task — rather than
reimplementing entity-frame kinematics a second time (the reimplementation
tax ADR 0005 exists to eliminate).
"""

from __future__ import annotations

import contextlib
import io
import re
from typing import Any, Callable


def _required_capacity(message: str, name: str) -> int | None:
    match = re.search(rf"{name} overflow \({name} must be >= (\d+)\)", message)
    return None if match is None else int(match.group(1))


def _next_capacity(required: int) -> int:
    return required + max(32, required // 8)


def _quiet_warp_module_loads() -> None:
    """Drop warp's per-kernel ``Module … load on device`` lines (88 of a 7-task build).

    ``log_level`` is the supported switch (``config.quiet`` is deprecated) and leaves
    warnings through. Only the default level is nudged, so
    ``warp.config.log_level = warp.LOG_DEBUG`` before the build brings them back.
    """
    import warp

    if warp.config.log_level == warp.LOG_INFO:
        warp.config.log_level = warp.LOG_WARNING


def build_mjlab_env(env_cfg: Any, *, device: str = "cpu") -> Any:
    """Build a ``ManagerBasedRlEnv``, growing ``nconmax``/``njmax`` until it fits.

    mjlab sizes those buffers from the task config, which is tuned for the training
    scene; a config re-used here (a single env, a different terrain patch) can need
    more, and mujoco_warp only reports how much once the build fails.

    mjlab's manager tables (~120 lines per env, printed unconditionally) are held back so
    they do not bury the build's progress, and replayed if the build fails.
    """
    from mjlab.envs import ManagerBasedRlEnv

    _quiet_warp_module_loads()
    tables = io.StringIO()
    while True:
        try:
            with contextlib.redirect_stdout(tables):
                return ManagerBasedRlEnv(cfg=env_cfg, device=device)
        except ValueError as exc:
            nconmax = _required_capacity(str(exc), "nconmax")
            njmax = _required_capacity(str(exc), "njmax")
            if nconmax is None and njmax is None:
                print(tables.getvalue(), end="")
                raise
            if nconmax is not None:
                env_cfg.sim.nconmax = _next_capacity(nconmax)
            if njmax is not None:
                env_cfg.sim.njmax = _next_capacity(njmax)
            tables.seek(0)
            tables.truncate(0)
        except Exception:
            print(tables.getvalue(), end="")
            raise


class TraceCommandManager:
    """Stand-in ``CommandManager`` serving trace-time values for browser-side commands.

    A traced term may read a command the *browser* owns and the trace env cannot
    build: a ``UiCommand`` (a slider has no Python side at all), or a native
    ``TrackingCommand`` whose clip lookup is data rather than math. The tracer only
    needs each read to hand back a real tensor of the right shape — the values bake
    nothing, they become graph *inputs* the runtime serves from the live command
    (``getStateField``). So a plain object with the right tensor attributes is
    enough, and this is what makes ``env.command_manager.get_term(name)`` find it.

    ``get_command(name)`` returns the term's ``command`` attribute, as mjlab's own
    ``CommandManager`` does.
    """

    def __init__(self, terms: dict[str, Any]):
        self._terms = dict(terms)

    def get_term(self, name: str) -> Any:
        if name not in self._terms:
            raise KeyError(
                f"Trace env has no command {name!r}; it knows "
                f"{sorted(self._terms)}. Pass it to "
                "`build_single_entity_trace_env(commands=...)`."
            )
        return self._terms[name]

    def get_command(self, name: str) -> Any:
        return self.get_term(name).command


def build_single_entity_trace_env(
    spec_fn: Callable[[], Any],
    *,
    entity_name: str = "robot",
    device: str = "cpu",
    zero_geom_margins: bool = True,
    commands: dict[str, Any] | None = None,
) -> Any:
    """Build a minimal single-entity ``ManagerBasedRlEnv`` for ONNX tracing.

    No observations/actions/terminations/commands are configured — this env
    is never stepped, only used as the tracer's ``env.scene[entity_name]``
    read/write target. Returns a live, ``reset()``-ed env; pass it to
    :meth:`mjswan.SceneHandle.set_trace_env`.

    Args:
        spec_fn: Zero-arg callable returning a fresh ``mujoco.MjSpec`` for the
            entity (mjlab's ``EntityCfg.spec_fn`` contract — called each time
            the spec is needed, so it must not share mutable state across
            calls).
        entity_name: The key ``env.scene[entity_name]`` resolves to. Match
            whatever your traced functions use as their ``entity_name``/
            ``asset_cfg.name``.
        device: Torch device for the entity's tensors.
        zero_geom_margins: Zero every geom's contact margin before compiling.
            mujoco_warp's collision backend rejects some robot XMLs with
            non-zero geom margins (``NotImplementedError: ... has non-zero
            margin ... enabled``); margins only affect contact dynamics, which
            this env never simulates (only entity kinematic state is read),
            so zeroing them is safe here. Set ``False`` if your spec is
            already margin-clean and you want the geoms untouched for some
            other reason.
        commands: Trace-time stand-ins for commands the browser owns, keyed by the
            name traced terms read (``env.command_manager.get_term(name)``). See
            :class:`TraceCommandManager`; omit when no term reads a command.

    Returns:
        A live ``mjlab.envs.ManagerBasedRlEnv``, already ``reset()``.
    """
    from mjlab.entity import EntityCfg
    from mjlab.envs import ManagerBasedRlEnvCfg
    from mjlab.scene import SceneCfg

    def _spec_fn():
        spec = spec_fn()
        if zero_geom_margins:
            for geom in spec.geoms:
                geom.margin = 0.0
        return spec

    # The browser resets to the keyframe, so `default_joint_pos` must match it — mjlab's
    # `{".*": 0.0}` would bake a zero default into every `*_rel` observation.
    keyframe_pos = _keyframe_joint_pos(_spec_fn())
    init_state = EntityCfg.InitialStateCfg()
    if keyframe_pos:
        init_state = EntityCfg.InitialStateCfg(joint_pos=keyframe_pos)
    entity_cfg = EntityCfg(spec_fn=_spec_fn, init_state=init_state)
    scene_cfg = SceneCfg(num_envs=1, entities={entity_name: entity_cfg})
    env_cfg = ManagerBasedRlEnvCfg(decimation=1, scene=scene_cfg)
    # Through `build_mjlab_env` for its quieting; warp's kernel loads and mjlab's tables
    # are noise here too.
    env = build_mjlab_env(env_cfg, device=device)
    env.reset()
    if commands:
        # After reset(), since mjlab builds its own empty manager during construction.
        env.command_manager = TraceCommandManager(commands)
    return env


def _keyframe_joint_pos(spec: Any) -> dict[str, float]:
    """Per-joint positions from the model's first keyframe, as ``EntityCfg`` wants them.

    mjlab has its own way to say this — ``init_state.joint_pos = None`` — but that
    path builds ``default_joint_pos`` from the raw float64 keyframe and then fails
    writing it into float32 ``qpos``. Naming the joints avoids it, and is explicit
    about the one-dof assumption ``InitialStateCfg.joint_pos`` makes anyway (its
    values are scalars, so a ball joint has no representation there).

    Returns an empty dict when the model has no keyframe, which leaves mjlab's own
    ``{".*": 0.0}`` default in place — correct for a model whose zero pose *is* its
    rest pose.
    """
    import re

    import mujoco

    if not spec.keys:
        return {}
    model = spec.compile()
    qpos = model.key_qpos[0]
    positions: dict[str, float] = {}
    for joint in range(model.njnt):
        if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
            continue  # the root pose, not a joint position
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        # Keys are regexes to mjlab (`resolve_expr`), and a joint name is not one.
        positions[re.escape(name)] = float(qpos[model.jnt_qposadr[joint]])
    return positions


__all__ = ["TraceCommandManager", "build_single_entity_trace_env"]
