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

from typing import Any, Callable


def build_single_entity_trace_env(
    spec_fn: Callable[[], Any],
    *,
    entity_name: str = "robot",
    device: str = "cpu",
    zero_geom_margins: bool = True,
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

    Returns:
        A live ``mjlab.envs.ManagerBasedRlEnv``, already ``reset()``.
    """
    from mjlab.entity import EntityCfg
    from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
    from mjlab.scene import SceneCfg

    def _spec_fn():
        spec = spec_fn()
        if zero_geom_margins:
            for geom in spec.geoms:
                geom.margin = 0.0
        return spec

    # The keyframe is what the browser resets to (`mj_resetDataKeyframe`), so it is
    # also what `default_joint_pos` has to be: mjlab's default is `{".*": 0.0}`,
    # which bakes a zero default into every `*_rel` observation and leaves the
    # policy reading its whole stand pose as error from the first frame.
    keyframe_pos = _keyframe_joint_pos(_spec_fn())
    init_state = EntityCfg.InitialStateCfg()
    if keyframe_pos:
        init_state = EntityCfg.InitialStateCfg(joint_pos=keyframe_pos)
    entity_cfg = EntityCfg(spec_fn=_spec_fn, init_state=init_state)
    scene_cfg = SceneCfg(num_envs=1, entities={entity_name: entity_cfg})
    env_cfg = ManagerBasedRlEnvCfg(decimation=1, scene=scene_cfg)
    env = ManagerBasedRlEnv(env_cfg, device=device)
    env.reset()
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


__all__ = ["build_single_entity_trace_env"]
