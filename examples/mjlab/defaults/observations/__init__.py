"""Mjlab-specific custom observation registrations for mjswan examples.

Import this module before calling ``builder.build()`` to register the
custom observation classes used in mjlab tasks.
"""

import os
from typing import Any

import mjswan
from mjswan import ObsFunc, register_obs_func
from mjswan.envs.mdp import observations as obs_fns

_OBS_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Task-specific declarative observations (Lift-Cube).
#
# These are DSL builders (ADR 0003): mjlab-style functions taking a symbolic
# ``env`` and returning a composition expression.  Registering them here keeps
# task-specific terms out of the core mjswan library — the core only carries
# generic terms.  Being ts_src-free, they stay declarative / Cloud-safe.
# ---------------------------------------------------------------------------


def ee_to_object_distance(
    env,
    *,
    object_name: str,
    site_name: str | None = None,
    entity_name: str = "robot",
    **_,
):
    """EE→object distance in the robot base frame (mjlab manipulation)."""
    from mjswan.dsl import body_pos, quat_apply_inv, site_pos

    base_quat = env.entity(entity_name).data.root_link_quat_w
    return quat_apply_inv(base_quat, body_pos(object_name) - site_pos(site_name or ""))


def object_to_goal_distance(
    env, *, object_name: str, command_name: str, entity_name: str = "robot", **_
):
    """Object→goal distance in the robot base frame (mjlab manipulation)."""
    from mjswan.dsl import body_pos, command_value, quat_apply_inv

    base_quat = env.entity(entity_name).data.root_link_quat_w
    return quat_apply_inv(
        base_quat, command_value(command_name) - body_pos(object_name)
    )


register_obs_func("ee_to_object_distance", ee_to_object_distance)
register_obs_func("object_to_goal_distance", object_to_goal_distance)
register_obs_func("pole_angle_cos_sin", obs_fns.joint_pos_cos_sin)


def get_policy_observations(task_id: str, env_cfg: Any) -> dict[str, Any]:
    """Return browser-safe policy observations for the given mjlab task."""
    if task_id not in {"Mjlab-Cartpole-Balance", "Mjlab-Cartpole-Swingup"}:
        return {"policy": env_cfg.observations["actor"]}

    actor = env_cfg.observations["actor"]
    terms = actor.terms
    return {
        "policy": mjswan.ObservationGroupCfg(
            terms={
                "cart_pos": mjswan.ObservationTermCfg(
                    func=obs_fns.joint_pos_rel,
                    params={
                        "entity_name": "cartpole",
                        "joint_names": ["slider"],
                    },
                    scale=getattr(terms["cart_pos"], "scale", None),
                    clip=getattr(terms["cart_pos"], "clip", None),
                    history_length=getattr(terms["cart_pos"], "history_length", 0),
                ),
                "pole_angle": mjswan.ObservationTermCfg(
                    func=obs_fns.joint_pos_cos_sin,
                    params={"joint_name": "cartpole/hinge_1"},
                    scale=getattr(terms["pole_angle"], "scale", None),
                    clip=getattr(terms["pole_angle"], "clip", None),
                    history_length=getattr(terms["pole_angle"], "history_length", 0),
                ),
                "cart_vel": mjswan.ObservationTermCfg(
                    func=obs_fns.joint_vel_rel,
                    params={
                        "entity_name": "cartpole",
                        "joint_names": ["slider"],
                    },
                    scale=getattr(terms["cart_vel"], "scale", None),
                    clip=getattr(terms["cart_vel"], "clip", None),
                    history_length=getattr(terms["cart_vel"], "history_length", 0),
                ),
                "pole_vel": mjswan.ObservationTermCfg(
                    func=obs_fns.joint_vel_rel,
                    params={
                        "entity_name": "cartpole",
                        "joint_names": ["hinge_1"],
                    },
                    scale=getattr(terms["pole_vel"], "scale", None),
                    clip=getattr(terms["pole_vel"], "clip", None),
                    history_length=getattr(terms["pole_vel"], "history_length", 0),
                ),
            },
            concatenate_terms=getattr(actor, "concatenate_terms", True),
            enable_corruption=getattr(actor, "enable_corruption", False),
            history_length=getattr(actor, "history_length", None),
        )
    }


def register_custom_observations(env_cfg: Any) -> None:
    """Register env_cfg-dependent observations (e.g. height_scan)."""
    terrain_scan = next(
        (
            sensor
            for sensor in (env_cfg.scene.sensors or ())
            if getattr(sensor, "name", None) == "terrain_scan"
        ),
        None,
    )
    if terrain_scan is None:
        return

    frame = terrain_scan.frame
    frame_ref_name = (
        f"{frame.entity}/{frame.name}" if getattr(frame, "entity", None) else frame.name
    )
    pattern = terrain_scan.pattern
    register_obs_func(
        "height_scan",
        ObsFunc(
            ts_name="HeightScan",
            ts_src=os.path.join(_OBS_DIR, "HeightScan.ts"),
            defaults={
                "frame_type": frame.type,
                "frame_ref_name": frame_ref_name,
                "ray_alignment": terrain_scan.ray_alignment,
                "pattern_size": list(pattern.size),
                "pattern_resolution": float(pattern.resolution),
                "pattern_direction": list(pattern.direction),
                "max_distance": float(terrain_scan.max_distance),
                "terrain_body_name": "terrain",
            },
        ),
    )
