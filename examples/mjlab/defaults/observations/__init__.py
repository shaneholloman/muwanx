"""Mjlab-specific custom observation registrations for mjswan examples.

Import this module before calling ``builder.build()`` to register the
custom observation classes used in mjlab tasks.

Per ADR 0005, mjswan traces each task's own observation functions to ONNX
directly against the scene's live env — mjlab's ``pole_angle_cos_sin``
(Cartpole), ``ee_to_object_distance``/``object_to_goal_distance``
(Lift-Cube manipulation) all trace exactly like ``joint_pos_rel`` or any
other mjlab-library function. There is nothing to reimplement or register
here for those, and nothing to reshape either: the task's own actor group goes
straight to ``add_policy_wandb(observations=...)``, which is why the
``get_policy_observations`` wrapper that used to live here is gone.

Only ``height_scan`` needs a registration, and only because its parameters come
off the task's sensor config rather than the function.
"""

import os
from typing import Any

from mjswan import ObservationBinding, register_observation

_OBS_DIR = os.path.dirname(os.path.abspath(__file__))


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
    register_observation(
        "height_scan",
        ObservationBinding(
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
