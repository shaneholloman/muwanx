"""Mjlab-specific custom command registrations for mjswan examples.

Import this module before calling ``builder.build()`` to register the
custom command term classes used in mjlab tasks.
"""

import os
from dataclasses import asdict
from typing import Any

from mjswan import CommandBinding, register_command

_CMD_DIR = os.path.dirname(os.path.abspath(__file__))

register_command(
    "LiftingCommandCfg",
    CommandBinding(
        ts_name="LiftingCommand",
        ts_src=os.path.join(_CMD_DIR, "LiftingCommand.ts"),
        serializer=asdict,
    ),
)


def _serialize_uniform_velocity_command(cfg: Any) -> dict[str, Any]:
    """Convert mjlab's sampled velocity command to manual browser sliders."""

    ranges = cfg.ranges
    return {
        "ui": {
            "inputs": [
                {
                    "type": "slider",
                    "name": "lin_vel_x",
                    "label": "Forward Velocity",
                    "min": ranges.lin_vel_x[0],
                    "max": ranges.lin_vel_x[1],
                    "step": 0.05,
                    "default": max(ranges.lin_vel_x[0], min(0.5, ranges.lin_vel_x[1])),
                },
                {
                    "type": "slider",
                    "name": "lin_vel_y",
                    "label": "Lateral Velocity",
                    "min": ranges.lin_vel_y[0],
                    "max": ranges.lin_vel_y[1],
                    "step": 0.05,
                    "default": max(ranges.lin_vel_y[0], min(0.0, ranges.lin_vel_y[1])),
                },
                {
                    "type": "slider",
                    "name": "ang_vel_z",
                    "label": "Yaw Rate",
                    "min": ranges.ang_vel_z[0],
                    "max": ranges.ang_vel_z[1],
                    "step": 0.05,
                    "default": max(ranges.ang_vel_z[0], min(0.0, ranges.ang_vel_z[1])),
                },
            ]
        }
    }


register_command(
    "UniformVelocityCommandCfg",
    CommandBinding(
        ts_name="UiCommand",
        serializer=_serialize_uniform_velocity_command,
    ),
)
