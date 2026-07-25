"""Task-side terrain spawn for the mjlab velocity-rough examples.

The generic mjlab reset events (``reset_joints_by_offset``, ``randomize_terrain``,
``reset_root_state_uniform``) are mjlab's own functions, traced directly to
ONNX (ADR 0005) — ``add_scene_mjlab``/``set_events`` adapt and hold them
un-serialized (:class:`mjswan.managers.event_manager.EventTermCfg`) until
build time, same timing as observations/terminations.

Patch-based spawning (placing the single browser env on a random flat terrain
tile) is NOT an mjlab term — it is a mjswan browser enhancement that reads
``terrainData`` (an engine capability outside bounded linear algebra).  So it
lives here, in the task, as a ``ts_src`` event:

- :func:`register_custom_events` registers the ``ResetRootStateFromFlatPatches``
  class from the local ``.ts`` file (this makes the build ``uses_custom_js``).
- :func:`apply_terrain_spawn` swaps the scene's ``reset_root_state_uniform``
  event term for the patch-based one, using the ``terrain_data`` that
  ``add_scene_mjlab`` attached to the scene config and the real ``pose_range``
  already carried on the mjlab term's own params.
"""

from __future__ import annotations

import os
from typing import Any

from mjswan import EventBinding, register_event
from mjswan.managers.event_manager import EventTermCfg

_EVENT_DIR = os.path.dirname(os.path.abspath(__file__))
_FLAT_PATCHES_TS = os.path.join(_EVENT_DIR, "ResetRootStateFromFlatPatches.ts")


def register_custom_events(env_cfg: Any | None = None) -> None:
    """Register the task-side patch-spawn event (ts_src)."""
    del env_cfg
    register_event(
        "reset_root_state_from_flat_patches",
        EventBinding(
            ts_name="ResetRootStateFromFlatPatches",
            ts_src=_FLAT_PATCHES_TS,
        ),
    )


def apply_terrain_spawn(scene_handle: Any) -> None:
    """Swap the scene's root-spawn reset for patch-based spawning.

    No-op unless ``add_scene_mjlab`` attached ``terrain_data`` with flat
    patches. Call after ``add_scene_mjlab`` for terrain-generator tasks.
    """
    config = scene_handle._config
    terrain_data = getattr(config, "terrain_data", None) or {}
    flat_patches = terrain_data.get("flat_patches", {})
    events = getattr(config, "events", None)
    if not flat_patches or not events:
        return
    patch_name = "spawn" if "spawn" in flat_patches else next(iter(flat_patches))
    for key, event in events.items():
        func = event.func
        if getattr(func, "__name__", None) != "reset_root_state_uniform":
            continue
        pose_range = dict(event.params.get("pose_range") or {})
        events[key] = EventTermCfg(
            func=EventBinding(
                ts_name="ResetRootStateFromFlatPatches",
                ts_src=_FLAT_PATCHES_TS,
            ),
            mode=event.mode,
            params={"pose_range": pose_range, "patch_name": patch_name},
        )
        break
