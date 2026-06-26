"""Task-side terrain spawn for the mjlab velocity-rough examples.

The generic mjlab reset events (``reset_joints_by_offset``, ``randomize_terrain``,
``reset_root_state_uniform``) are declarative built-ins in
``mjswan.envs.mdp.events`` (ADR 0003) and are auto-wired by ``add_mjlab_scene``.

Patch-based spawning (placing the single browser env on a random flat terrain
tile) is NOT an mjlab term — it is a mjswan browser enhancement that reads
``terrainData`` (an engine capability outside bounded linear algebra).  So it
lives here, in the task, as a ``ts_src`` event:

- :func:`register_custom_events` registers the ``ResetRootStateFromFlatPatches``
  class from the local ``.ts`` file (this makes the build ``uses_custom_js``).
- :func:`apply_terrain_spawn` swaps the scene's declarative root-spawn event for
  the patch-based one, using the ``terrain_data`` that ``add_mjlab_scene``
  attached to the scene config.
"""

from __future__ import annotations

import os
from typing import Any

from mjswan import EventFunc, register_event_func

_EVENT_DIR = os.path.dirname(os.path.abspath(__file__))
_FLAT_PATCHES_TS = os.path.join(_EVENT_DIR, "ResetRootStateFromFlatPatches.ts")


def register_custom_events(env_cfg: Any | None = None) -> None:
    """Register the task-side patch-spawn event (ts_src)."""
    del env_cfg
    register_event_func(
        "reset_root_state_from_flat_patches",
        EventFunc(
            ts_name="ResetRootStateFromFlatPatches",
            ts_src=_FLAT_PATCHES_TS,
        ),
    )


def _root_spawn_params(event: dict[str, Any]) -> dict[str, Any] | None:
    """Spawn params if *event* is a root-state spawn reset, else ``None``.

    Recognizes the declarative DSL spawn event (a ``{"kind": "event", ...}``
    envelope with a ``freejoint_pos`` / ``freejoint_yaw`` mutation) and the
    legacy named ``ResetRootStateUniform`` event, reconstructing ``pose_range``
    from the DSL mutation samples.
    """
    if event.get("name") == "ResetRootStateUniform":
        return dict(event.get("params") or {})
    if event.get("kind") != "event":
        return None
    mutations = event.get("mutations") or []
    if not any(
        m.get("target") in ("freejoint_pos", "freejoint_yaw") for m in mutations
    ):
        return None
    pose_range: dict[str, Any] = {}
    for m in mutations:
        sample = m.get("sample") or {}
        if m.get("target") == "freejoint_pos" and sample.get("dist") == "uniform_xyz":
            for axis in ("x", "y", "z"):
                if axis in sample:
                    pose_range[axis] = sample[axis]
        elif m.get("target") == "freejoint_yaw":
            pose_range["yaw"] = [sample.get("low", 0.0), sample.get("high", 0.0)]
    return {"pose_range": pose_range}


def apply_terrain_spawn(scene_handle: Any) -> None:
    """Swap the scene's root-spawn reset for patch-based spawning.

    No-op unless ``add_mjlab_scene`` attached ``terrain_data`` with flat
    patches.  Call after ``add_mjlab_scene`` for terrain-generator tasks.
    """
    config = scene_handle._config
    terrain_data = getattr(config, "terrain_data", None) or {}
    flat_patches = terrain_data.get("flat_patches", {})
    events = getattr(config, "events", None)
    if not flat_patches or not events:
        return
    patch_name = "spawn" if "spawn" in flat_patches else next(iter(flat_patches))
    for i, event in enumerate(events):
        params = _root_spawn_params(event)
        if params is None:
            continue
        params["patch_name"] = patch_name
        events[i] = {"name": "ResetRootStateFromFlatPatches", "params": params}
