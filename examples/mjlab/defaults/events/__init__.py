"""Task-side terrain spawn for the mjlab velocity-rough examples.

mjlab's own reset events (``reset_joints_by_offset``, ``reset_root_state_uniform``,
``randomize_terrain``) need nothing from this package — they are mjlab's functions,
traced to ONNX at build time (ADR 0005).

What does live here is the one spawn behaviour mjlab has no term for. mjlab trains
with many parallel envs spread across the terrain, so per-env spawn jitter is small;
the browser has a single env, so spawning it on a random flat patch is what covers
the terrain at all. :func:`apply_terrain_spawn` swaps the scene's
``reset_root_state_uniform`` for :func:`reset_root_state_on_flat_patch`.

That term is an ordinary traced body, not a hand-written TS class. The patch table
comes off the terrain generator at build time, so it bakes into the graph as a
constant and the two draws (which patch, which yaw) become the graph's ``rand``
input, fed from the orchestrator's seeded PRNG — where the retired
``ResetRootStateFromFlatPatches.ts`` used ``Math.random()`` and so could not replay.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

from mjswan.managers.event_manager import EventTermCfg


def reset_root_state_on_flat_patch(
    env: Any,
    env_ids: Any,
    *,
    asset_cfg: Any,
    patches: list[list[float]],
    yaw_range: tuple[float, float] = (-math.pi, math.pi),
) -> None:
    """Place the root on a uniformly-chosen flat terrain patch, at a random yaw.

    ``patches`` is the terrain generator's sampled ``(x, y, z)`` list, fixed once the
    terrain is built, so it bakes into the graph rather than being read at runtime.
    """
    asset = env.scene[asset_cfg.name]
    root_pos_w = asset.data.root_link_pos_w
    device = root_pos_w.device

    table = torch.tensor(patches, dtype=torch.float, device=device)
    count = table.shape[0]
    # A uniform draw scaled to an index, rather than a randint: `sample_uniform` is what
    # the tracer records, so the browser redraws it from the same seeded stream.
    pick = sample_uniform(0.0, 1.0, (1,), device=device)
    index = torch.clamp((pick * count).long(), max=count - 1)
    patch = table.index_select(0, index)

    # x/y come from the patch; z is the patch height plus the root's standing height.
    root_pos = torch.cat(
        [patch[:, 0:2], patch[:, 2:3] + root_pos_w[:, 2:3]],
        dim=-1,
    )
    yaw = sample_uniform(yaw_range[0], yaw_range[1], (1,), device=device)
    zeros = torch.zeros_like(yaw)
    root_quat = quat_mul(
        quat_from_euler_xyz(zeros, zeros, yaw), asset.data.root_link_quat_w
    )
    asset.write_root_link_pose_to_sim(
        torch.cat([root_pos, root_quat], dim=-1), env_ids=env_ids
    )


def apply_terrain_spawn(scene_handle: Any) -> None:
    """Swap the scene's root-spawn reset for patch-based spawning.

    No-op unless ``add_scene_mjlab`` attached ``terrain_data`` with flat patches. Call
    after ``add_scene_mjlab`` for terrain-generator tasks.
    """
    config = scene_handle._config
    terrain_data = getattr(config, "terrain_data", None) or {}
    flat_patches = terrain_data.get("flat_patches", {})
    events = getattr(config, "events", None)
    if not flat_patches or not events:
        return
    patch_name = "spawn" if "spawn" in flat_patches else next(iter(flat_patches))
    patches = flat_patches[patch_name]
    if not patches:
        return
    for key, event in events.items():
        func = event.func
        if getattr(func, "__name__", None) != "reset_root_state_uniform":
            continue
        # The mjlab term's own yaw range, so replacing the spawn does not also widen it.
        pose_range = dict(event.params.get("pose_range") or {})
        entity = getattr(event.params.get("asset_cfg"), "name", None) or "robot"
        events[key] = EventTermCfg(
            func=reset_root_state_on_flat_patch,
            mode=event.mode,
            params={
                "asset_cfg": SceneEntityCfg(entity),
                "patches": patches,
                "yaw_range": tuple(pose_range.get("yaw", (-math.pi, math.pi))),
            },
        )
        break
