"""Env-derived params for the terrain-related terminations.

``out_of_terrain_bounds`` and ``terrain_edge_reached`` are now declarative
built-ins (see ADR 0003), resolved by the mjlab adapter directly from
``mjswan.envs.mdp.terminations``.  The terrain generator constants
(``limit_x``/``limit_y``, ``half_x``/``half_y``) are still env-derived, so
this helper injects them into the mjlab ``TerminationTermCfg.params`` before
the build runs.
"""

from __future__ import annotations

from typing import Any


def register_custom_terminations(env_cfg: Any) -> None:
    """Inject terrain-generator-derived constants into env_cfg terminations.

    Mutates ``env_cfg.terminations["out_of_terrain_bounds"].params`` and
    ``env_cfg.terminations["terrain_edge_reached"].params`` in place so the
    matching declarative built-ins receive their per-build limits.  Safe to
    call when no terrain generator is present (no-op).
    """
    terrain = getattr(env_cfg.scene, "terrain", None)
    terrain_generator = getattr(terrain, "terrain_generator", None)
    is_generator = (
        terrain is not None
        and getattr(terrain, "terrain_type", None) == "generator"
        and terrain_generator is not None
    )
    if not is_generator:
        return

    out_term = env_cfg.terminations.get("out_of_terrain_bounds")
    if out_term is not None:
        out_params = dict(getattr(out_term, "params", None) or {})
        margin = float(out_params.get("margin", 0.3))
        half_x = 0.5 * terrain_generator.num_rows * terrain_generator.size[0]
        half_y = 0.5 * terrain_generator.num_cols * terrain_generator.size[1]
        out_params["limit_x"] = max(0.0, half_x - margin)
        out_params["limit_y"] = max(0.0, half_y - margin)
        out_term.params = out_params

    edge_term = env_cfg.terminations.get("terrain_edge_reached")
    if edge_term is not None:
        edge_params = dict(getattr(edge_term, "params", None) or {})
        edge_params["half_x"] = 0.5 * terrain_generator.size[0]
        edge_params["half_y"] = 0.5 * terrain_generator.size[1]
        edge_term.params = edge_params
