"""Custom observation registrations for the MuscleMimic Fullbody demo.

Import this module before calling ``builder.build()`` to register all
myoMimicFullbody-v0 observation terms.

The mimic task uses Python closures (all named ``_fn``) so the adapter
cannot resolve them by function name.  We register by *term name* instead,
which the adapter uses as a fallback when the function-name lookup fails.
"""

from __future__ import annotations

from pathlib import Path

from myosuite.integrations.musclemimic.fullbody_model import (
    FULLBODY_BODY2SITES_FOR_MIMIC,
)

from mjswan.envs.mdp.observations import ObsFunc, register_obs_func

_OBS_DIR = Path(__file__).resolve().parent

FPS = 100
SITE_NAMES: list[str] = list(FULLBODY_BODY2SITES_FOR_MIMIC.values())
BODY_NAMES: list[str] = [
    f"mimic_fullbody_robot/{body}" for body in FULLBODY_BODY2SITES_FOR_MIMIC.keys()
]

register_obs_func(
    "qpos",
    ObsFunc(
        ts_name="MimicQpos",
        ts_src=str(_OBS_DIR / "MimicObservations.ts"),
    ),
)

register_obs_func(
    "qvel",
    ObsFunc(
        ts_name="MimicQvel",
        ts_src=str(_OBS_DIR / "MimicObservations.ts"),
        defaults={"fps": FPS},
    ),
)

register_obs_func(
    "act",
    ObsFunc(
        ts_name="MimicAct",
        ts_src=str(_OBS_DIR / "MimicObservations.ts"),
    ),
)

register_obs_func(
    "mimic_site_pos",
    ObsFunc(
        ts_name="MimicSitePos",
        ts_src=str(_OBS_DIR / "MimicObservations.ts"),
        defaults={"site_names": SITE_NAMES, "body_names": BODY_NAMES},
    ),
)

register_obs_func(
    "mimic_site_target",
    ObsFunc(
        ts_name="MimicSiteTarget",
        ts_src=str(_OBS_DIR / "MimicObservations.ts"),
        defaults={"site_names": SITE_NAMES, "fps": FPS},
    ),
)

register_obs_func(
    "mimic_site_err",
    ObsFunc(
        ts_name="MimicSiteErr",
        ts_src=str(_OBS_DIR / "MimicObservations.ts"),
        defaults={
            "site_names": SITE_NAMES,
            "body_names": BODY_NAMES,
            "fps": FPS,
        },
    ),
)

register_obs_func(
    "clip_ref_qpos",
    ObsFunc(
        ts_name="MimicClipRefQpos",
        ts_src=str(_OBS_DIR / "MimicObservations.ts"),
        defaults={"fps": FPS},
    ),
)

register_obs_func(
    "clip_ref_qvel",
    ObsFunc(
        ts_name="MimicClipRefQvel",
        ts_src=str(_OBS_DIR / "MimicObservations.ts"),
        defaults={"fps": FPS},
    ),
)

register_obs_func(
    "clip_phase",
    ObsFunc(
        ts_name="MimicClipPhase",
        ts_src=str(_OBS_DIR / "MimicObservations.ts"),
        defaults={"fps": FPS},
    ),
)

register_obs_func(
    "mimic_lookahead",
    ObsFunc(
        ts_name="MimicLookahead",
        ts_src=str(_OBS_DIR / "MimicObservations.ts"),
        defaults={
            "k": 5,
            "stride": 20,
            "fps": FPS,
            "n_clip_sites": 17,
            "site_names": SITE_NAMES,
        },
    ),
)
