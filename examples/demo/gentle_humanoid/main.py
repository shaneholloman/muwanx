"""Gentle Humanoid tracking policy demo."""

from __future__ import annotations

import io
import os
import subprocess
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import onnx
import yaml

import mjswan
from mjswan.envs.mdp.actions import JointPositionActionCfg
from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

from .dsl_terms import (
    gentle_humanoid_target_joint_pos,
    gentle_humanoid_target_projected_gravity,
    gentle_humanoid_target_root_z,
    gentle_humanoid_tracking,
)

HERE = Path(__file__).resolve().parent
GENTLE_HUMANOID_REPO_URL = os.getenv(
    "MJSWAN_GENTLE_HUMANOID_REPO_URL",
    "https://github.com/Axellwppr/motion_tracking.git",
)
GENTLE_HUMANOID_REPO_COMMIT = os.getenv(
    "MJSWAN_GENTLE_HUMANOID_REPO_COMMIT",
    "5684a5e192cf5fe803bc83fc863e75e45e026a40",
)
GENTLE_HUMANOID_DEP_REPO = HERE / ".dep" / "motion_tracking"

COMMAND_TS = HERE / "commands" / "GentleHumanoidTrackingCommand.ts"
OBS_TS = HERE / "observations" / "GentleHumanoidObservations.ts"


def _run_git(args: list[str], cwd: Path) -> None:
    env = os.environ.copy()
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    try:
        subprocess.run(["git", *args], cwd=cwd, env=env, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("git is required to fetch Gentle Humanoid assets") from exc
    except subprocess.CalledProcessError as exc:
        command = " ".join(["git", *args])
        raise RuntimeError(f"Failed to run `{command}` in {cwd}") from exc


def _ensure_gentle_humanoid_repo() -> Path:
    repo = GENTLE_HUMANOID_DEP_REPO
    if not (repo / ".git").exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        _run_git(["clone", GENTLE_HUMANOID_REPO_URL, str(repo)], cwd=HERE)
    else:
        _run_git(["remote", "set-url", "origin", GENTLE_HUMANOID_REPO_URL], cwd=repo)
        _run_git(["fetch", "--tags", "origin"], cwd=repo)
    _run_git(["checkout", "--detach", GENTLE_HUMANOID_REPO_COMMIT], cwd=repo)
    return repo


def _resolve_gentle_humanoid_root() -> Path:
    configured_root = os.getenv("MJSWAN_GENTLE_HUMANOID_ROOT")
    if configured_root:
        return Path(configured_root).expanduser()
    return _ensure_gentle_humanoid_repo() / "sim2real"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def _map_by_name(
    values: list[float],
    source_names: list[str],
    target_names: list[str],
    *,
    default: float = 0.0,
) -> list[float]:
    by_name = {name: float(values[i]) for i, name in enumerate(source_names)}
    return [by_name.get(name, default) for name in target_names]


def _default_motion_npz(tracking_cfg: dict[str, Any]) -> bytes:
    clips = tracking_cfg.get("motion_clips", [])
    if not isinstance(clips, list):
        raise ValueError("tracking.yaml motion_clips must be a list")
    default_clip = next((clip for clip in clips if clip.get("name") == "default"), None)
    if default_clip is None:
        raise ValueError("tracking.yaml motion_clips must include a default clip")

    root_quat_wxyz = np.asarray(default_clip["root_quat"], dtype=np.float32)
    root_rot_xyzw = np.asarray(
        [root_quat_wxyz[1], root_quat_wxyz[2], root_quat_wxyz[3], root_quat_wxyz[0]],
        dtype=np.float32,
    ).reshape(1, 4)
    payload = io.BytesIO()
    np.savez(
        payload,
        fps=np.asarray(50.0, dtype=np.float32),
        root_pos=np.asarray(default_clip["root_pos"], dtype=np.float32).reshape(1, 3),
        root_rot=root_rot_xyzw,
        dof_pos=np.asarray(default_clip["joint_pos"], dtype=np.float32).reshape(1, -1),
        joint_names=np.asarray(tracking_cfg["dataset_joint_names"], dtype="S"),
    )
    return payload.getvalue()


def _ensure_default_motion_file(tracking_cfg: dict[str, Any]) -> Path:
    path = HERE / ".dep" / "generated" / "gentle_humanoid_default_motion.npz"
    payload = _default_motion_npz(tracking_cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_bytes() != payload:
        path.write_bytes(payload)
    return path


def _register_gentle_humanoid_extensions() -> dict[str, mjswan.ObservationBinding]:
    mjswan.register_command(
        "GentleHumanoidTrackingCommandCfg",
        mjswan.CommandBinding(
            ts_name="GentleHumanoidTrackingCommand",
            serializer=lambda _cfg: {},
            ts_src=str(COMMAND_TS),
        ),
    )

    # The four motion-command-coupled obs are now declarative DSL builders
    # (see dsl_terms.py, issue #79); only the remaining terms stay ts_src.
    obs_names = {
        "boot": "GentleHumanoidBootIndicator",
        "compliance": "GentleHumanoidComplianceFlagObs",
        "root_ang_vel": "GentleHumanoidRootAngVelBHistory",
        "projected_gravity": "GentleHumanoidProjectedGravityBHistory",
        "joint_pos": "GentleHumanoidJointPosHistory",
        "joint_vel": "GentleHumanoidJointVelHistory",
        "prev_actions": "GentleHumanoidPrevActions",
    }
    obs_funcs = {
        key: mjswan.ObservationBinding(ts_name=ts_name, ts_src=str(OBS_TS))
        for key, ts_name in obs_names.items()
    }
    for key, obs_func in obs_funcs.items():
        mjswan.register_observation(f"gentle_humanoid_{key}", obs_func)
    return obs_funcs


def setup_builder() -> mjswan.Builder:
    """Create the builder for the Gentle Humanoid tracking demo."""
    gentle_humanoid_root = _resolve_gentle_humanoid_root()
    if not gentle_humanoid_root.exists():
        raise FileNotFoundError(
            f"Gentle Humanoid asset root not found: {gentle_humanoid_root}. "
            "Set MJSWAN_GENTLE_HUMANOID_ROOT to override it."
        )

    tracking_cfg = _load_yaml(gentle_humanoid_root / "config" / "tracking.yaml")
    controller_cfg = _load_yaml(gentle_humanoid_root / "config" / "controller.yaml")
    obs_funcs = _register_gentle_humanoid_extensions()

    action_joint_names = list(tracking_cfg["action_joint_names"])
    dataset_joint_names = list(tracking_cfg["dataset_joint_names"])
    real_joint_names = list(controller_cfg["real_joint_names"])
    default_joint_pos = _map_by_name(
        list(controller_cfg["default_qpos_real"]),
        real_joint_names,
        action_joint_names,
    )
    stiffness = {
        name: value
        for name, value in zip(
            real_joint_names,
            list(tracking_cfg.get("kps_real", controller_cfg["kps_real"])),
            strict=True,
        )
    }
    damping = {
        name: value
        for name, value in zip(
            real_joint_names,
            list(tracking_cfg.get("kds_real", controller_cfg["kds_real"])),
            strict=True,
        )
    }
    default_compliance_force = min(
        20.0,
        max(10.0, float(tracking_cfg.get("compliance_flag_threshold", 10.0))),
    )

    builder = mjswan.Builder()
    project = builder.add_project(name="Gentle Humanoid Tracking")
    scene = project.add_scene(
        name="Unitree G1",
        spec=mujoco.MjSpec.from_file(
            str(gentle_humanoid_root / "assets" / "g1" / "g1.xml")
        ),
    )
    scene.set_viewer(
        mjswan.ViewerConfig(
            lookat=(0, 0, 0),
            distance=3,
            elevation=-10,
            azimuth=30,
        )
    )

    policy_path = gentle_humanoid_root / tracking_cfg["policy_path"]
    policy_json = policy_path.with_suffix(".json")
    policy = scene.add_policy(
        name="Gentle Humanoid Tracking",
        policy=onnx.load(str(policy_path), load_external_data=True),
        config_path=str(policy_json),
        commands={
            "motion": mjswan.CommandTermConfig(
                term_name="GentleHumanoidTrackingCommand",
                params={
                    "joint_names": action_joint_names,
                    "future_steps": list(tracking_cfg["future_steps"]),
                    "switch_tail_keep_steps": int(
                        tracking_cfg.get("switch_tail_keep_steps", 8)
                    ),
                    "transition_steps": int(tracking_cfg.get("transition_steps", 100)),
                    "ref_max_len": int(tracking_cfg.get("ref_max_len", 2048)),
                },
            ),
            "compliance": mjswan.ui_command(
                [
                    mjswan.CheckboxConfig(
                        name="enabled",
                        label=(
                            "Compliance (turn off for motions with hand-ground contact)"
                        ),
                        default=bool(
                            float(tracking_cfg.get("compliance_flag_value", 1.0))
                        ),
                    ),
                    mjswan.SliderConfig(
                        name="force",
                        label="Force",
                        range=(10.0, 20.0),
                        default=default_compliance_force,
                        step=0.5,
                        enabled_when="enabled",
                    ),
                ]
            ),
        },
        observations={
            "policy": ObservationGroupCfg(
                terms={
                    "boot": ObservationTermCfg(func=obs_funcs["boot"]),
                    "tracking": ObservationTermCfg(
                        func=gentle_humanoid_tracking,
                        params={"future_steps": list(tracking_cfg["future_steps"])},
                    ),
                    "compliance": ObservationTermCfg(
                        func=obs_funcs["compliance"],
                        params={
                            "command_name": "compliance",
                            "default_enabled": float(
                                tracking_cfg.get("compliance_flag_value", 1.0)
                            ),
                            "default_force": float(default_compliance_force),
                        },
                    ),
                    "target_joint_pos": ObservationTermCfg(
                        func=gentle_humanoid_target_joint_pos,
                        params={
                            "future_steps": list(tracking_cfg["future_steps"]),
                            "num_joints": len(action_joint_names),
                        },
                    ),
                    "target_root_z": ObservationTermCfg(
                        func=gentle_humanoid_target_root_z,
                        params={"future_steps": list(tracking_cfg["future_steps"])},
                    ),
                    "target_projected_gravity": ObservationTermCfg(
                        func=gentle_humanoid_target_projected_gravity,
                        params={"future_steps": list(tracking_cfg["future_steps"])},
                    ),
                    "root_ang_vel": ObservationTermCfg(
                        func=obs_funcs["root_ang_vel"],
                        params={
                            "history_steps": list(
                                tracking_cfg["root_angvel_history_steps"]
                            )
                        },
                    ),
                    "projected_gravity": ObservationTermCfg(
                        func=obs_funcs["projected_gravity"],
                        params={
                            "history_steps": list(
                                tracking_cfg["projected_gravity_history_steps"]
                            )
                        },
                    ),
                    "joint_pos": ObservationTermCfg(
                        func=obs_funcs["joint_pos"],
                        params={
                            "history_steps": list(
                                tracking_cfg["joint_pos_history_steps"]
                            ),
                            "num_joints": len(action_joint_names),
                        },
                    ),
                    "joint_vel": ObservationTermCfg(
                        func=obs_funcs["joint_vel"],
                        params={
                            "history_steps": list(
                                tracking_cfg["joint_vel_history_steps"]
                            ),
                            "num_joints": len(action_joint_names),
                        },
                    ),
                    "prev_actions": ObservationTermCfg(
                        func=obs_funcs["prev_actions"],
                        params={
                            "history_steps": int(tracking_cfg["prev_action_steps"])
                        },
                    ),
                }
            )
        },
        actions={
            "joint_pos": JointPositionActionCfg(
                actuator_names=(".*",),
                scale=list(tracking_cfg["action_scale"]),
                use_default_offset=True,
                stiffness=stiffness,
                damping=damping,
            )
        },
        policy_joint_names=action_joint_names,
        default_joint_pos=default_joint_pos,
        default=True,
    )

    policy.add_motion(
        name="default",
        source=str(_ensure_default_motion_file(tracking_cfg)),
        fps=50.0,
        anchor_body_name="pelvis",
        body_names=("pelvis",),
        dataset_joint_names=dataset_joint_names,
        default=True,
        loop=False,
    )
    for motion_cfg in tracking_cfg["motions"]:
        source = gentle_humanoid_root / motion_cfg["path"]
        policy.add_motion(
            name=motion_cfg["name"],
            source=str(source),
            fps=50.0,
            anchor_body_name="pelvis",
            body_names=("pelvis",),
            dataset_joint_names=dataset_joint_names,
            loop=False,
        ).set_metadata("start", int(motion_cfg.get("start", 0))).set_metadata(
            "end", int(motion_cfg.get("end", -1))
        )

    return builder


def main() -> None:
    """Build and optionally launch the Gentle Humanoid tracking demo."""
    app = setup_builder().build()
    if os.getenv("MJSWAN_NO_LAUNCH") == "1":
        return
    app.launch()


if __name__ == "__main__":
    main()
