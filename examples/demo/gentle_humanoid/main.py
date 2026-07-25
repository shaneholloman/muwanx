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

from .dsl_terms import BUILDERS

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


def _body_world_npz(
    root_pos: np.ndarray,  # (N, 3)
    root_quat_wxyz: np.ndarray,  # (N, 4), wxyz
    dof_pos: np.ndarray,  # (N, n_source), source joint order
    source_joint_names: list[str],
    target_joint_names: list[str],
    *,
    fps: float = 50.0,
) -> bytes:
    """Convert a root+dof clip to the engine's ``body_world`` format (#79):
    reorder joints source→policy order, pelvis as the single body, zero velocities.
    """
    n = root_pos.shape[0]
    src_idx = {name: i for i, name in enumerate(source_joint_names)}
    joint_pos = np.zeros((n, len(target_joint_names)), dtype=np.float32)
    for j, name in enumerate(target_joint_names):
        i = src_idx.get(name)
        if i is not None:
            joint_pos[:, j] = dof_pos[:, i]

    def _c(a: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(a, dtype=np.float32)

    payload = io.BytesIO()
    np.savez(
        payload,
        fps=np.asarray(float(fps), dtype=np.float32),
        joint_pos=_c(joint_pos),
        joint_vel=_c(np.zeros_like(joint_pos)),
        body_pos_w=_c(root_pos.reshape(n, 1, 3)),
        body_quat_w=_c(root_quat_wxyz.reshape(n, 1, 4)),
        body_lin_vel_w=_c(np.zeros((n, 1, 3))),
        body_ang_vel_w=_c(np.zeros((n, 1, 3))),
    )
    return payload.getvalue()


def _default_clip_bytes(tracking_cfg: dict[str, Any]) -> bytes:
    clips = tracking_cfg.get("motion_clips", [])
    if not isinstance(clips, list):
        raise ValueError("tracking.yaml motion_clips must be a list")
    clip = next((c for c in clips if c.get("name") == "default"), None)
    if clip is None:
        raise ValueError("tracking.yaml motion_clips must include a default clip")
    return _body_world_npz(
        root_pos=np.asarray(clip["root_pos"], dtype=np.float32).reshape(1, 3),
        root_quat_wxyz=np.asarray(clip["root_quat"], dtype=np.float32).reshape(1, 4),
        dof_pos=np.asarray(clip["joint_pos"], dtype=np.float32).reshape(1, -1),
        source_joint_names=list(tracking_cfg["dataset_joint_names"]),
        target_joint_names=list(tracking_cfg["action_joint_names"]),
    )


def _clip_file_bytes(
    path: Path, start: int, end: int, target_joint_names: list[str]
) -> bytes:
    """Load a dataset clip (``root_pos``/``root_rot`` xyzw/``dof_pos``) and window
    ``[start:end]``, converting to the engine's ``body_world`` format."""
    with np.load(path) as npz:
        root_pos = np.asarray(npz["root_pos"], dtype=np.float32)
        root_rot_xyzw = np.asarray(npz["root_rot"], dtype=np.float32)
        dof_pos = np.asarray(npz["dof_pos"], dtype=np.float32)
        source_joint_names = [
            s.decode() if isinstance(s, bytes) else str(s) for s in npz["joint_names"]
        ]
    hi = end if end >= 0 else root_pos.shape[0]
    root_quat_wxyz = root_rot_xyzw[start:hi][:, [3, 0, 1, 2]]  # xyzw -> wxyz
    return _body_world_npz(
        root_pos[start:hi],
        root_quat_wxyz,
        dof_pos[start:hi],
        source_joint_names,
        target_joint_names,
    )


def _write_generated(name: str, payload: bytes) -> Path:
    path = HERE / ".dep" / "generated" / f"gentle_humanoid_{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_bytes() != payload:
        path.write_bytes(payload)
    return path


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

    action_joint_names = list(tracking_cfg["action_joint_names"])
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
            # Built-in engine motion player; the demo's clips are converted to
            # its body_world format at build time (see _clip_file_bytes, #79).
            "motion": mjswan.CommandTermConfig(term_name="TrackingCommand"),
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
                    "boot": ObservationTermCfg(func=BUILDERS["gentle_humanoid_boot"]),
                    "tracking": ObservationTermCfg(
                        func=BUILDERS["gentle_humanoid_tracking"],
                        params={"future_steps": list(tracking_cfg["future_steps"])},
                    ),
                    "compliance": ObservationTermCfg(
                        func=BUILDERS["gentle_humanoid_compliance"],
                        params={"command_name": "compliance"},
                    ),
                    "target_joint_pos": ObservationTermCfg(
                        func=BUILDERS["gentle_humanoid_target_joint_pos"],
                        params={
                            "future_steps": list(tracking_cfg["future_steps"]),
                            "num_joints": len(action_joint_names),
                        },
                    ),
                    "target_root_z": ObservationTermCfg(
                        func=BUILDERS["gentle_humanoid_target_root_z"],
                        params={"future_steps": list(tracking_cfg["future_steps"])},
                    ),
                    "target_projected_gravity": ObservationTermCfg(
                        func=BUILDERS["gentle_humanoid_target_projected_gravity"],
                        params={"future_steps": list(tracking_cfg["future_steps"])},
                    ),
                    "root_ang_vel": ObservationTermCfg(
                        func=BUILDERS["gentle_humanoid_root_ang_vel"],
                        params={
                            "history_steps": list(
                                tracking_cfg["root_angvel_history_steps"]
                            )
                        },
                    ),
                    "projected_gravity": ObservationTermCfg(
                        func=BUILDERS["gentle_humanoid_projected_gravity"],
                        params={
                            "history_steps": list(
                                tracking_cfg["projected_gravity_history_steps"]
                            )
                        },
                    ),
                    "joint_pos": ObservationTermCfg(
                        func=BUILDERS["gentle_humanoid_joint_pos"],
                        params={
                            "history_steps": list(
                                tracking_cfg["joint_pos_history_steps"]
                            ),
                            "num_joints": len(action_joint_names),
                        },
                    ),
                    "joint_vel": ObservationTermCfg(
                        func=BUILDERS["gentle_humanoid_joint_vel"],
                        params={
                            "history_steps": list(
                                tracking_cfg["joint_vel_history_steps"]
                            ),
                            "num_joints": len(action_joint_names),
                        },
                    ),
                    "prev_actions": ObservationTermCfg(
                        func=BUILDERS["gentle_humanoid_prev_actions"],
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

    # Clips are converted to the engine's body_world format (joints reordered
    # into action order), so the bundled npz's joint order IS action order.
    policy.add_motion(
        name="default",
        source=str(_write_generated("default", _default_clip_bytes(tracking_cfg))),
        fps=50.0,
        anchor_body_name="pelvis",
        body_names=("pelvis",),
        dataset_joint_names=action_joint_names,
        default=True,
        loop=False,
    )
    for motion_cfg in tracking_cfg["motions"]:
        payload = _clip_file_bytes(
            gentle_humanoid_root / motion_cfg["path"],
            int(motion_cfg.get("start", 0)),
            int(motion_cfg.get("end", -1)),
            action_joint_names,
        )
        policy.add_motion(
            name=motion_cfg["name"],
            source=str(_write_generated(motion_cfg["name"], payload)),
            fps=50.0,
            anchor_body_name="pelvis",
            body_names=("pelvis",),
            dataset_joint_names=action_joint_names,
            loop=False,
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
