"""Gentle Humanoid motion clips convert to the engine's body_world format (#79).

The demo migrated off its custom ts_src motion command to the built-in
`TrackingCommand`; its dataset clips (`root_pos`/`root_rot` xyzw/`dof_pos`) are
converted to body_world at build time.  This pins the two load-bearing
transforms — joint reorder (dataset→action order) and the xyzw→wxyz quaternion
convention — plus the exact array schema the built-in loader requires.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from examples.demo.gentle_humanoid.main import _body_world_npz, _clip_file_bytes

# The six arrays TrackingCommand's body_world loader requires (else it throws).
_REQUIRED = {
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
}


def _load(payload: bytes) -> dict:
    with np.load(io.BytesIO(payload)) as npz:
        return {k: npz[k] for k in npz.files}


def test_body_world_reorders_joints_and_keeps_wxyz():
    root_pos = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    root_quat_wxyz = np.array([[0.9, 0.1, 0.2, 0.3]], dtype=np.float32)
    dof_pos = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)  # source a,b,c
    out = _load(
        _body_world_npz(
            root_pos, root_quat_wxyz, dof_pos, ["a", "b", "c"], ["c", "a", "b"]
        )
    )
    # reordered into target (policy) order [c, a, b] = [30, 10, 20]
    np.testing.assert_array_equal(out["joint_pos"], [[30.0, 10.0, 20.0]])
    # pelvis is the single tracked body; quaternion passes through as wxyz
    np.testing.assert_array_equal(out["body_pos_w"], root_pos.reshape(1, 1, 3))
    np.testing.assert_array_equal(out["body_quat_w"], root_quat_wxyz.reshape(1, 1, 4))
    assert not out["joint_vel"].any()
    assert not out["body_lin_vel_w"].any() and not out["body_ang_vel_w"].any()
    assert not any(out[k].dtype.kind == "S" for k in out)


def test_body_world_is_c_contiguous_even_for_fortran_input():
    n = 4
    root_pos = np.asfortranarray(np.arange(n * 3, dtype=np.float32).reshape(n, 3))
    root_quat = np.asfortranarray(np.arange(n * 4, dtype=np.float32).reshape(n, 4))
    dof_pos = np.asfortranarray(np.arange(n * 2, dtype=np.float32).reshape(n, 2))
    out = _load(_body_world_npz(root_pos, root_quat, dof_pos, ["a", "b"], ["a", "b"]))
    for k in _REQUIRED:
        assert out[k].flags["C_CONTIGUOUS"], f"{k} is not C-contiguous"


def test_body_world_emits_all_required_arrays():
    out = _load(
        _body_world_npz(
            np.zeros((2, 3), np.float32),
            np.zeros((2, 4), np.float32),
            np.zeros((2, 1), np.float32),
            ["a"],
            ["a"],
        )
    )
    assert _REQUIRED <= set(out)


def test_target_joint_absent_from_source_is_zeroed():
    out = _load(
        _body_world_npz(
            np.zeros((1, 3), np.float32),
            np.zeros((1, 4), np.float32),
            np.array([[5.0]], np.float32),
            ["a"],
            ["a", "missing"],
        )
    )
    np.testing.assert_array_equal(out["joint_pos"], [[5.0, 0.0]])


def test_clip_file_windows_and_converts_xyzw(tmp_path: Path):
    n = 10
    root_pos = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
    root_rot_xyzw = np.tile(np.array([0.1, 0.2, 0.3, 0.9], np.float32), (n, 1))
    dof_pos = np.arange(n * 2, dtype=np.float32).reshape(n, 2)  # source a,b
    src = tmp_path / "clip.npz"
    np.savez(
        src,
        root_pos=root_pos,
        root_rot=root_rot_xyzw,
        dof_pos=dof_pos,
        joint_names=np.asarray(["a", "b"], dtype="S"),
    )

    out = _load(_clip_file_bytes(src, start=2, end=5, target_joint_names=["b", "a"]))
    assert out["joint_pos"].shape == (3, 2)  # windowed [2:5]
    # xyzw [.1,.2,.3,.9] -> wxyz [.9,.1,.2,.3]
    np.testing.assert_allclose(
        out["body_quat_w"][0, 0], [0.9, 0.1, 0.2, 0.3], atol=1e-6
    )
    # reorder to [b, a]; frame 2 source dof = [4, 5] -> [b=5, a=4]
    np.testing.assert_array_equal(out["joint_pos"][0], [5.0, 4.0])
    np.testing.assert_array_equal(out["body_pos_w"][0, 0], root_pos[2])


def test_clip_file_end_negative_means_to_end(tmp_path: Path):
    n = 4
    src = tmp_path / "clip.npz"
    np.savez(
        src,
        root_pos=np.zeros((n, 3), np.float32),
        root_rot=np.zeros((n, 4), np.float32),
        dof_pos=np.zeros((n, 1), np.float32),
        joint_names=np.asarray(["a"], dtype="S"),
    )
    out = _load(_clip_file_bytes(src, start=1, end=-1, target_joint_names=["a"]))
    assert out["joint_pos"].shape == (3, 1)  # [1:4]
