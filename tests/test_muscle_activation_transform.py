"""Numeric specification for the muscle activation transform.

The TypeScript runtime applies the per-actuator transform inline in
``runtime.ts`` (``muscle_activation`` branch).  There is no Python
implementation in production code, so these tests pin the spec via a
local reference function and exercise the canonical values.  Update both
the reference and the TS branch together whenever the formula changes.
"""

from __future__ import annotations

import math


def muscle_activation_reference(
    action: float,
    scale: float = 1.0,
    offset: float = 0.0,
    normalize: bool = True,
) -> float:
    """Mirror of the ``muscle_activation`` branch in ``runtime.ts``.

    Shared pre-step: ``raw = scale * action + offset``.

    ``normalize=True``  → MyoSuite canonical sigmoid ``σ(5 * (raw - 0.5))``.
    ``normalize=False`` → clip ``raw`` to ``[0, 1]`` (myosuite4 准拠).

    Keep aligned with ``src/mjswan/template/src/core/engine/runtime.ts`` —
    if the TS branch changes, update this helper too.
    """
    raw = action * scale + offset
    if normalize:
        return 1.0 / (1.0 + math.exp(-5.0 * (raw - 0.5)))
    return max(0.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# normalize=True — canonical sigmoid σ(5(raw - 0.5))
# ---------------------------------------------------------------------------


class TestNormalizeTrueSigmoid:
    def test_action_zero_maps_below_decile(self):
        # σ(-2.5) ≈ 0.0759
        assert muscle_activation_reference(0.0) == pytest_approx(0.07585818, abs=1e-6)

    def test_action_half_is_fixed_point(self):
        # σ(0) = 0.5
        assert muscle_activation_reference(0.5) == pytest_approx(0.5, abs=1e-12)

    def test_action_one_maps_above_ninth_decile(self):
        # σ(2.5) ≈ 0.9241
        assert muscle_activation_reference(1.0) == pytest_approx(0.92414182, abs=1e-6)

    def test_negative_action_clamped_by_sigmoid_tail(self):
        # σ(-5) ≈ 0.0067
        assert muscle_activation_reference(-0.5) == pytest_approx(0.00669285, abs=1e-6)

    def test_above_one_action_approaches_unity(self):
        # σ(5) ≈ 0.9933
        assert muscle_activation_reference(1.5) == pytest_approx(0.99330715, abs=1e-6)

    def test_monotonic_in_action(self):
        prev = -1.0
        for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
            v = muscle_activation_reference(a)
            assert v > prev
            prev = v


# ---------------------------------------------------------------------------
# normalize=False — clip(raw, 0, 1)
# ---------------------------------------------------------------------------


class TestNormalizeFalseClip:
    def test_within_unit_interval_is_identity(self):
        for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert muscle_activation_reference(a, normalize=False) == pytest_approx(
                a, abs=1e-12
            )

    def test_above_one_clipped_to_one(self):
        assert muscle_activation_reference(1.5, normalize=False) == 1.0

    def test_negative_clipped_to_zero(self):
        assert muscle_activation_reference(-0.5, normalize=False) == 0.0

    def test_zero_action_maps_to_zero(self):
        # In normalize=False, action=0 stays at 0 (not 0.5 like sigmoid mode).
        assert muscle_activation_reference(0.0, normalize=False) == 0.0


# ---------------------------------------------------------------------------
# Shared pre-step: scale/offset applied identically in both modes
# ---------------------------------------------------------------------------


class TestScaleOffsetPreStep:
    def test_scale_offset_recover_fixed_point_sigmoid(self):
        # raw = 2.0 * 0.5 + (-0.5) = 0.5 → σ(0) = 0.5
        assert muscle_activation_reference(
            0.5, scale=2.0, offset=-0.5, normalize=True
        ) == pytest_approx(0.5, abs=1e-12)

    def test_scale_offset_recover_fixed_point_clip(self):
        # raw = 2.0 * 0.5 + (-0.5) = 0.5 → clip(0.5) = 0.5
        assert muscle_activation_reference(
            0.5, scale=2.0, offset=-0.5, normalize=False
        ) == pytest_approx(0.5, abs=1e-12)

    def test_offset_shifts_sigmoid_center(self):
        # offset=0.5 makes action=0 map to σ(0) = 0.5
        assert muscle_activation_reference(
            0.0, offset=0.5, normalize=True
        ) == pytest_approx(0.5, abs=1e-12)

    def test_scale_two_doubles_dynamic_range_in_clip_mode(self):
        # raw = 2.0 * 0.3 + 0 = 0.6
        assert muscle_activation_reference(
            0.3, scale=2.0, normalize=False
        ) == pytest_approx(0.6, abs=1e-12)

    def test_scale_two_pushes_action_above_clip_upper(self):
        # raw = 2.0 * 0.7 = 1.4 → clip to 1.0
        assert muscle_activation_reference(0.7, scale=2.0, normalize=False) == 1.0


# ---------------------------------------------------------------------------
# Per-actuator scale/offset semantics (mirror runtime.ts loop)
# ---------------------------------------------------------------------------


class TestPerActuatorParameters:
    def test_per_actuator_scale_applied_index_wise_sigmoid(self):
        # Three actuators: scale=[1.0, 2.0, 0.5], action=[0.5, 0.5, 0.5].
        # raw=[0.5, 1.0, 0.25] → σ(0)=0.5, σ(2.5)≈0.9241, σ(-1.25)≈0.2227
        actions = [0.5, 0.5, 0.5]
        scales = [1.0, 2.0, 0.5]
        expected = [0.5, 0.92414182, 0.22270014]
        for i, (a, s, e) in enumerate(zip(actions, scales, expected)):
            assert muscle_activation_reference(a, scale=s) == pytest_approx(
                e, abs=1e-6
            ), f"index {i}"

    def test_per_actuator_offset_applied_index_wise_clip(self):
        # Two actuators: offset=[0.2, -0.2], action=[0.5, 0.5].
        # raw=[0.7, 0.3] → clip → [0.7, 0.3]
        actions = [0.5, 0.5]
        offsets = [0.2, -0.2]
        expected = [0.7, 0.3]
        for i, (a, o, e) in enumerate(zip(actions, offsets, expected)):
            assert muscle_activation_reference(
                a, offset=o, normalize=False
            ) == pytest_approx(e, abs=1e-12), f"index {i}"


# ---------------------------------------------------------------------------
# Tiny pytest-approx shim so we do not require ``pytest`` symbol at module
# top-level (keeps the reference function importable in isolation if needed).
# ---------------------------------------------------------------------------


def pytest_approx(expected, abs=1e-12):
    import pytest

    return pytest.approx(expected, abs=abs)
