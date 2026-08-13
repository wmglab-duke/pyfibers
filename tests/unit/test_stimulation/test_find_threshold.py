"""Tests for Stimulation.find_threshold and related validation.

The copyrights of this software are owned by Duke University.
See LICENSE for licensing instructions.
Source code: https://github.com/wmglab-duke/pyfibers
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from pyfibers import (
    BisectionMean,
    BoundsSearchMode,
    FiberModel,
    Stimulation,
    TerminationMode,
    ThresholdCondition,
    build_fiber,
)


@pytest.fixture(scope="module")
def fiber():
    return build_fiber(fiber_model=FiberModel.MRG_INTERPOLATION, diameter=10.0, n_nodes=5)


class StubStim(Stimulation):
    """Stimulation subclass with scripted threshsim / run_sim (no NEURON loop)."""

    def __init__(self, supra_fn, checker_result=True, **kwargs):
        super().__init__(**kwargs)
        self.supra_fn = supra_fn
        self.checker_result = checker_result
        self.threshsim_calls = []
        self.run_sim_calls = []

    def threshsim(self, stimamp, fiber, **kwargs):
        self.threshsim_calls.append(float(stimamp))
        supra = bool(self.supra_fn(float(stimamp)))
        return supra, (int(supra), 2.0 if supra else None)

    def run_sim(self, stimamp, fiber, **kwargs):
        self.run_sim_calls.append(float(stimamp))
        return 1, 2.0

    def threshold_checker(self, *args, **kwargs):
        return self.checker_result


def magnitude_supra(threshold):
    return lambda amp: abs(amp) >= threshold


def test_bounds_found_immediately(fiber):
    stim = StubStim(magnitude_supra(0.5), dt=0.001, tstop=1)
    stim.find_threshold(
        fiber,
        stimamp_top=-1,
        stimamp_bottom=-0.01,
        termination_tolerance=100,
    )
    assert stim.threshsim_calls[:2] == [-1.0, -0.01]
    assert stim.run_sim_calls


def test_contradictory_bounds_raises(fiber):
    results = [False, True]

    def scripted(_amp):
        return results.pop(0)

    stim = StubStim(scripted, dt=0.001, tstop=1)
    with pytest.raises(RuntimeError, match="unexpected"):
        stim.find_threshold(fiber, stimamp_top=-1, stimamp_bottom=-0.01)


def test_both_sub_percent_expands_top(fiber):
    stim = StubStim(magnitude_supra(1.05), dt=0.001, tstop=1)
    stim.find_threshold(
        fiber,
        stimamp_top=-1,
        stimamp_bottom=-0.01,
        bounds_search_mode=BoundsSearchMode.PERCENT_INCREMENT,
        bounds_search_step=10,
        termination_tolerance=100,
    )
    assert stim.threshsim_calls[:3] == pytest.approx([-1.0, -0.01, -1.1])


def test_both_sub_absolute_expands_top(fiber):
    """Absolute increment grows top magnitude, including for cathodic bounds."""
    stim = StubStim(lambda _amp: False, dt=0.001, tstop=1)
    with pytest.raises(RuntimeError, match="max_iterations"):
        stim.find_threshold(
            fiber,
            stimamp_top=-1,
            stimamp_bottom=-0.01,
            bounds_search_mode=BoundsSearchMode.ABSOLUTE_INCREMENT,
            bounds_search_step=0.1,
            max_iterations=1,
        )
    assert stim.threshsim_calls[2] == pytest.approx(-1.1)


def test_both_supra_percent_shrinks_bottom(fiber):
    stim = StubStim(magnitude_supra(0.005), dt=0.001, tstop=1)
    stim.find_threshold(
        fiber,
        stimamp_top=-1,
        stimamp_bottom=-0.01,
        bounds_search_mode=BoundsSearchMode.PERCENT_INCREMENT,
        bounds_search_step=10,
        termination_tolerance=100,
    )
    assert stim.threshsim_calls[2] == pytest.approx(-0.009)


def test_both_supra_absolute_shrinks_bottom(fiber):
    """Absolute increment shrinks bottom magnitude, including for cathodic bounds."""
    stim = StubStim(lambda _amp: True, dt=0.001, tstop=1)
    with pytest.raises(RuntimeError, match="max_iterations"):
        stim.find_threshold(
            fiber,
            stimamp_top=-1,
            stimamp_bottom=-0.5,
            bounds_search_mode=BoundsSearchMode.ABSOLUTE_INCREMENT,
            bounds_search_step=0.1,
            max_iterations=1,
        )
    assert stim.threshsim_calls[2] == pytest.approx(-0.4)


def test_max_iterations_raises(fiber):
    stim = StubStim(lambda _amp: False, dt=0.001, tstop=1)
    with pytest.raises(RuntimeError, match="max_iterations=3"):
        stim.find_threshold(
            fiber,
            stimamp_top=-1,
            stimamp_bottom=-0.01,
            max_iterations=3,
        )


def test_exit_t_set_on_activation_supra(fiber):
    stim = StubStim(magnitude_supra(0.5), dt=0.001, tstop=1)
    stim.find_threshold(
        fiber,
        condition=ThresholdCondition.ACTIVATION,
        stimamp_top=-1,
        stimamp_bottom=-0.01,
        exit_t_shift=5,
        termination_tolerance=100,
    )
    assert stim._exit_t == pytest.approx(7.0)


def test_exit_t_not_set_for_block(fiber):
    stim = StubStim(magnitude_supra(0.5), dt=0.001, tstop=1)
    with pytest.warns(UserWarning, match="lacks intrinsic activity"):
        stim.find_threshold(
            fiber,
            condition=ThresholdCondition.BLOCK,
            stimamp_top=-1,
            stimamp_bottom=-0.01,
            exit_t_shift=5,
            termination_tolerance=100,
        )
    assert math.isinf(stim._exit_t)


def test_arithmetic_midpoint(fiber):
    stim = StubStim(magnitude_supra(0.5), dt=0.001, tstop=1)
    stim.find_threshold(
        fiber,
        stimamp_top=-1,
        stimamp_bottom=-0.01,
        bisection_mean=BisectionMean.ARITHMETIC,
        termination_tolerance=100,
    )
    assert stim.threshsim_calls[2] == pytest.approx((-0.01 + -1) / 2)


def test_geometric_midpoint(fiber):
    stim = StubStim(magnitude_supra(0.5), dt=0.001, tstop=1)
    stim.find_threshold(
        fiber,
        stimamp_top=-1,
        stimamp_bottom=-0.01,
        bisection_mean=BisectionMean.GEOMETRIC,
        termination_tolerance=100,
    )
    assert stim.threshsim_calls[2] == pytest.approx(-math.sqrt(0.01 * 1))


def test_percent_termination(fiber):
    stim = StubStim(magnitude_supra(0.5), dt=0.001, tstop=1)
    stim.find_threshold(
        fiber,
        stimamp_top=-1.0,
        stimamp_bottom=-0.995,
        termination_mode=TerminationMode.PERCENT_DIFFERENCE,
        termination_tolerance=1,
    )
    assert len(stim.run_sim_calls) == 1


def test_absolute_termination(fiber):
    stim = StubStim(magnitude_supra(0.5), dt=0.001, tstop=1)
    stim.find_threshold(
        fiber,
        stimamp_top=-1.0,
        stimamp_bottom=-0.995,
        termination_mode=TerminationMode.ABSOLUTE_DIFFERENCE,
        termination_tolerance=0.01,
    )
    assert len(stim.run_sim_calls) == 1


def test_converged_subthreshold_uses_prev_top(fiber):
    stim = StubStim(lambda amp: abs(amp) >= 0.9, dt=0.001, tstop=1)
    amp, _ = stim.find_threshold(
        fiber,
        stimamp_top=-1,
        stimamp_bottom=-0.01,
        termination_tolerance=100,
    )
    assert amp == pytest.approx(-1.0)


def test_final_validation_failure(fiber):
    stim = StubStim(magnitude_supra(0.5), checker_result=False, dt=0.001, tstop=1)
    with pytest.raises(RuntimeError, match="expected action potential condition"):
        stim.find_threshold(
            fiber,
            stimamp_top=-1,
            stimamp_bottom=-0.01,
            termination_tolerance=100,
        )


def test_both_sub_absolute_expands_anodic_top(fiber):
    stim = StubStim(lambda _amp: False, dt=0.001, tstop=1)
    with pytest.raises(RuntimeError, match="max_iterations"):
        stim.find_threshold(
            fiber,
            stimamp_top=1,
            stimamp_bottom=0.01,
            bounds_search_mode=BoundsSearchMode.ABSOLUTE_INCREMENT,
            bounds_search_step=0.1,
            max_iterations=1,
        )
    assert stim.threshsim_calls[2] == pytest.approx(1.1)


def test_anodic_same_sign_bounds(fiber):
    stim = StubStim(magnitude_supra(0.5), dt=0.001, tstop=1)
    stim.find_threshold(
        fiber,
        stimamp_top=1,
        stimamp_bottom=0.01,
        bisection_mean=BisectionMean.GEOMETRIC,
        termination_tolerance=100,
    )
    assert stim.threshsim_calls[2] == pytest.approx(math.sqrt(0.01 * 1))


def test_top_must_exceed_bottom_magnitude(fiber):
    stim = StubStim(magnitude_supra(0.5), dt=0.001, tstop=1)
    with pytest.raises(ValueError, match="greater in magnitude"):
        stim.find_threshold(fiber, stimamp_top=-0.01, stimamp_bottom=-1)


def test_opposite_signs_rejected(fiber):
    stim = StubStim(magnitude_supra(0.5), dt=0.001, tstop=1)
    with pytest.raises(ValueError, match="same sign"):
        stim.find_threshold(fiber, stimamp_top=-1, stimamp_bottom=0.01)


def test_exit_t_shift_nonpositive_rejected(fiber):
    stim = StubStim(magnitude_supra(0.5), dt=0.001, tstop=1)
    with pytest.raises(ValueError, match="exit_t_shift"):
        stim.find_threshold(fiber, stimamp_top=-1, stimamp_bottom=-0.01, exit_t_shift=0)


def test_warn_activation_with_intrinsic(fiber):
    fiber.stim = MagicMock()
    stim = StubStim(magnitude_supra(0.5), dt=0.001, tstop=1)
    try:
        with pytest.warns(UserWarning, match="intrinsic activity"):
            stim.find_threshold(
                fiber,
                condition=ThresholdCondition.ACTIVATION,
                stimamp_top=-1,
                stimamp_bottom=-0.01,
                termination_tolerance=100,
            )
    finally:
        fiber.stim = None


def test_warn_block_without_intrinsic(fiber):
    stim = StubStim(magnitude_supra(0.5), dt=0.001, tstop=1)
    with pytest.warns(UserWarning, match="lacks intrinsic activity"):
        stim.find_threshold(
            fiber,
            condition=ThresholdCondition.BLOCK,
            stimamp_top=-1,
            stimamp_bottom=-0.01,
            termination_tolerance=100,
        )


def test_invalid_enum_strings(fiber):
    stim = StubStim(magnitude_supra(0.5), dt=0.001, tstop=1)
    with pytest.raises(ValueError, match="Invalid threshold condition"):
        stim.find_threshold(fiber, condition="not_a_condition", stimamp_top=-1, stimamp_bottom=-0.01)
    with pytest.raises(ValueError, match="Invalid bounds search mode"):
        stim.find_threshold(fiber, bounds_search_mode="nope", stimamp_top=-1, stimamp_bottom=-0.01)
    with pytest.raises(ValueError, match="Invalid termination mode"):
        stim.find_threshold(fiber, termination_mode="nope", stimamp_top=-1, stimamp_bottom=-0.01)
    with pytest.raises(ValueError, match="Invalid bisection mean"):
        stim.find_threshold(fiber, bisection_mean="nope", stimamp_top=-1, stimamp_bottom=-0.01)


def test_silent_kwarg_future_warning(fiber):
    stim = StubStim(magnitude_supra(0.5), dt=0.001, tstop=1)
    with pytest.warns(FutureWarning, match="silent"):
        stim.find_threshold(
            fiber,
            stimamp_top=-1,
            stimamp_bottom=-0.01,
            termination_tolerance=100,
            silent=True,
        )


def test_exit_t_reset_to_inf(fiber):
    stim = StubStim(magnitude_supra(0.5), dt=0.001, tstop=1)
    stim._exit_t = 12.0
    stim.find_threshold(
        fiber,
        stimamp_top=-1,
        stimamp_bottom=-0.01,
        exit_t_shift=5,
        termination_tolerance=100,
    )
    # Validation resets to Inf, then activation supra sets t+shift
    assert stim._exit_t == pytest.approx(7.0)
