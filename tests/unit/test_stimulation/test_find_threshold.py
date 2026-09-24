"""Tests for find_threshold and its extracted search helpers.

The copyrights of this software are owned by Duke University.
See LICENSE for licensing instructions.
Source code: https://github.com/wmglab-duke/pyfibers
"""

from __future__ import annotations

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
    """Stimulation subclass with scripted threshsim (no NEURON loop)."""

    def __init__(self, supra_fn, confirm=True, **kwargs):
        super().__init__(**kwargs)
        self.supra_fn = supra_fn
        self.confirm = confirm
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
        return self.confirm


def _bounds_search(stim, fiber, stimamp_top, stimamp_bottom, **overrides):
    params = {
        "condition": ThresholdCondition.ACTIVATION,
        "bounds_search_mode": BoundsSearchMode.PERCENT_INCREMENT,
        "bounds_search_step": 10,
        "stimamp_top": stimamp_top,
        "stimamp_bottom": stimamp_bottom,
        "max_iterations": 50,
        "exit_t_shift": 5,
        "block_delay": 5.0,
        "thresh_num_aps": 1,
        "kwargs": {},
    }
    params.update(overrides)
    return stim._bounds_search(fiber, **params)


def _bisection_search(stim, fiber, stimamp_top, stimamp_bottom, **overrides):
    params = {
        "condition": ThresholdCondition.ACTIVATION,
        "termination_mode": TerminationMode.ABSOLUTE_DIFFERENCE,
        "termination_tolerance": 0.3,
        "stimamp_top": stimamp_top,
        "stimamp_bottom": stimamp_bottom,
        "bisection_mean": BisectionMean.ARITHMETIC,
        "block_delay": 5.0,
        "thresh_num_aps": 1,
        "kwargs": {},
    }
    params.update(overrides)
    return stim._bisection_search(fiber, **params)


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


def test_both_sub_absolute_expands_anodic_top(fiber):
    """Absolute increment grows anodic top in the positive direction."""
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


@pytest.mark.parametrize(
    ("bad_kwargs", "match"),
    [
        ({"condition": "not-a-condition"}, "Invalid threshold condition"),
        ({"bounds_search_mode": "not-a-mode"}, "Invalid bounds search mode"),
        ({"termination_mode": "not-a-mode"}, "Invalid termination mode"),
        ({"bisection_mean": "not-a-mean"}, "Invalid bisection mean"),
    ],
)
def test_validate_threshold_enums_rejects_invalid(bad_kwargs, match):
    stim = Stimulation(dt=0.001, tstop=1)
    kwargs = {
        "condition": ThresholdCondition.ACTIVATION,
        "bounds_search_mode": BoundsSearchMode.PERCENT_INCREMENT,
        "termination_mode": TerminationMode.PERCENT_DIFFERENCE,
        "bisection_mean": BisectionMean.ARITHMETIC,
    }
    kwargs.update(bad_kwargs)
    with pytest.raises(ValueError, match=match):
        stim._validate_threshold_enums(**kwargs)


def _mock_block_fiber():
    fiber = MagicMock()
    fiber.stim = MagicMock()  # pretend intrinsic activity present
    return fiber


def test_validate_threshold_args_rejects_inconsistent_bounds():
    stim = Stimulation(dt=0.001, tstop=1)
    fiber = MagicMock()
    fiber.stim = None
    with pytest.raises(ValueError, match="greater in magnitude"):
        stim._validate_threshold_args(ThresholdCondition.ACTIVATION, -0.1, -1.0, 5, fiber)
    with pytest.raises(ValueError, match="opposite signs"):
        stim._validate_threshold_args(ThresholdCondition.ACTIVATION, -1.0, 0.01, 5, fiber)
    with pytest.raises(ValueError, match="exit_t_shift must be nonzero and positive"):
        stim._validate_threshold_args(ThresholdCondition.ACTIVATION, -1.0, -0.01, 0, fiber)


def test_validate_threshold_args_warns_for_intrinsic_activity():
    stim = Stimulation(dt=0.001, tstop=1)
    fiber = MagicMock()
    fiber.stim = object()
    with pytest.warns(UserWarning, match="intrinsic activity"):
        stim._validate_threshold_args(ThresholdCondition.ACTIVATION, -1.0, -0.01, 5, fiber)
    fiber.stim = None
    with pytest.warns(UserWarning, match="lacks intrinsic activity"):
        stim._validate_threshold_args(ThresholdCondition.BLOCK, -1.0, -0.01, 5, fiber, block_delay=5.0)
    assert stim._exit_t == float("Inf")


def test_bounds_search_returns_initial_straddling_bounds(fiber):
    stim = StubStim(lambda amp: abs(amp) >= 0.5, dt=0.001, tstop=1)
    top, bottom = _bounds_search(stim, fiber, -1.0, -0.01)
    assert top == pytest.approx(-1.0)
    assert bottom == pytest.approx(-0.01)
    assert stim.threshsim_calls == pytest.approx([-1.0, -0.01])
    assert stim._exit_t == pytest.approx(7.0)


def test_bounds_search_percent_expands_top_when_both_sub(fiber):
    stim = StubStim(lambda amp: abs(amp) >= 1.1, dt=0.001, tstop=1)
    top, bottom = _bounds_search(stim, fiber, -1.0, -0.01)
    assert top == pytest.approx(-1.1)
    assert bottom == pytest.approx(-1.0)


def test_bounds_search_percent_shrinks_bottom_when_both_supra(fiber):
    stim = StubStim(lambda amp: abs(amp) >= 0.48, dt=0.001, tstop=1)
    top, bottom = _bounds_search(stim, fiber, -1.0, -0.5)
    assert top == pytest.approx(-0.5)
    assert bottom == pytest.approx(-0.45)


def test_bounds_search_raises_on_contradictory_bounds(fiber):
    stim = StubStim(lambda amp: abs(amp) < 0.5, dt=0.001, tstop=1)
    with pytest.raises(RuntimeError, match="unexpected"):
        _bounds_search(stim, fiber, -1.0, -0.01)


def test_bounds_search_block_does_not_set_exit_t(fiber):
    stim = StubStim(lambda amp: abs(amp) >= 0.5, dt=0.001, tstop=1)
    _bounds_search(stim, fiber, -1.0, -0.01, condition=ThresholdCondition.BLOCK, block_delay=5.0)
    assert stim._exit_t is None


def test_bisection_search_uses_arithmetic_midpoint(fiber):
    stim = StubStim(lambda amp: abs(amp) >= 0.6, dt=0.001, tstop=1)
    amp, n_aps, aptime = _bisection_search(stim, fiber, -1.0, -0.5)
    assert stim.threshsim_calls == pytest.approx([-0.75])
    assert amp == pytest.approx(-0.75)
    assert (n_aps, aptime) == (1, 2.0)
    assert stim.run_sim_calls == pytest.approx([-0.75])


def test_bisection_search_uses_geometric_midpoint(fiber):
    stim = StubStim(lambda amp: abs(amp) >= 0.6, dt=0.001, tstop=1)
    amp, n_aps, aptime = _bisection_search(stim, fiber, -1.0, -0.5, bisection_mean=BisectionMean.GEOMETRIC)
    assert stim.threshsim_calls == pytest.approx([-(0.5**0.5)])
    assert amp == pytest.approx(-(0.5**0.5))
    assert (n_aps, aptime) == (1, 2.0)


def test_bisection_search_keeps_top_when_midpoint_is_sub(fiber):
    stim = StubStim(lambda amp: abs(amp) >= 0.8, dt=0.001, tstop=1)
    amp, n_aps, aptime = _bisection_search(stim, fiber, -1.0, -0.5)
    assert amp == pytest.approx(-1.0)
    assert (n_aps, aptime) == (1, 2.0)


def test_bisection_search_confirms_when_already_converged(fiber):
    stim = StubStim(lambda _amp: True, dt=0.001, tstop=1)
    amp, n_aps, aptime = _bisection_search(
        stim,
        fiber,
        -1.0,
        -0.995,
        termination_mode=TerminationMode.PERCENT_DIFFERENCE,
        termination_tolerance=1,
    )
    assert stim.threshsim_calls == []
    assert amp == pytest.approx(-1.0)
    assert stim.run_sim_calls == pytest.approx([-1.0])
    assert (n_aps, aptime) == (1, 2.0)


def test_bisection_search_raises_if_confirmation_fails(fiber):
    stim = StubStim(lambda _amp: True, confirm=False, dt=0.001, tstop=1)
    with pytest.raises(RuntimeError, match="expected action potential condition"):
        _bisection_search(
            stim,
            fiber,
            -1.0,
            -0.995,
            termination_mode=TerminationMode.PERCENT_DIFFERENCE,
            termination_tolerance=1,
        )


def test_find_threshold_returns_confirmed_amplitude(fiber):
    stim = StubStim(lambda amp: abs(amp) >= 0.5, dt=0.001, tstop=1)
    amp, (n_aps, aptime) = stim.find_threshold(
        fiber,
        stimamp_top=-1,
        stimamp_bottom=-0.01,
        bounds_search_mode=BoundsSearchMode.ABSOLUTE_INCREMENT,
        bounds_search_step=0.1,
        termination_mode=TerminationMode.ABSOLUTE_DIFFERENCE,
        termination_tolerance=0.05,
    )
    assert abs(amp) >= 0.5
    assert n_aps == 1
    assert aptime == 2.0
    assert stim.run_sim_calls[-1] == pytest.approx(amp)


def test_block_delay_none_errors():
    """Block searches with default block_delay=None should raise ValueError."""
    fiber = _mock_block_fiber()
    stim = StubStim(lambda _amp: True, dt=0.001, tstop=1)
    with pytest.raises(ValueError, match="positive block_delay"):
        stim.find_threshold(
            fiber,
            condition="block",
            stimamp_top=-1,
            stimamp_bottom=-0.01,
            max_iterations=1,
        )


def test_block_delay_nonpositive_errors():
    """Block searches with block_delay <= 0 should raise ValueError."""
    fiber = _mock_block_fiber()
    stim = StubStim(lambda _amp: True, dt=0.001, tstop=1)
    for delay in (0, -1.0):
        with pytest.raises(ValueError, match="positive block_delay"):
            stim.find_threshold(
                fiber,
                condition="block",
                stimamp_top=-1,
                stimamp_bottom=-0.01,
                block_delay=delay,
                max_iterations=1,
            )


def test_activation_block_delay_none_ok():
    """Activation searches may leave block_delay as None."""
    fiber = _mock_block_fiber()
    fiber.stim = None  # avoid intrinsic-activity warning for activation
    stim = StubStim(lambda _amp: True, dt=0.001, tstop=1)
    with pytest.raises(RuntimeError, match="max_iterations"):
        stim.find_threshold(
            fiber,
            condition="activation",
            stimamp_top=-1,
            stimamp_bottom=-0.01,
            block_delay=None,
            max_iterations=1,
        )
