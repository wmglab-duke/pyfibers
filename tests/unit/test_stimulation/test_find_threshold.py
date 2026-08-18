"""Tests for absolute-increment bound search sign handling in find_threshold.

The copyrights of this software are owned by Duke University.
See LICENSE for licensing instructions.
Source code: https://github.com/wmglab-duke/pyfibers
"""

from __future__ import annotations

import pytest

from pyfibers import BoundsSearchMode, FiberModel, Stimulation, build_fiber


@pytest.fixture(scope="module")
def fiber():
    return build_fiber(fiber_model=FiberModel.MRG_INTERPOLATION, diameter=10.0, n_nodes=5)


class StubStim(Stimulation):
    """Stimulation subclass with scripted threshsim (no NEURON loop)."""

    def __init__(self, supra_fn, **kwargs):
        super().__init__(**kwargs)
        self.supra_fn = supra_fn
        self.threshsim_calls = []

    def threshsim(self, stimamp, fiber, **kwargs):
        self.threshsim_calls.append(float(stimamp))
        supra = bool(self.supra_fn(float(stimamp)))
        return supra, (int(supra), 2.0 if supra else None)

    def run_sim(self, stimamp, fiber, **kwargs):
        return 1, 2.0


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
