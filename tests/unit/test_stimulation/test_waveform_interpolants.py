"""Tests for deprecated array → interp1d waveform conversion.

The copyrights of this software are owned by Duke University.
See LICENSE for licensing instructions.
Source code: https://github.com/wmglab-duke/pyfibers
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest

from pyfibers import ScaledStim


@pytest.fixture
def mock_neuron():
    h = Mock()
    h.Vector = Mock(return_value=Mock(record=Mock()))
    h._ref_t = 0
    h.load_file = Mock()
    return h


def _array_stim(mock_neuron, waveform, dt=0.01, tstop=0.05, **kwargs):
    with patch("pyfibers.stimulation.h", mock_neuron), pytest.warns(FutureWarning, match="lists/arrays is deprecated"):
        return ScaledStim(waveform=waveform, dt=dt, tstop=tstop, **kwargs)


def test_previous_step_holds_between_samples(mock_neuron):
    # samples at 0, 0.01, 0.02, 0.03, 0.04 → values 0, 1, -1, 0, 0
    stim = _array_stim(mock_neuron, [0, 1, -1, 0, 0])
    wf = stim.waveform[0]

    assert wf(0.0) == pytest.approx(0.0)
    assert wf(0.005) == pytest.approx(0.0)
    assert wf(0.01) == pytest.approx(1.0)
    assert wf(0.015) == pytest.approx(1.0)
    assert wf(0.02) == pytest.approx(-1.0)
    assert wf(0.025) == pytest.approx(-1.0)
    assert wf(0.03) == pytest.approx(0.0)


def test_last_sample_holds_until_tstop(mock_neuron):
    # Last grid sample is nonzero; previous-step should keep it until tstop.
    stim = _array_stim(mock_neuron, [0, 0, 0, 0, 1])
    wf = stim.waveform[0]

    assert wf(0.04) == pytest.approx(1.0)
    assert wf(0.049) == pytest.approx(1.0)
    assert wf(0.049999999999) == pytest.approx(1.0)
    assert wf(0.05) == pytest.approx(0.0)


def test_accepts_time_just_past_last_sample_knot(mock_neuron):
    """FP-drifted h.t can exceed (n-1)*dt while still being < tstop (CI failure case)."""
    dt, tstop = 0.001, 20.0
    n = int(tstop / dt)
    waveform = np.zeros(n)
    waveform[0] = 1.0
    stim = _array_stim(mock_neuron, waveform, dt=dt, tstop=tstop)
    wf = stim.waveform[0]

    last_sample_t = (n - 1) * dt  # 19.999
    assert last_sample_t < tstop

    # Slightly above last sample abscissa, still under tstop — must not raise.
    drifted = last_sample_t + 1e-12
    assert drifted > last_sample_t
    assert drifted < tstop
    assert wf(drifted) == pytest.approx(0.0)

    # Same pattern as failed CI: value just under tstop that float-prints near tstop.
    assert wf(tstop - 1e-14) == pytest.approx(0.0)


def test_rejects_time_above_tstop(mock_neuron):
    stim = _array_stim(mock_neuron, [0, 1, 0, 0, 0])
    wf = stim.waveform[0]
    with pytest.raises(ValueError, match="above the interpolation range"):
        wf(0.0500001)


def test_rejects_time_below_zero(mock_neuron):
    stim = _array_stim(mock_neuron, [0, 1, 0, 0, 0])
    wf = stim.waveform[0]
    with pytest.raises(ValueError, match="below the interpolation range"):
        wf(-1e-12)


def test_sample_waveforms_matches_array_grid(mock_neuron):
    waveform = [0, 1, -1, 0.5, 0]
    stim = _array_stim(mock_neuron, waveform)
    assert np.array_equal(stim.sample_waveforms(), np.atleast_2d(waveform))


def test_single_timestep_array_waveform(mock_neuron):
    stim = _array_stim(mock_neuron, [1.0], dt=0.01, tstop=0.01)
    wf = stim.waveform[0]
    assert wf(0.0) == pytest.approx(1.0)
    assert wf(0.005) == pytest.approx(1.0)
    # Domain max is tstop (== dt); evaluating at tstop is in range.
    assert wf(0.01) == pytest.approx(1.0)


def test_potentials_at_time_uses_grid_time(mock_neuron):
    fiber = Mock()
    fiber.potentials = np.array([[1.0, 2.0, 3.0]])
    fiber.coordinates = [0, 1, 2]

    with patch("pyfibers.stimulation.h", mock_neuron):
        with pytest.warns(FutureWarning, match="lists/arrays is deprecated"):
            stim = ScaledStim(waveform=[0, 1, 0, 0, 0], dt=0.01, tstop=0.05)

        mock_neuron.t = 0.049999999999  # drifted clock must not be used
        potentials = stim._potentials_at_time(fiber, np.array([2.0]), 0.01)
        assert np.allclose(potentials, np.array([2.0, 4.0, 6.0]))

        potentials = stim._potentials_at_time(fiber, np.array([1.0]), 0.04)
        assert np.allclose(potentials, np.zeros(3))
