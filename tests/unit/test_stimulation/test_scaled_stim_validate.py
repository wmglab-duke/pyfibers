"""Tests for ScaledStim validation, padding, and multi-source scaling.

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


@pytest.fixture
def mock_fiber():
    fiber = Mock()
    fiber.temperature = 37
    fiber.v_rest = -70
    fiber.potentials = np.array([0.1, 0.2, 0.3])
    fiber.coordinates = [0, 1, 2]
    return fiber


def _stim(mock_neuron, waveform=None, dt=0.01, tstop=0.05, **kwargs):
    wf = waveform if waveform is not None else (lambda t: 1.0 if 0 < t <= 0.02 else 0.0)
    with patch("pyfibers.stimulation.h", mock_neuron):
        return ScaledStim(waveform=wf, dt=dt, tstop=tstop, **kwargs)


def test_prep_potentials_none_raises(mock_neuron, mock_fiber):
    stim = _stim(mock_neuron)
    mock_fiber.potentials = None
    with pytest.raises(ValueError, match="No fiber potentials"):
        stim._prep_potentials(mock_fiber)


def test_prep_potentials_length_mismatch(mock_neuron, mock_fiber):
    stim = _stim(mock_neuron)
    mock_fiber.potentials = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match="match the length"):
        stim._prep_potentials(mock_fiber)


def test_prep_potentials_mutates_1d_to_2d(mock_neuron, mock_fiber):
    stim = _stim(mock_neuron)
    stim._prep_potentials(mock_fiber)
    assert mock_fiber.potentials.ndim == 2
    assert mock_fiber.potentials.shape == (1, 3)


def test_waveform_length_mismatch(mock_neuron):
    with (
        patch("pyfibers.stimulation.h", mock_neuron),
        pytest.warns(FutureWarning, match="lists/arrays is deprecated"),
        pytest.raises(ValueError, match="Processed waveform length"),
    ):
        ScaledStim(
            waveform=[1, 0],
            dt=0.01,
            tstop=0.05,
            pad_waveform=False,
            truncate_waveform=False,
        )


def test_mixed_callable_and_array_rejected(mock_neuron):
    with patch("pyfibers.stimulation.h", mock_neuron), pytest.raises(TypeError, match="callable or a list of callables"):
        ScaledStim(waveform=[lambda t: 1, [0, 1, 0]], dt=0.01, tstop=0.05)


def test_pad_nonzero_end_warns(mock_neuron):
    with patch("pyfibers.stimulation.h", mock_neuron), pytest.warns(UserWarning, match="Padding a waveform"):
        ScaledStim(waveform=[1, 1, 1], dt=0.01, tstop=0.05, pad_waveform=True)


def test_truncate_drops_nonzero_warns(mock_neuron):
    with patch("pyfibers.stimulation.h", mock_neuron), pytest.warns(UserWarning, match="Truncating waveform"):
        ScaledStim(waveform=np.ones(20), dt=0.01, tstop=0.05, truncate_waveform=True)


def test_unit_peak_warn(mock_neuron):
    with patch("pyfibers.stimulation.h", mock_neuron), pytest.warns(UserWarning, match="max absolute value of 1"):
        ScaledStim(waveform=lambda t: 0.5, dt=0.01, tstop=0.05)


def test_array_waveform_future_warning(mock_neuron):
    with patch("pyfibers.stimulation.h", mock_neuron), pytest.warns(FutureWarning, match="lists/arrays is deprecated"):
        ScaledStim(waveform=[0, 1, 0, 0, 0], dt=0.01, tstop=0.05)


def test_source_count_mismatch(mock_neuron, mock_fiber):
    stim = _stim(mock_neuron, waveform=[lambda t: 1, lambda t: 0])
    mock_fiber.potentials = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="does not match number of waveforms"):
        stim._validate_scaling_inputs(mock_fiber, np.array(1.0))


def test_all_zero_potentials(mock_neuron, mock_fiber):
    stim = _stim(mock_neuron)
    mock_fiber.potentials = np.zeros(3)
    with pytest.raises(ValueError, match="non-zero fiber potential"):
        stim._validate_scaling_inputs(mock_fiber, np.array(1.0))


def test_all_zero_waveform(mock_neuron, mock_fiber):
    with pytest.warns(UserWarning, match="max absolute value of 1"):
        stim = _stim(mock_neuron, waveform=lambda t: 0.0)
    with pytest.raises(ValueError, match="non-zero waveform"):
        stim._validate_scaling_inputs(mock_fiber, np.array(1.0))


def test_stimamp_list_length_mismatch(mock_neuron, mock_fiber):
    stim = _stim(mock_neuron, waveform=[lambda t: 1, lambda t: 1])
    mock_fiber.potentials = [np.array([0.1, 0.2, 0.3]), np.array([0.2, 0.3, 0.4])]
    with pytest.raises(ValueError, match="Number of stimamps"):
        stim._validate_scaling_inputs(mock_fiber, np.array([1.0, 2.0, 3.0]))


def test_scalar_stimamp_broadcast(mock_neuron, mock_fiber):
    stim = _stim(mock_neuron, waveform=[lambda t: 1, lambda t: 1])
    mock_fiber.potentials = [np.array([0.1, 0.2, 0.3]), np.array([0.2, 0.3, 0.4])]
    result = stim._validate_scaling_inputs(mock_fiber, np.array(2.0))
    np.testing.assert_array_equal(result, [2.0, 2.0])


def test_callable_raises_runtime_error(mock_neuron):
    def bad(_t):
        raise ValueError("bad time")

    with patch("pyfibers.stimulation.h", mock_neuron), pytest.raises(RuntimeError, match="processing callable"):
        ScaledStim(waveform=bad, dt=0.01, tstop=0.05)
