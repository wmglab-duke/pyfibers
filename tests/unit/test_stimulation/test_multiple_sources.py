from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest

from pyfibers import ScaledStim


@pytest.fixture
def mock_fiber():
    fiber = Mock()
    fiber.temperature = 37
    fiber.v_rest = -70
    fiber.potentials = [np.array([0.1, 0.2, 0.3]), np.array([0.2, 0.3, 0.4])]
    fiber.coordinates = [0, 1, 2]
    fiber.apc = [Mock(time=5.0, n=1), Mock(time=0, n=0), Mock(time=0, n=0)]
    fiber.loc_index = Mock(return_value=1)
    fiber.balance = Mock()
    fiber.apcounts = Mock()
    return fiber


@pytest.fixture
def mock_neuron():
    h = Mock()
    h.Vector = Mock(return_value=Mock(record=Mock()))
    h.trainIClamp = Mock()
    h.fadvance = Mock()
    h._ref_t = 0
    h.load_file = Mock()
    h.celsius = 37
    h.finitialize = Mock()
    h.frecord_init = Mock()
    h.t = 0
    h.dt = 0.001
    return h


def test_initialize_scaled_stim(mock_neuron):
    with patch('pyfibers.stimulation.h', mock_neuron):
        waveforms = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]]
        stim = ScaledStim(waveforms, dt=0.01, tstop=50, pad_waveform=True, truncate_waveform=True)

        assert stim.dt == 0.01
        assert stim.tstop == 50
        assert stim.pad is True
        assert stim.truncate is True
        padded_waveforms = np.vstack([np.concatenate([wf, np.zeros(4995)]) for wf in waveforms])
        assert np.array_equal(stim._prepped_waveform, padded_waveforms)


def test_prep_waveform(mock_neuron):
    with patch('pyfibers.stimulation.h', mock_neuron):
        waveforms = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]]
        stim = ScaledStim(waveforms, dt=0.01, tstop=0.05, pad_waveform=True, truncate_waveform=True)
        stim._prep_waveform()

        expected_waveforms = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]])
        assert np.array_equal(stim.waveform, expected_waveforms)


def test_potentials_at_time(mock_fiber, mock_neuron):
    with patch('pyfibers.stimulation.h', mock_neuron):
        waveforms = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]]
        stim = ScaledStim(waveforms, dt=0.01, tstop=0.05, pad_waveform=True, truncate_waveform=True)
        stim._prep_waveform()
        stim._prep_potentials(mock_fiber)

        potentials = stim._potentials_at_time(0, mock_fiber, [1, 1])
        assert np.allclose(potentials, np.array([0.11, 0.17, 0.23]))
        potentials = stim._potentials_at_time(0, mock_fiber, [1, 2])
        assert np.allclose(potentials, [0.21, 0.32, 0.43])


if __name__ == '__main__':
    pytest.main(['-v'])
