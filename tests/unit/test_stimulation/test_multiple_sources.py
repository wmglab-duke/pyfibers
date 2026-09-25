"""Tests for multiple stimulation sources.

The copyrights of this software are owned by Duke University.
See LICENSE for licensing instructions.
Source code: https://github.com/wmglab-duke/pyfibers
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest
from scipy.interpolate import interp1d

from pyfibers import ScaledStim


@pytest.fixture
def mock_fiber():
    fiber = Mock()
    fiber.temperature = 37
    fiber.v_rest = -70
    fiber.potentials = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
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
        waveforms = [
            interp1d(
                [0.0, 0.01, 0.02, 50.0], [0.1, 0.2, 0.3, 0.0], kind='previous', bounds_error=False, fill_value=0.0
            ),
            interp1d(
                [0.0, 0.01, 0.02, 50.0], [0.5, 0.4, 0.3, 0.0], kind='previous', bounds_error=False, fill_value=0.0
            ),
        ]
        stim = ScaledStim(waveforms, dt=0.01, tstop=50)

        assert stim.dt == 0.01
        assert stim.tstop == 50
        assert len(stim.waveform) == 2
        assert stim.waveform[0](0.0) == 0.1
        assert stim.waveform[0](0.02) == 0.3
        assert stim.waveform[1](0.0) == 0.5
        assert stim.waveform[1](0.02) == 0.3


def test_initialize_array_waveform(mock_neuron):
    with patch('pyfibers.stimulation.h', mock_neuron):
        waveforms = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]]
        stim = ScaledStim(waveforms, dt=0.01, tstop=50, pad_waveform=True, truncate_waveform=True)

        assert stim.dt == 0.01
        assert stim.tstop == 50
        assert stim.pad is True
        assert stim.truncate is True
        padded_waveforms = np.vstack([np.concatenate([wf, np.zeros(4995)]) for wf in waveforms])
        assert np.array_equal(stim._sample_waveforms(stim.waveform), padded_waveforms)
        assert len(stim.waveform) == 2
        assert all(callable(wf) for wf in stim.waveform)


def test_waveform_setter(mock_neuron):
    with patch('pyfibers.stimulation.h', mock_neuron):
        waveforms = [
            interp1d(
                [0.0, 0.01, 0.02, 0.05], [0.1, 0.2, 0.3, 0.0], kind='previous', bounds_error=False, fill_value=0.0
            ),
            interp1d(
                [0.0, 0.01, 0.02, 0.05], [0.5, 0.4, 0.3, 0.0], kind='previous', bounds_error=False, fill_value=0.0
            ),
        ]
        stim = ScaledStim(waveforms, dt=0.01, tstop=0.05)

        assert stim.waveform is waveforms
        assert len(stim.waveform) == 2
        assert all(callable(wf) for wf in stim.waveform)
        assert stim.waveform[0](0.0) == 0.1
        assert stim.waveform[0](0.02) == 0.3
        assert stim.waveform[1](0.0) == 0.5
        assert stim.waveform[1](0.02) == 0.3

        single_waveform = interp1d(
            [0.0, 0.01, 0.02, 0.05],
            [1.0, 0.5, 0.0, 0.0],
            kind='previous',
            bounds_error=False,
            fill_value=0.0,
        )
        stim.waveform = single_waveform
        assert len(stim.waveform) == 1
        assert stim.waveform[0] is single_waveform
        assert stim.waveform[0](0.0) == 1.0
        assert stim.waveform[0](0.02) == 0.0


def test_waveform_array_prep(mock_neuron):
    with patch('pyfibers.stimulation.h', mock_neuron):
        waveforms = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]]
        stim = ScaledStim(waveforms, dt=0.01, tstop=0.05, pad_waveform=True, truncate_waveform=True)

        expected_waveforms = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]])
        assert np.array_equal(stim._sample_waveforms(stim.waveform), expected_waveforms)
        assert all(callable(wf) for wf in stim.waveform)


def test_tstop_setter_repads_array_waveform(mock_neuron):
    with patch('pyfibers.stimulation.h', mock_neuron):
        waveforms = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]]
        stim = ScaledStim(waveforms, dt=0.01, tstop=0.05, pad_waveform=True, truncate_waveform=True)
        assert stim._sample_waveforms(stim.waveform).shape[1] == 5

        stim.tstop = 50
        padded_waveforms = np.vstack([np.concatenate([wf, np.zeros(4995)]) for wf in waveforms])
        assert np.array_equal(stim._sample_waveforms(stim.waveform), padded_waveforms)


def test_potentials_at_time(mock_fiber, mock_neuron):
    with patch('pyfibers.stimulation.h', mock_neuron):
        waveforms = [
            interp1d(
                [0.0, 0.01, 0.02, 0.05], [0.1, 0.2, 0.3, 0.3], kind='previous', bounds_error=False, fill_value=0.3
            ),
            interp1d(
                [0.0, 0.01, 0.02, 0.05], [0.5, 0.4, 0.3, 0.3], kind='previous', bounds_error=False, fill_value=0.3
            ),
        ]
        stim = ScaledStim(waveforms, dt=0.01, tstop=0.05)

        mock_neuron.t = 0.0
        potentials = stim._potentials_at_time(mock_fiber, np.array([1, 1]))
        assert np.allclose(potentials, np.array([0.11, 0.17, 0.23]))
        potentials = stim._potentials_at_time(mock_fiber, np.array([1, 2]))
        assert np.allclose(potentials, [0.21, 0.32, 0.43])


def test_potentials_at_time_array(mock_fiber, mock_neuron):
    with patch('pyfibers.stimulation.h', mock_neuron):
        waveforms = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]]
        stim = ScaledStim(waveforms, dt=0.01, tstop=0.05, pad_waveform=True, truncate_waveform=True)

        mock_neuron.t = 0.0
        potentials = stim._potentials_at_time(mock_fiber, [1, 1])
        assert np.allclose(potentials, np.array([0.11, 0.17, 0.23]))
        potentials = stim._potentials_at_time(mock_fiber, [1, 2])
        assert np.allclose(potentials, [0.21, 0.32, 0.43])
        mock_neuron.t = 0.01
        potentials = stim._potentials_at_time(mock_fiber, [1, 1])
        assert np.allclose(potentials, [0.10, 0.16, 0.22])


def test_potentials_at_time_strict_zip(mock_fiber, mock_neuron):
    with patch('pyfibers.stimulation.h', mock_neuron):
        waveforms = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]]
        stim = ScaledStim(waveforms, dt=0.01, tstop=0.05, pad_waveform=True, truncate_waveform=True)
        with pytest.raises(ValueError, match="zip\\(\\)"):
            stim._potentials_at_time(mock_fiber, [1])


def test_pad_setter_reprocesses_waveform(mock_neuron):
    with patch('pyfibers.stimulation.h', mock_neuron):
        # Start with pad=False, tstop matching waveform length exactly
        waveforms = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]]
        stim = ScaledStim(waveforms, dt=0.01, tstop=0.05, pad_waveform=False, truncate_waveform=False)
        assert stim.pad is False
        assert stim._sample_waveforms(stim.waveform).shape[1] == 5

        # Extend tstop so waveform is now short, then enable pad
        stim._tstop = 0.1  # bypass setter to avoid reprocessing
        stim.pad = True
        assert stim.pad is True
        padded_waveforms = np.vstack([np.concatenate([wf, np.zeros(5)]) for wf in waveforms])
        assert np.array_equal(stim._sample_waveforms(stim.waveform), padded_waveforms)


def test_truncate_setter_reprocesses_waveform(mock_neuron):
    with patch('pyfibers.stimulation.h', mock_neuron):
        # Start with truncate=False, waveform longer than n_timesteps — requires pad=False truncate=False and exact len
        # Use a 6-element waveform with tstop matching 6 elements
        waveforms = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]]
        stim = ScaledStim(waveforms, dt=0.01, tstop=0.06, pad_waveform=False, truncate_waveform=False)
        assert stim.truncate is False
        assert stim._sample_waveforms(stim.waveform).shape[1] == 6

        # Shorten tstop so waveform is now long, then enable truncate
        stim._tstop = 0.05  # bypass setter
        stim.truncate = True
        assert stim.truncate is True
        expected = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]])
        assert np.array_equal(stim._sample_waveforms(stim.waveform), expected)


def test_array_and_callable_waveforms_equivalent(mock_fiber, mock_neuron):
    """Deprecated array input must match an equivalent interpolant at every time step."""
    with patch('pyfibers.stimulation.h', mock_neuron):
        array_waveforms = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]]
        times = np.arange(5) * 0.01
        callable_waveforms = [interp1d(times, row, kind='previous', bounds_error=True) for row in array_waveforms]
        array_stim = ScaledStim(array_waveforms, dt=0.01, tstop=0.05)
        callable_stim = ScaledStim(callable_waveforms, dt=0.01, tstop=0.05)

        np.testing.assert_array_equal(
            array_stim._sample_waveforms(array_stim.waveform),
            callable_stim._sample_waveforms(callable_stim.waveform),
        )
        for t in times:
            mock_neuron.t = t
            np.testing.assert_allclose(
                array_stim._potentials_at_time(mock_fiber, [1, 1]),
                callable_stim._potentials_at_time(mock_fiber, [1, 1]),
            )


def test_waveform_getter_wraps_single_callable(mock_neuron):
    with patch('pyfibers.stimulation.h', mock_neuron):

        def pulse(t):
            return 1.0 if t < 0.02 else 0.0

        stim = ScaledStim(waveform=pulse, dt=0.01, tstop=0.05)
        assert isinstance(stim.waveform, list)
        assert len(stim.waveform) == 1
        assert stim.waveform[0] is pulse
        assert stim.waveform[0](0.0) == 1.0
        assert stim.waveform[0](0.02) == 0.0


def test_waveform_array_converted_to_interpolants(mock_neuron):
    with patch('pyfibers.stimulation.h', mock_neuron):
        waveform = [0.0, 1.0, -1.0, 0.0, 0.5]
        stim = ScaledStim(waveform=waveform, dt=0.01, tstop=0.05)
        assert all(callable(wf) for wf in stim.waveform)
        assert len(stim.waveform) == 1
        assert stim.waveform[0](0.0) == 0.0
        assert stim.waveform[0](0.01) == 1.0
        assert stim.waveform[0](0.02) == -1.0
        assert stim.waveform[0](0.04) == 0.5
        with pytest.raises(ValueError, match="interpolation range"):
            stim.waveform[0](-1.0)
        with pytest.raises(ValueError, match="interpolation range"):
            stim.waveform[0](0.05)


def test_waveform_dt_setter_reprocesses_array(mock_neuron):
    with patch('pyfibers.stimulation.h', mock_neuron):
        waveforms = [[0.1, 0.2, 0.3, 0.4, 0.5]]
        stim = ScaledStim(waveforms, dt=0.01, tstop=0.05, pad_waveform=True)
        assert stim._sample_waveforms(stim.waveform).shape[1] == 5

        stim.dt = 0.005
        sampled = stim._sample_waveforms(stim.waveform)
        assert sampled.shape[1] == 10
        np.testing.assert_array_equal(sampled[0], np.concatenate([[0.1, 0.2, 0.3, 0.4, 0.5], np.zeros(5)]))


def test_waveform_input_preserved_across_reprocess(mock_neuron):
    with patch('pyfibers.stimulation.h', mock_neuron):
        waveform = np.array([1.0, -1.0, 0.0, 0.0, 0.0])
        stim = ScaledStim(waveform=waveform, dt=0.01, tstop=0.05)
        np.testing.assert_array_equal(stim._waveform_input, waveform)
        stim.tstop = 0.1
        np.testing.assert_array_equal(stim._waveform_input, waveform)
        sampled = stim._sample_waveforms(stim.waveform)
        assert sampled.shape[1] == 10
        np.testing.assert_array_equal(sampled[0], np.concatenate([waveform, np.zeros(5)]))


def test_callable_waveform_clears_waveform_input(mock_neuron):
    with patch('pyfibers.stimulation.h', mock_neuron):
        array_waveform = np.array([1.0, -1.0, 0.0, 0.0, 0.0])
        stim = ScaledStim(waveform=array_waveform, dt=0.01, tstop=0.05)
        assert stim._waveform_input is not None

        stim.waveform = lambda t: 1.0 if t < 0.02 else 0.0
        assert stim._waveform_input is None
        stim.tstop = 0.1  # no-op reprocess for callables
        assert stim._waveform_input is None
        assert stim.waveform[0](0.0) == 1.0


if __name__ == '__main__':
    pytest.main(['-v'])
