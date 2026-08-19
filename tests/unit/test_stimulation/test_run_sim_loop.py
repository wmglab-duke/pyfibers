"""Tests for ScaledStim/IntraStim run_sim loop branches.

The copyrights of this software are owned by Duke University.
See LICENSE for licensing instructions.
Source code: https://github.com/wmglab-duke/pyfibers
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from pyfibers import IntraStim, ScaledStim


class _Section:
    def __init__(self, v=-80.0):
        self.v = v
        self._seg = SimpleNamespace(e_extracellular=0)

    def __call__(self, _loc):
        return self._seg


def _patch_scaled_loop(stim, fiber, n_timesteps=8):
    stim._validate_scaling_inputs = lambda _fiber, amps: np.atleast_1d(np.array(amps, dtype=float))
    stim._potentials_at_time = lambda _i, _fiber, _amps: np.zeros(len(fiber.sections))
    stim._update_extracellular = lambda *_a, **_k: None
    stim.pre_run_setup = lambda *_a, **_k: setattr(stim, "_n_timesteps", n_timesteps)
    stim.ap_checker = lambda *_a, **_k: (0, None)
    stim.end_excitation_checker = Mock(return_value=False)


def _patch_intra_loop(istim, n_timesteps=8):
    def fake_add(_fiber):
        istim.istim = SimpleNamespace(amp=1.0)

    istim._add_istim = fake_add
    istim._validate_inputs = lambda *_a, **_k: None
    istim.pre_run_setup = lambda *_a, **_k: setattr(istim, "_n_timesteps", n_timesteps)
    istim.ap_checker = lambda *_a, **_k: (0, None)
    istim.end_excitation_checker = Mock(return_value=False)


@pytest.fixture
def mock_neuron():
    h = Mock()
    h.Vector = Mock(return_value=Mock(record=Mock()))
    h._ref_t = 0
    h.load_file = Mock()
    h.t = 0
    h.fadvance = Mock()
    return h


@pytest.fixture
def mock_fiber():
    fiber = Mock()
    fiber.sections = [_Section() for _ in range(3)]
    fiber.potentials = np.array([0.1, 0.2, 0.3])
    fiber.coordinates = [0, 1, 2]
    return fiber


def test_scaled_run_sim_raises_on_nan(mock_neuron, mock_fiber):
    with patch("pyfibers.stimulation.h", mock_neuron):
        stim = ScaledStim(waveform=lambda t: 1.0, dt=0.01, tstop=0.05)
        mock_fiber.sections[0].v = np.nan
        _patch_scaled_loop(stim, mock_fiber)
        with pytest.raises(RuntimeError, match="NaN"):
            stim.run_sim(1.0, mock_fiber, fail_on_end_excitation=None)


def test_scaled_run_sim_exit_func_breaks(mock_neuron, mock_fiber):
    with patch("pyfibers.stimulation.h", mock_neuron):
        stim = ScaledStim(waveform=lambda t: 1.0, dt=0.01, tstop=0.05)
        _patch_scaled_loop(stim, mock_fiber, n_timesteps=20)
        stim.run_sim(
            1.0,
            mock_fiber,
            exit_func=lambda *_a, **_k: True,
            exit_func_interval=1,
            fail_on_end_excitation=None,
        )
        assert mock_neuron.fadvance.call_count == 1


def test_scaled_run_sim_use_exit_t(mock_neuron, mock_fiber):
    with patch("pyfibers.stimulation.h", mock_neuron):
        stim = ScaledStim(waveform=lambda t: 1.0, dt=0.01, tstop=0.05)
        _patch_scaled_loop(stim, mock_fiber, n_timesteps=20)
        stim._exit_t = 0.02

        def fadvance():
            mock_neuron.t += 0.01

        mock_neuron.fadvance.side_effect = fadvance
        mock_neuron.t = 0
        stim.run_sim(1.0, mock_fiber, use_exit_t=True, fail_on_end_excitation=None)
        assert mock_neuron.fadvance.call_count == 2


def test_use_exit_t_ignored_when_exit_t_falsy(mock_neuron, mock_fiber):
    with patch("pyfibers.stimulation.h", mock_neuron):
        stim = ScaledStim(waveform=lambda t: 1.0, dt=0.01, tstop=0.05)
        _patch_scaled_loop(stim, mock_fiber, n_timesteps=6)
        stim._exit_t = None
        mock_neuron.t = 10
        stim.run_sim(1.0, mock_fiber, use_exit_t=True, fail_on_end_excitation=None)
        assert mock_neuron.fadvance.call_count == 5


def test_scaled_run_sim_calls_end_excitation_checker(mock_neuron, mock_fiber):
    with patch("pyfibers.stimulation.h", mock_neuron):
        stim = ScaledStim(waveform=lambda t: 1.0, dt=0.01, tstop=0.05)
        _patch_scaled_loop(stim, mock_fiber, n_timesteps=2)
        stim.run_sim(1.0, mock_fiber, fail_on_end_excitation=True)
        stim.end_excitation_checker.assert_called_once()


def test_intra_run_sim_nan(mock_neuron, mock_fiber):
    with patch("pyfibers.stimulation.h", mock_neuron):
        istim = IntraStim(istim_loc=0.5, dt=0.01, tstop=0.05)
        _patch_intra_loop(istim, n_timesteps=4)
        mock_fiber.sections[0].v = np.nan
        with pytest.raises(RuntimeError, match="NaN"):
            istim.run_sim(1.0, mock_fiber, fail_on_end_excitation=None)


def test_intra_run_sim_exit_func(mock_neuron, mock_fiber):
    with patch("pyfibers.stimulation.h", mock_neuron):
        istim = IntraStim(istim_loc=0.5, dt=0.01, tstop=0.05)
        _patch_intra_loop(istim, n_timesteps=20)
        istim.run_sim(
            1.0,
            mock_fiber,
            exit_func=lambda *_a, **_k: True,
            exit_func_interval=1,
            fail_on_end_excitation=None,
        )
        assert mock_neuron.fadvance.call_count == 1


def test_intra_run_sim_use_exit_t(mock_neuron, mock_fiber):
    with patch("pyfibers.stimulation.h", mock_neuron):
        istim = IntraStim(istim_loc=0.5, dt=0.01, tstop=0.05)
        _patch_intra_loop(istim, n_timesteps=20)
        istim._exit_t = 0.02

        def fadvance():
            mock_neuron.t += 0.01

        mock_neuron.fadvance.side_effect = fadvance
        mock_neuron.t = 0
        istim.run_sim(1.0, mock_fiber, use_exit_t=True, fail_on_end_excitation=None)
        assert mock_neuron.fadvance.call_count == 2


def test_intra_run_sim_precision_from_dt(mock_neuron, mock_fiber):
    with patch("pyfibers.stimulation.h", mock_neuron):
        istim = IntraStim(istim_loc=0.5, dt=0.0001, tstop=0.001)
        captured = {}

        def fake_ap(_fiber, ap_detect_location=0.9, precision=3, **_k):
            captured["precision"] = precision
            return 0, None

        _patch_intra_loop(istim, n_timesteps=2)
        istim.ap_checker = fake_ap
        istim.run_sim(1.0, mock_fiber, fail_on_end_excitation=None)
        assert captured["precision"] == 4
