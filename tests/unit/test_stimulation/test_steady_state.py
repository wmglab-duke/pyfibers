"""Tests for Stimulation._steady_state, pre_run_setup, and extracellular I/O.

The copyrights of this software are owned by Duke University.
See LICENSE for licensing instructions.
Source code: https://github.com/wmglab-duke/pyfibers
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from neuron import h

from pyfibers import FiberModel, Stimulation, build_fiber


@pytest.fixture(scope="module")
def fiber():
    return build_fiber(fiber_model=FiberModel.MRG_INTERPOLATION, diameter=10.0, n_nodes=5)


def _stim_with_mock_h(mock_h, **kwargs):
    mock_h.Vector.return_value.record.return_value = Mock()
    mock_h._ref_t = 0
    return Stimulation(dt=0.001, tstop=1, t_init_ss=-10, dt_init_ss=5, **kwargs)


def test_steady_state_vm_still_changing():
    with patch("pyfibers.stimulation.h") as mock_h:
        stim = _stim_with_mock_h(mock_h)
        seg = SimpleNamespace(v=-80.0)
        fiber = Mock(v_rest=-80.0)
        fiber.side_effect = lambda loc: seg

        def fadvance():
            mock_h.t += mock_h.dt
            seg.v += 1.5

        mock_h.fadvance.side_effect = fadvance
        with pytest.raises(RuntimeError, match="stable Vm"):
            stim._steady_state(fiber)


def test_steady_state_rest_mismatch():
    with patch("pyfibers.stimulation.h") as mock_h:
        stim = _stim_with_mock_h(mock_h)
        seg = SimpleNamespace(v=-80.0)
        fiber = Mock(v_rest=-80.0)
        fiber.side_effect = lambda loc: seg

        def fadvance():
            mock_h.t += mock_h.dt
            seg.v += 0.7

        mock_h.fadvance.side_effect = fadvance
        with pytest.raises(RuntimeError, match="are different"):
            stim._steady_state(fiber)


def test_steady_state_drift_from_start():
    with patch("pyfibers.stimulation.h") as mock_h:
        stim = _stim_with_mock_h(mock_h)
        seg = SimpleNamespace(v=-80.0)
        fiber = Mock(v_rest=-70.0)
        fiber.side_effect = lambda loc: seg

        def fadvance():
            mock_h.t += mock_h.dt
            seg.v = -70.0

        mock_h.fadvance.side_effect = fadvance
        with pytest.raises(RuntimeError, match="specified as"):
            stim._steady_state(fiber)


def test_pre_run_setup_sets_temp_apc_extra(fiber):
    stim = Stimulation(dt=0.001, tstop=1)
    stim._steady_state = lambda _fiber: None
    h.celsius = 0
    stim.pre_run_setup(fiber, ap_detect_threshold=-40)
    assert h.celsius == fiber.temperature
    assert len(fiber.apc) == len(fiber.nodes)
    assert fiber.apc[0].thresh == -40
    assert all(section(0.5).e_extracellular == 0 for section in fiber.sections)
    assert stim._n_timesteps == 1000
    assert fiber.time is stim.time


def test_initialize_extracellular_zeros(fiber):
    for section in fiber.sections:
        section(0.5).e_extracellular = 12.0
    Stimulation._initialize_extracellular(fiber)
    assert all(section(0.5).e_extracellular == 0 for section in fiber.sections)


def test_update_extracellular_writes(fiber):
    values = list(range(len(fiber.sections)))
    Stimulation._update_extracellular(fiber, values)
    assert [section(0.5).e_extracellular for section in fiber.sections] == values
    Stimulation._initialize_extracellular(fiber)
