"""Tests for base Stimulation custom_run_sim and construction.

The copyrights of this software are owned by Duke University.
See LICENSE for licensing instructions.
Source code: https://github.com/wmglab-duke/pyfibers
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from pyfibers import FiberModel, Stimulation, build_fiber


@pytest.fixture(scope="module")
def fiber():
    return build_fiber(fiber_model=FiberModel.MRG_INTERPOLATION, diameter=10.0, n_nodes=5)


def test_base_run_sim_not_implemented(fiber):
    stim = Stimulation(dt=0.001, tstop=1)
    with pytest.raises(NotImplementedError, match="overridden by the subclass"):
        stim.run_sim(1.0, fiber)


def test_custom_run_sim_dispatched(fiber):
    received = {}

    def custom(self, stimamp, fiber_arg):
        received["self"] = self
        received["stimamp"] = stimamp
        received["fiber"] = fiber_arg
        return 3, 4.5

    stim = Stimulation(dt=0.001, tstop=1, custom_run_sim=custom)
    assert stim.run_sim(1.5, fiber) == (3, 4.5)
    assert received["self"] is stim
    assert received["stimamp"] == 1.5
    assert received["fiber"] is fiber


def test_init_without_sections_raises():
    with patch("pyfibers.stimulation.h") as mock_h:
        mock_h.Vector.return_value.record.side_effect = RuntimeError("no time")
        with pytest.raises(RuntimeError, match="created a fiber"):
            Stimulation(dt=0.001, tstop=1)


def test_str_repr(fiber):
    stim = Stimulation(dt=0.002, tstop=12, t_init_ss=-50, dt_init_ss=2.5)
    text = str(stim)
    assert "Stimulation" in text
    assert "12 ms" in text
    assert "dt=0.002 ms" in text
    assert "t_init_ss=-50 ms" in text
    assert "dt_init_ss=2.5 ms" in text
    assert repr(stim) == text
