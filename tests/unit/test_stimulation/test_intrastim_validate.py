"""Tests for IntraStim construction and input validation.

The copyrights of this software are owned by Duke University.
See LICENSE for licensing instructions.
Source code: https://github.com/wmglab-duke/pyfibers
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from pyfibers import FiberModel, IntraStim, build_fiber


@pytest.fixture(scope="module")
def fiber():
    return build_fiber(fiber_model=FiberModel.MRG_INTERPOLATION, diameter=10.0, n_nodes=5)


def test_init_requires_exactly_one_of_ind_loc(fiber):
    with pytest.raises(ValueError, match="either ind or loc"):
        IntraStim(dt=0.001, tstop=0.01)
    with pytest.raises(ValueError, match="either ind or loc"):
        IntraStim(istim_ind=0, istim_loc=0.5, dt=0.001, tstop=0.01)


def test_clamp_kws_merge_defaults(fiber):
    istim = IntraStim(istim_loc=0.5, dt=0.001, tstop=0.01, clamp_kws={"delay": 2, "pw": 0.2})
    assert istim.istim_params["delay"] == 2
    assert istim.istim_params["pw"] == 0.2
    assert istim.istim_params["amp"] == 1
    assert istim.istim_params["freq"] == 100


def test_validate_nonzero_potentials_raises(fiber):
    istim = IntraStim(istim_loc=0.5, dt=0.001, tstop=0.01)
    istim.istim = MagicMock()
    fiber.potentials = np.ones(len(fiber.coordinates))
    try:
        with pytest.raises(ValueError, match="must be zero"):
            istim._validate_inputs(1.0, fiber)
    finally:
        fiber.potentials = np.zeros(len(fiber.coordinates))


def test_validate_missing_istim_raises(fiber):
    istim = IntraStim(istim_loc=0.5, dt=0.001, tstop=0.01)
    with pytest.raises(RuntimeError, match="not enabled"):
        istim._validate_inputs(1.0, fiber)


def test_validate_nonscalar_stimamp_raises(fiber):
    istim = IntraStim(istim_loc=0.5, dt=0.001, tstop=0.01)
    istim.istim = MagicMock()
    with pytest.raises(TypeError, match="stimamp must be a single float or int"):
        istim._validate_inputs([1.0, 2.0], fiber)


def test_negative_stimamp_warns(fiber):
    istim = IntraStim(istim_loc=0.5, dt=0.001, tstop=0.01)
    istim.istim = MagicMock()
    with pytest.warns(UserWarning, match="Negative intracellular"):
        istim._validate_inputs(-1.0, fiber)


def test_add_istim_passive_end_warns(fiber):
    istim = IntraStim(istim_ind=0, dt=0.001, tstop=0.01)
    with pytest.warns(UserWarning, match="passive node"):
        istim._add_istim(fiber)
    istim._cleanup_istim()


def test_run_sim_calls_cleanup(fiber):
    istim = IntraStim(istim_loc=0.5, dt=0.001, tstop=0.01, t_init_ss=-10, dt_init_ss=5)
    istim.run_sim(0.01, fiber, fail_on_end_excitation=None)
    assert istim.istim is None
    assert istim.istim_record is None
