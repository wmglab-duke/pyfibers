"""Tests for IntraStim stimamp type check before sign warning.

The copyrights of this software are owned by Duke University.
See LICENSE for licensing instructions.
Source code: https://github.com/wmglab-duke/pyfibers
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock

import pytest

from pyfibers import FiberModel, IntraStim, build_fiber


@pytest.fixture(scope="module")
def fiber():
    return build_fiber(fiber_model=FiberModel.MRG_INTERPOLATION, diameter=10.0, n_nodes=5)


def test_validate_nonscalar_stimamp_raises(fiber):
    istim = IntraStim(istim_loc=0.5, dt=0.001, tstop=0.01)
    istim.istim = MagicMock()
    with pytest.raises(TypeError, match="stimamp must be a single float or int"):
        istim._validate_inputs([1.0, 2.0], fiber)


def test_validate_nonscalar_negative_raises_typeerror_not_warning(fiber):
    """Type is checked before sign so a negative list does not emit the amp warning."""
    istim = IntraStim(istim_loc=0.5, dt=0.001, tstop=0.01)
    istim.istim = MagicMock()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with pytest.raises(TypeError, match="stimamp must be a single float or int"):
            istim._validate_inputs([-1.0, -2.0], fiber)
    assert not [w for w in record if "Negative intracellular" in str(w.message)]


def test_negative_stimamp_warns(fiber):
    istim = IntraStim(istim_loc=0.5, dt=0.001, tstop=0.01)
    istim.istim = MagicMock()
    with pytest.warns(UserWarning, match="Negative intracellular"):
        istim._validate_inputs(-1.0, fiber)
