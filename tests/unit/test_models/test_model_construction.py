"""Tests for fiber-model construction rules (not MOD electrophysiology).

The copyrights of this software are owned by Duke University.
See LICENSE for licensing instructions.
Source code: https://github.com/wmglab-duke/pyfibers
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from pyfibers import FiberModel, build_fiber


def test_mrg_discrete_invalid_diameter():
    with pytest.raises(ValueError, match="Diameter chosen not valid for MRG_DISCRETE"):
        build_fiber(fiber_model=FiberModel.MRG_DISCRETE, diameter=3.0, n_nodes=5)


def test_mrg_interpolation_diameter_bounds():
    with pytest.raises(ValueError, match="between 2 and 16"):
        build_fiber(fiber_model=FiberModel.MRG_INTERPOLATION, diameter=1.0, n_nodes=5)
    with pytest.raises(ValueError, match="between 2 and 16"):
        build_fiber(fiber_model=FiberModel.MRG_INTERPOLATION, diameter=17.0, n_nodes=5)


def test_mrg_cannot_pass_delta_z():
    with pytest.raises(ValueError, match="Cannot specify delta_z"):
        build_fiber(fiber_model=FiberModel.MRG_INTERPOLATION, diameter=10.0, n_nodes=5, delta_z=100)


def test_pena_diameter_bounds():
    with pytest.raises(ValueError, match="between 1.011 and 16"):
        build_fiber(fiber_model=FiberModel.PENA, diameter=0.5, n_nodes=5)
    with pytest.raises(ValueError, match="between 1.011 and 16"):
        build_fiber(fiber_model=FiberModel.PENA, diameter=17.0, n_nodes=5)


def test_pena_warns_above_5_7(caplog):
    with caplog.at_level(logging.WARNING, logger="pyfibers.models.mrg"):
        build_fiber(fiber_model=FiberModel.PENA, diameter=10.0, n_nodes=5)
    assert "not recommended for fiber diameters above 5.7" in caplog.text


def test_small_mrg_deprecated():
    with pytest.warns(FutureWarning, match="SMALL_MRG_INTERPOLATION is deprecated"):
        fiber = build_fiber(fiber_model=FiberModel.SMALL_MRG_INTERPOLATION, diameter=2.0, n_nodes=5)
    assert fiber.nodecount == 5


def test_pena_sets_gnabar():
    fiber = build_fiber(fiber_model=FiberModel.PENA, diameter=2.0, n_nodes=5)
    active = fiber.nodes[2]
    assert np.isclose(active.gnabar_axnode_myel, 2.333333)
    assert np.isclose(active.gkbar_axnode_myel, 0.115556)


def test_sweeney_delta_z_is_100_times_d():
    with pytest.raises(AssertionError):
        build_fiber(fiber_model=FiberModel.SWEENEY, diameter=5.7, n_nodes=5, delta_z=100)
    fiber = build_fiber(fiber_model=FiberModel.SWEENEY, diameter=5.7, n_nodes=5)
    assert fiber.delta_z == pytest.approx(5.7 * 100)
    assert len(fiber.sections) == 2 * (fiber.nodecount - 1) + 1


def test_thio_ignores_passive_end_nodes():
    with pytest.warns(UserWarning, match="Ignoring passive_end_nodes"):
        fiber = build_fiber(fiber_model=FiberModel.THIO_AUTONOMIC, diameter=1.0, n_nodes=5, passive_end_nodes=True)
    assert fiber.passive_end_nodes is False


def test_tigerholm_ignores_passive_end_nodes():
    with pytest.warns(UserWarning, match="Ignoring passive_end_nodes"):
        fiber = build_fiber(fiber_model=FiberModel.TIGERHOLM, diameter=1.0, n_nodes=5, passive_end_nodes=True)
    assert fiber.passive_end_nodes is False


def test_schild94_vs_97_gating_maps():
    f94 = build_fiber(fiber_model=FiberModel.SCHILD94, diameter=1.0, n_nodes=5)
    f97 = build_fiber(fiber_model=FiberModel.SCHILD97, diameter=1.0, n_nodes=5)
    assert "j_naf" in f94.gating_variables
    assert "j_naf" not in f97.gating_variables
    assert f97.gating_variables["m_naf"] == "m_naf97mean"
    assert f94.gating_variables["m_naf"] == "m_naf"


def test_balance_sets_balanced_true():
    tiger = build_fiber(fiber_model=FiberModel.TIGERHOLM, diameter=1.0, n_nodes=5)
    thio = build_fiber(fiber_model=FiberModel.THIO_AUTONOMIC, diameter=1.0, n_nodes=5)
    assert tiger.balanced is True
    assert thio.balanced is True


def test_fibermodel_includes_pena():
    assert "PENA" in FiberModel.__members__
    assert "SMALL_MRG_INTERPOLATION" in FiberModel.__members__
    assert FiberModel.PENA.value is FiberModel.SMALL_MRG_INTERPOLATION.value
