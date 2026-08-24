"""Tests for ap_checker, threshold_checker, and supra_exit.

The copyrights of this software are owned by Duke University.
See LICENSE for licensing instructions.
Source code: https://github.com/wmglab-duke/pyfibers
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pyfibers import FiberModel, Stimulation, build_fiber
from pyfibers.fiber import Fiber


class FakeSection:
    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name


class FakeAPC:
    def __init__(self, n, time=0.0):
        self.n = n
        self.time = time


def make_fiber(ap_ns, ap_times=None, names=None):
    fiber = MagicMock(spec=Fiber)
    n_nodes = len(ap_ns)
    times = ap_times if ap_times is not None else [0.0] * n_nodes
    fiber.apc = [FakeAPC(n, t) for n, t in zip(ap_ns, times)]
    fiber.nodes = [FakeSection(names[i] if names else f"node {i}") for i in range(n_nodes)]
    fiber.sections = fiber.nodes
    fiber.loc_index.side_effect = lambda loc, target="nodes": int(loc * (n_nodes - 1))
    fiber.__getitem__.side_effect = lambda item: fiber.nodes[item]
    return fiber


def test_activation_supra_when_n_ge_thresh():
    fiber = make_fiber([0] * 8 + [2, 0])
    assert Stimulation.threshold_checker(fiber, ap_detect_location=0.9, thresh_num_aps=1) is True
    assert Stimulation.threshold_checker(fiber, ap_detect_location=0.9, thresh_num_aps=3) is False


def test_thresh_num_aps_must_be_positive():
    fiber = make_fiber([1] * 10)
    with pytest.raises(ValueError, match="thresh_num_aps must be positive"):
        Stimulation.threshold_checker(fiber, thresh_num_aps=0)


def test_block_requires_thresh_num_aps_one():
    fiber = make_fiber([1] * 10, ap_times=[5.0] * 10)
    with pytest.raises(NotImplementedError, match="thresh_num_aps=1"):
        Stimulation.threshold_checker(fiber, block=True, thresh_num_aps=2)


def test_block_no_aps_raises():
    fiber = make_fiber([0] * 10)
    with pytest.raises(RuntimeError, match="No APs detected for block threshold"):
        Stimulation.threshold_checker(fiber, block=True)


def test_block_supra_when_last_ap_before_delay():
    fiber = make_fiber([0] * 8 + [1, 0], ap_times=[0] * 8 + [4.0, 0])
    assert Stimulation.threshold_checker(fiber, block=True, block_delay=5, ap_detect_location=0.9) is True
    assert Stimulation.threshold_checker(fiber, block=True, block_delay=3, ap_detect_location=0.9) is False


def test_ap_checker_passive_node_warns():
    names = [f"node {i}" for i in range(10)]
    names[8] = "passive node 8"
    fiber = make_fiber([1] * 10, ap_times=[5.0] * 10, names=names)
    with pytest.warns(UserWarning, match="passive node"):
        Stimulation.ap_checker(fiber, ap_detect_location=0.9)


def test_ap_checker_t_le_0_raises():
    fiber = make_fiber([0] * 8 + [1, 0], ap_times=[0] * 8 + [-0.1, 0])
    with pytest.raises(RuntimeError, match="t<=0"):
        Stimulation.ap_checker(fiber, ap_detect_location=0.9)


def test_ap_checker_t_zero_does_not_raise():
    """Time==0 is falsy, so the t<=0 guard does not fire."""
    fiber = make_fiber([0] * 8 + [1, 0], ap_times=[0] * 8 + [0.0, 0])
    n, t = Stimulation.ap_checker(fiber, ap_detect_location=0.9)
    assert n == 1
    assert t == 0.0


def test_ap_checker_virtual_anode_warns():
    ns = [1] + [0] * 9
    fiber = make_fiber(ns, ap_times=[5.0] + [0] * 9)
    with pytest.warns(UserWarning, match="virtual anode"):
        Stimulation.ap_checker(fiber, ap_detect_location=0.9, check_all_apc=True)


def test_supra_exit_disables_check_all_apc():
    ns = [1] + [0] * 9
    fiber = make_fiber(ns, ap_times=[5.0] + [0] * 9)
    stim = object.__new__(Stimulation)
    result = stim.supra_exit(fiber, ap_detect_location=0.9, thresh_num_aps=1)
    assert result is False


@pytest.fixture(scope="module")
def real_fiber():
    return build_fiber(fiber_model=FiberModel.MRG_INTERPOLATION, diameter=10.0, n_nodes=5)


def test_threshsim_unknown_condition_returns_none(real_fiber):
    stim = Stimulation(dt=0.001, tstop=1)
    assert stim.threshsim(1.0, real_fiber, condition="not_a_condition") is None
