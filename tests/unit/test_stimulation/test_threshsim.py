"""Tests for Stimulation.threshsim branching.

The copyrights of this software are owned by Duke University.
See LICENSE for licensing instructions.
Source code: https://github.com/wmglab-duke/pyfibers
"""

from __future__ import annotations

import pytest

from pyfibers import FiberModel, Stimulation, ThresholdCondition, build_fiber


@pytest.fixture(scope="module")
def fiber():
    return build_fiber(fiber_model=FiberModel.MRG_INTERPOLATION, diameter=10.0, n_nodes=5)


def _stim_with_captured_run():
    stim = Stimulation(dt=0.001, tstop=1)
    captured = {}

    def fake_run_sim(stimamp, fiber_arg, **kwargs):
        captured.update(kwargs)
        captured["stimamp"] = stimamp
        captured["fiber"] = fiber_arg
        return 1, 2.0

    def fake_checker(*_args, **kwargs):
        captured["checker_kwargs"] = kwargs
        return True

    stim.run_sim = fake_run_sim
    stim.threshold_checker = fake_checker
    return stim, captured


def test_threshsim_activation_uses_supra_exit(fiber):
    stim, captured = _stim_with_captured_run()
    is_supra, ap_info = stim.threshsim(1.0, fiber, condition=ThresholdCondition.ACTIVATION, thresh_num_aps=1)
    assert is_supra is True
    assert ap_info == (1, 2.0)
    assert captured["fail_on_end_excitation"] is None
    assert captured["use_exit_t"] is True
    assert captured["exit_func"] == stim.supra_exit
    assert captured["exit_func_kws"] == {"thresh_num_aps": 1}


def test_threshsim_activation_multi_ap_disables_exit(fiber):
    stim, captured = _stim_with_captured_run()
    stim.threshsim(1.0, fiber, condition=ThresholdCondition.ACTIVATION, thresh_num_aps=2)
    assert captured["use_exit_t"] is True
    assert captured["exit_func"] is not stim.supra_exit
    assert captured["exit_func"](fiber, 0.9) is False


def test_threshsim_block_skips_early_exit(fiber):
    stim, captured = _stim_with_captured_run()
    stim.threshsim(1.0, fiber, condition=ThresholdCondition.BLOCK, block_delay=3.0)
    assert captured["fail_on_end_excitation"] is None
    assert "use_exit_t" not in captured
    assert "exit_func" not in captured
    assert captured["checker_kwargs"]["block"] is True
    assert captured["checker_kwargs"]["block_delay"] == 3.0
