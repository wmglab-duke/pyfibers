"""Tests for wmglab-neuron.

The copyrights of this software are owned by Duke University. Please
refer to the LICENSE and README.md files for licensing instructions. The
source code can be found on the following GitHub repository:
https://github.com/wmglab-duke/ascent
"""

from __future__ import annotations

import numpy as np
import pytest  # noqa: I900
import scipy.signal as sg
from scipy.stats import norm

from wmglab_neuron import (
    BisectionMean,
    BoundsSearchMode,
    FiberModel,
    ScaledStim,
    TerminationMode,
    ThresholdCondition,
    build_fiber,
)

# TODO Change all c fiber model to 1 um


def get_fiber(diameter=5.7, fiber_model=FiberModel.MRG_INTERPOLATION, temperature=37, n_sections=133):
    return build_fiber(diameter=diameter, fiber_model=fiber_model, temperature=temperature, n_sections=n_sections)


def get_activation_threshold(model, nodecount=133, diameter=5.7, **kwargs):  # TODO test range of diameters
    """Get activation threshold."""

    # create curve of potentials
    fiber = build_fiber(diameter=diameter, fiber_model=model, temperature=37, n_sections=nodecount)
    fiber.potentials = norm.pdf(np.linspace(-1, 1, nodecount), 0, 0.05) * 100

    waveform = np.concatenate((np.ones(200), -np.ones(200), np.zeros(49600)))

    # parameters
    time_step = 0.001
    time_stop = 50
    stimulation = ScaledStim(waveform=waveform, dt=time_step, tstop=time_stop)

    stimulation.run_sim(0, fiber)  # TODO why do I need to run this first for correct result

    amp, ap = stimulation.find_threshold(fiber, **kwargs)

    return amp


def get_activation_threshold_ps(model, nodecount=133, diameter=5.7, **kwargs):  # TODO test range of diameters
    """Get activation threshold.

    Using point source
    """

    # create curve of potentials
    fiber = build_fiber(diameter=diameter, fiber_model=model, temperature=37, n_sections=nodecount)
    fiber.potentials = fiber.point_source_potentials(0, 250, fiber.length / 2, 1, 0.01)

    waveform = np.concatenate((np.ones(200), -np.ones(200), np.zeros(49600)))

    # parameters
    time_step = 0.001
    time_stop = 50
    stimulation = ScaledStim(waveform=waveform, dt=time_step, tstop=time_stop)

    stimulation.run_sim(0, fiber)  # TODO why do I need to run this first for correct result

    amp, ap = stimulation.find_threshold(fiber, **kwargs)

    return amp


def get_amp_responses(model, stimamps, save=False):
    """Get activation threshold."""
    nodecount = 133
    # create curve of potentials
    fiber = build_fiber(diameter=5.7, fiber_model=model, temperature=37, n_sections=133)
    fiber.potentials = norm.pdf(np.linspace(-1, 1, nodecount), 0, 0.05) * 100

    if save:
        fiber.set_save_gating()
        fiber.set_save_vm()

    waveform = np.concatenate((np.ones(200), -np.ones(200), np.zeros(49600)))

    # parameters
    time_step = 0.001
    time_stop = 5
    stimulation = ScaledStim(waveform=waveform, dt=time_step, tstop=time_stop)

    stimulation.run_sim(0, fiber)  # TODO why do I need to run this first for correct result

    result = [stimulation.run_sim(stimamp, fiber) for stimamp in stimamps]

    aps = [r[0] for r in result]

    return aps, fiber


def test_end_excitation():
    fiber = get_fiber()
    fiber.potentials = norm.pdf(np.linspace(-1, 1, 133), 0, 0.5) * 100
    waveform = np.concatenate((np.ones(200), -np.ones(200), np.zeros(49600)))
    time_step = 0.001
    time_stop = 5
    stimulation = ScaledStim(waveform=waveform, dt=time_step, tstop=time_stop)
    stimulation.run_sim(0, fiber)  # TODO why do I need to run this for correct result
    with pytest.raises(RuntimeError):
        stimulation.find_threshold(fiber)


def test_mrg_discrete():
    assert np.isclose(get_activation_threshold(FiberModel.MRG_DISCRETE), -0.023414306640625)


def test_geometric_mean():
    assert np.isclose(
        get_activation_threshold(FiberModel.MRG_INTERPOLATION, bisection_mean=BisectionMean.GEOMETRIC),
        -0.023501400846134893,
    )


def test_both_subthreshold():
    assert np.isclose(
        get_activation_threshold(FiberModel.MRG_INTERPOLATION, stimamp_top=-0.01, stimamp_bottom=-0.001),
        -0.023512489759687515,
    )


def test_both_suprathreshold():
    assert np.isclose(
        get_activation_threshold(FiberModel.MRG_INTERPOLATION, stimamp_top=-1, stimamp_bottom=-0.1),
        -0.02351225891204326,
    )


def test_mrg_interp():
    assert np.isclose(get_activation_threshold(FiberModel.MRG_INTERPOLATION), -0.02353515625)


def test_allsupra():
    assert np.isclose(
        get_activation_threshold(FiberModel.MRG_INTERPOLATION, stimamp_top=-0.025, stimamp_bottom=-0.02), -0.023515625
    )


def test_absolutes():
    assert np.isclose(
        get_activation_threshold(
            FiberModel.MRG_INTERPOLATION,
            termination_mode=TerminationMode.ABSOLUTE_DIFFERENCE,
            bounds_search_mode=BoundsSearchMode.ABSOLUTE_INCREMENT,
            termination_tolerance=0.0001,
        ),
        -0.02350494384765625,
    )


def test_tigerholm():
    assert np.isclose(get_activation_threshold(FiberModel.TIGERHOLM), -1.2818437500000002)


def test_rattay():
    assert np.isclose(get_activation_threshold(FiberModel.RATTAY), -0.042266845703125)


def test_sundt():
    assert np.isclose(get_activation_threshold(FiberModel.SUNDT, diameter=0.2, nodecount=665), -0.6867578125)


def test_schild94():
    assert np.isclose(
        get_activation_threshold_ps(FiberModel.SCHILD94, diameter=1, nodecount=265, stimamp_top=-50), -17.828701171875
    )


def test_schild97():
    assert np.isclose(
        get_activation_threshold_ps(FiberModel.SCHILD97, diameter=1, nodecount=265, stimamp_top=-50), -43.1654296875
    )


def test_finite_amps():
    assert np.array_equal(np.array(get_amp_responses(FiberModel.MRG_INTERPOLATION, [0.01, -0.1, -1])[0]), [0, 1, 1])


def test_vm():
    _, fiber = get_amp_responses(FiberModel.MRG_INTERPOLATION, [-1], save=True)
    assert np.isclose(fiber.vm[6][200], 388.5680657889212)


def test_gating():
    _, fiber = get_amp_responses(FiberModel.MRG_INTERPOLATION, [-1], save=True)
    assert np.isclose(fiber.gating['h'][6][200], 0.0014151250180505406)
    assert np.isclose(fiber.gating['m'][6][200], 0.9999924898989334)
    assert np.isclose(fiber.gating['mp'][6][200], 0.9999961847550823)
    assert np.isclose(fiber.gating['s'][6][200], 0.9090909090909091)


def test_block_threshold():
    time_step = 0.0005  # ms
    time_stop = 50  # ms
    frequency = 20  # khz (because our time units are ms)
    # create time vector to make waveform with sg
    t = np.arange(0, time_stop, time_step)  # ms
    waveform = sg.square(2 * np.pi * frequency * t)

    n_sections = 265

    fiber = build_fiber(FiberModel.MRG_INTERPOLATION, diameter=10, n_sections=n_sections)
    fiber.set_save_vm()
    fiber.potentials = fiber.point_source_potentials(0, 250, fiber.length / 2, 1, 0.01)

    # Create new stimulation object
    stimulation = ScaledStim(waveform=waveform, dt=time_step, tstop=time_stop)

    stimulation.set_intracellular_stim(delay=10, pw=0.5, dur=40, freq=100, amp=1, ind=2)

    amp, _ = stimulation.find_threshold(fiber, stimamp_top=-3, istim_delay=10, condition=ThresholdCondition.BLOCK)

    assert np.isclose(amp, -2.7897656250000002)

    n, t = stimulation.run_sim(0, fiber)

    assert n == 4.0 and np.isclose(t, 40.5215)
