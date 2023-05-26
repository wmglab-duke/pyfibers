"""Tests for wmglab-neuron.

The copyrights of this software are owned by Duke University. Please
refer to the LICENSE and README.md files for licensing instructions. The
source code can be found on the following GitHub repository:
https://github.com/wmglab-duke/ascent
"""

import numpy as np
import pytest  # noqa: I900
from scipy.stats import norm

from wmglab_neuron import BoundsSearchMode, FiberModel, ScaledStim, TerminationMode, build_fiber

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


def test_bad_fiber_model():
    """Test that a bad fiber model raises an error."""
    with pytest.raises(ValueError):
        build_fiber(diameter=5.7, fiber_model='bad_model', temperature=37, n_sections=133)


def test_len():
    assert len(get_fiber()) == (133 - 1) / 11 + 1


def test_getitem():
    fiber = get_fiber()
    assert fiber[0] is fiber.nodes[0]


def test_iter():
    fiber = get_fiber()
    for i, node in enumerate(fiber):
        assert node is fiber.nodes[i]


def test_contains():
    fiber = get_fiber()
    assert fiber.nodes[0] in fiber
    assert fiber.sections[1] in fiber


def test_loc():
    fiber = get_fiber()
    assert fiber.loc(0) is fiber.nodes[0]
    assert fiber.loc(1) is fiber.nodes[-1]
    assert fiber.loc(0.5) is fiber.nodes[6]


def test_end_excitation():
    fiber = get_fiber()
    fiber.potentials = norm.pdf(np.linspace(-1, 1, 133), 0, 0.5) * 100
    waveform = np.concatenate((np.ones(200), -np.ones(200), np.zeros(49600)))
    time_step = 0.001
    time_stop = 5
    stimulation = ScaledStim(waveform=waveform, dt=time_step, tstop=time_stop)
    stimulation.run_sim(0, fiber)  # TODO why do I need to run this for correct result
    with pytest.raises(AssertionError):
        stimulation.find_threshold(fiber)


def test_pointsource():
    fiber = get_fiber()
    fiber.potentials = fiber.point_source_potentials(0, 100, 3000, 1, 1)
    assert np.isclose(fiber.potentials[66], 0.000753537379490885)


def test_waveform_pad_truncate():
    fiber = get_fiber()  # TODO figure out why this is needed and then delete # noqa: F841
    waveform = np.concatenate((np.ones(200), -np.ones(200), np.zeros(49600)))
    stimulation = ScaledStim(waveform=waveform, dt=0.001, tstop=5)
    assert len(stimulation.waveform) == 5000

    waveform = np.concatenate((np.ones(200), -np.ones(200), np.zeros(100)))
    stimulation = ScaledStim(waveform=waveform, dt=0.001, tstop=5)
    assert len(stimulation.waveform) == 5000


def test_mrg_discrete():
    assert np.isclose(get_activation_threshold(FiberModel.MRG_DISCRETE), -0.023414306640625)


def test_geometric_mean():
    assert np.isclose(
        get_activation_threshold(FiberModel.MRG_INTERPOLATION, bisection_mean='geometric'), -0.023501400846134893
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
