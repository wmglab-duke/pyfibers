"""Tests for wmglab-neuron.

The copyrights of this software are owned by Duke University. Please
refer to the LICENSE and README.md files for licensing instructions. The
source code can be found on the following GitHub repository:
https://github.com/wmglab-duke/ascent
"""

import numpy as np
from scipy.stats import norm

from src.wmglab_neuron import FiberBuilder, FiberModel, Stimulation

# TODO: maybe remove this append?


def get_activation_threshold(model):
    """Get activation threshold."""
    nodecount = 133
    # create curve of potentials
    potentials = norm.pdf(np.linspace(-1, 1, nodecount), 0, 0.05) * 100
    fiber = FiberBuilder.generate(diameter=5.7, fiber_model=model, temperature=37, n_fiber_coords=133)

    waveform = np.concatenate((np.ones(200), -np.ones(200), np.zeros(49600)))

    # parameters
    time_step = 0.001
    time_stop = 50
    stimulation = Stimulation(fiber, waveform=waveform, potentials=potentials, dt=time_step, tstop=time_stop)

    amp, ap = stimulation.find_threshold()

    return amp


def get_amp_responses(model, stimamps, save=False):
    """Get activation threshold."""
    nodecount = 133
    # create curve of potentials
    potentials = norm.pdf(np.linspace(-1, 1, nodecount), 0, 0.05) * 100
    fiber = FiberBuilder.generate(diameter=5.7, fiber_model=model, temperature=37, n_fiber_coords=133)

    if save:
        fiber.set_save_gating()
        fiber.set_save_vm()

    waveform = np.concatenate((np.ones(200), -np.ones(200), np.zeros(49600)))

    # parameters
    time_step = 0.001
    time_stop = 50
    stimulation = Stimulation(fiber, waveform=waveform, potentials=potentials, dt=time_step, tstop=time_stop)

    aps = [stimulation.run_sim(stimamp) for stimamp in stimamps]

    return aps, fiber


def test_mrg_discrete():
    assert np.isclose(get_activation_threshold(FiberModel.MRG_DISCRETE), -0.023414306640625)


def test_mrg_interp():
    assert np.isclose(get_activation_threshold(FiberModel.MRG_INTERPOLATION), -0.02353515625)


def test_tigerholm():
    assert np.isclose(get_activation_threshold(FiberModel.TIGERHOLM), -1.2818437500000002)


def test_rattay():
    assert np.isclose(get_activation_threshold(FiberModel.RATTAY), -0.042266845703125)


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


# TODO: below tests enable
# def test_sundt():
#     assert np.isclose(get_activation_threshold(FiberModel.SUNDT), float)

# def test_schild94():
#     assert np.isclose(get_activation_threshold(FiberModel.SCHILD94), float)

# def test_schild97():
#     assert np.isclose(get_activation_threshold(FiberModel.SCHILD97), float)
