"""Tests for wmglab-neuron.

The copyrights of this software are owned by Duke University. Please
refer to the LICENSE and README.md files for licensing instructions. The
source code can be found on the following GitHub repository:
https://github.com/wmglab-duke/ascent
"""

import sys

import numpy as np
from scipy.stats import norm

from src.wmglab_neuron import Fiber, FiberModel, Recording, Stimulation

sys.path.append(r'C:\nrn\lib\python')
sys.path.append('..')
# TODO: maybe remove this append?


def test_activation_threshold():
    """Test activation threshold."""
    nodecount = 133
    # create curve of potentials
    potentials = norm.pdf(np.linspace(-1, 1, nodecount), 0, 0.05) * 100
    fiber = Fiber(
        diameter=8.7, fiber_model=FiberModel.MRG_DISCRETE, temperature=37, n_fiber_coords=133, potentials=potentials
    )

    waveform = np.concatenate((np.ones(200), -np.ones(200), np.zeros(49600)))

    # initialize recoding class
    recording = Recording(fiber)

    # decide what to save
    recording.set_save(vm=True, gating=True, istim=False)

    # parameters
    time_step = 0.001
    time_stop = 50
    stimulation = Stimulation(fiber, waveform=waveform, dt=time_step, tstop=time_stop)

    amp, ap = stimulation.find_threshold(recording)

    assert np.equal(amp, -0.023837280273437504)


# todo: test recording, test block threshold, and add coverage/test report to gitlab ci,
# TODO: also need to test everything option occuring in enums (including fiber models)
def test_stimamp():
    """Test activation threshold."""
    nodecount = 133
    # create curve of potentials
    potentials = norm.pdf(np.linspace(-1, 1, nodecount), 0, 0.05) * 100
    fiber = Fiber(
        diameter=8.7, fiber_model=FiberModel.MRG_DISCRETE, temperature=37, n_fiber_coords=133, potentials=potentials
    )

    waveform = np.concatenate((np.ones(200), -np.ones(200), np.zeros(49600)))

    # initialize recoding class
    recording = Recording(fiber)

    # decide what to save
    recording.set_save(vm=True, gating=True, istim=False)

    # parameters
    time_step = 0.001
    time_stop = 50
    stimulation = Stimulation(fiber, waveform=waveform, dt=time_step, tstop=time_stop)

    amp, ap = stimulation.find_threshold(recording)

    assert np.equal(amp, -0.023837280273437504)
