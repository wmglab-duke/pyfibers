"""Example use case of wmglab_neuron."""

import sys

import numpy as np
from scipy.stats import norm

sys.path.append(r'C:\nrn\lib\python')

from src.wmglab_neuron import Fiber, Recording, Stimulation  # noqa: E402

# TODO: remove the sys.path.append (installation should add to python path) and the noqa: E402
nodecount = 133
# create curve of potentials
potentials = norm.pdf(np.linspace(-1, 1, nodecount), 0, 0.05)
# create biphasic square wave
waveform = np.concatenate((np.zeros(100), np.ones(100), np.zeros(100), -np.ones(100)))

# parameters
time_step = 0.001
time_stop = 50

# create fiber
fiber = Fiber(diameter=8.7, fiber_mode='MRG_DISCRETE', temperature=37, n_fiber_coords=nodecount, potentials=potentials)

# Create instance of Stimulation class
stimulation = Stimulation(fiber, waveform=waveform, dt=time_step, tstop=time_stop)

# initialize recoding class
recording = Recording(fiber)

# decide what to save
recording.set_save(vm=True, gating=False, istim=False)

# run threshold search
amp, ap = stimulation.find_threshold(recording)

print(f'Threshold: {amp} (mA)')

# get saved data
vm, gating, istim, aptimes, apcounts = recording.get_variables(fiber)

# run a finite amp
ap = stimulation.run_sim(5, recording)

# get data as a tuple
ampdata = recording.get_variables(fiber)
