"""Example use case of wmglab_neuron."""

import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

sys.path.append(r'C:\nrn\lib\python')

from src.wmglab_neuron import FiberBuilder, FiberModel, Recording, Stimulation  # noqa: E402

# TODO: remove the sys.path.append (installation should add to python path) and the noqa: E402
nodecount = 133
# create curve of potentials
potentials = norm.pdf(np.linspace(-1, 1, nodecount), 0, 0.05) * 100

plt.plot(potentials)
# create biphasic square wave
waveform = np.concatenate((np.ones(200), -np.ones(200), np.zeros(49600)))

# parameters
time_step = 0.001
time_stop = 50

model = FiberModel.RATTAY

# create fiber
fiber = FiberBuilder.generate(diameter=5.7, fiber_model=model, temperature=37, n_fiber_coords=nodecount)

# Create instance of Stimulation class
stimulation = Stimulation(fiber, waveform=waveform, potentials=potentials, dt=time_step, tstop=time_stop)

# initialize recoding class
recording = Recording(fiber)

# decide what to save
recording.set_save(vm=True, gating=False, istim=False)

# run threshold search
amp, ap = stimulation.find_threshold(recording)

print(f'Threshold for 5.7 micron {model}: {amp} (mA)')

# get saved data
vm, gating, istim, aptimes, apcounts = recording.get_variables(fiber)

# run a finite amp
ap = stimulation.run_sim(5, recording)

# get data as a tuple
ampdata = recording.get_variables(fiber)
