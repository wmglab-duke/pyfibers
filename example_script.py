"""Example use case of wmglab_neuron."""

import os
import shutil
import subprocess
import sys
import time

import numpy as np

from src.wmglab_neuron import Fiber, Recording, Saving, Stimulation

# TODO need to add installation instructions for windows. cannot use defailt pip install neuron


def auto_compile(override: bool = False):
    """Compile NEURON files if they have not been compiled yet.

    :param override: if True, compile regardless of whether the files have already been compiled
    :return: True if ran compilation, False if not
    """  # TODO: do this during install?
    operating_sys = 'UNIX-LIKE' if any([s in sys.platform for s in ['darwin', 'linux']]) else 'WINDOWS'
    if (
        (not os.path.exists(os.path.join('x86_64')) and operating_sys == 'UNIX-LIKE')
        or (not os.path.exists(os.path.join('src', 'MOD_Files', 'nrnmech.dll')) and operating_sys == 'WINDOWS')
        or override
    ):
        print('compiling')
        os.chdir(os.path.join('src', 'MOD_Files'))
        exit_data = subprocess.run(['nrnivmodl'], shell=True, capture_output=True, text=True)
        if exit_data.returncode != 0:
            print(exit_data.stderr)
            sys.exit("Error in compiling of NEURON files. Exiting...")
        os.chdir('../..')

        shutil.copytree('src/MOD_Files/x86_64', 'x86_64')
        compiled = True
    else:
        print('skipped compile')
        compiled = False
    return compiled


auto_compile()

if not os.path.exists('data'):
    os.mkdir('data')

start_time = time.time()

# create gaussian curve of potentials
potentials = np.random.normal(0, 1, 1000)
# create biphasic square wave
waveform = np.concatenate((np.zeros(100), np.ones(100), np.zeros(100), -np.ones(100)))

time_step = 0.001
time_stop = 50

# todo: error if len(potentials) != node_count. Fiber generation occurs as part of init
fiber = Fiber(diameter=8.7, fiber_mode='MRG_DISCRETE', temperature=37)
fiber.generate(n_fiber_coords=133)

# Create instance of Stimulation class
stimulation = Stimulation(potentials_list=potentials, waveform_list=waveform, dt=time_step, tstop=time_stop)
stimulation.apply_intracellular(
    fiber,
    delay=0,
    pw=0,
    dur=0,
    freq=0,
    amp=0,
    ind=0,
)

saving = Saving(
    dt=time_step,
    fiber=fiber,
    space_vm=True,
    space_gating=True,
    space_times=[0, 0.1, 0.3],
    time_vm=True,
    time_gating=True,
    istim=True,
    locs=[0.1, 0.5, 0.9],
    end_ap_times=True,
    loc_min=0.1,
    loc_max=0.9,
    ap_end_thresh=-30,
    ap_loctime=True,
    runtime=True,
    output_path=os.path.join(os.getcwd(), 'data'),
)

recording = Recording(fiber)

fiber.run_protocol(
    stimulation,
    saving,
    recording,
    start_time,
    protocol_mode='ACTIVATION_THRESHOLD',
    amps=None,
    t_init_ss=-200,
    dt_init_ss=10,
)
