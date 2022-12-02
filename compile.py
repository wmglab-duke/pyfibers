"""Compile."""

import os
import shutil
import subprocess
import sys

sys.path.append(r'C:\nrn\lib\python')

# TODO need to add installation instructions for windows. cannot use defailt pip install neuron


def auto_compile(override: bool = False):
    """Compile NEURON files if they have not been compiled yet.

    :param override: if True, compile regardless of whether the files have already been compiled
    :return: True if ran compilation, False if not
    """  # TODO: do this during install?
    operating_sys = 'UNIX-LIKE' if any([s in sys.platform for s in ['darwin', 'linux']]) else 'WINDOWS'
    if (
        (not os.path.exists(os.path.join('x86_64')) and operating_sys == 'UNIX-LIKE')
        or (not os.path.exists(os.path.join('src', 'MOD', 'nrnmech.dll')) and operating_sys == 'WINDOWS')
        or override
    ):
        print('compiling')
        os.chdir(os.path.join('src', 'MOD'))
        exit_data = subprocess.run(['nrnivmodl'], shell=True, capture_output=True, text=True)
        if exit_data.returncode != 0:
            print(exit_data.stderr)
            sys.exit("Error in compiling of NEURON files. Exiting...")
        os.chdir('../..')

        shutil.copytree('src/MOD/x86_64', 'x86_64')
        compiled = True
    else:
        print('skipped compile')
        compiled = False
    return compiled


auto_compile()
