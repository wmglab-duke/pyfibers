# PyFibers
*PyFibers paper citation will be available soon.*

[![CI](https://github.com/wmglab-duke/pyfibers/workflows/CI/badge.svg)](https://github.com/wmglab-duke/pyfibers/actions)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/pyfibers.svg)](https://pypi.org/project/pyfibers/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/pyfibers.svg)](https://pypi.org/project/pyfibers/)
<!-- [![PyPI - License](https://img.shields.io/pypi/l/pyfibers.svg)](https://pypi.org/project/pyfibers/) -->
[![Development Status](https://img.shields.io/badge/development%20status-beta-yellow.svg)](https://pypi.org/classifiers/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Documentation](https://img.shields.io/badge/docs-github%20pages-blue.svg)](https://wmglab-duke.github.io/pyfibers/)
[![Contributors](https://img.shields.io/github/contributors/wmglab-duke/pyfibers.svg)](https://github.com/wmglab-duke/pyfibers/graphs/contributors)
[![Issues](https://img.shields.io/github/issues/wmglab-duke/pyfibers.svg)](https://github.com/wmglab-duke/pyfibers/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/wmglab-duke/pyfibers.svg)](https://github.com/wmglab-duke/pyfibers/pulls)

This package implements biophysical models of axons in the NEURON simulation environment using Python. With our package, you can model the responses of axons to electrical stimulation (e.g., find the minimum current amplitude required to activate or block an action potential). You can add your own fiber models and simulations protocols. You can use analytical tools for extracellular potentials, or import from FEM (e.g., COMSOL).

We have implemented the following models:
- MRG (McIntyre, Richardson, and Grill)
   - MRG-discrete
   - MRG-interpolation
   - Small MRG-interpolation
- Thio
   - Autonomic
   - Cutaneous
- Sundt
- Tigerholm
- Rattay
- Schild
   - Schild 1994
   - Schild 1997

## Installation
Note: It is recommended you create a new virtual environment for PyFibers. For example, using Anaconda/Miniconda:
  - `conda create -n pyfibers`
  - `conda activate pyfibers`
1. Install NEURON and add to PATH (https://nrn.readthedocs.io/en/latest/)
   - Make sure your NEURON and Python versions are compatible (https://nrn.readthedocs.io/en/latest/changelog.html)
   - Check your installation by running the following command: `python -c "import neuron; neuron.test(); quit()"`. If successful, test outputs along with "OK" should be printed to the terminal.
2. Using git, clone PyFibers to your computer (Once the package is released, this step will be eliminated. Also skip if you downloaded or were provided a zipped copy of the repository)
3. Install the package using pip
   - Open your terminal where Python is callable (e.g., Anaconda Prompt)
   - Navigate to the repository root directory
   - Run the command `pip install .` if you do not plan to develop PyFibers code.
   - If you do plan to develop the package, instead do the following:
     - Install PyFibers alongside development dependencies with `pip install .[dev]` (If using a zsh shell, use the command `pip install .'[dev]'`)
     - Install pre-commit hooks with `pre-commit install`
4. To compile the mod files, in Anaconda prompt, run the command `pyfibers_compile`
   - It is normal to see the following message during compilation: `NEURON mechanisms not found in <path>.` Check the NEURON output that follows for a message that the mechanisms were compiled successfully. (e.g., for Windows: `nrnmech.dll was built successfully.`). In the future while using PyFibers, if you see the `NEURON mechanisms not found in <path>.` message, this is cause for concern, as this means PyFibers cannot find the compiled mechanisms. Failed compiles will commonly cause the error message `Argument not a density mechanism name` to appear when trying to create fibers.
   - Careful! Make sure that that the correct NEURON installation is in your path, as the first found installation will be used for compilation. The version used for compilation must be the same version used to run PyFibers code.
   - Note: if you receive a message that the `pyfibers_compile` command is not found, find the executable for this command in the `Scripts` path of your python directory (e.g. `C:\Users\<username>\Anaconda3\envs\pyfibers\Scripts`) and run the executable (e.g., `pyfibers_compile.exe`).


## Usage
📖 **Documentation**: For detailed information on usage, see our [documentation](https://wmglab-duke.github.io/pyfibers/).

- [Tutorials](https://wmglab-duke.github.io/pyfibers/tutorials/index.html) on various operations.
- [API Documentation](https://wmglab-duke.github.io/pyfibers/autodoc/index.html) on function/class arguments and outputs.

The basic steps for running a PyFibers simulation are as follows:
### Creating a model fiber
Use the build_fiber function to create fiber object. The fiber object consists of NEURON sections with ion channel mechanisms inserted for the fiber model chosen when the object is initialized. Users can add custom fiber models as well as using our provided models (See [Custom Fiber Models](https://wmglab-duke.github.io/pyfibers/custom_fiber.html))

```python
from pyfibers import build_fiber

fiber = build_fiber(
    fiber_model=FiberModel.MRG_DISCRETE,
    diameter=10,  # um
    n_nodes=25,  # um
    temperature=37,  # C
)
```
### Running a Simulation
The fiber object can be run in simulations using our provided `ScaledStim` class; alternatively, users can create their own custom simulation setup (See [Custom Simulations](https://wmglab-duke.github.io/pyfibers/custom_stim.html)). Once you have a fiber object, you can create a `ScaledStim` instance, which is a set of instructions for stimulating model fibers.

```python
# Add extracellular potentials
fiber.potentials = potential_values

# Create instance of ScaledStim class
stimulation = ScaledStim(waveform=waveform, dt=time_step, tstop=time_stop)

# run threshold search
amp, _ = stimulation.find_threshold(fiber)
print(f"Threshold for 5.7 micron fiber: {amp} (mA)")
```
For more examples, see the [documentation](https://wmglab-duke.github.io/pyfibers/).

## Contributing
If you develop additional functionality that would be generally useful to our users, please open a Pull Request for us to review.

## Authors and acknowledgment
   - Developed at Duke University by Daniel Marshall, Elie Farrah, and Eric Musselman
   - Please see: \<Paper REF Forthcoming>

## License
See LICENSE
