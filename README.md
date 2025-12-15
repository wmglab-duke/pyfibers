# PyFibers

**PyFibers Paper**: Marshall DP, Farah ES, Musselman ED, Pelot NA, Grill WM (2025) PyFibers: An open-source NEURON-Python package to simulate responses of model nerve fibers to electrical stimulation. PLoS Comput Biol 21(12): e1013764. https://doi.org/10.1371/journal.pcbi.1013764

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/wmglab-duke/pyfibers)
[![Documentation](https://img.shields.io/badge/Documentation-Latest-blue?logo=read-the-docs)](https://wmglab-duke.github.io/pyfibers/)
[![DOI](https://zenodo.org/badge/1010198505.svg)](https://doi.org/10.5281/zenodo.17068760)
[![Contributors](https://img.shields.io/github/contributors/wmglab-duke/pyfibers.svg)](https://github.com/wmglab-duke/pyfibers/graphs/contributors)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/pyfibers.svg)](https://pypi.org/project/pyfibers/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/pyfibers.svg)](https://pypi.org/project/pyfibers/)
<!-- [![PyPI - License](https://img.shields.io/pypi/l/pyfibers.svg)](https://pypi.org/project/pyfibers/) -->
[![Development Status](https://img.shields.io/badge/development%20status-beta-yellow.svg)](https://pypi.org/classifiers/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Documentation](https://img.shields.io/badge/docs-github%20pages-blue.svg)](https://wmglab-duke.github.io/pyfibers/)
[![Issues](https://img.shields.io/github/issues/wmglab-duke/pyfibers.svg)](https://github.com/wmglab-duke/pyfibers/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/wmglab-duke/pyfibers.svg)](https://github.com/wmglab-duke/pyfibers/pulls)
[![CI](https://github.com/wmglab-duke/pyfibers/workflows/CI/badge.svg)](https://github.com/wmglab-duke/pyfibers/actions)

This package implements biophysical models of axons in the NEURON simulation environment using Python.
With our package, you can model the responses of axons to electrical stimulation (e.g., find the minimum current amplitude required to activate or block an action potential).
You can add your own fiber models and simulations protocols.
You can use analytical tools for extracellular potentials, or import from finite element models (FEM) such as COMSOL, ANSYS, or FEniCS.

| Feature | Description |
|---------|-------------|
| **Flexible stimulation** | Support for custom waveforms and extracellular potential distributions |
| **FEM integration** | Easy import of high-resolution potentials from finite element simulations |
| **1D and 3D fibers** | Support for both straight and curved fiber geometries |
| **Advanced analysis** | Built-in threshold search, conduction velocity measurement, and comprehensive data recording |
| **Extensible** | Add your own fiber models and simulation protocols |
| **Simulate recording** | Simple tools to calculate single fiber action potentials |
| **Library of built-in fiber models** | **MRG** (Myelinated): MRG-discrete, MRG-interpolation, Peña (Small MRG-interpolation)<br>**Sweeney** (Myelinated)<br>**Thio** (Unmyelinated): Autonomic, Cutaneous<br>**Sundt** (Unmyelinated)<br>**Tigerholm** (Unmyelinated)<br>**Rattay** (Unmyelinated)<br>**Schild** (Unmyelinated): Schild 1994, Schild 1997 |

## Installation

Note that these installation instructions are for users. Developer instructions are available in [contributing.md](https://github.com/wmglab-duke/pyfibers/blob/main/contributing.md).

It is recommended (But not required) you create a new virtual environment for PyFibers. For example, using Anaconda/Miniconda:
  - `conda create -n pyfibers`
  - `conda activate pyfibers`
1. Install NEURON and add to PATH ([https://nrn.readthedocs.io/en/latest/](https://nrn.readthedocs.io/en/latest/))
   - Make sure your NEURON and Python versions are compatible ([https://nrn.readthedocs.io/en/latest/changelog.html](https://nrn.readthedocs.io/en/latest/changelog.html))
   - Check your installation by running the following command: `python -c "import neuron; neuron.test(); quit()"`. If successful, test outputs along with "OK" should be printed to the terminal.
2. Install PyFibers from PyPI and compile the `.mod` files.
   ```bash
   pip install pyfibers
   pyfibers_compile
   ```

Some notes for `pyfibers_compile`:
- It is normal to see the following message during compilation: `NEURON mechanisms not found in <path>.` Check the NEURON output that follows for a message that the mechanisms were compiled successfully (e.g., for Windows: `nrnmech.dll was built successfully.`) In the future while using PyFibers, if you see the `NEURON mechanisms not found in <path>.` message, this is cause for concern, as this means PyFibers cannot find the compiled mechanisms. Failed compiles will commonly cause the error message `Argument not a density mechanism name` to appear when trying to create fibers.
- Careful! Make sure that the correct NEURON installation is in your path, as the first found installation will be used for compilation. The version used for compilation must be the same version used to run PyFibers code.
- If you receive a message that the `pyfibers_compile` command is not found, find the executable for this command in the `Scripts` path of your python directory (e.g. `C:\Users\<username>\Anaconda3\envs\pyfibers\Scripts`) and run the executable (e.g., `pyfibers_compile.exe`).


## Usage
📖 **Documentation**: For detailed information on usage, see our [documentation](https://wmglab-duke.github.io/pyfibers/):
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
# Add extracellular potentials to the fiber.
fiber.potentials = potential_values

# Create instance of ScaledStim class
stimulation = ScaledStim(waveform=waveform, dt=time_step, tstop=time_stop)

# run threshold search
amp, _ = stimulation.find_threshold(fiber)
print(f"Threshold for 5.7 micron fiber: {amp} (mA)")
```
For more examples, see the [documentation](https://wmglab-duke.github.io/pyfibers/).

## Logging

PyFibers provides helpful logging messages during simulations. By default, logging is disabled to avoid interfering with your application's logging configuration.

To enable logging:

```python
import pyfibers

pyfibers.enable_logging()  # Enable INFO level logging
```

For detailed logging configuration options, see the [API documentation](https://wmglab-duke.github.io/pyfibers/autodoc/index.html).

## Contributing
If you develop additional functionality that would be generally useful to our users, please open a Pull Request for us to review. For detailed contribution guidelines, see our [contributing guide](https://github.com/wmglab-duke/pyfibers/blob/main/contributing.md).

## Authors and acknowledgment
   - Developed at Duke University by Daniel Marshall, Elie Farah, and Eric Musselman
   - Associated publication: [https://doi.org/10.1371/journal.pcbi.1013764](https://doi.org/10.1371/journal.pcbi.1013764)

## License
See LICENSE
