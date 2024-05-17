Note add badges

# PyFibers
This package implements biophysical models of axons in the NEURON simulation environment using Python. With our package, you can model the responses of axons to electrical stimulation (e.g., find the minimum current amplitude required to activate or block an action potential). You can add your own fiber models and simulations protocols.

We have implemented the following models:
- MRG
- SUNDT
- TIGERHOLM
- RATTAY

## Installation
*Note that these instructions are temporary for development*
*Currently, it is recommended that you use NEURON v8.2.3 and Python 3.11*
1. Install NEURON and add to PATH (https://nrn.readthedocs.io/en/latest/)
2. Clone the package to your machine
3. Switch to the current stable branch (currently v0.0.4, only use main if you are sure you need it.)
4. Install the package using pip
   - Open anaconda prompt
   - Navigate to the repo root directory
   - `pip install .` or if you plan to do package development `pip install .[dev]`, which will also install development dependencies
   - If you plan to develop the package also:
     - Install the dev dependencies with `pip install .[dev]`
     - Install pre-commit hooks with `pre-commit install`
5. To compile the mod files, in Anaconda prompt, run the command `pyfibers_compile`
   - Careful! Make sure that there is not another NEURON installation in your path (this could happen, for example, if you were adding one to your path in your `.bash_profile`). To check, you can run `which nrnivmodl`, and the path that it prints should be to your python installation of NEURON.
   - Note: You cannot run compilation on DCC Desktop nodes, as there are libraries needed which are not available there. Make sure you are on a login node when running this command.
   - Note: if you receive a message that the pyfibers_compile command is not found, find the executable for this command in the `bin` path of your python directory and run it there (i.e., double-click on it)

## Usage
See the Documentation for detailed information on usage:
- [Tutorials](https://wmglab.pages.oit.duke.edu/pyfibers/tutorials/index.html) on various operations.
- [API Documentation](https://wmglab.pages.oit.duke.edu/pyfibers/autodoc/index.html) on function/class arguments and outputs.

### Creating a fiber model
Use the build_fiber function to create fiber object. The fiber object consists of NEURON sections with ion channel mechanisms inserted for the fiber model chosen when the object is initialized. The fiber object can be run in simulations using our provided `ScaledStim` class; alternatively, users can create their own custom simulation setup.

```python
# create fiber
from pyfibers import build_fiber

fiber = build_fiber(
    fiber_model=FiberModel.MRG_DISCRETE,
    n_sections=133,
    temperature=37,
    diameter=5.7,
)
```
### Running a Simulatiion
Once you have a fiber object, you can create a `ScaledStim` instance. Provide a list of potentials to accompany the fiber and a time-varying waveform.
A `ScaledStim` instance requires the following arguments: fiber object (see previous step), a list of extracellular potential values at the midpoint of each fiber section, and a time varying waveform.
```python
# Create instance of ScaledStim class
fiber.potentials = potential_values
stimulation = ScaledStim(waveform=waveform, dt=time_step, tstop=time_stop)

# run threshold search
amp, _ = stimulation.find_threshold(fiber)
print(f"Threshold for 5.7 micron fiber: {amp} (mA)")
```
For more examples, see the documentation.

## Contributing
If you develop additional functionality that would be generally useful to our users, please open a Pull Request for us to review.

## Authors and acknowledgment
   - Developed at Duke University by Elie Farrah, Daniel Marshall, and Eric Musselman
   - Please see: \<Paper REF Forthcoming>

## License
See LICENSE
