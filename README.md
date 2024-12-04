# PyFibers
This package implements biophysical models of axons in the NEURON simulation environment using Python. With our package, you can model the responses of axons to electrical stimulation (e.g., find the minimum current amplitude required to activate or block an action potential). You can add your own fiber models and simulations protocols.

We have implemented the following models:
- MRG (McIntyre, Richardson, and Grill)
   - MRG-discrete
   - MRG-interpolation
   - Small MRG-interpolation
- Sundt
- Tigerholm
- Rattay
- Schild
   - Schild 1994
   - Schild 1997

## Installation
*Note that these instructions are temporary for development*
1. Install NEURON and add to PATH (https://nrn.readthedocs.io/en/latest/)
   - Make sure your NEURON and Python versions are compatible (https://nrn.readthedocs.io/en/latest/changelog.html)\
   - Check your installation by running the following command: `python -c "import neuron; neuron.test(); quit()"`. If successful, test outputs along with "OK" should be printed to the terminal.
2. Using git, clone PyFibers to your computer (Once the package is released, this step will be eliminated. Also skip if you downloaded or were provided a zipped copy of the repository)
3. Install the package using pip
   - Open anaconda prompt
   - Recommended: Create a new environment for running PyFibers code:
      - `conda create -n pyfibers`
      - `conda activate pyfibers`
   - Navigate to the repository root directory
   - Run the command `pip install .` if you do not plan to develop PyFibers code.
   - If you do plan to develop the package, instead do the following:
     - Install PyFibers alongside development dependencies with `pip install .[dev]` (If using a zsh shell, use the command `pip install .'[dev]'`)
     - Install pre-commit hooks with `pre-commit install`
4. To compile the mod files, in Anaconda prompt, run the command `pyfibers_compile`
   - Careful! Make sure that that the correct NEURON installation is in your path, as the first found installation will be used for compilation. The version used for compilation must be the same version used to run PyFibers code.
   - Note: if you receive a message that the `pyfibers_compile` command is not found, find the executable for this command in the `Scripts` path of your python directory (e.g. `C:\Users\<username>\Anaconda3\envs\pyfibers\Scripts`) and run the executable (e.g., `pyfibers_compile.exe`).

## Usage
See the Documentation for detailed information on usage:
- [Tutorials](https://wmglab.pages.oit.duke.edu/wmglab-neuron/tutorials/index.html) on various operations.
- [API Documentation](https://wmglab.pages.oit.duke.edu/wmglab-neuron/autodoc/index.html) on function/class arguments and outputs.

The basic steps for running a PyFibers simulation are as follows:
### Creating a fiber model
Use the build_fiber function to create fiber object. The fiber object consists of NEURON sections with ion channel mechanisms inserted for the fiber model chosen when the object is initialized. Users can add custom fiber models as well as using our provided models (See [Custom Fiber Models](https://wmglab.pages.oit.duke.edu/wmglab-neuron/custom.html#how-to-create-a-new-fiber-model))

```python
from pyfibers import build_fiber

fiber = build_fiber(
    fiber_model=FiberModel.MRG_DISCRETE,
    n_sections=133,
    temperature=37,
    diameter=5.7,
)
```
### Running a Simulatiion
The fiber object can be run in simulations using our provided `ScaledStim` class; alternatively, users can create their own custom simulation setup (See [Custom Simulations](https://wmglab.pages.oit.duke.edu/wmglab-neuron/custom.html#custom-simulations)). Once you have a fiber object, you can create a `ScaledStim` instance, which is a set of instructions for stimulating model fibers.

```python
# Add extracellular potentials
fiber.potentials = potential_values

# Create instance of ScaledStim class
stimulation = ScaledStim(waveform=waveform, dt=time_step, tstop=time_stop)

# run threshold search
amp, _ = stimulation.find_threshold(fiber)
print(f"Threshold for 5.7 micron fiber: {amp} (mA)")
```
For more examples, see the documentation.

## Contributing
If you develop additional functionality that would be generally useful to our users, please open a Pull Request for us to review.

## Authors and acknowledgment
   - Developed at Duke University by Daniel Marshall, Elie Farrah, and Eric Musselman
   - Please see: \<Paper REF Forthcoming>

## License
See LICENSE
