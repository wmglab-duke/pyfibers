Note add badges

# wmglab-neuron
This package implements biophysical models of axons in the NEURON simulation environment using Python. With our package, you can model the responses of axons to electrical stimulation (e.g., find the minimum current amplitude required to activate or block an action potential). You can add your own fiber models and simulations protocols.

We have implemented the following models:
- MRG
- SUNDT
- TIGERHOLM
- RATTAY

## Installation
*Note that these instructions are temporary for development*
1. Install NEURON and add to PATH
2. Clone the package to your machine
3. Install the package using pip
   - Open anaconda prompt
   - Navigate to the repo root directory
   - `pip install -r requirements.txt` (check that dependencies are installed successfully)
   - `pip install .`
4. To compile the mod files, in Anaconda prompt, run the command `wmglab_neuron_install`
   - Careful! Make sure that there is not another NEURON installation in your path (this could happen, for example, if you were adding one to your path in your `.bash_profile`). To check, you can run `which nrnivmodl`, and the path that it prints should be to your python installation of NEURON.
   - Note: You cannot run compilation on DCC Desktop nodes, as there are libraries needed which are not available there. Make sure you are on a login node when running this command.
   - Note: if you receive a message that the wmglab_neuron_install command is not found, find the executable for this command in the `bin` path of your python directory and run it there (i.e., double-click on it)
   - After a few seconds, your terminal window should say "Successfully installed wmglab-neuron-\<Major>.\<Minor>.\<Patch>".
### Migrating from v0.0.1 to v0.0.2
1. Change all usages and imports of the Stimulation class to ScaledStim.
  - `from wmglab_neuron import ScaledStim`
  - `stimulation = ScaledStim(args)`
2. Remove potentials from instantiation of Stimulation, instead add to the relevant fiber (`fiber.potentials=potentials`)
3. Remove fiber as an argument to instantiation of Stimulation, instead pass the fiber when calling:
  - run_sim() (`stimulation.run_sim(stimamp,fiber)`)
  - find_threshold(`stimulation.find_threshold(fiber)`).
4. Replace imports of FiberBuilder and calls of FiberBuilder.generate() with the "build_fiber" function.
  - `from wmglab_neuron import build_fiber`
  - `fiber = build_fiber(args)`
## Usage
See the Documentation for detailed information on usage:
- [Tutorials](https://wmglab.pages.oit.duke.edu/wmglab-neuron/tutorials/index.html) on various operations.
- [API Documentation](https://wmglab.pages.oit.duke.edu/wmglab-neuron/autodoc/index.html) on function/class arguments and outputs.

### Creating a fiber model
Use the build_fiber function to create fiber object. The fiber object consists of NEURON sections with ion channel mechanisms inserted for the fiber model chosen when the object is initialized. The fiber object can be run in simulations using our provided `ScaledStim` class; alternatively, users can create their own custom simulation setup.

```python
# create fiber
from wmglab_neuron import build_fiber

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
