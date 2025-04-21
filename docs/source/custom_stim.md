# Custom Simulation Code

This section provides examples of how to run custom simulations using fibers. These examples demonstrate different methods, including directly using NEURON's :class:`IClamp <neuron:IClamp>` and `h.continuerun()`, leveraging the {py:class}`~pyfibers.stimulation.Stimulation` class with a custom `run_sim()` function, and using the {py:meth}`~pyfibers.stimulation.Stimulation.pre_run_setup` method with manual assignment of extracellular potentials.

Before running any simulations, we must create a model fiber. See the [Fiber Tutorial](tutorials/1_create_fiber.ipynb) for information on how to do so. The tutorials assume that you have already created a model fiber called `fiber`.

## Custom Simulation with a Custom `run_sim()` Function

In this example, we demonstrate how to set up a custom simulation by providing a custom `run_sim` function to the {py:class}`~pyfibers.stimulation.Stimulation` class. You could also achieve this by creating a subclass and overriding the {py:meth}`~pyfibers.stimulation.Stimulation.run_sim` method.

```{note}
Note, to use custom `run_sim()` methods with threshold searches, the custom method should take stimulation amplitude as the first argument, and return the number of action potentials generated and the time of the last action potential.
```

2. **Define the custom `run_sim()` function**:

```python
def custom_run_sim(self, stimamp, fiber):
    print("Running custom simulation.")

    # Set up the simulation using Stimulation.pre_run_setup
    stimulation.pre_run_setup(fiber)

    # Example of a custom simulation loop
    for i in range(int(stimulation.tstop / stimulation.dt)):
        # Custom simulation logic here

        # Advance to the next time step
        h.fadvance()

    return n_aps, last_ap_time
```

3. **Set up the {py:class}`~pyfibers.stimulation.Stimulation` instance with the custom `run_sim()` function**:
```python
from pyfibers import Stimulation

stimulation = Stimulation(dt=0.001, tstop=20, custom_run_sim=custom_run_sim)

stimulation.run_sim(fiber)
```

---

## Custom `run_sim` in a Subclass of {py:class}`~pyfibers.stimulation.Stimulation`

In this example, we demonstrate how to create a subclass of {py:class}`~pyfibers.stimulation.Stimulation` and override the {py:meth}`~pyfibers.stimulation.Stimulation.run_sim` method to provide custom simulation logic. See {py:meth}`pyfibers.stimulation.ScaledStim.run_sim` and {py:meth}`pyfibers.stimulation.IntraStim.run_sim` for examples.

1. **Create a subclass of {py:class}`~pyfibers.stimulation.Stimulation`**:
```python
from pyfibers import Stimulation


class CustomStimulation(Stimulation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
```

2. **Override the {py:meth}`~pyfibers.stimulation.Stimulation.run_sim` method**:
```python
def run_sim(self, stimamp, fiber):  # defined as a method of CustomStimulation
    print("Running custom simulation:")

    self.pre_run_setup(fiber)

    # Example of a custom simulation loop
    for i in range(int(self.tstop / self.dt)):
        # Custom simulation logic here

        # Advance to next time step
        h.fadvance()

    return n_aps, last_ap_time
```

3. **Set up the `CustomStimulation` instance and run the simulation**:
```python
stimulation = CustomStimulation(dt=0.001, tstop=20)
stimulation.run_sim(fiber)
```

---

## Custom Simulation Using NEURON `IClamp` and `h.continuerun()`

In this example, we demonstrate how to set up a custom simulation using NEURON's :class:`IClamp <neuron:IClamp>` and `h.continuerun()` without using the simulation classes included in PyFibers.

2. **Set up and configure intracellular stimulation**:
```python
from neuron import h

# Apply stimulation to the 10th section of the fiber
stim = h.IClamp(fiber.sections[10](0.5))
stim.amp = 0.1  # nA
stim.delay = 1  # ms
stim.dur = 5  # ms
```

4. **Run the simulation**:
```python
# Initialize membrane potential using h.finitialize
h.finitialize(fiber.v_rest)
# Run the simulation for 20 ms using h.continuerun
h.continuerun(20)
```

---

## Custom Simulation Using {py:meth}`~pyfibers.stimulation.Stimulation.pre_run_setup` and Manual Extracellular Potentials

In this example, we demonstrate how to set up a custom simulation by using the {py:meth}`~pyfibers.stimulation.Stimulation.pre_run_setup` method and then manually assigning extracellular potentials in a loop.

2. **Set up the {py:class}`~pyfibers.stimulation.Stimulation` instance**:
```python
from pyfibers import Stimulation

stimulation = Stimulation(dt=0.001, tstop=20)

# Pre-run setup using Stimulation.pre_run_setup
stimulation.pre_run_setup(fiber)
```

3. **Manually assign extracellular potentials in the simulation loop**:
```python
stimamp = 0.1  # Example stimulation amplitude

for i in range(int(stimulation.tstop / stimulation.dt)):
    # Manually assign extracellular potentials using Stimulation._update_extracellular
    e_stims = [stimamp * pot for pot in fiber.potentials]
    stimulation._update_extracellular(fiber, e_stims)

    # Advance the simulation using h.fadvance
    h.fadvance()
```

---

These examples demonstrate different ways to set up and run custom simulations using fibers—with or without the {py:class}`~pyfibers.stimulation.Stimulation` class. Each method allows for flexibility in how the simulations are defined and executed.
