# Custom Simulation Code
This section provides examples of how to run custom simulations using fibers. These examples demonstrate different methods, including directly using NEURON's IClamp and h.run, leveraging the Stimulation class with a custom run_sim function, and using the Stimulation.pre_run_setup method with manual assignment of extracellular potentials.

Before running any simulations, we must create a model fiber. See the [Fiber Tutorial](tutorials/1_create_fiber.ipynb) for information on how to do so. The tutorials will assume that you have already created a model fiber called `fiber`.

## Custom Simulation with a Custom `run_sim` Function

In this example, we demonstrate how to set up a custom simulation by providing a custom `run_sim` function to the `Stimulation` class. You could also achieve this by creating a subclass and overriding the `run_sim` method.

```{note}
Note, to use custom `run_sim` methods with threshold searches, the custom method should take stimulation amplitude as the first argument, and return the number of action potentials generated and the time of the last action potential.
```

2. **Define the custom `run_sim` function**:


    ```python
    def custom_run_sim(self, stimamp, fiber):
        print("Running custom simulation.")

        stimulation.pre_run_setup(fiber)

        # Example of a custom simulation loop
        for i in range(int(stimulation.tstop / stimulation.dt)):
            # Custom simulation logic here

            # Advance to next time step
            h.fadvance()

        return n_aps, last_ap_time
    ```

3. **Set up the `Stimulation` instance with the custom `run_sim` function**:
    ```python
    from pyfibers import Stimulation

    stimulation = Stimulation(dt=0.001, tstop=20, custom_run_sim=custom_run_sim)

    stimulation.run_sim(fiber)
    ```

---

## Custom `run_sim` in a subclass of `Stimulation`

In this example, we demonstrate how to create a subclass of `Stimulation` and override the `run_sim` method to provide custom simulation logic. See  {py:class}`pyfibers.stimulation.ScaledStim` and {py:class}`pyfibers.stimulation.IntraStim` for examples of how to do this.

1. **Create a subclass of `Stimulation`**:
    ```python
    from pyfibers import Stimulation


    class CustomStimulation(Stimulation):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
    ```

2. **Override the `run_sim` method**:
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

## Custom Simulation Using NEURON IClamp and `h.run`

In this example, we demonstrate how to set up a custom simulation using the NEURON `IClamp` and `h.run` without using the simulation classes included in PyFibers.

2. **Set up and configure intracellular stimulation**:
    ```python
    from neuron import h

    stim = h.IClamp(fiber.sections[10](0.5))  # Apply stimulation to the 10th section
    stim.amp = 0.1  # nA
    stim.delay = 1  # ms
    stim.dur = 5  # ms
    ```

4. **Run the simulation**:
    ```python
    h.finitialize(fiber.v_rest)  # Initialize membrane potential
    h.continuerun(20)  # Run the simulation for 20 ms
    ```

---

## Custom Simulation Using `Stimulation.pre_run_setup` and Manual Extracellular Potentials
In this example, we demonstrate how to set up a custom simulation by using the `Stimulation.pre_run_setup` method and then manually assigning extracellular potentials in a loop.

2. **Set up the `Stimulation` instance**:
    ```python
    from pyfibers import Stimulation

    stimulation = Stimulation(dt=0.001, tstop=20)

    # Pre-run setup
    stimulation.pre_run_setup(fiber)
    ```

3. **Manually assign extracellular potentials in the simulation loop**:
    ```python
    stimamp = 0.1  # Example stimulation amplitude

    for i in range(int(stimulation.tstop / stimulation.dt)):
        # Manually assign extracellular potentials
        e_stims = [stimamp * pot for pot in fiber.potentials]
        stimulation._update_extracellular(fiber, e_stims)

        # Advance the simulation
        h.fadvance()
    ```

---

These examples demonstrate different ways to set up and run custom simulations using fibers with or without the `Stimulation` class. Each method allows for flexibility in how the simulations are defined and executed.
