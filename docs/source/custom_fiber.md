# Custom Fiber Models

Creating a new fiber model involves defining a subclass of {py:class}`~pyfibers.fiber.Fiber` and implementing methods that describe the specific mechanisms and ultrastructure of the fiber. This section provides a step-by-step guide on how to create a new fiber model.

```{note}
You may create a fiber model in the `/models` directory, or in a separate folder (e.g., `/plugins`) and make it discoverable as a plugin (recommended). To make your fiber model discoverable as a plugin, follow [these instructions](#making-your-fiber-model-discoverable-as-a-plugin).
```

The implementation of a custom fiber model requires custom .MOD file(s) that describe the membrane mechanisms (e.g., ion channels) of your model, as well as a Python class that defines the fiber model.

If you wish to create a new fiber model in the `PyFibers` structure instead of as a plugin, it is recommended to download the package from GitHub (link). Your new files should be placed in the following directory structure within `PyFibers`:
```{image} images/source1.png
:alt: Directory structure for custom fiber models
:align: center
:width: 300px
```
After your files are all created and in the proper location, you should install your custom version of `PyFibers` (`pip install .` from the root directory) and run `pyfibers_compile` to compile the new mechanisms.

## Steps to Create a Fiber Model Subclass

1. **Inherit from {py:class}`~pyfibers.fiber.Fiber`**: Your new class should inherit from {py:class}`~pyfibers.fiber.Fiber`.

2. **Specify the Submodels**: Define the ``submodels`` attribute as a list of the submodels that your fiber model uses. Often, this list will contain only a single item. Each item in the list should be capitalized and contain only letters and underscores; these strings are used to define the name of each fiber model as accessed from the {py:class}`FiberModel` enum (for example, one would access the fiber model by calling `FiberModel.MY_HOMOGENEOUS_FIBER_MODEL`). If your class has multiple subtypes (such as in ``{py:class}`` pyfibers.models.MRGFiber``), you can define multiple submodels. When you create an instance of your fiber model, it will gain the enum as the ``self.fiber_model`` attribute. Check the name by accessing ``self.fiber_model.name``.

3. **Initialize the Subclass**: Define the ``__init__`` method, call the superclass initializer, and set any model-specific parameters. At minimum, set ``self.v_rest`` (resting membrane potential) and ``self.myelinated`` (whether the fiber is myelinated). It is also recommended to specify gating variables if you want to be able to record these values during simulations. Finally, set the ``delta_z`` parameter, which is the distance from the center of one node to the next node.

4. **Define the Node Creation Method(s)**: Implement method(s) that create the specific sections of the fiber model. For a homogeneous fiber model, you will create a single method. For a heterogeneous fiber model, you will create multiple methods. These methods should return a NEURON {py:class}`h.Section` object representing the node or section.

5. **Define the `generate` Method**: Implement the `generate` method, which calls the superclass {py:meth}`~pyfibers.fiber.Fiber.generate` method with a list of functions that create the specific sections of the fiber model. For a homogeneous fiber model, you will pass a single function in the list. For a heterogeneous fiber model, you will pass a list of functions.

6. **Make the class discoverable**: This *only* applies if you are not creating a fiber model as a plugin. To make your fiber model discoverable within the `PyFibers` structure, you must open `/models/__init__.py` and add an import statement for your new fiber model class, as well as add the class name to the ``__all__`` list. This will allow `PyFibers` to automatically discover your fiber model and add it to the list of available models.

## Examples

Note that the class construction will differ if you are creating a "homogeneous" fiber model (i.e., all sections of the fiber are identical, typically unmyelinated fibers) or a "heterogeneous" fiber model (i.e., sections of the fiber have different properties, such as nodes and myelin). See the figure below for a visual representation of the construction of homogeneous and heterogeneous fiber models. We will begin with an example of creating a homogeneous fiber model and then extend it to a heterogeneous fiber model.

```{figure} images/fiber_construction.png
:name: fiber-construction
:alt: Homogeneous and heterogeneous fiber models
:align: center

Construction of a model fiber. **A\)** For unmyelinated fibers and myelinated fibers with only one section type (i.e., homogeneous fiber), a fiber model describes a single set of mechanisms and ultrastructure for a node, which is then repeated to the target number of sections. **B\)** For fibers with multiple section types (e.g., MRG as shown), a repeating series of sections with heterogeneous membrane mechanisms and ultrastructure is described from one node until just before the next node. This sequence is repeated to one less than the target number of nodes, and a node is added to the end of the fiber for symmetry. Example shown: MRG model fiber with 4 nodes and 11 sections per node, resulting in a final section count of n<sub>sections</sub> = (n<sub>nodes</sub> - 1) * 11 + 1 = (4 - 1) * 11 + 1 = 34. Figure is not to scale.
```

### Creating a New Homogeneous Fiber Model

If your model consists of identical sections throughout (typically unmyelinated fibers), you will pass a single function in the list to the {py:meth}`~pyfibers.fiber.Fiber.generate` method. For these models it is recommended to use {py:meth}`~pyfibers.fiber.Fiber.nodebuilder` to create the nodes, then add/override any mechanisms or properties as needed.

```python
from neuron import h
from pyfibers.fiber import Fiber, nodebuilder

h.load_file("stdrun.hoc")


class MyHomogeneousFiber(Fiber):
    submodels = ["MY_HOMOGENEOUS_FIBER_MODEL"]

    def __init__(self, diameter: float, delta_z: float = 8.333, **kwargs):
        """Initialize MyHomogeneousFiber class.

        :param diameter: Fiber diameter [microns]
        :param delta_z: Length of each segment [microns]
        :param kwargs: Keyword arguments to pass to the base class
        """
        super().__init__(fiber_model=fiber_model, diameter=diameter, **kwargs)
        self.gating_variables = {
            "h": "h_myModel",
            "m": "m_myModel",
            "n": "n_myModel",
        }
        self.myelinated = False
        self.v_rest = -65  # resting membrane potential [mV]
        self.delta_z = delta_z  # distance from the center of one node to the next [um]

    def generate(self, **kwargs) -> Fiber:
        """Generate the fiber model sections with NEURON.

        :param kwargs: Arguments to pass to the base class generate method
        :return: Fiber object
        """
        return super().generate([self.create_my_node], **kwargs)

    def create_my_node(self, ind: int, node_type: str) -> h.Section:
        """Create a node for MyHomogeneousFiber model.

        :param ind: Index of the node
        :param node_type: Type of the node
        :return: NEURON section representing the node
        """
        node = self.nodebuilder(ind, node_type)
        node.insert("my_mechanism")
        node.Ra = 100
        node.cm = 1.0
        # Edit any other necessary properties here
        return node
```

### Creating a New Heterogeneous Fiber Model

For fibers with multiple section types (e.g., nodes, myelin), the process is very similar to the homogeneous fiber model, with a few changes:
- You will pass a list of functions to the {py:meth}`~pyfibers.fiber.Fiber.generate` method.
- You **must** calculate ``delta_z`` (the distance from the center of one node to the next node) here in order to allow PyFibers to verify that the fiber is correctly built.
- You may still use {py:meth}`~pyfibers.fiber.Fiber.nodebuilder` to create sections and override the relevant properties, but it is recommended to avoid the use of this function for heterogeneous fibers (as shown below).

```python
from neuron import h
from pyfibers.fiber import Fiber

h.load_file("stdrun.hoc")


class MyHeterogeneousFiber(Fiber):
    submodels = ["MY_HETEROGENEOUS_FIBER_MODEL"]

    def __init__(self, diameter: float, **kwargs):
        """Initialize MyHeterogeneousFiber class.

        :param diameter: Fiber diameter [microns]
        :param kwargs: Keyword arguments to pass to the base class
        """
        super().__init__(fiber_model=fiber_model, diameter=diameter, **kwargs)
        self.gating_variables = {
            "h": "h_myModel",
            "m": "m_myModel",
            "s": "s_myModel",
        }
        self.myelinated = True
        self.v_rest = -80  # resting membrane potential [mV]
        self.delta_z = (
            1.5 + self.diameter * 100
        )  # distance from the center of one node to the next [um]

    def generate(self, **kwargs) -> Fiber:
        """Generate the fiber model sections with NEURON.

        :param kwargs: Arguments to pass to the base class generate method
        :return: Fiber object
        """
        function_list = [
            self.create_node,
            self.create_myelin,
        ]

        return super().generate(function_list, **kwargs)

    def create_node(self, ind: int, node_type: str) -> h.Section:
        """Create a node for MyHeterogeneousFiber model.

        :param ind: Index of the node
        :param node_type: Type of the node
        :return: NEURON section representing the node
        """
        name = f"{node_type} node {ind}"
        node = h.Section(name=name)
        node.cm = 2.5  # uF/cm^2
        node.Ra = 54.7  # ohm-cm
        node.diam = self.diameter * node_diam_factor
        node.L = 1.5  # um
        node.insert("my_mechanism")
        node.insert("extracellular")
        node.xc[0] = 0  # short circuit extracellular
        node.xg[0] = 1e10  # short circuit extracellular
        # Edit any other necessary properties here
        return node

    def create_myelin(self, ind: int, node_type: str) -> h.Section:
        """Create a myelin section for MyHeterogeneousFiber model.

        :param ind: Index of the section
        :param node_type: Type of the section
        :return: NEURON section representing the myelin section
        """
        name = f"myelin {ind}"
        myel = h.Section(name=name)
        myel.cm = 0  # uF/cm^2
        myel.Ra = 54.7  # ohm-cm
        myel.diam = self.diameter
        myel.L = self.diameter * myel_length_factor
        myel.insert("extracellular")
        myel.xc[0] = 0  # short circuit extracellular
        myel.xg[0] = 1e10  # short circuit extracellular
        # Edit any other necessary properties here
        return myel
```

By following these steps, you can create new fiber models based on the specific characteristics of your desired fiber. Define the necessary methods and parameters that match your fiber model requirements.

## Making Your Fiber Model Discoverable as a Plugin

Other research groups may wish to create their own fiber models in the `PyFibers` environment and publish them as a separate public repository. Such fiber models can be made discoverable as plugins which will become automatically available in `PyFibers` after installation. To make your fiber model discoverable as a plugin, follow these steps:

1. **Create a Python package for your fiber model**: Create a Python package for your fiber model and include the necessary files and classes. We recommend following Python's [packaging tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/). An example directory structure for a fiber model package is shown below:
```{image} images/source2.png
:alt: Directory structure for a fiber model package
:align: center
:width: 300px
```

2. **Specify the Entry Point for Your Fiber Model**: The entry point tells `PyFibers` that your fiber model should be imported and where to find the class that defines it. Entry points can be specified in multiple ways; here we provide an example using **pyproject.toml**:

```toml
[project.entry-points."pyfibers.fiber_plugins"]
my_fiber_model = "my_plugin_folder.custom_model_class:MyFiberModelClass"
```

3. Install your package and verify that it works with `PyFibers`. Upon installation, your fiber model should be automatically discovered and available for use in `PyFibers`. The necessary MOD files should be compiled automatically when `PyFibers` is imported.

4. **Publish Your Fiber Model Package**: Publish your fiber model package to a repository such as PyPI. Once published, other users can install your fiber model package using pip. After installation, your fiber model will be automatically discovered and available for use in `PyFibers`. The name of the fiber model will be given by the ``submodels`` attribute of your class (see the instructions for creating fiber models above).
