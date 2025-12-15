# Implementations of Fiber Models

In PyFibers, we have implemented a number of models describing the ultrastructure and membrane properties of both myelinated and unmyelinated peripheral nerve fibers. Herein we list the available fiber models, and note any changes from the original publications. Unless otherwise specified, the model code was adapted from ASCENT {cite:p}`musselman_ascent_2021`. Validation of the included models and more details on their implementation was documented in the PyFibers paper {cite:p}`Marshall2025`.

## Myelinated Fiber Models

We provide four variants of myelinated fiber models:

**MRG Models**: Three variants of the McIntyre–Richardson–Grill (MRG) model {cite:p}`McIntyre2002` for myelinated fibers. The MRG model is a biophysically detailed model of a myelinated nerve fiber, and is widely used in the field of computational neuroscience. The three variants are:
- {py:attr}`~pyfibers.model_enum.FiberModel.MRG_DISCRETE`: The original MRG model, as described in {cite:p}`McIntyre2002`.
- {py:attr}`~pyfibers.model_enum.FiberModel.MRG_INTERPOLATION`: An interpolation of the MRG model, which allows for modeling of any fiber diameter between 2 and 16 microns, as described in {cite:p}`musselman_ascent_2021`.
- {py:attr}`~pyfibers.model_enum.FiberModel.PENA`: A modification of MRG parameters to better replicate the behavior of thinly myelinated fibers, as described in {cite:p}`pena_cap_2024`.
- {py:attr}`~pyfibers.model_enum.FiberModel.SMALL_MRG_INTERPOLATION`: **Deprecated.** Old name for {py:attr}`~pyfibers.model_enum.FiberModel.PENA`.

**Sweeney Model**:
- {py:attr}`~pyfibers.model_enum.FiberModel.SWEENEY`: A myelinated fiber model based on the Sweeney model, as described in {cite:p}`sweeney_modeling_1987`.

## Unmyelinated Fiber Models

We provide several variants of unmyelinated C‑fiber models.
- {py:attr}`~pyfibers.model_enum.FiberModel.TIGERHOLM`: A biophysically detailed model of a C‑fiber, as described in {cite:p}`Tigerholm2014`.
- {py:attr}`~pyfibers.model_enum.FiberModel.RATTAY`: A simplified version of the Tigerholm model, as described in {cite:p}`Rattay1993`.
- {py:attr}`~pyfibers.model_enum.FiberModel.SCHILD94`: A simple model of a C‑fiber, as described in {cite:p}`Schild1994`.
- {py:attr}`~pyfibers.model_enum.FiberModel.SCHILD97`: A modification of the Schild 1994 model, as described in {cite:p}`Schild1997`.
- {py:attr}`~pyfibers.model_enum.FiberModel.SUNDT`: A modification of the Schild 1994 model, as described in {cite:p}`Sundt2015`.

## Construction of Model Fibers

Fibers in `PyFibers` are composed of serially connected NEURON section objects. Sections may or may not be designated as nodes:
- Section: A NEURON object which represents a discrete geometric length of a model fiber
- Node: A special section that typically represents an excitable portion of a fiber’s geometry (i.e., nodes of Ranvier). Active nodes have nonlinear conductance mechanisms and are thus excitable. Passive nodes have the same ultrastructure parameters, but the nonlinear mechanisms have been removed, and are not excitable. For unmyelinated fibers in PyFibers, nodes and sections are synonymous. See the figure below for an example.

```{figure} images/fiber_construction.png
:name: fiber-construction-example
:alt: Homogeneous and heterogeneous fiber models
:align: center

Construction of a model fiber. **A\)** For unmyelinated fibers and myelinated fibers with only one section type (i.e., homogeneous fiber), a fiber model describes a single set of mechanisms and ultrastructure for a node, which is then repeated to the target number of sections. **B\)** For fibers with multiple section types (e.g., MRG as shown), a repeating series of sections with heterogeneous membrane mechanisms and ultrastructure is described from one node until just before the next node. This sequence is repeated to one less than the target number of nodes, and a node is added to the end of the fiber for symmetry. Example shown: MRG model fiber with 4 nodes and 11 sections per node, resulting in a final section count of n<sub>sections</sub> = (n<sub>nodes</sub> - 1) * 11 + 1 = (4 - 1) * 11 + 1 = 34. Figure is not to scale.
```

## Passive End Nodes

When constructing a fiber, users can specify the number of passive end nodes at each end of the fiber. In this case, the fiber is constructed as normal. For end nodes designated as passive, all mechanisms other than extracellular (i.e. the mechanism named "extracellular") are removed from the node. Then, the following properties are set as described in {cite:p}`Pelot2021`:
- The passive (``pas``) mechanism is added to the node.
- The passive reversal potential is set to the resting potential of the fiber.
- The membrane capacitance is set to 1 [uF/cm^2] (a typical value for membranes).
- The membrane resistance is set to 0.0001 [S/cm^2] (slightly lower than the typical value, which dulls current entry into the node).
- The axial resistivity is set to 1e10 [Ohm*cm] (effectively infinite so that no current flows axially, thereby avoiding the creation of action potentials by restricting current flow to adjacent nodes).
