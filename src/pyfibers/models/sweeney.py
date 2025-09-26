"""Implementation of the Sweeney fiber model.

Based on:
Sweeney, J. D., Mortimer, J. T., & Durand, D. (1987).
Modeling of mammalian myelinated nerve for functional neuromuscular stimulation.
"""

from __future__ import annotations

from neuron import h

from pyfibers.fiber import Fiber

h.load_file("stdrun.hoc")


class SweeneyFiber(Fiber):
    """Implementation of the Sweeney fiber model."""

    submodels = ["SWEENEY"]

    def __init__(self: SweeneyFiber, diameter: float, **kwargs) -> None:
        """Initialize SweeneyFiber class.

        :param diameter: fiber diameter [microns]
        :param kwargs: keyword arguments to pass to the base class
        """
        assert "delta_z" not in kwargs, "Cannot specify delta_z for Sweeney Fiber"
        super().__init__(diameter=diameter, **kwargs)
        self.gating_variables = {
            "h": "h_sweeney",
            "m": "m_sweeney",
        }
        self.myelinated = True
        self.delta_z = self.diameter * 100
        self.v_rest = -80  # millivolts

    def generate(self: SweeneyFiber, **kwargs) -> Fiber:
        """Build fiber model sections with NEURON.

        :param kwargs: passed to superclass generate method
        :return: Fiber object
        """
        # Function list for section order
        function_list = [
            self.create_node,
            self.create_myelin,
        ]

        return super().generate(function_list, **kwargs)

    def create_node(self: SweeneyFiber, index: int, node_type: str) -> h.Section:
        """Create a node of Ranvier.

        :param index: Section index in the fiber.
        :param node_type: Node type ('active' or 'passive').
        :return: Created node with Sweeney mechanisms
        """
        name = f"{node_type} node {index}"
        node = h.Section(name=name)
        node.nseg = 1
        node.diam = self.diameter * 0.6
        node.L = 1.5  # um
        node.insert("sweeney")
        node.cm = 2.5  # uF/cm^2
        node.Ra = 54.7  # ohm-cm
        node.insert('extracellular')
        node.xc[0] = 0  # short circuit
        node.xg[0] = 1e10  # short circuit
        node.ena = 35.64  # mV
        return node

    def create_myelin(self: SweeneyFiber, index: int) -> h.Section:
        """Create a myelin section.

        :param index: Section index in the fiber.
        :return: Created myelin section
        """
        name = f"myelin {index}"
        section = h.Section(name=name)
        section.nseg = 1
        section.diam = self.diameter * 0.6
        section.L = 100 * self.diameter - 1.5  # um, -1.5 since 100*D is internodal length
        section.cm = 0
        section.Ra = 54.7  # ohm-cm
        section.insert('extracellular')
        section.xc[0] = 0  # short circuit
        section.xg[0] = 1e10  # short circuit
        return section
