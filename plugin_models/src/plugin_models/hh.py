"""The copyrights of this software are owned by Duke University."""

from __future__ import annotations

from neuron import h

from pyfibers.fiber import Fiber

h.load_file('stdrun.hoc')


class HHFiber(Fiber):
    """Hodgkin-Huxley fiber model."""

    submodels = ['HH']

    def __init__(self: HHFiber, diameter: float, delta_z: float = 8.333, **kwargs) -> None:
        """Initialize UnmyelinatedFiber class.

        :param diameter: fiber diameter [microns]
        :param delta_z: node spacing [microns]
        :param kwargs: keyword arguments to pass to the base class
        """
        super().__init__(diameter=diameter, **kwargs)
        self.gating_variables = {  # TODO Figure out hh gating variables
            "h": "h_hh",
            "m": "m_hh",
            "n": "n_hh",
        }
        self.myelinated = False
        self.delta_z = delta_z  # microns
        self.v_rest = -65  # millivolts #TODO find rationale for this value

    def generate(self: HHFiber, **kwargs) -> Fiber:  # noqa D102
        return super().generate([self.create_hh], **kwargs)

    def create_hh(self: HHFiber, ind: int, node_type: str) -> h.Section:
        """Create a Hodgkin-Huxley node.

        :param ind: Node index in the fiber.
        :param node_type: Node type ('active' or 'passive').
        :return: Created node with Hodgkin-Huxley mechanisms
        """
        node = self.nodebuilder(ind, node_type)
        node.insert('hh')

        node.Ra = 100  # TODO find rationale for this value
        node.cm = 1  # TODO find rationale for this value
        return node
