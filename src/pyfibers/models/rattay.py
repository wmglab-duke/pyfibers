"""Implementation of Rattay fiber model.

As described in Rattay 1993: https://doi.org/10.1109/10.250575
"""

from __future__ import annotations

from neuron import h

from pyfibers.fiber import Fiber

h.load_file('stdrun.hoc')


class RattayFiber(Fiber):
    """Rattay fiber model."""

    submodels = ['RATTAY']

    def __init__(self: RattayFiber, diameter: float, delta_z: float = 8.333, **kwargs) -> None:
        """Initialize RattayFiber class.

        :param diameter: Fiber diameter [microns].
        :param delta_z: Node spacing [microns].
        :param kwargs: Keyword arguments to pass to the base class.
        """
        super().__init__(diameter=diameter, **kwargs)
        self.gating_variables = {
            "h": "h_RattayAberham",
            "m": "m_RattayAberham",
            "n": "n_RattayAberham",
        }
        self.myelinated = False
        self.v_rest = -70  # millivolts
        self.delta_z = delta_z

    def generate(self: RattayFiber, **kwargs) -> Fiber:  # noqa: D102
        return super().generate([self.create_rattay], **kwargs)

    def create_rattay(self: RattayFiber, ind: int, node_type: str) -> h.Section:
        """Create a RATTAY node.

        :param ind: Node index in the fiber.
        :param node_type: Node type ('active' or 'passive').
        :return: Created node with Rattay mechanisms.
        """
        node = self.nodebuilder(ind, node_type)
        node.insert('RattayAberham')

        node.Ra = 100  # Ohm*cm
        node.cm = 1  # uF/cm^2
        node.ena = 45  # mV, sodium reversal potential
        node.ek = -82  # mV, potassium reversal potential
        return node
