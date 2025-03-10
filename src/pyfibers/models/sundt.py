"""Implementation of Sundt Fiber model.

As described in Sundt 2015: https://doi.org/10.1152/jn.00226.2015
"""

from __future__ import annotations

from neuron import h

from pyfibers.fiber import Fiber

h.load_file('stdrun.hoc')


class SundtFiber(Fiber):
    """Sundt fiber model."""

    submodels = ['SUNDT']

    def __init__(self: SundtFiber, diameter: float, delta_z: float = 8.333, **kwargs) -> None:
        """Initialize SundtFiber class.

        :param diameter: fiber diameter [microns]
        :param delta_z: node spacing [microns]
        :param kwargs: keyword arguments to pass to the base class
        """
        super().__init__(diameter=diameter, **kwargs)
        self.gating_variables = {
            "h": "h_nahh",
            "m": "m_nahh",
            "n": "n_borgkdr",
            "l": "l_borgkdr",
        }
        self.myelinated = False
        self.v_rest = -60  # millivolts
        self.delta_z = delta_z

    def generate(self: SundtFiber, **kwargs) -> Fiber:  # noqa: D102
        return super().generate([self.create_sundt], **kwargs)

    def create_sundt(self: SundtFiber, ind: int, node_type: str) -> h.Section:
        """Create a SUNDT node.

        :param ind: node index in the fiber
        :param node_type: node type ('active' or 'passive')
        :return: created node with  Sundt mechanisms
        """
        node = self.nodebuilder(ind, node_type)
        node.insert('nahh')
        node.insert('borgkdr')  # insert delayed rectified K channels
        node.insert('pas')  # insert leak channels

        node.gnabar_nahh = 0.04
        node.mshift_nahh = -6  # NaV1.7/1.8 channelshift
        node.hshift_nahh = 6  # NaV1.7/1.8 channelshift
        node.gkdrbar_borgkdr = 0.04  # density of K channels
        node.ek = -90  # K equilibrium potential
        node.g_pas = 1 / 10000  # set Rm = 10000 ohms-cm2
        node.Ra = 100  # intracellular resistance
        node.e_pas = node.v + (node.ina + node.ik) / node.g_pas  # calculate leak equilibrium potential
        return node
