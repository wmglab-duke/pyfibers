"""The copyrights of this software are owned by Duke University."""

from __future__ import annotations

from neuron import h

from wmglab_neuron import FiberModel
from wmglab_neuron.fiber import _Fiber, _HomogeneousFiber

h.load_file('stdrun.hoc')


class SundtFiber(_HomogeneousFiber):
    """Sundt fiber model."""

    def __init__(self: SundtFiber, fiber_model: FiberModel, diameter: float, **kwargs) -> None:
        """Initialize UnmyelinatedFiber class.

        :param fiber_model: enum of fiber model type
        :param diameter: fiber diameter [microns]
        :param kwargs: keyword arguments to pass to the base class
        """
        super().__init__(fiber_model=fiber_model, diameter=diameter, **kwargs)
        self.myelinated = False
        self.delta_z = 8.333  # microns
        self.v_rest = -60  # millivolts

    def generate(self: SundtFiber, n_sections: int, length: float) -> _Fiber:  # noqa D102
        return self.generate_homogeneous(n_sections, length, self.create_sundt)

    @staticmethod
    def create_sundt(node: h.Section) -> None:
        """Create a SUNDT node.

        :param node: NEURON section
        """
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
