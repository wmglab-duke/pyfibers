"""The copyrights of this software are owned by Duke University."""

from __future__ import annotations

from neuron import h

from pyfibers import FiberModel
from pyfibers.fiber import _Fiber, _HomogeneousFiber

h.load_file('stdrun.hoc')


class RattayFiber(_HomogeneousFiber):
    """Rattay fiber model."""

    def __init__(self: RattayFiber, fiber_model: FiberModel, diameter: float, **kwargs) -> None:
        """Initialize UnmyelinatedFiber class.

        :param fiber_model: name of fiber model type
        :param diameter: fiber diameter [microns]
        :param kwargs: keyword arguments to pass to the base class
        """
        super().__init__(fiber_model=fiber_model, diameter=diameter, **kwargs)
        self.gating_variables = {
            "h": "h_RattayAberham",
            "m": "m_RattayAberham",
            "n": "n_RattayAberham",
        }
        self.myelinated = False
        self.delta_z = 8.333  # microns
        self.v_rest = -70  # millivolts

    def generate(self: RattayFiber, n_sections: int, length: float) -> _Fiber:  # noqa D102
        return self.generate_homogeneous(n_sections, length, self.create_rattay)

    @staticmethod
    def create_rattay(node: h.Section) -> None:
        """Create a RATTAY node.

        :param node: NEURON section
        """
        node.insert('RattayAberham')

        node.Ra = 100  # required for propagation; less than 100 does not propagate
        node.cm = 1
        node.ena = 45
        node.ek = -82
