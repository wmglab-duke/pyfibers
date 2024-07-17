"""The copyrights of this software are owned by Duke University."""

from __future__ import annotations

from neuron import h

from pyfibers.fiber import Fiber, _HomogeneousFiber

h.load_file('stdrun.hoc')


class RattayFiber(_HomogeneousFiber):
    """Rattay fiber model."""

    submodels = ['RATTAY']

    def __init__(self: RattayFiber, diameter: float, **kwargs) -> None:
        """Initialize RattayFiber class.

        :param diameter: Fiber diameter [microns].
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

    def generate(self: RattayFiber, **kwargs) -> Fiber:  # noqa D102
        return self.generate_homogeneous(self.create_rattay, **kwargs)

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
