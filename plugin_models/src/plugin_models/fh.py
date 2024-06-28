"""The copyrights of this software are owned by Duke University."""

from __future__ import annotations

from neuron import h

from pyfibers.fiber import Fiber, _HomogeneousFiber

h.load_file('stdrun.hoc')


class FHFiber(_HomogeneousFiber):
    """Frankenhauser-Huxley fiber model."""

    submodels = ['FH']

    def __init__(self: FHFiber, diameter: float, **kwargs) -> None:
        """Initialize UnmyelinatedFiber class.

        :param diameter: fiber diameter [microns]
        :param kwargs: keyword arguments to pass to the base class
        """
        super().__init__(diameter=diameter, **kwargs)
        self.gating_variables = {
            "m": "m_fh",
            "h": "h_fh",
            "n": "n_fh",
            "p": "p_fh",
        }
        self.myelinated = False
        self.delta_z = 8.333  # microns
        self.v_rest = -70  # millivolts #TODO find rationale for this value

    def generate(self: FHFiber, n_sections: int, length: float) -> Fiber:  # noqa D102
        return self.generate_homogeneous(n_sections, length, self.create_hh)

    @staticmethod
    def create_hh(node: h.Section) -> None:
        """Create a RATTAY node.

        :param node: NEURON section
        """
        node.insert('fh')
        node.cm = 2
        node.nai = 13.74
        node.nao = 114.5
        node.ki = 120
        node.ko = 2.5
