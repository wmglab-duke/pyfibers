"""The copyrights of this software are owned by Duke University."""

from __future__ import annotations

from neuron import h

from pyfibers.fiber import Fiber, _HomogeneousFiber

h.load_file('stdrun.hoc')


class HHFiber(_HomogeneousFiber):
    """Hodgkin-Huxley fiber model."""

    submodels = ['HH']

    def __init__(self: HHFiber, diameter: float, **kwargs) -> None:
        """Initialize UnmyelinatedFiber class.

        :param diameter: fiber diameter [microns]
        :param kwargs: keyword arguments to pass to the base class
        """
        super().__init__(diameter=diameter, **kwargs)
        self.gating_variables = {  # TODO Figure out hh gating variables
            "h": "h_hh",
            "m": "m_hh",
            "n": "n_hh",
        }
        self.myelinated = False
        self.delta_z = 8.333  # microns
        self.v_rest = -65  # millivolts #TODO find rationale for this value

    def generate(self: HHFiber, n_sections: int, length: float) -> Fiber:  # noqa D102
        return self.generate_homogeneous(n_sections, length, self.create_hh)

    @staticmethod
    def create_hh(node: h.Section) -> None:
        """Create a Hodgkin-Huxley node.

        :param node: NEURON section
        """
        node.insert('hh')

        node.Ra = 100  # TODO find rationale for this value
        node.cm = 1  # TODO find rationale for this value
