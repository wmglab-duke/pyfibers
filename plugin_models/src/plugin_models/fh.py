"""The copyrights of this software are owned by Duke University."""

from __future__ import annotations

from neuron import h

from pyfibers.fiber import Fiber

h.load_file('stdrun.hoc')


class FHFiber(Fiber):
    """Frankenhauser-Huxley fiber model."""

    submodels = ['FH']

    def __init__(self: FHFiber, diameter: float, delta_z: float = 8.333, **kwargs) -> None:
        """Initialize UnmyelinatedFiber class.

        :param diameter: fiber diameter [microns]
        :param delta_z: node spacing [microns]
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
        self.delta_z = delta_z  # microns
        self.v_rest = -70  # millivolts #TODO find rationale for this value

    def generate(self: FHFiber, **kwargs) -> Fiber:  # noqa: D102
        return super().generate([self.create_fh], **kwargs)

    def create_fh(self: FHFiber, ind: int, node_type: str) -> h.Section:
        """Create a RATTAY node.

        :param ind: Node index in the fiber.
        :param node_type: Node type ('active' or 'passive').
        :return: Created node with Frankenhauser-Huxley mechanisms.
        """
        node = self.nodebuilder(ind, node_type)
        node.insert('fh')
        node.cm = 2
        node.nai = 13.74
        node.nao = 114.5
        node.ki = 120
        node.ko = 2.5
        return node
