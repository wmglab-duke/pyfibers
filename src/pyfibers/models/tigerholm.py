"""Implementation of Tigerholm fiber model.

As described in Tigerholm 2014: https://doi.org/10.1152/jn.00777.2012
"""

from __future__ import annotations

import warnings

from neuron import h

from pyfibers.fiber import Fiber

h.load_file('stdrun.hoc')


class TigerholmFiber(Fiber):
    """Tigerholm Fiber model."""

    submodels = ['TIGERHOLM']

    def __init__(self: TigerholmFiber, diameter: float, delta_z: float = 8.333, **kwargs) -> None:
        """Initialize TigerholmFiber class.

        :param diameter: fiber diameter [microns]
        :param delta_z: node spacing [microns]
        :param kwargs: keyword arguments to pass to the base class
        """
        super().__init__(diameter=diameter, **kwargs)
        self.gating_variables = {
            "m1.7": "m_nattxs",
            "h1.7": "h_nattxs",
            "s1.7": "s_nattxs",
            "m1.8": "m_nav1p8",
            "h1.8": "h_nav1p8",
            "s1.8": "s_nav1p8",
            "u1.8": "u_nav1p8",
            "m1.9": "m_nav1p9",
            "h1.9": "h_nav1p9",
            "s1.9": "s_nav1p9",
            "n_kdr": "n_kdrTiger",
            "ns_km": "ns_ks",
            "nf_km": "nf_ks",
            "n_ka": "m_kf",
            "h_ka": "h_kf",
            "ns_h": "ns_h",
            "nf_h": "nf_h",
            "w_kna": "w_kna",
        }
        self.myelinated = False
        self.v_rest = -55  # millivolts
        self.delta_z = delta_z

        if self.passive_end_nodes:
            warnings.warn('Ignoring passive_end_nodes for Tigerholm fiber', UserWarning, stacklevel=2)
            self.passive_end_nodes: bool = False

    def generate(self: TigerholmFiber, **kwargs) -> Fiber:  # noqa: D102
        return super().generate([self.create_tigerholm], **kwargs)

    def create_tigerholm(self: TigerholmFiber, ind: int, node_type: str) -> h.Section:
        """Create a TIGERHOLM node.

        :param ind: node index in fiber
        :param node_type: node type ('active' or 'passive')
        :return: NEURON section with TIGERHOLM mechanism
        """
        node = self.nodebuilder(ind, node_type)
        node.insert('ks')
        node.insert('kf')
        node.insert('h')
        node.insert('nattxs')
        node.insert('nav1p8')
        node.insert('nav1p9')
        node.insert('nakpump')
        node.insert('kdrTiger')
        node.insert('kna')
        node.insert('naoiTiger')
        node.insert('koiTiger')
        node.insert('leak')
        node.insert('extrapump')

        node.gbar_kna = 0.00042
        node.Ra = 35.4
        node.cm = 1
        node.gbar_ks = 0.0069733
        node.gbar_kf = 0.012756
        node.gbar_h = 0.0025377
        node.gbar_nattxs = 0.10664
        node.gbar_nav1p8 = 0.24271
        node.gbar_nav1p9 = 9.4779e-05
        node.smalla_nakpump = -0.0047891
        node.gbar_kdrTiger = 0.018002

        node.celsiusT_ks = self.temperature
        node.celsiusT_kf = self.temperature
        node.celsiusT_h = self.temperature
        node.celsiusT_nattxs = self.temperature
        node.celsiusT_nav1p8 = self.temperature
        node.celsiusT_nav1p9 = self.temperature
        node.celsiusT_nakpump = self.temperature
        node.celsiusT_kdrTiger = self.temperature

        return node

    def balance(self: TigerholmFiber) -> None:
        """Balance membrane currents."""
        v_rest = self.v_rest
        for s in self.sections:
            if (-(s.ina_nattxs + s.ina_nav1p9 + s.ina_nav1p8 + s.ina_h + s.ina_nakpump) / (v_rest - s.ena)) < 0:
                s.pumpina_extrapump = -(s.ina_nattxs + s.ina_nav1p9 + s.ina_nav1p8 + s.ina_h + s.ina_nakpump)
            else:
                s.gnaleak_leak = -(s.ina_nattxs + s.ina_nav1p9 + s.ina_nav1p8 + s.ina_h + s.ina_nakpump) / (
                    v_rest - s.ena
                )

            if (-(s.ik_ks + s.ik_kf + s.ik_h + s.ik_kdrTiger + s.ik_nakpump + s.ik_kna) / (v_rest - s.ek)) < 0:
                s.pumpik_extrapump = -(s.ik_ks + s.ik_kf + s.ik_h + s.ik_kdrTiger + s.ik_nakpump + s.ik_kna)
            else:
                s.gkleak_leak = -(s.ik_ks + s.ik_kf + s.ik_h + s.ik_kdrTiger + s.ik_nakpump + s.ik_kna) / (
                    v_rest - s.ek
                )
