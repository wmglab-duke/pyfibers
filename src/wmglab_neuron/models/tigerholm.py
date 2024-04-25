"""The copyrights of this software are owned by Duke University."""

from __future__ import annotations

import warnings

from neuron import h

from wmglab_neuron import FiberModel
from wmglab_neuron.fiber import _Fiber, _HomogeneousFiber

h.load_file('stdrun.hoc')


class TigerholmFiber(_HomogeneousFiber):
    """Tigerholm Fiber model."""

    def __init__(self: TigerholmFiber, fiber_model: FiberModel, diameter: float, **kwargs) -> None:
        """Initialize UnmyelinatedFiber class.

        :param fiber_model: name of fiber model type
        :param diameter: fiber diameter [microns]
        :param kwargs: keyword arguments to pass to the base class
        """
        super().__init__(fiber_model=fiber_model, diameter=diameter, **kwargs)
        self.myelinated = False
        self.delta_z = 8.333  # micronss
        self.v_rest = -55  # millivolts

        if self.passive_end_nodes:
            warnings.warn('Ignoring passive_end_nodes for Tigerholm fiber', UserWarning, stacklevel=2)
            self.passive_end_nodes: bool = False

    def generate(self: TigerholmFiber, n_sections: int, length: float) -> _Fiber:  # noqa D102
        return self.generate_homogeneous(n_sections, length, self.create_tigerholm, celsius=self.temperature)

    @staticmethod
    def create_tigerholm(node: h.Section, celsius: int) -> None:
        """Create a TIGERHOLM node.

        :param celsius: model temperature [celsius]
        :param node: NEURON section
        """
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
        node.celsiusT_ks = celsius
        node.celsiusT_kf = celsius
        node.celsiusT_h = celsius
        node.celsiusT_nattxs = celsius
        node.celsiusT_nav1p8 = celsius
        node.celsiusT_nav1p9 = celsius
        node.celsiusT_nakpump = celsius
        node.celsiusT_kdrTiger = celsius
        node.gbar_ks = 0.0069733
        node.gbar_kf = 0.012756
        node.gbar_h = 0.0025377
        node.gbar_nattxs = 0.10664
        node.gbar_nav1p8 = 0.24271
        node.gbar_nav1p9 = 9.4779e-05
        node.smalla_nakpump = -0.0047891
        node.gbar_kdrTiger = 0.018002

    def balance(self: TigerholmFiber) -> None:
        """Balance membrane currents for Tigerholm model."""
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
