"""Tests for pyfibers.

The copyrights of this software are owned by Duke University. Please
refer to the LICENSE and README.md files for licensing instructions. The
source code can be found on the following GitHub repository:
https://github.com/wmglab-duke/ascent
"""

from __future__ import annotations

import numpy as np
import pytest  # noqa: I900

from pyfibers import FiberModel, IntraStim, build_fiber  # BisectionMean,; BoundsSearchMode,; TerminationMode,

# TODO Change all c fiber model to 1 um


def get_fiber(diameter=5.7, fiber_model=FiberModel.MRG_INTERPOLATION, temperature=37, n_sections=133):
    return build_fiber(diameter=diameter, fiber_model=fiber_model, temperature=temperature, n_sections=n_sections)


def get_activation_threshold(model, nodecount=133, diameter=5.7, **kwargs):  # TODO test range of diameters
    """Get activation threshold.

    Using intracellular stim
    """

    # create curve of potentials
    fiber = build_fiber(diameter=diameter, fiber_model=model, temperature=37, n_sections=nodecount)

    # parameters
    time_step = 0.001
    time_stop = 20
    stimulation = IntraStim(istim_loc=0.5, dt=time_step, tstop=time_stop)

    stimulation.run_sim(0, fiber)  # TODO why do I need to run this first for correct result

    amp, ap = stimulation.find_threshold(fiber, stimamp_top=1, stimamp_bottom=0, **kwargs)

    return amp


def test_mrg_discrete():
    assert np.isclose(get_activation_threshold(FiberModel.MRG_DISCRETE), 0.15576171875)


def test_mrg_interpolation():
    assert np.isclose(get_activation_threshold(FiberModel.MRG_INTERPOLATION), 0.17333984375)


def test_tigerholm():
    assert np.isclose(
        get_activation_threshold(FiberModel.TIGERHOLM, diameter=1),
        0.419921875,
    )


def test_rattay():
    assert np.isclose(
        get_activation_threshold(FiberModel.RATTAY, diameter=1),
        0.18603515625,
    )


def test_sundt():
    assert np.isclose(
        get_activation_threshold(FiberModel.SUNDT, diameter=1),
        0.16650390625,
    )


def test_schild94():
    assert np.isclose(
        get_activation_threshold(FiberModel.SCHILD94, diameter=1),
        0.2138671875,
    )


def test_schild97():
    assert np.isclose(
        get_activation_threshold(FiberModel.SCHILD97, diameter=1),
        0.4609375,
    )


if __name__ == "__main__":
    pytest.main()
