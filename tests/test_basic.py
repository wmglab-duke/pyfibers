"""Tests for wmglab-neuron.

The copyrights of this software are owned by Duke University. Please
refer to the LICENSE and README.md files for licensing instructions. The
source code can be found on the following GitHub repository:
https://github.com/wmglab-duke/ascent
"""

from __future__ import annotations

import numpy as np
import pytest  # noqa: I900

from wmglab_neuron import FiberModel, ScaledStim, build_fiber

# TODO Change all c fiber model to 1 um


def get_fiber(diameter=5.7, fiber_model=FiberModel.MRG_INTERPOLATION, temperature=37, n_sections=133):
    return build_fiber(diameter=diameter, fiber_model=fiber_model, temperature=temperature, n_sections=n_sections)


def test_bad_fiber_model():
    """Test that a bad fiber model raises an error."""
    with pytest.raises(ValueError):
        build_fiber(diameter=5.7, fiber_model='bad_model', temperature=37, n_sections=133)


def test_len():
    assert len(get_fiber()) == (133 - 1) / 11 + 1


def test_getitem():
    fiber = get_fiber()
    assert fiber[0] is fiber.nodes[0]


def test_iter():
    fiber = get_fiber()
    for i, node in enumerate(fiber):
        assert node is fiber.nodes[i]


def test_contains():
    fiber = get_fiber()
    assert fiber.nodes[0] in fiber
    assert fiber.sections[1] in fiber


def test_loc():
    fiber = get_fiber()
    assert fiber.loc(0) is fiber.nodes[0]
    assert fiber.loc(1) is fiber.nodes[-1]
    assert fiber.loc(0.5) is fiber.nodes[6]


def test_pointsource():
    fiber = get_fiber()
    fiber.potentials = fiber.point_source_potentials(0, 100, 3000, 1, 1)
    assert np.isclose(fiber.potentials[66], 0.753537379490885)


def test_waveform_pad_truncate():
    fiber = get_fiber()  # TODO figure out why this is needed and then delete # noqa: F841
    waveform = np.concatenate((np.ones(200), -np.ones(200), np.zeros(49600)))
    stimulation = ScaledStim(waveform=waveform, dt=0.001, tstop=5)
    assert len(stimulation.waveform) == 5000

    waveform = np.concatenate((np.ones(200), -np.ones(200), np.zeros(100)))
    stimulation = ScaledStim(waveform=waveform, dt=0.001, tstop=5)
    assert len(stimulation.waveform) == 5000
