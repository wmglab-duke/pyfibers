"""Tests for PyFibers.

The copyrights of this software are owned by Duke University.
See LICENSE for licensing instructions.
Source code: https://github.com/wmglab-duke/pyfibers
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import interp1d

from pyfibers import FiberModel, ScaledStim, build_fiber


def get_fiber(diameter=5.7, fiber_model=FiberModel.MRG_INTERPOLATION, temperature=37, n_sections=133):
    return build_fiber(diameter=diameter, fiber_model=fiber_model, temperature=temperature, n_sections=n_sections)


def test_bad_fiber_model():
    """Test that a bad fiber model raises an error."""
    with pytest.raises(AttributeError):
        build_fiber(diameter=5.7, fiber_model='bad_model', temperature=37, n_sections=133)


def test_passive_end_nodes_fiber_too_short():
    """Raise when passive end nodes would leave no active nodes."""
    with pytest.raises(ValueError, match="too short for the requested number of passive end nodes"):
        build_fiber(
            diameter=5.7,
            fiber_model=FiberModel.MRG_INTERPOLATION,
            temperature=37,
            n_nodes=5,
            passive_end_nodes=3,
        )


def test_passive_end_nodes_allowed_when_active_node_remains():
    """Allow passive ends when at least one active node remains in the middle."""
    fiber = build_fiber(
        diameter=5.7,
        fiber_model=FiberModel.MRG_INTERPOLATION,
        temperature=37,
        n_nodes=5,
        passive_end_nodes=2,
    )
    assert fiber.nodecount == 5
    assert fiber.passive_end_nodes == 2
    assert 'passive' in fiber.nodes[0].name()
    assert 'passive' in fiber.nodes[1].name()
    assert 'passive' not in fiber.nodes[2].name()
    assert 'passive' in fiber.nodes[3].name()
    assert 'passive' in fiber.nodes[4].name()


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
    assert np.isclose(fiber.potentials[66], 753.537379490885)


def test_waveform_pad_truncate():
    fiber = get_fiber()  # noqa: F841
    waveform = np.concatenate((np.ones(200), -np.ones(200), np.zeros(49600)))
    stimulation = ScaledStim(waveform=waveform, dt=0.001, tstop=5)
    assert stimulation._prepped_waveform.shape[1] == 5000

    waveform = np.concatenate((np.ones(200), -np.ones(200), np.zeros(100)))
    stimulation = ScaledStim(waveform=waveform, dt=0.001, tstop=5)
    assert stimulation._prepped_waveform.shape[1] == 5000


def test_waveform_callable():
    fiber = get_fiber()  # TODO figure out why this is needed and then delete # noqa: F841
    dt = 0.005  # ms
    start = 0  # ms
    up = 1  # ms
    down = 2  # ms
    off = 3  # ms
    stop = 4  # ms
    concat_waveform = np.concatenate(
        (
            np.zeros(int((up - start) / dt)),
            np.ones(int((down - up) / dt)),
            -np.ones(int((off - down) / dt)),
            np.zeros(int((stop - off) / dt)),
        )
    )
    concat_stimulation = ScaledStim(waveform=concat_waveform, dt=dt, tstop=stop)

    callable_waveform = interp1d([start, up, down, off, stop], [0, 1, -1, 0, 0], kind="previous")
    callable_stimulation = ScaledStim(waveform=callable_waveform, dt=dt, tstop=stop)
    assert np.array_equal(concat_stimulation._prepped_waveform, callable_stimulation._prepped_waveform)


def test_multiple_waveforms():
    fiber = get_fiber()  # TODO figure out why this is needed and then delete # noqa: F841
    dt = 0.005  # ms
    start = 0  # ms
    up_list = np.arange(1, 10)  # ms
    down_list = np.arange(10, 20)  # ms
    off_list = np.arange(20, 30)  # ms
    stop = 35  # ms
    concat_waveforms = []
    callable_waveforms = []
    for up, down, off in zip(up_list, down_list, off_list):
        concat_waveforms.append(
            np.concatenate(
                (
                    np.zeros(int((up - start) / dt)),
                    np.ones(int((down - up) / dt)),
                    -np.ones(int((off - down) / dt)),
                    np.zeros(int((stop - off) / dt)),
                )
            )
        )
        callable_waveforms.append(interp1d([start, up, down, off, stop], [0, 1, -1, 0, 0], kind="previous"))

    concat_stimulation = ScaledStim(waveform=concat_waveforms, dt=dt, tstop=stop)
    callable_stimulation = ScaledStim(waveform=callable_waveforms, dt=dt, tstop=stop)
    assert np.array_equal(concat_stimulation._prepped_waveform, callable_stimulation._prepped_waveform)
