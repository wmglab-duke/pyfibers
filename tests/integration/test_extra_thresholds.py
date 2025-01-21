"""Tests for pyfibers.

The copyrights of this software are owned by Duke University. Please
refer to the LICENSE and README.md files for licensing instructions. The
source code can be found on the following GitHub repository:
https://github.com/wmglab-duke/ascent
"""

from __future__ import annotations

import numpy as np
import pytest  # noqa: I900
import scipy.signal as sg
from scipy.stats import norm

from pyfibers import (  # BisectionMean,; BoundsSearchMode,; TerminationMode,
    FiberModel,
    ScaledStim,
    ThresholdCondition,
    build_fiber,
)

# TODO Change all c fiber model to 1 um


def get_fiber(diameter=5.7, fiber_model=FiberModel.MRG_INTERPOLATION, temperature=37, n_sections=133):
    return build_fiber(diameter=diameter, fiber_model=fiber_model, temperature=temperature, n_sections=n_sections)


def get_activation_threshold(model, nodecount=133, diameter=5.7, **kwargs):  # TODO test range of diameters
    """Get activation threshold.

    Using point source
    """

    # create curve of potentials
    fiber = build_fiber(diameter=diameter, fiber_model=model, temperature=37, n_sections=nodecount)
    fiber.potentials = fiber.point_source_potentials(0, 100, fiber.length / 2, 1, 10)

    waveform = np.concatenate((np.ones(200), -np.ones(200), np.zeros(49600)))

    # parameters
    time_step = 0.001
    time_stop = 20
    stimulation = ScaledStim(waveform=waveform, dt=time_step, tstop=time_stop)

    amp, ap = stimulation.find_threshold(fiber, **kwargs)

    return amp


def get_amp_responses(model, stimamps, save=False):
    """Get activation threshold."""
    nodecount = 133
    # create curve of potentials
    fiber = build_fiber(diameter=5.7, fiber_model=model, temperature=37, n_sections=133)
    fiber.potentials = norm.pdf(np.linspace(-1, 1, nodecount), 0, 0.05) * 100

    if save:
        fiber.record_gating()
        fiber.record_vm()

    waveform = np.concatenate((np.ones(200), -np.ones(200), np.zeros(49600)))

    # parameters
    time_step = 0.001
    time_stop = 5
    stimulation = ScaledStim(waveform=waveform, dt=time_step, tstop=time_stop)

    result = [stimulation.run_sim(stimamp, fiber) for stimamp in stimamps]

    aps = [r[0] for r in result]

    return aps, fiber


def test_end_excitation():
    fiber = get_fiber()
    fiber.potentials = norm.pdf(np.linspace(-1, 1, 133), 0, 0.5) * 100
    waveform = np.concatenate((np.ones(200), -np.ones(200), np.zeros(49600)))
    time_step = 0.001
    time_stop = 5
    stimulation = ScaledStim(waveform=waveform, dt=time_step, tstop=time_stop)
    with pytest.raises(RuntimeError):
        stimulation.find_threshold(fiber)


def test_mrg_discrete():
    assert np.isclose(get_activation_threshold(FiberModel.MRG_DISCRETE), -0.2691015625)


def test_mrg_interpolation():
    assert np.isclose(get_activation_threshold(FiberModel.MRG_INTERPOLATION), -0.27006835937499996)


def test_tigerholm():
    assert np.isclose(
        get_activation_threshold(FiberModel.TIGERHOLM, diameter=1, nodecount=265, stimamp_top=-5),
        -1.9056152343749997,
    )


def test_rattay():
    assert np.isclose(
        get_activation_threshold(FiberModel.RATTAY, diameter=1, nodecount=265, stimamp_top=-5),
        -1.359833984375,
    )


def test_sundt():
    assert np.isclose(
        get_activation_threshold(FiberModel.SUNDT, diameter=1, nodecount=265, stimamp_top=-5),
        -2.11515625,
    )


def test_schild94():
    assert np.isclose(
        get_activation_threshold(FiberModel.SCHILD94, diameter=1, nodecount=265, stimamp_top=-5),
        -2.08591796875,
    )


def test_schild97():
    assert np.isclose(
        get_activation_threshold(FiberModel.SCHILD97, diameter=1, nodecount=265, stimamp_top=-6),
        -5.8128125,
    )


def test_amp_response_and_var_save():
    resp, fiber = get_amp_responses(FiberModel.MRG_INTERPOLATION, [0.01, -0.1, -1], save=True)
    assert np.array_equal(resp, [0, 1, 1])

    assert np.isclose(fiber.vm[6][200], 389.23435435930253)

    assert np.isclose(fiber.gating['h'][6][200], 0.03842278091747966)
    assert np.isclose(fiber.gating['m'][6][200], 0.9999924888915432)
    assert np.isclose(fiber.gating['mp'][6][200], 0.9216810668251406)
    assert np.isclose(fiber.gating['s'][6][200], 0.10491253677621307)


def test_block_threshold():
    time_step = 0.0005  # ms
    time_stop = 50  # ms
    frequency = 20  # khz (because our time units are ms)
    # create time vector to make waveform with sg
    t = np.arange(0, time_stop, time_step)  # ms
    waveform = sg.square(2 * np.pi * frequency * t)

    n_sections = 265

    fiber = build_fiber(FiberModel.MRG_INTERPOLATION, diameter=10, n_sections=n_sections)
    fiber.record_vm()
    fiber.potentials = fiber.point_source_potentials(0, 250, fiber.length / 2, 1, 10)

    # Create new stimulation object
    stimulation = ScaledStim(waveform=waveform, dt=time_step, tstop=time_stop)

    # Add intrinsic activity
    fiber.add_intrinsic_activity(loc=0.1, avg_interval=5, num_stims=10000, start_time=10, noise=0)

    # Find threshold
    amp, _ = stimulation.find_threshold(fiber, stimamp_top=-3, block_delay=10, condition=ThresholdCondition.BLOCK)

    assert np.isclose(amp, -2.696328125)

    # Run simulation with no stimulation
    n, t = stimulation.run_sim(0, fiber)

    assert n == 8.0
    assert np.isclose(t, 45.477)


# def test_geometric_mean(): #should check that all of these are close to each other.
#     assert np.isclose(
#         get_activation_threshold(FiberModel.MRG_INTERPOLATION, bisection_mean=BisectionMean.GEOMETRIC),
#         -0.023501400846134893,
#     )

# def test_both_subthreshold():
#     assert np.isclose(
#         get_activation_threshold(FiberModel.MRG_INTERPOLATION, stimamp_top=-0.01, stimamp_bottom=-0.001),
#         -0.023512489759687515,
#     )


# def test_both_suprathreshold():
#     assert np.isclose(
#         get_activation_threshold(FiberModel.MRG_INTERPOLATION, stimamp_top=-1, stimamp_bottom=-0.1),
#         -0.02351225891204326,
#     )


# def test_allsupra():
#     assert np.isclose(
#         get_activation_threshold(FiberModel.MRG_INTERPOLATION, stimamp_top=-0.025, stimamp_bottom=-0.02), -0.023515625
#     )


# def test_absolutes():
#     assert np.isclose(
#         get_activation_threshold(
#             FiberModel.MRG_INTERPOLATION,
#             termination_mode=TerminationMode.ABSOLUTE_DIFFERENCE,
#             bounds_search_mode=BoundsSearchMode.ABSOLUTE_INCREMENT,
#             termination_tolerance=0.0001,
#         ),
#         -0.02350494384765625,
#     )
