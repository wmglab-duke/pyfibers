"""Tests for fiber recording functionality.

The copyrights of this software are owned by Duke University.
See LICENSE for licensing instructions.
Source code: https://github.com/wmglab-duke/pyfibers
"""

from __future__ import annotations

import pytest
from neuron import h

from pyfibers import FiberModel, build_fiber


@pytest.fixture
def setup_fiber():
    """Set up an actual MRG Fiber instance for testing."""
    return build_fiber(fiber_model=FiberModel.MRG_INTERPOLATION, diameter=10, n_nodes=10)


def test_apcounts(setup_fiber):
    """Test the apcounts method."""
    fiber = setup_fiber
    fiber.apcounts(thresh=-30)
    assert len(fiber.apc) == len(fiber.nodes)
    for apc in fiber.apc:
        assert apc.thresh == -30


def test_record_values_missing_attr_error(setup_fiber):
    """Test that AttributeError is raised if the attribute is not in the section and allow_missing is False."""
    fiber = setup_fiber
    ref_attr = '_ref_non_existent_attr'
    with pytest.raises(AttributeError):
        fiber.record_values(ref_attr=ref_attr, allsec=True, allow_missing=False)


def test_record_values_missing_attr_none(setup_fiber):
    """Test that None is returned if the attribute is not in the section and allow_missing is True."""
    fiber = setup_fiber
    ref_attr = '_ref_non_existent_attr'
    indices = [0, 1, 2]
    recorded_values = fiber.record_values(ref_attr=ref_attr, allsec=True, indices=indices, allow_missing=True)
    assert len(recorded_values) == len(indices)
    for record in recorded_values:
        assert record is None


def test_record_values_allsec(setup_fiber):
    """Test the record_values method with allsec=True."""
    fiber = setup_fiber
    ref_attr = '_ref_v'
    indices = [0, 1, 2]
    recorded_values = fiber.record_values(ref_attr=ref_attr, allsec=True, indices=indices)
    assert len(recorded_values) == len(indices)
    for record in recorded_values:
        assert record is not None


def test_record_values_nodes(setup_fiber):
    """Test the record_values method with allsec=False."""
    fiber = setup_fiber
    ref_attr = '_ref_v'
    indices = [0, 1, 2]
    recorded_values = fiber.record_values(ref_attr=ref_attr, allsec=False, indices=indices)
    assert len(recorded_values) == len(indices)
    for record in recorded_values:
        assert record is not None


def test_record_vm(setup_fiber):
    """Test the record_vm method."""
    fiber = setup_fiber
    indices = [0, 1, 2]
    vm = fiber.record_vm(allsec=True, indices=indices)
    assert len(vm) == len(indices)
    for v in vm:
        assert v is not None


def test_record_im(setup_fiber):
    """Test the record_im method."""
    fiber = setup_fiber
    indices = [0, 1, 2]
    im = fiber.record_im(allsec=True, indices=indices)
    assert len(im) == len(indices)
    for i in im:
        assert i is not None


def test_record_vext(setup_fiber):
    """Test the record_vext method."""
    fiber = setup_fiber
    vext = fiber.record_vext()
    assert len(vext) == len(fiber.sections)
    for ve in vext:
        assert ve is not None


def test_record_gating(setup_fiber):
    """Test the record_gating method."""
    fiber = setup_fiber
    fiber.gating_variables = {'Na': 'm', 'K': 'h'}
    gating = fiber.record_gating(allsec=False)
    assert len(gating) == len(fiber.gating_variables)
    for name in fiber.gating_variables:
        assert gating[name] is not None
        assert len(gating[name]) == len(fiber.nodes)


def test_record_values_empty_indices(setup_fiber):
    """Test the record_values method with empty indices."""
    fiber = setup_fiber
    ref_attr = '_ref_v'
    with pytest.raises(
        ValueError,
        match="Indices cannot be an empty list. If you want to record from all nodes, omit the indices argument.",
    ):
        fiber.record_values(ref_attr=ref_attr, allsec=True, indices=[])


def test_bad_indices(setup_fiber):
    """Test that ValueError is raised if the indices are out of bounds."""
    fiber = setup_fiber
    ref_attr = '_ref_v'
    indices = [len(fiber.sections)]
    with pytest.raises(IndexError):
        fiber.record_values(ref_attr=ref_attr, allsec=True, indices=indices)


def test_record_values_dt_and_tvec_conflict(setup_fiber):
    fiber = setup_fiber
    with pytest.raises(ValueError, match="Cannot specify both recording_dt and recording_tvec"):
        fiber.record_values("_ref_v", recording_dt=0.1, recording_tvec=h.Vector([0, 1]))


def test_record_values_recording_dt(setup_fiber):
    fiber = setup_fiber
    recorded = fiber.record_values("_ref_v", recording_dt=0.1, indices=[0, 1])
    assert len(recorded) == 2
    assert all(record is not None for record in recorded)


def test_record_gating_empty_raises(setup_fiber):
    fiber = setup_fiber
    fiber.gating_variables = {}
    with pytest.raises(RuntimeError, match="Gating variables not defined"):
        fiber.record_gating()


def test_record_gating_uses_model_map(setup_fiber):
    fiber = setup_fiber
    gating = fiber.record_gating()
    assert set(gating) == {"h", "m", "mp", "s"}
    assert len(gating["h"]) == len(fiber.nodes)


def test_set_save_aliases(setup_fiber):
    from pyfibers.fiber import Fiber

    assert Fiber.set_save_vm is Fiber.record_vm
    assert Fiber.set_save_im is Fiber.record_im
    assert Fiber.set_save_vext is Fiber.record_vext
    assert Fiber.set_save_gating is Fiber.record_gating


def test_record_values_recording_tvec(setup_fiber):
    fiber = setup_fiber
    tvec = h.Vector([0, 1, 2])
    recorded = fiber.record_values("_ref_v", recording_tvec=tvec, indices=[0, 1])
    assert len(recorded) == 2
    assert all(record is not None for record in recorded)


def test_time_recorder_survives_fiber_replacement():
    """Replacing a fiber must not drop time recording.

    See https://github.com/neuronsimulator/nrn/issues/3603#issuecomment-3299874561
    """
    h.dt = 0.025
    fiber = build_fiber(fiber_model=FiberModel.MRG_INTERPOLATION, diameter=10, n_nodes=5)
    h.finitialize(fiber.v_rest)
    h.frecord_init()
    for _ in range(5):
        h.fadvance()
    assert len(fiber.time) > 1

    fiber = build_fiber(fiber_model=FiberModel.MRG_INTERPOLATION, diameter=10, n_nodes=5)
    h.finitialize(fiber.v_rest)
    h.frecord_init()
    for _ in range(5):
        h.fadvance()
    assert len(fiber.time) > 1


if __name__ == "__main__":
    pytest.main()
