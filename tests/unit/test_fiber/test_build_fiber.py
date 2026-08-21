"""Tests for build_fiber, generate, loc_index, and related construction.

The copyrights of this software are owned by Duke University.
See LICENSE for licensing instructions.
Source code: https://github.com/wmglab-duke/pyfibers
"""

from __future__ import annotations

import pytest

from pyfibers import FiberModel, build_fiber


@pytest.fixture(scope="module")
def fiber():
    return build_fiber(fiber_model=FiberModel.MRG_INTERPOLATION, diameter=10.0, n_nodes=5)


def test_exactly_one_of_length_n_sections_n_nodes():
    with pytest.raises(ValueError, match="exactly one of length, n_sections, or n_nodes"):
        build_fiber(fiber_model=FiberModel.MRG_INTERPOLATION, diameter=10.0)
    with pytest.raises(ValueError, match="exactly one of length, n_sections, or n_nodes"):
        build_fiber(fiber_model=FiberModel.MRG_INTERPOLATION, diameter=10.0, n_nodes=5, length=1000)


def test_is_3d_kwarg_rejected():
    with pytest.raises(ValueError, match="build_fiber_3d"):
        build_fiber(fiber_model=FiberModel.MRG_INTERPOLATION, diameter=10.0, n_nodes=5, is_3d=True)


def test_n_nodes_sets_nodecount(fiber):
    assert len(fiber) == 5
    assert fiber.nodecount == 5


def test_length_floors_to_delta_z_pattern():
    probe = build_fiber(fiber_model=FiberModel.MRG_INTERPOLATION, diameter=10.0, n_nodes=5)
    fiber = build_fiber(
        fiber_model=FiberModel.MRG_INTERPOLATION,
        diameter=10.0,
        length=probe.delta_z * 4.2,
    )
    assert fiber.nodecount == 5


def test_enforce_odd_nodecount_decrements(caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="pyfibers.fiber"):
        fiber = build_fiber(
            fiber_model=FiberModel.MRG_INTERPOLATION,
            diameter=10.0,
            n_nodes=4,
            enforce_odd_nodecount=True,
        )
    assert fiber.nodecount == 3
    assert "Altering node count" in caplog.text


def test_enforce_odd_false_keeps_even():
    fiber = build_fiber(
        fiber_model=FiberModel.MRG_INTERPOLATION,
        diameter=10.0,
        n_nodes=4,
        enforce_odd_nodecount=False,
    )
    assert fiber.nodecount == 4


def test_n_sections_must_match_pattern():
    with pytest.raises(ValueError, match="n_sections must be 1 \\+"):
        build_fiber(fiber_model=FiberModel.MRG_INTERPOLATION, diameter=10.0, n_sections=13)


def test_fewer_than_three_nodes_warns():
    with pytest.warns(UserWarning, match="fewer than 3 nodes"):
        build_fiber(
            fiber_model=FiberModel.MRG_INTERPOLATION,
            diameter=10.0,
            n_nodes=1,
            passive_end_nodes=False,
        )


def test_unmyelinated_diameter_gt_3_warns():
    with pytest.warns(UserWarning, match="Unmyelinated fibers are typically"):
        build_fiber(fiber_model=FiberModel.TIGERHOLM, diameter=5.0, n_nodes=5)


def test_loc_and_loc_index_invalid_range(fiber):
    with pytest.raises(ValueError, match="between 0 and 1"):
        fiber.loc_index(-0.1)
    with pytest.raises(ValueError, match="between 0 and 1"):
        fiber(1.1)


def test_loc_index_bad_target(fiber):
    with pytest.raises(ValueError, match='target can either be "nodes" or "sections"'):
        fiber.loc_index(0.5, target="axons")
    with pytest.raises(ValueError, match='target can either be "nodes" or "sections"'):
        fiber(0.5, target="axons")
    with pytest.raises(ValueError, match='target can either be "nodes" or "sections"'):
        fiber.__len__(target="axons")


def test_len_target_sections(fiber):
    assert fiber.__len__(target="sections") == len(fiber.sections)
    assert fiber.__len__(target="sections") > len(fiber)


def test_call_target_sections(fiber):
    section = fiber(0.5, target="sections")
    assert section is fiber.sections[fiber.loc_index(0.5, target="sections")]


def test_add_intrinsic_requires_loc_xor_index(fiber):
    with pytest.raises(ValueError, match="Must specify either loc or loc_index"):
        fiber.add_intrinsic_activity(loc=0.5, loc_index=1)
    with pytest.raises(ValueError, match="Must specify either loc or loc_index"):
        fiber.add_intrinsic_activity(loc=None, loc_index=None)


def test_add_intrinsic_sets_nc_syn_stim(fiber):
    fiber.add_intrinsic_activity(loc=None, loc_index=2)
    assert fiber.nc is not None
    assert fiber.syn is not None
    assert fiber.stim is not None
    fiber.nc = None
    fiber.syn = None
    fiber.stim = None


def test_add_intrinsic_loc_selects_node():
    fiber = build_fiber(fiber_model=FiberModel.MRG_INTERPOLATION, diameter=10.0, n_nodes=5)
    fiber.add_intrinsic_activity(loc=0.5, loc_index=None)
    assert fiber.syn.get_segment().sec is fiber(0.5)


def test_add_intrinsic_applies_stim_syn_nc_params():
    fiber = build_fiber(fiber_model=FiberModel.MRG_INTERPOLATION, diameter=10.0, n_nodes=5)
    fiber.add_intrinsic_activity(
        loc=None,
        loc_index=2,
        avg_interval=2,
        num_stims=3,
        start_time=4,
        noise=0.5,
        synapse_tau=0.2,
        synapse_reversal_potential=-10,
        netcon_weight=0.3,
    )
    assert fiber.stim.interval == 2
    assert fiber.stim.number == 3
    assert fiber.stim.start == 4
    assert fiber.stim.noise == 0.5
    assert fiber.syn.tau == 0.2
    assert fiber.syn.e == -10
    assert fiber.nc.weight[0] == pytest.approx(0.3)


def test_generate_requires_n_sections_if_others_none():
    fiber_class = FiberModel.MRG_INTERPOLATION.value
    fiber = fiber_class(diameter=10.0, fiber_model=FiberModel.MRG_INTERPOLATION)
    with pytest.raises(ValueError, match="n_sections must be specified"):
        fiber.generate()


def test_passive_end_nodes_false_all_active():
    fiber = build_fiber(
        fiber_model=FiberModel.MRG_INTERPOLATION,
        diameter=10.0,
        n_nodes=5,
        passive_end_nodes=False,
    )
    assert all("passive" not in node.name() for node in fiber.nodes)


def test_passive_end_nodes_int_marks_n_each_end():
    fiber = build_fiber(
        fiber_model=FiberModel.MRG_INTERPOLATION,
        diameter=10.0,
        n_nodes=7,
        passive_end_nodes=2,
    )
    names = [node.name() for node in fiber.nodes]
    assert "passive" in names[0]
    assert "passive" in names[1]
    assert "passive" not in names[3]
    assert "passive" in names[5]
    assert "passive" in names[6]


def test_nodebuilder_sets_geometry_and_extracellular():
    fiber = build_fiber(fiber_model=FiberModel.SUNDT, diameter=1.0, n_nodes=5)
    node = fiber.nodebuilder(0, "active")
    assert "active node 0" in node.name()
    assert node.L == pytest.approx(fiber.delta_z)
    assert node.diam == pytest.approx(fiber.diameter)
    assert node.nseg == 1
    assert node.xc[0] == 0
    assert node.xg[0] == pytest.approx(1e10)
    assert node.v == pytest.approx(fiber.v_rest)


def test_make_passive_rejects_nonpassive_name():
    fiber = build_fiber(fiber_model=FiberModel.MRG_INTERPOLATION, diameter=10.0, n_nodes=5)
    with pytest.raises(ValueError, match="must contain 'passive'"):
        fiber._make_passive(fiber.nodes[2])


def test_make_passive_sets_pas_params():
    fiber = build_fiber(fiber_model=FiberModel.SUNDT, diameter=1.0, n_nodes=5)
    node = fiber.nodebuilder(0, "passive")
    fiber._make_passive(node)
    assert node.g_pas == pytest.approx(0.0001)
    assert node.e_pas == pytest.approx(fiber.v_rest)
    assert node.Ra == pytest.approx(1e10)
    assert node.cm == pytest.approx(1)


def test_len_nodecount_mismatch():
    fiber = build_fiber(fiber_model=FiberModel.MRG_INTERPOLATION, diameter=10.0, n_nodes=5)
    fiber.nodecount = 99
    with pytest.raises(RuntimeError, match="Node count does not match"):
        len(fiber)


def test_fiber_str_repr(fiber):
    text = str(fiber)
    assert "MRG_INTERPOLATION" in text
    assert "10.0" in text
    assert "not 3d" in text
    assert "MRGFiber" in repr(fiber)


def test_set_xyz_warns_when_xy_vary():
    fiber = build_fiber(fiber_model=FiberModel.MRG_INTERPOLATION, diameter=10.0, n_nodes=5)
    fiber.coordinates[0, 0] = 1.0
    with pytest.warns(UserWarning, match="3D fiber path"):
        fiber.set_xyz(0, 0, 0)
