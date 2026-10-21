"""Tests for Fiber.ve_second_difference.

The copyrights of this software are owned by Duke University.
See LICENSE for licensing instructions.
Source code: https://github.com/wmglab-duke/pyfibers
"""

from __future__ import annotations

import numpy as np
import pytest

from pyfibers import FiberModel, build_fiber


def _node_z(fiber):
    return np.array([z for z, sec in zip(fiber.longitudinal_coordinates, fiber.sections) if sec in fiber.nodes])


@pytest.fixture
def mrg():
    return build_fiber(diameter=5.7, fiber_model=FiberModel.MRG_DISCRETE, temperature=37, n_nodes=11)


@pytest.fixture
def unmyel():
    return build_fiber(diameter=1, fiber_model=FiberModel.RATTAY, temperature=37, n_sections=21)


def _interior_coords(fiber):
    node_mask = np.array([sec in fiber.nodes for sec in fiber.sections])
    return fiber.longitudinal_coordinates[node_mask][1:-1], fiber.coordinates[node_mask][1:-1]


def test_linear_ve_gives_zeros(mrg):
    mrg.potentials = 3.0 * mrg.longitudinal_coordinates + 1.0
    result, z, xyz = mrg.ve_second_difference()
    expected_z, expected_xyz = _interior_coords(mrg)
    assert result.shape == (1, len(mrg.nodes) - 2)
    np.testing.assert_allclose(result, 0, atol=1e-6)
    np.testing.assert_allclose(z, expected_z)
    np.testing.assert_allclose(xyz, expected_xyz)


def test_quadratic_ve_gives_constant(mrg):
    mrg.potentials = 1e-6 * mrg.longitudinal_coordinates**2
    dz = np.diff(_node_z(mrg))[0]
    result, _, _ = mrg.ve_second_difference()
    np.testing.assert_allclose(result, 2e-6 * dz**2)


def test_uses_nodes_only(mrg):
    # Non-node sections get large values that must not leak into the result
    potentials = np.full(len(mrg.sections), 1e3)
    node_mask = np.array([sec in mrg.nodes for sec in mrg.sections])
    node_ve = np.arange(node_mask.sum(), dtype=float) ** 2
    potentials[node_mask] = node_ve
    mrg.potentials = potentials
    result, z, xyz = mrg.ve_second_difference()
    np.testing.assert_allclose(result, np.diff(node_ve, n=2)[None, :])
    np.testing.assert_allclose(z, mrg.longitudinal_coordinates[node_mask][1:-1])
    np.testing.assert_allclose(xyz, mrg.coordinates[node_mask][1:-1])


def test_multiple_sources(unmyel):
    z = unmyel.longitudinal_coordinates
    unmyel.potentials = [z, z**2]
    result, node_z, xyz = unmyel.ve_second_difference()
    expected_z, expected_xyz = _interior_coords(unmyel)
    assert result.shape == (2, len(unmyel.nodes) - 2)
    np.testing.assert_allclose(result[0], 0, atol=1e-6)
    np.testing.assert_allclose(result[1], np.diff(_node_z(unmyel) ** 2, n=2))
    np.testing.assert_allclose(node_z, expected_z)
    np.testing.assert_allclose(xyz, expected_xyz)


def test_error_without_potentials(mrg):
    with pytest.raises(ValueError, match="No fiber potentials set"):
        mrg.ve_second_difference()
