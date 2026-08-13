"""Tests for 3D fiber functionality.

The copyrights of this software are owned by Duke University.
See LICENSE for licensing instructions.
Source code: https://github.com/wmglab-duke/pyfibers
"""

from __future__ import annotations

import numpy as np
import pytest

from pyfibers import FiberModel, build_fiber, build_fiber_3d


class TestFiber3D:

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.diameter = 10.0  # Example diameter in um
        self.path_coordinates = np.array(
            [[0, 0, 0], [1000, 0, 0], [2000, 1000, 0], [3000, 1000, 1000]]
        )  # Example path coordinates in um, much longer fiber
        self.non_3d_diameter = 10.0
        self.non_3d_length = 10000.0  # Length in um, much longer fiber

    def test_build_fiber_3d(self):
        fiber = build_fiber_3d(
            fiber_model=FiberModel.MRG_INTERPOLATION,
            diameter=self.diameter,
            path_coordinates=self.path_coordinates,
            enforce_odd_nodecount=False,
        )
        assert fiber is not None
        assert fiber.diameter == self.diameter
        expected_length = 3367.9
        assert np.isclose(fiber.length, expected_length)  # Length calculated based on given path_coordinates

    def test_resample_potentials_3d(self):
        fiber = build_fiber_3d(
            fiber_model=FiberModel.MRG_INTERPOLATION,
            diameter=self.diameter,
            path_coordinates=self.path_coordinates,
            enforce_odd_nodecount=False,
        )
        # Assuming some dummy potentials and coordinates for testing
        potentials = np.array([1, 2, 3, 4])
        potential_coords = np.array([[0, 0, 0], [1000, 0, 0], [2000, 1000, 0], [3000, 1000, 1000]])
        resampled_potentials = fiber.resample_potentials_3d(potentials, potential_coords)
        assert resampled_potentials is not None
        assert len(resampled_potentials) == len(fiber.coordinates)
        expected_potentials = np.array(
            [
                1.0005,
                1.0025,
                1.0273669,
                1.1358865,
                1.3061919,
                1.4764973,
                1.6468027,
                1.8171081,
                1.9874135,
                2.06783495,
                2.0854185,
                2.08683271,
                2.08824693,
                2.10583048,
                2.18256542,
                2.30298953,
                2.42341363,
                2.54383773,
                2.66426184,
                2.78468594,
                2.86142089,
                2.87900444,
                2.88041865,
                2.88183287,
                2.89941642,
                2.97615137,
                3.09657547,
                3.21699957,
                3.33742368,
                3.45784778,
                3.57827188,
                3.65500683,
                3.67259038,
                3.67400459,
            ]
        )
        assert np.allclose(resampled_potentials, expected_potentials)

    def test_set_xyz_non_3d_fiber(self):
        fiber = build_fiber(
            fiber_model=FiberModel.MRG_INTERPOLATION, diameter=self.non_3d_diameter, length=self.non_3d_length
        )
        coordsave = fiber.coordinates.copy()
        fiber.set_xyz(1, 1, 1000)
        assert np.allclose(fiber.coordinates[:, 0], 1)
        assert np.allclose(fiber.coordinates[:, 1], 1)
        assert np.allclose(fiber.coordinates[:, 2], coordsave[:, 2] + 1000)

    def test_longitudinal_coordinates(self):
        fiber3d = build_fiber_3d(
            fiber_model=FiberModel.MRG_INTERPOLATION, diameter=self.diameter, path_coordinates=self.path_coordinates
        )
        fiber = build_fiber(fiber_model=FiberModel.MRG_INTERPOLATION, diameter=self.diameter, length=fiber3d.length + 1)
        assert np.allclose(fiber.longitudinal_coordinates, fiber3d.longitudinal_coordinates)
        assert np.isclose(fiber.length, fiber3d.length)

    def test_is_3d_flag(self):
        fiber3d = build_fiber_3d(
            fiber_model=FiberModel.MRG_INTERPOLATION, diameter=self.diameter, path_coordinates=self.path_coordinates
        )
        fiber1d = build_fiber(fiber_model=FiberModel.MRG_INTERPOLATION, diameter=self.diameter, n_nodes=5)
        assert fiber3d.is_3d() is True
        assert fiber1d.is_3d() is False

    def test_path_coordinates_required(self):
        with pytest.raises(ValueError, match="path_coordinates must be provided"):
            build_fiber_3d(fiber_model=FiberModel.MRG_INTERPOLATION, diameter=self.diameter, path_coordinates=None)

    def test_3d_rejects_n_nodes_length(self):
        with pytest.raises(ValueError, match="cannot specify n_sections, n_nodes, or length"):
            build_fiber_3d(
                fiber_model=FiberModel.MRG_INTERPOLATION,
                diameter=self.diameter,
                path_coordinates=self.path_coordinates,
                n_nodes=5,
            )

    def test_3d_shift_changes_coordinates(self):
        fiber0 = build_fiber_3d(
            fiber_model=FiberModel.MRG_INTERPOLATION,
            diameter=self.diameter,
            path_coordinates=self.path_coordinates,
            shift=0,
        )
        fiber1 = build_fiber_3d(
            fiber_model=FiberModel.MRG_INTERPOLATION,
            diameter=self.diameter,
            path_coordinates=self.path_coordinates,
            shift=100,
        )
        assert not np.allclose(fiber0.coordinates, fiber1.coordinates)

    def test_3d_shift_and_shift_ratio_conflict(self):
        with pytest.raises(ValueError, match="Cannot specify both shift and shift_ratio"):
            build_fiber_3d(
                fiber_model=FiberModel.MRG_INTERPOLATION,
                diameter=self.diameter,
                path_coordinates=self.path_coordinates,
                shift=10,
                shift_ratio=0.1,
            )

    def test_set_xyz_raises_on_3d(self):
        fiber = build_fiber_3d(
            fiber_model=FiberModel.MRG_INTERPOLATION, diameter=self.diameter, path_coordinates=self.path_coordinates
        )
        with pytest.raises(ValueError, match="not compatible with 3D"):
            fiber.set_xyz(1, 1, 1)

    def test_resample_potentials_3d_rejects_1d_fiber(self):
        fiber = build_fiber(fiber_model=FiberModel.MRG_INTERPOLATION, diameter=self.diameter, n_nodes=5)
        with pytest.raises(ValueError, match="only compatible with 3D"):
            fiber.resample_potentials_3d(np.array([1, 2, 3]), np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]))

    def test_resample_potentials_3d_shape_errors(self):
        fiber = build_fiber_3d(
            fiber_model=FiberModel.MRG_INTERPOLATION, diameter=self.diameter, path_coordinates=self.path_coordinates
        )
        with pytest.raises(ValueError, match="2D array"):
            fiber.resample_potentials_3d(np.array([1, 2, 3, 4]), np.array([0, 1, 2, 3]))
        with pytest.raises(ValueError, match="exactly 3 coordinates"):
            fiber.resample_potentials_3d(np.array([1, 2, 3, 4]), np.array([[0, 0], [1, 0], [2, 0], [3, 0]]))


if __name__ == "__main__":
    pytest.main()
