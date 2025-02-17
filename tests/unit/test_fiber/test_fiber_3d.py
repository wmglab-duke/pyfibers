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
            fiber_model=FiberModel.MRG_INTERPOLATION, diameter=self.diameter, path_coordinates=self.path_coordinates
        )
        assert fiber is not None
        assert fiber.diameter == self.diameter
        expected_length = 3367.9
        assert np.isclose(fiber.length, expected_length)  # Length calculated based on given path_coordinates

    def test_resample_potentials_3d(self):
        fiber = build_fiber_3d(
            fiber_model=FiberModel.MRG_INTERPOLATION, diameter=self.diameter, path_coordinates=self.path_coordinates
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


if __name__ == "__main__":
    pytest.main()
