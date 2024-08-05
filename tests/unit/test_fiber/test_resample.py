from __future__ import annotations

import numpy as np
import pytest
from neuron import h

from pyfibers.fiber import Fiber


# Mock class to test Fiber without dependencies
class MockFiber(Fiber):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sections = [h.Section(name=f"section{i}") for i in range(5)]
        self.nodes = self.sections
        self.delta_z = 10
        self.nodecount = 5
        self.longitudinal_coordinates = np.array([0, 10, 20, 30, 40])
        self.length = 40


@pytest.fixture
def mock_fiber():
    return MockFiber(fiber_model=None, diameter=1.0)


def test_resample_potentials_not_on_coords(mock_fiber):
    potentials = np.array([0, 3, 6, 9, 12])
    potential_coords = np.array([-10, 0, 20, 40, 50])
    resampled_potentials = mock_fiber.resample_potentials(potentials, potential_coords)
    expected_potentials = np.array([0.0, 3.0, 4.5, 6.0, 7.5])
    assert np.allclose(resampled_potentials, expected_potentials)


def test_resample_potentials_high_res_not_on_coords(mock_fiber):
    potentials = np.array([0, 1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12])
    potential_coords = np.array([-10, 0, 5, 10, 20, 30, 40, 50, 60])
    resampled_potentials = mock_fiber.resample_potentials(potentials, potential_coords)
    expected_potentials = np.array([0.0, 1.5, 4.5, 6.0, 7.5])
    assert np.allclose(resampled_potentials, expected_potentials)


def test_resample_potentials_low_res_not_on_coords(mock_fiber):
    potentials = np.array([0, 6, 12])
    potential_coords = np.array([-20, 20, 60])
    resampled_potentials = mock_fiber.resample_potentials(potentials, potential_coords)
    expected_potentials = np.array([0.0, 1.5, 3.0, 4.5, 6.0])
    assert np.allclose(resampled_potentials, expected_potentials)


def test_resample_potentials_not_on_coords_inplace(mock_fiber):
    potentials = np.array([0, 3, 6, 9, 12])
    potential_coords = np.array([-10, 0, 20, 40, 50])
    mock_fiber.resample_potentials(potentials, potential_coords, inplace=True)
    expected_potentials = np.array([0.0, 3.0, 4.5, 6.0, 7.5])
    assert np.allclose(mock_fiber.potentials, expected_potentials)


def test_resample_potentials_not_on_coords_center(mock_fiber):
    potentials = np.array([0, 3, 6, 9, 12])
    potential_coords = np.array([-30, -10, 0, 10, 30])
    resampled_potentials = mock_fiber.resample_potentials(potentials, potential_coords, center=True)
    expected_potentials = np.array([1.5, 3.0, 6.0, 9.0, 10.5])
    assert np.allclose(resampled_potentials, expected_potentials)


def test_resample_potentials_not_on_coords_inplace_center(mock_fiber):
    potentials = np.array([0, 3, 6, 9, 12])
    potential_coords = np.array([-30, -10, 0, 10, 30])
    mock_fiber.resample_potentials(potentials, potential_coords, center=True, inplace=True)
    expected_potentials = np.array([1.5, 3.0, 6.0, 9.0, 10.5])
    assert np.allclose(mock_fiber.potentials, expected_potentials)


if __name__ == "__main__":
    pytest.main()
