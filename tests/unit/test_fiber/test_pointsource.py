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
        self.longitudinal_coordinates = np.array([0, 10, 20, 30, 40], dtype=float)
        self.coordinates = np.hstack([np.zeros((5, 1)), np.zeros((5, 1)), self.longitudinal_coordinates[:, None]])
        self.length = 40


@pytest.fixture
def mock_fiber():
    return MockFiber(fiber_model=None, diameter=1.0)


def test_point_source_potentials_isotropic(mock_fiber):
    x, y, z, i0, sigma = 0, 0, 0, 1, 1.0
    potentials = mock_fiber.point_source_potentials(x, y, z, i0, sigma)
    expected_potentials = i0 / (4 * np.pi * sigma * np.sqrt(np.array([0, 100, 400, 900, 1600]) * 1e-12))
    assert np.allclose(potentials, expected_potentials, atol=1e-10)


def test_point_source_potentials_anisotropic(mock_fiber):
    x, y, z, i0, sigma = 0, 0, 0, 1, (1.0, 1.0, 1.0)
    potentials = mock_fiber.point_source_potentials(x, y, z, i0, sigma)
    expected_potentials = i0 / (4 * np.pi * np.sqrt(np.array([0, 100, 400, 900, 1600]) * 1e-12))
    assert np.allclose(potentials, expected_potentials, atol=1e-10)


def test_point_source_potentials_offset(mock_fiber):
    x, y, z, i0, sigma = 10, 10, 10, 1, 1.0
    potentials = mock_fiber.point_source_potentials(x, y, z, i0, sigma)
    expected_potentials = np.array([4594.40746185, 5626.97697598, 4594.40746185, 3248.73667181, 2399.35104439])
    assert np.allclose(potentials, expected_potentials, atol=1e-10)


def test_point_source_potentials_inplace(mock_fiber):
    x, y, z, i0, sigma = 0, 0, 0, 1, 1.0
    mock_fiber.point_source_potentials(x, y, z, i0, sigma, inplace=True)
    expected_potentials = i0 / (4 * np.pi * sigma * np.sqrt(np.array([0, 100, 400, 900, 1600]) * 1e-12))
    assert np.allclose(mock_fiber.potentials, expected_potentials, atol=1e-10)


if __name__ == "__main__":
    pytest.main()
