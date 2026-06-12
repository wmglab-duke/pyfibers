from __future__ import annotations

import numpy as np
import pytest
from neuron import h

from pyfibers.fiber import Fiber


# Mock class to test Fiber without dependencies
class MockFiber(Fiber):
    # Override properties to allow setting for testing
    @property
    def length(self: Fiber) -> float:
        if hasattr(self, '_mock_length'):
            return self._mock_length
        return super().length

    @length.setter
    def length(self: Fiber, value: float) -> None:
        self._mock_length = value

    @property
    def longitudinal_coordinates(self: Fiber) -> np.ndarray:
        if hasattr(self, '_mock_longitudinal_coordinates'):
            return self._mock_longitudinal_coordinates
        return super().longitudinal_coordinates

    @longitudinal_coordinates.setter
    def longitudinal_coordinates(self: Fiber, value: np.ndarray) -> None:
        self._mock_longitudinal_coordinates = value

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sections = [h.Section(name=f"section{i}") for i in range(5)]
        self.nodes = self.sections
        self.delta_z = 10
        self.nodecount = 5
        self.longitudinal_coordinates = np.array([0, 10, 20, 30, 40])
        self.length = 40
        self.apc = [h.APCount(sec(0.5)) for sec in self.sections]


@pytest.fixture
def mock_fiber():
    return MockFiber(fiber_model=None, diameter=1.0)


def test_measure_cv_linear_conduction(mock_fiber):
    times = [1, 2, 3, 4, 5]
    for apc, time in zip(mock_fiber.apc, times):
        apc.time = time
        apc.n = 1

    cv = mock_fiber.measure_cv(start=0, end=1)
    expected_cv = (40e-6) / (4e-3)  # distance in meters, time in seconds
    assert np.isclose(cv, expected_cv, atol=1e-6)


def test_measure_cv_non_linear_conduction(mock_fiber):
    times = [1, 3, 8, 9, 10]
    for apc, time in zip(mock_fiber.apc, times):
        apc.time = time
        apc.n = 1

    with pytest.raises(ValueError, match="Conduction is not linear between the specified nodes."):
        mock_fiber.measure_cv(start=0, end=1)


def test_measure_cv_partial_fiber(mock_fiber):
    times = [2, 3, 4]
    for apc, time in zip(mock_fiber.apc[1:-1], times):
        apc.time = time
        apc.n = 1

    cv = mock_fiber.measure_cv(start=0.25, end=0.75)
    expected_cv = (20e-6) / (2e-3)  # distance in meters, time in seconds
    assert np.isclose(cv, expected_cv, atol=1e-6)


def test_measure_cv_no_ap(mock_fiber):
    mock_fiber.apc[1].time = 2
    mock_fiber.apc[1].n = 1
    mock_fiber.apc[2].time = 0  # No AP detected

    with pytest.raises(RuntimeError, match="No detected APs at node"):
        mock_fiber.measure_cv(start=0.25, end=0.75)


if __name__ == "__main__":
    pytest.main()
