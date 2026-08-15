"""Tests for action potential counting.

The copyrights of this software are owned by Duke University.
See LICENSE for licensing instructions.
Source code: https://github.com/wmglab-duke/pyfibers
"""

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
        self.coordinates = np.array([0, 10, 20, 30, 40])
        self.length = 40
        self.apc = None


@pytest.fixture
def mock_fiber():
    return MockFiber(fiber_model=None, diameter=1.0)


def test_apcounts_initialization(mock_fiber):
    mock_fiber.apcounts()
    assert mock_fiber.apc is not None
    assert len(mock_fiber.apc) == len(mock_fiber.nodes)
    assert all(isinstance(apc, type(h.APCount)) for apc in mock_fiber.apc)


def test_apcounts_threshold(mock_fiber):
    threshold = -20
    mock_fiber.apcounts(thresh=threshold)
    assert all(apc.thresh == threshold for apc in mock_fiber.apc)


def test_apcounts_action_potential_detection(mock_fiber):
    mock_fiber.apcounts()
    stim = h.IClamp(mock_fiber.nodes[0](0.5))
    stim.delay = 5
    stim.dur = 1
    stim.amp = 100

    h.finitialize(-65)
    h.continuerun(20)

    assert np.any([apc.n for apc in mock_fiber.apc])


if __name__ == "__main__":
    pytest.main()
