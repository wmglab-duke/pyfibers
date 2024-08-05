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
        for sec in self.sections:
            sec.L = 10
            sec.diam = 1
            sec.insert('extracellular')
            sec.insert('pas')
        self.nodes = self.sections
        self.delta_z = 10
        self.nodecount = 5
        self.coordinates = np.array([0, 10, 20, 30, 40])
        self.length = 40
        self.passive_end_nodes = 1


@pytest.fixture
def mock_fiber():
    return MockFiber(fiber_model=None, diameter=1.0)


def test_record_values_vm(mock_fiber):
    mock_fiber.set_save_vm()
    assert mock_fiber.vm is not None
    assert len(mock_fiber.vm) == len(mock_fiber.nodes)

    for i in range(len(mock_fiber.nodes)):
        if i < mock_fiber.passive_end_nodes or i >= len(mock_fiber.nodes) - mock_fiber.passive_end_nodes:
            assert mock_fiber.vm[i] is None
        else:
            assert isinstance(mock_fiber.vm[i], type(h.Vector))


def test_record_values_im(mock_fiber):
    mock_fiber.set_save_im()
    assert mock_fiber.im is not None
    assert len(mock_fiber.im) == len(mock_fiber.nodes)

    for i in range(len(mock_fiber.nodes)):
        if i < mock_fiber.passive_end_nodes or i >= len(mock_fiber.nodes) - mock_fiber.passive_end_nodes:
            assert mock_fiber.im[i] is None
        else:
            assert isinstance(mock_fiber.im[i], type(h.Vector))


def test_record_values_custom_variable(mock_fiber):
    mock_fiber.gating_variables = {"i_pas": "i_pas"}
    mock_fiber.set_save_gating()
    assert mock_fiber.gating is not None
    assert len(mock_fiber.gating) == 1
    assert "i_pas" in mock_fiber.gating

    for recordings in mock_fiber.gating.values():
        assert len(recordings) == len(mock_fiber.nodes)
        for i in range(len(mock_fiber.nodes)):
            if i < mock_fiber.passive_end_nodes or i >= len(mock_fiber.nodes) - mock_fiber.passive_end_nodes:
                assert recordings[i] is None
            else:
                assert isinstance(recordings[i], type(h.Vector))


if __name__ == "__main__":
    pytest.main()
