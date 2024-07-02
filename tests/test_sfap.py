from __future__ import annotations

import numpy as np
from neuron import h

from pyfibers.fiber import Fiber


def create_mock_section(vext_value, length, xraxial_value):
    sec = h.Section()
    sec.L = length
    sec.diam = 1  # Set a default diameter
    sec.insert('extracellular')
    for seg in sec:
        seg.xraxial[0] = xraxial_value
        seg.vext[0] = vext_value
    return sec


def test_calculate_periaxonal_current():
    from_sec = create_mock_section(vext_value=5.0, length=100, xraxial_value=1.0)
    to_sec = create_mock_section(vext_value=0.0, length=100, xraxial_value=1.0)

    result = Fiber.calculate_periaxonal_current(from_sec, to_sec)

    expected_result = 5.0 / (1e6 * 1e-2)  # [mA]

    assert np.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"


def test_membrane_currents():
    class MockFiber(Fiber):
        def __init__(self):
            self.im = [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]]  # section 1  # section 2
            self.time = [0, 1, 2]
            self.sections = [
                create_mock_section(vext_value=5.0, length=100, xraxial_value=1.0),
                create_mock_section(vext_value=0.0, length=200, xraxial_value=1.0),
            ]
            self.myelinated = True

    fiber = MockFiber()
    membrane_currents_result = fiber.membrane_currents(downsample=1)

    expected_result = np.array([[-0.0003333, 0.00033358], [-0.00033327, 0.00033365], [-0.00033324, 0.00033371]])

    assert np.allclose(
        membrane_currents_result, expected_result
    ), f"Expected {expected_result}, but got {membrane_currents_result}"


def test_sfap():
    current_matrix = np.array([[1, 2], [3, 4]])
    potentials = np.array([0.5, 1.5])

    result = Fiber.sfap(current_matrix, potentials)

    expected_result = 1e3 * np.array([3.5, 7.5])  # [uV]

    assert np.allclose(result, expected_result), f"Expected {expected_result}, but got {result}"


def test_record_sfap():
    class MockFiber(Fiber):
        def __init__(self):
            self.im = [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]]  # section 1  # section 2
            self.time = [0, 1, 2]
            self.sections = [
                create_mock_section(vext_value=5.0, length=100, xraxial_value=1.0),
                create_mock_section(vext_value=0.0, length=200, xraxial_value=1.0),
            ]
            self.myelinated = True

        def membrane_currents(self, downsample=1):
            return np.array([[0.001, 0.002], [0.003, 0.004], [0.005, 0.006]])

    fiber = MockFiber()
    rec_potentials = [0.5, 1.5]

    result = fiber.record_sfap(rec_potentials, downsample=1)

    expected_result = 1e3 * np.array([0.0035, 0.0075, 0.0115])  # [uV]

    assert np.allclose(result, expected_result), f"Expected {expected_result}, but got {result}"
