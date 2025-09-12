"""Tests for IntraStim cleanup functionality.

The copyrights of this software are owned by Duke University. Please
refer to the LICENSE and README.md files for licensing instructions. The
source code can be found on the following GitHub repository:
https://github.com/wmglab-duke/ascent
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pyfibers import IntraStim


@pytest.fixture
def mock_neuron():
    """Create a mock NEURON module."""
    h = MagicMock()
    h.Vector = MagicMock(return_value=MagicMock(record=MagicMock()))
    h.trainIClamp = MagicMock()
    h.fadvance = MagicMock()
    h._ref_t = 0
    h.load_file = MagicMock()
    h.celsius = 37
    h.finitialize = MagicMock()
    h.frecord_init = MagicMock()
    h.t = 0
    h.dt = 0.001
    return h


def test_intrastim_cleanup_method(mock_neuron):
    """Test the _cleanup_istim method directly."""
    with patch('pyfibers.stimulation.h', mock_neuron):
        istim = IntraStim(istim_loc=0.5, dt=0.001, tstop=10)

        # Set some mock objects
        istim.istim = MagicMock()
        istim.istim_record = MagicMock()

        # Call cleanup
        istim._cleanup_istim()

        # Verify cleanup
        assert istim.istim is None
        assert istim.istim_record is None


def test_intrastim_cleanup_with_existing_objects(mock_neuron):
    """Test cleanup when istim objects already exist."""
    with patch('pyfibers.stimulation.h', mock_neuron):
        istim = IntraStim(istim_loc=0.5, dt=0.001, tstop=10)

        # Set up existing objects
        istim.istim = MagicMock()
        istim.istim_record = MagicMock()

        # Call cleanup
        istim._cleanup_istim()

        # Verify cleanup
        assert istim.istim is None
        assert istim.istim_record is None


if __name__ == "__main__":
    pytest.main()
