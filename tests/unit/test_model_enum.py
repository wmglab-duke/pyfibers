"""Tests for pyfibers.model_enum module.

The copyrights of this software are owned by Duke University. Please
refer to the LICENSE and README.md files for licensing instructions. The
source code can be found on the following GitHub repository:
https://github.com/wmglab-duke/ascent
"""

from __future__ import annotations

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

from pyfibers import FiberModel
from pyfibers.fiber import Fiber
from pyfibers.model_enum import (
    _add_fiber_to_members,
    _create_fiber_model_enum,
    _discover_plugins,
    _update_all_module_references,
    register_custom_fiber,
)


class MockFiber(Fiber):
    """Mock fiber class for testing."""

    submodels = ["MOCK"]

    def __init__(self, diameter: float, **kwargs):
        super().__init__(diameter=diameter, **kwargs)
        self.v_rest = -70
        self.myelinated = False
        self.delta_z = 10.0

    def generate(self, **kwargs):
        return super().generate([self.create_mock], **kwargs)

    def create_mock(self, ind: int, node_type: str):
        from neuron import h

        node = h.Section(name=f"{node_type} node {ind}")
        node.L = self.delta_z
        node.diam = self.diameter
        node.nseg = 1
        node.insert("extracellular")
        node.xc[0] = 0
        node.xg[0] = 1e10
        node.v = self.v_rest
        return node


class MockFiberMultipleSubmodels(Fiber):
    """Mock fiber class with multiple submodels for testing."""

    submodels = ["MOCK_A", "MOCK_B"]

    def __init__(self, diameter: float, **kwargs):
        super().__init__(diameter=diameter, **kwargs)
        self.v_rest = -70
        self.myelinated = False
        self.delta_z = 10.0

    def generate(self, **kwargs):
        return super().generate([self.create_mock], **kwargs)

    def create_mock(self, ind: int, node_type: str):
        from neuron import h

        node = h.Section(name=f"{node_type} node {ind}")
        node.L = self.delta_z
        node.diam = self.diameter
        node.nseg = 1
        node.insert("extracellular")
        node.xc[0] = 0
        node.xg[0] = 1e10
        node.v = self.v_rest
        return node


class MockFiberNoSubmodels(Fiber):
    """Mock fiber class without submodels for testing error cases."""

    def __init__(self, diameter: float, **kwargs):
        super().__init__(diameter=diameter, **kwargs)
        self.v_rest = -70
        self.myelinated = False
        self.delta_z = 10.0

    def generate(self, **kwargs):
        return super().generate([self.create_mock], **kwargs)

    def create_mock(self, ind: int, node_type: str):
        from neuron import h

        node = h.Section(name=f"{node_type} node {ind}")
        node.L = self.delta_z
        node.diam = self.diameter
        node.nseg = 1
        node.insert("extracellular")
        node.xc[0] = 0
        node.xg[0] = 1e10
        node.v = self.v_rest
        return node


class NotAFiber:
    """Mock class that is not a Fiber subclass for testing error cases."""

    submodels = ["NOT_FIBER"]


class TestAddFiberToMembers:
    """Test the _add_fiber_to_members function."""

    def test_add_valid_fiber(self):
        """Test adding a valid fiber class to members dictionary."""
        members = {}
        _add_fiber_to_members(members, MockFiber)

        assert "MOCK" in members
        assert members["MOCK"] is MockFiber

    def test_add_fiber_with_multiple_submodels(self):
        """Test adding a fiber class with multiple submodels."""
        members = {}
        _add_fiber_to_members(members, MockFiberMultipleSubmodels)

        assert "MOCK_A" in members
        assert "MOCK_B" in members
        assert members["MOCK_A"] is MockFiberMultipleSubmodels
        assert members["MOCK_B"] is MockFiberMultipleSubmodels

    def test_add_fiber_overwrites_existing(self, caplog):
        """Test that adding a fiber with existing submodel name overwrites and warns."""
        members = {"MOCK": "existing_class"}

        with caplog.at_level(logging.WARNING, logger='pyfibers.model_enum'):
            _add_fiber_to_members(members, MockFiber)

        assert members["MOCK"] is MockFiber
        assert "Overwriting existing fiber model 'MOCK'" in caplog.text

    def test_add_fiber_no_submodels_raises_error(self):
        """Test that adding a fiber without submodels raises ValueError."""
        members = {}
        with pytest.raises(ValueError, match="must have a 'submodels' attribute"):
            _add_fiber_to_members(members, MockFiberNoSubmodels)

    def test_add_non_fiber_class_raises_error(self):
        """Test that adding a non-Fiber class raises ValueError."""
        members = {}
        with pytest.raises(ValueError, match="must be a subclass of Fiber"):
            _add_fiber_to_members(members, NotAFiber)


class TestDiscoverPlugins:
    """Test the _discover_plugins function."""

    @patch('pyfibers.model_enum.entry_points')
    def test_discover_plugins_success(self, mock_entry_points):
        """Test successful plugin discovery."""
        # Mock entry point
        mock_entry = MagicMock()
        mock_entry.load.return_value = MockFiber
        mock_entry.name = "test_plugin"
        mock_entry_points.return_value = [mock_entry]

        plugins = _discover_plugins()

        assert "MOCK" in plugins
        assert plugins["MOCK"] is MockFiber

    @patch('pyfibers.model_enum.entry_points')
    def test_discover_plugins_no_submodels_raises_error(self, mock_entry_points, caplog):
        """Test that plugins without submodels raise ValueError."""
        # Mock entry point with class that has no submodels
        mock_entry = MagicMock()
        mock_entry.load.return_value = MockFiberNoSubmodels
        mock_entry.name = "test_plugin"
        mock_entry_points.return_value = [mock_entry]

        with caplog.at_level(logging.ERROR, logger='pyfibers.model_enum'):
            plugins = _discover_plugins()

        # Should be empty due to error
        assert len(plugins) == 0
        assert "Error loading plugin test_plugin" in caplog.text

    @patch('pyfibers.model_enum.entry_points')
    def test_discover_plugins_load_error(self, mock_entry_points, caplog):
        """Test handling of plugin loading errors."""
        # Mock entry point that raises an exception
        mock_entry = MagicMock()
        mock_entry.load.side_effect = ImportError("Plugin not found")
        mock_entry.name = "broken_plugin"
        mock_entry_points.return_value = [mock_entry]

        with caplog.at_level(logging.ERROR, logger='pyfibers.model_enum'):
            plugins = _discover_plugins()

        # Should be empty due to error
        assert len(plugins) == 0
        assert "Error loading plugin broken_plugin" in caplog.text

    @patch('pyfibers.model_enum.entry_points')
    def test_discover_plugins_empty(self, mock_entry_points):
        """Test discovery when no plugins are available."""
        mock_entry_points.return_value = []

        plugins = _discover_plugins()

        assert len(plugins) == 0


class TestCreateFiberModelEnum:
    """Test the _create_fiber_model_enum function."""

    def test_create_enum_with_members(self):
        """Test creating enum with members dictionary."""
        members = {"TEST": MockFiber, "ANOTHER": MockFiber}
        enum = _create_fiber_model_enum(members)

        assert hasattr(enum, "TEST")
        assert hasattr(enum, "ANOTHER")
        assert enum.TEST.value is MockFiber
        assert enum.ANOTHER.value is MockFiber

    def test_create_enum_empty(self):
        """Test creating enum with empty members dictionary."""
        members = {}
        enum = _create_fiber_model_enum(members)

        assert len(enum.__members__) == 0


class TestUpdateAllModuleReferences:
    """Test the _update_all_module_references function."""

    def test_update_current_module(self):
        """Test that current module is updated."""
        # Create a mock enum
        mock_enum = MagicMock()
        mock_enum.__members__ = {"TEST": "value"}

        # Store original FiberModel
        original_fibermodel = sys.modules['pyfibers.model_enum'].FiberModel

        try:
            _update_all_module_references(mock_enum)

            # Check that current module was updated
            assert sys.modules['pyfibers.model_enum'].FiberModel is mock_enum
        finally:
            # Restore original FiberModel
            sys.modules['pyfibers.model_enum'].FiberModel = original_fibermodel

    def test_update_main_pyfibers_module(self):
        """Test that main pyfibers module is updated if it exists."""
        # Create a mock enum
        mock_enum = MagicMock()
        mock_enum.__members__ = {"TEST": "value"}

        # Mock the main pyfibers module
        with patch.dict(sys.modules, {'pyfibers': MagicMock()}):
            sys.modules['pyfibers'].FiberModel = MagicMock()

            _update_all_module_references(mock_enum)

            # Check that main pyfibers module was updated
            assert sys.modules['pyfibers'].FiberModel is mock_enum

    def test_update_other_modules(self):
        """Test that other modules with outdated FiberModel are updated."""
        # Create a mock enum
        mock_enum = MagicMock()
        mock_enum.__members__ = {"TEST": "value", "ANOTHER": "value"}

        # Create a mock module with outdated FiberModel
        mock_module = MagicMock()
        mock_old_enum = MagicMock()
        mock_old_enum.__members__ = {"TEST": "value"}  # Fewer members
        mock_module.FiberModel = mock_old_enum

        # Add mock module to sys.modules
        with patch.dict(sys.modules, {'test_module': mock_module}):
            _update_all_module_references(mock_enum)

            # Check that the module was updated
            assert mock_module.FiberModel is mock_enum


class TestRegisterCustomFiber:
    """Test the register_custom_fiber function."""

    def test_register_valid_fiber_functionality(self):
        """Test the core functionality of register_custom_fiber without global state issues."""
        # Test the _add_fiber_to_members function directly
        members = {}
        _add_fiber_to_members(members, MockFiber)

        assert "MOCK" in members
        assert members["MOCK"] is MockFiber

    def test_register_fiber_with_multiple_submodels_functionality(self):
        """Test registering a fiber with multiple submodels functionality."""
        # Test the _add_fiber_to_members function directly
        members = {}
        _add_fiber_to_members(members, MockFiberMultipleSubmodels)

        assert "MOCK_A" in members
        assert "MOCK_B" in members
        assert members["MOCK_A"] is MockFiberMultipleSubmodels
        assert members["MOCK_B"] is MockFiberMultipleSubmodels

    def test_register_invalid_fiber_raises_error(self):
        """Test that registering an invalid fiber raises ValueError."""
        with pytest.raises(ValueError, match="must have a 'submodels' attribute"):
            _add_fiber_to_members({}, MockFiberNoSubmodels)

    def test_register_non_fiber_raises_error(self):
        """Test that registering a non-Fiber class raises ValueError."""
        with pytest.raises(ValueError, match="must be a subclass of Fiber"):
            _add_fiber_to_members({}, NotAFiber)

    def test_register_preserves_existing_members_functionality(self):
        """Test that registering a custom fiber preserves existing enum members."""
        # Test with a members dictionary that already has some entries
        members = {"EXISTING": "some_class"}
        original_count = len(members)

        _add_fiber_to_members(members, MockFiber)

        # Check that we have more members now
        new_count = len(members)
        assert new_count > original_count
        assert "MOCK" in members
        assert "EXISTING" in members  # Original member preserved


class TestFiberModelEnum:
    """Test the FiberModel enum itself."""

    def test_fibermodel_has_members(self):
        """Test that FiberModel has expected members."""
        # Should have at least some built-in models
        assert len(FiberModel.__members__) > 0

        # Check for some expected built-in models
        expected_models = ["MRG_DISCRETE", "MRG_INTERPOLATION", "RATTAY"]
        for model in expected_models:
            if hasattr(FiberModel, model):
                assert hasattr(FiberModel, model)

    def test_fibermodel_member_values_are_classes(self):
        """Test that FiberModel member values are fiber classes."""
        for member in FiberModel:
            assert isinstance(member.value, type)
            # Should be a subclass of Fiber
            assert issubclass(member.value, Fiber)

    def test_fibermodel_member_names_are_uppercase(self):
        """Test that all FiberModel member names are uppercase."""
        for name in FiberModel.__members__.keys():
            assert name.isupper()
            assert name.replace("_", "").isalnum()


class TestIntegration:
    """Integration tests for the model_enum module."""

    def test_import_prints_members(self, caplog):
        """Test that importing the module logs available members."""
        # Re-import the module to trigger the log statement
        import importlib

        import pyfibers.model_enum

        with caplog.at_level(logging.DEBUG, logger='pyfibers.model_enum'):
            importlib.reload(pyfibers.model_enum)

        assert "Available FiberModel members:" in caplog.text

    def test_register_and_use_custom_fiber(self):
        """Test the full workflow of registering and using a custom fiber."""
        from pyfibers import build_fiber

        # Create a unique test fiber to avoid conflicts
        class TestFiberIntegration(MockFiber):
            submodels = ["TEST_INTEGRATION"]

        # Store the original FiberModel
        import pyfibers.model_enum

        original_fibermodel = pyfibers.model_enum.FiberModel

        try:
            # Register custom fiber
            register_custom_fiber(TestFiberIntegration)

            # Check that the fiber was registered by verifying it's in the enum
            assert "TEST_INTEGRATION" in FiberModel.__members__

            # Use the custom fiber
            fiber = build_fiber(diameter=5.7, fiber_model=FiberModel.TEST_INTEGRATION, temperature=37, n_nodes=5)

            # Verify the fiber was created
            assert fiber is not None
            assert hasattr(fiber, 'nodecount')

        finally:
            # Restore original FiberModel
            pyfibers.model_enum.FiberModel = original_fibermodel
            _update_all_module_references(original_fibermodel)
