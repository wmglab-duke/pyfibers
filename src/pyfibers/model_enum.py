"""Creates enum with all fiber models."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enum import Enum as NoAliasEnum
else:
    from aenum import NoAliasEnum

from . import models

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points


# Discover plugins using entry points
def discover_plugins() -> dict[str, type]:
    """Discover plugins using entry points for pyfibers.fiber_plugins.

    :raises ValueError: if plugin does not have a submodels attribute
    :return: dictionary of model names and plugin classes
    """
    plugins = {}
    for entry_point in entry_points(group='pyfibers.fiber_plugins'):
        try:
            plugin_class = entry_point.load()
        except Exception as e:  # noqa: PIE786, B902
            print(f"Error loading plugin {entry_point.name}: {e}")
            continue
        if hasattr(plugin_class, 'submodels'):
            for submodel in plugin_class.submodels:
                plugins[submodel.upper()] = plugin_class
        else:
            raise ValueError(f"Plugin {plugin_class} does not have a submodels attribute")
    return plugins


# Create a dictionary to hold the enum members
members: dict[str, type] = {}

# Populate the dictionary with submodels from the fiber classes
for fiber_class_name in models.__all__:
    fiber_class = getattr(models, fiber_class_name)
    if hasattr(fiber_class, 'submodels'):
        for submodel in fiber_class.submodels:
            members[submodel.upper()] = fiber_class
    else:
        raise ValueError(f"Class {fiber_class} does not have a submodels attribute")

# Discover and add plugins to the members dictionary
plugin_members = discover_plugins()
members.update(plugin_members)

# Create the FiberModel enum dynamically
FiberModel = NoAliasEnum('FiberModel', members)  # type: ignore
