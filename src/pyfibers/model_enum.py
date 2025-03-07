"""Dynamically create an enum of all fiber models, including built-in and plugin models.

This module imports fiber model classes from the local ``models`` package,
discovers additional plugin models via Python entry points, and aggregates
all submodels into a single ``FiberModel`` enum. This allows users to refer to
fiber models in a uniform way and makes it easier to extend the codebase with
new fiber models or external plugins.

Classes:
    FiberModel: A dynamically generated enum of fiber models.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import TYPE_CHECKING

from . import models

# Depending on whether we're type-checking, import the appropriate Enum class.
# aenum.NoAliasEnum allows dynamic creation of enum members without duplication.
# This means that a single fiber class can define multiple submodels (e.g., MRG) without conflicts.
if TYPE_CHECKING:
    from enum import Enum as NoAliasEnum
else:
    from aenum import NoAliasEnum


def discover_plugins() -> dict[str, type]:
    """Discover plugin classes using entry points under the 'pyfibers.fiber_plugins' group.

    This function looks for any package registering an entry point with the group
    'pyfibers.fiber_plugins'. Each plugin class must define a ``submodels`` attribute (list)
    that enumerates the specific fiber model(s) identifiers the plugin provides.

    :raises ValueError: If a discovered plugin class does not contain a ``submodels`` attribute.
    :return: A dictionary mapping each submodel name (converted to uppercase) to the plugin class.
    """
    plugins = {}
    # Iterate over all entry points in the specified group
    for entry_point in entry_points(group='pyfibers.fiber_plugins'):
        try:
            # Dynamically load the plugin class from the entry point
            plugin_class = entry_point.load()
        except Exception as e:  # noqa: PIE786, B902
            # If loading fails for any reason, print a warning and skip
            print(f"Error loading plugin {entry_point.name}: {e}")
            continue

        # Ensure the plugin class declares a submodels attribute
        if hasattr(plugin_class, 'submodels'):
            # Add each submodel to the dictionary, converting the name to uppercase
            for submodel in plugin_class.submodels:
                plugins[submodel.upper()] = plugin_class
        else:
            raise ValueError(f"Plugin {plugin_class} does not have a submodels attribute")

    return plugins


# Prepare a dictionary to accumulate all possible fiber model classes (enum members).
members: dict[str, type] = {}

# Populate 'members' with submodels defined by the built-in fiber classes from models.
# models.__all__ contains names of classes that define fiber models.
for fiber_class_name in models.__all__:
    fiber_class = getattr(models, fiber_class_name)
    # Each fiber class must define a 'submodels' attribute listing possible variants.
    if hasattr(fiber_class, 'submodels'):
        for submodel in fiber_class.submodels:
            members[submodel.upper()] = fiber_class
    else:
        raise ValueError(f"Class {fiber_class} does not have a submodels attribute")

# Discover plugin-based fiber classes and merge them into the same dictionary.
plugin_members = discover_plugins()
members.update(plugin_members)

# Finally, create an enum named 'FiberModel' with all discovered submodels.
# The NoAliasEnum is used to prevent redefinition conflicts if multiple submodels are defined in one class.
FiberModel = NoAliasEnum('FiberModel', members)  # type: ignore
