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

import logging
import sys
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any

from . import models

# Set up module-level logger
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from enum import Enum
    from typing import TypeAlias

    FiberModel: TypeAlias = Enum
else:
    from aenum import NoAliasEnum


def _add_fiber_to_members(members: dict[str, type], fiber_class: type) -> None:
    """Add a fiber class and its submodels to a members dictionary.

    :param members: Dictionary to add the fiber class to.
    :param fiber_class: The fiber class to add.
    :raises ValueError: If the fiber class does not have a 'submodels' attribute or is not a subclass of Fiber.
    """
    # Validate the fiber class
    # Import here to avoid circular imports
    from .fiber import Fiber

    if not hasattr(fiber_class, 'submodels'):
        raise ValueError(f"Fiber class {fiber_class} must have a 'submodels' attribute")

    if not issubclass(fiber_class, Fiber):
        raise ValueError(f"Fiber class {fiber_class} must be a subclass of Fiber")

    for submodel in fiber_class.submodels:  # type: ignore[attr-defined]
        submodel_upper = submodel.upper()
        if submodel_upper in members:
            logger.warning("Overwriting existing fiber model '%s' with %s", submodel_upper, fiber_class)
        members[submodel_upper] = fiber_class


def _discover_plugins() -> dict[str, type]:
    """Discover plugin classes using entry points under the 'pyfibers.fiber_plugins' group.

    :return: A dictionary mapping each submodel name (converted to uppercase) to the plugin class.
    :raises ValueError: If a discovered plugin class does not contain a submodels attribute.
    """
    plugins: dict[str, type] = {}
    for entry_point in entry_points(group='pyfibers.fiber_plugins'):
        try:
            plugin_class = entry_point.load()
            if hasattr(plugin_class, 'submodels'):
                _add_fiber_to_members(plugins, plugin_class)
            else:
                raise ValueError(f"Plugin {plugin_class} does not have a submodels attribute")
        except Exception as e:  # noqa: PIE786, B902
            logger.error("Error loading plugin %s: %s", entry_point.name, e)
    return plugins


def _update_all_module_references(new_enum: Any) -> None:  # noqa: ANN401
    """Update all modules that have imported FiberModel with the new enum.

    This function is called by register_custom_fiber whenever a new enum is created.
    Since a new enum is always created when registering, we update all module references
    to ensure consistency across the codebase.

    :param new_enum: The new FiberModel enum to propagate to all modules.
    """
    current_module = sys.modules[__name__]
    current_module.FiberModel = new_enum  # type: ignore[attr-defined]
    if "pyfibers" in sys.modules and hasattr(sys.modules["pyfibers"], "FiberModel"):
        sys.modules["pyfibers"].FiberModel = new_enum  # type: ignore[attr-defined]
    for module in sys.modules.values():
        if (
            hasattr(module, "FiberModel")
            and module.FiberModel is not new_enum
            and hasattr(module.FiberModel, "__members__")
            and hasattr(new_enum, "__members__")
        ):
            # Always update if enum is different (register_custom_fiber always creates a new enum)
            module.FiberModel = new_enum  # type: ignore[attr-defined]


def _create_fiber_model_enum(members_dict: dict[str, type]) -> Any:  # noqa: ANN401
    """Create the FiberModel enum with the given members dictionary.

    :param members_dict: Dictionary mapping submodel names to fiber classes.
    :return: The created FiberModel enum.
    """
    return NoAliasEnum("FiberModel", members_dict)  # type: ignore[name-defined]


def register_custom_fiber(fiber_class: type) -> None:
    """Register a custom fiber model class with the FiberModel enum at runtime.

    This function allows users to dynamically add custom fiber models to the
    FiberModel enum without needing to modify the main package or create a plugin.

    :param fiber_class: The fiber model class to register. Must have a 'submodels' attribute.
    """
    global FiberModel

    # Reconstruct members dictionary from current enum
    current_members: dict[str, type] = {name: member.value for name, member in FiberModel.__members__.items()}

    # Add the custom fiber class
    _add_fiber_to_members(current_members, fiber_class)

    # Recreate enum and update all references
    FiberModel = _create_fiber_model_enum(current_members)
    _update_all_module_references(FiberModel)

    logger.info(
        "Successfully registered custom fiber model: %s with submodels: %s",
        fiber_class.__name__,
        fiber_class.submodels,  # type: ignore[attr-defined]
    )


# Build the initial members dictionary
members: dict[str, type] = {}
for fiber_class_name in models.__all__:
    fiber_class = getattr(models, fiber_class_name)
    _add_fiber_to_members(members, fiber_class)
members.update(_discover_plugins())

# Create the FiberModel enum with all discovered submodels
FiberModel: Any = _create_fiber_model_enum(members)  # type: ignore[no-redef,misc]

logger.debug("Available FiberModel members: %s", list(FiberModel.__members__.keys()))
