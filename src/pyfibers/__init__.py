# isort: skip_file
"""Initializer for core PyFibers code."""
import logging
import os
import sys
from typing import TextIO

from contextlib import suppress

import neuron
import importlib.metadata

# assign version number from pyproject.toml
__version__ = importlib.metadata.version(__package__ or __name__)

# Configure package-level logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def enable_logging(level: int = logging.INFO, format_string: str = None, stream: TextIO = None) -> None:
    """Enable logging output for pyfibers.

    :param level: Logging level. Defaults to logging.INFO.
        Common values: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR
    :param format_string: Custom format string for log messages.
        Defaults to a standard format showing level, logger name, and message.
    :param stream: Output stream for logging. Defaults to stderr (standard for logging).
        Use sys.stdout for notebooks or when you want logging mixed with regular output.
        If None and PYFIBERS_DOCS_BUILD=1, automatically uses stdout for documentation builds.

    .. code-block:: python

        import pyfibers

        pyfibers.enable_logging()  # Enable INFO level logging (stderr in normal use, stdout in docs)
        pyfibers.enable_logging(stream=sys.stdout)  # Force stdout
        pyfibers.enable_logging(level=pyfibers.logging.DEBUG)  # Enable DEBUG level
        pyfibers.enable_logging(level=pyfibers.logging.WARNING)  # Only show warnings and errors
    """
    if format_string is None:
        format_string = '%(levelname)s:%(name)s:%(message)s'

    # Remove any existing StreamHandlers to avoid duplicates
    existing_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
    for handler in existing_handlers:
        logger.removeHandler(handler)

    # Set stream to stdout for docs builds
    if stream is None and os.getenv("PYFIBERS_DOCS_BUILD") == "1":
        stream = sys.stdout

    # Create a new console handler
    console_handler = logging.StreamHandler(stream)
    console_handler.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(format_string)
    console_handler.setFormatter(formatter)

    # Add the new handler
    logger.addHandler(console_handler)
    logger.setLevel(level)


# load all NEURON files
MOD_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MOD')
with suppress(RuntimeError):
    neuron.load_mechanisms(MOD_dir)

# load all python files
from .fiber import Fiber, build_fiber, build_fiber_3d  # noqa: E402
from .model_enum import FiberModel, register_custom_fiber  # noqa: E402
from .stimulation import (  # noqa: E402
    ScaledStim,
    IntraStim,
    Stimulation,
    BoundsSearchMode,
    TerminationMode,
    ThresholdCondition,
    BisectionMean,
)

__all__ = [
    'build_fiber',
    'build_fiber_3d',
    'Fiber',
    'ScaledStim',
    'IntraStim',
    'Stimulation',
    'FiberModel',
    'register_custom_fiber',
    'BoundsSearchMode',
    'TerminationMode',
    'ThresholdCondition',
    'BisectionMean',
    'logger',
    'enable_logging',
]
