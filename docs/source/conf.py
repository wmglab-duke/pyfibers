"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options.
For a full list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Path setup --------------------------------------------------------------

from __future__ import annotations

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from unittest import mock

os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../src/pyfibers'))


# -- Project information -----------------------------------------------------

project = 'pyfibers'
copyright_info = '2023, Duke University'

# The full version, including alpha/beta/rc tags
release = 'alpha'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'nbsphinx',
    'sphinx_copybutton',
    'sphinx_rtd_dark_mode',
    'sphinx.ext.autosummary',
    'enum_tools.autoenum',
]

MOCK_MODULES = ['numpy', 'pandas']

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

autodoc_mock_imports = ['neuron', 'scipy']

# Add any paths that contain templates here, relative to this directory.


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []
# uncomment the line below to exclude all tutorials from the documentation build
# exclude_patterns = ['tutorials/**.ipynb']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".


html_show_copyright = True

html_show_sphinx = False


# -- Options for extensions -------------------------------------------------

myst_heading_anchors = 4

default_dark_mode = False

# Comment out line below to not generate autosummary files
autosummary_generate = True
