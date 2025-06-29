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

os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../src/pyfibers'))
sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------

project = 'pyfibers'
copyright_info = '2023, Duke University'

# The full version, including alpha/beta/rc tags
release = "0.4.0"

html_short_title = "PyFibers Documentation"
# html_logo = './static/logo.png' noqa: E800
# pygments_style = "stata-dark" noqa: E800
# html_theme_options = {
#     "source_url": 'https://github.com/pyfibers_url'
# }

# -- General configuration ---------------------------------------------------
nitpicky = True  # raise warnings for missing references #noqa: E800
nitpick_ignore = [
    (
        'py:class',
        'h.Vector',
    ),  # Because neuron's objects.inv does not build properly, and intersphinx cannot find these classes
    ('py:class', 'h.Section'),
    ('py:class', 'neuron.h.Section'),
    ('py:class', 'h.NetStim'),
    ('py:class', 'h.NetCon'),
    ('py:class', 'h.IClamp'),
    ('py:class', 'h.trainIClamp'),  # added from a pyfibers mod file, so no object.inv
    ('py:class', 'neuron.h.Vector'),
    ('py:class', 'nd_line'),  # no sphinx docs for this package
    ('py:class', 'FiberModel'),  # Type references broken due to type hinting block in fiber.py
    ('py:data', 'collections.abc.Callable'),  # Broken by sphinx_autodoc_typehints, post issue on GH
]
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'myst_nb',
    'sphinx_copybutton',
    'sphinx.ext.autosummary',
    'enum_tools.autoenum',
    'sphinxcontrib.bibtex',
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "piccolo_theme",
]

autodoc_mock_imports = ['neuron', 'scipy', 'numpy.typing']
# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "neuron": ("https://nrn.readthedocs.io/en/latest", None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    # "multiprocess": ("https://multiprocess.readthedocs.io/en/latest/", None),
    # "nd_line": ("https://github.com/thedannymarsh/nd_line", None),
    # "neuron" : ("https://neuronsimulator.github.io/nrn/", None),
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []
# uncomment the line below to exclude all tutorials from the documentation build
# exclude_patterns = ['tutorials/**.ipynb']  # noqa: E800

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = 'piccolo_theme'
html_static_path = ['_static']
html_css_files = ["custom.css"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".


html_show_copyright = True

html_show_sphinx = False


# -- Options for extensions -------------------------------------------------

myst_heading_anchors = 4

# default_dark_mode = False
# Add the extension

# Make sure the target is unique
autosectionlabel_prefix_document = True

# Comment out line below to not generate autosummary files
autosummary_generate = True

bibtex_bibfiles = ['refs.bib']

bibtex_reference_style = 'author_year'

# mystnb
nb_execution_raise_on_error = True
nb_execution_show_tb = True
nb_output_stderr = 'remove-warn'
nb_execution_timeout = 600
nb_execution_mode = "cache"
nb_merge_streams = True
suppress_warnings = ["mystnb.stderr"]

# latex
latex_engine = 'xelatex'
