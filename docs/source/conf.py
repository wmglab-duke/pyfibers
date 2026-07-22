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

# Set environment variable to indicate documentation build
# This makes PyFibers log to stdout instead of stderr for notebook output
os.environ["PYFIBERS_DOCS_BUILD"] = "1"

sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../src/pyfibers'))
sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------

project = 'pyfibers'
copyright_info = '2023, Duke University'

# The full version, including alpha/beta/rc tags
release = "0.8.5"

html_short_title = "PyFibers documentation"
# html_logo = './static/logo.png' noqa: E800
# pygments_style = "stata-dark" noqa: E800
# html_theme_options = {
#     "source_url": 'https://github.com/pyfibers_url'
# }

# -- General configuration ---------------------------------------------------
nitpicky = True  # raise warnings for missing references #noqa: E800
nitpick_ignore = [
    # Because neuron's objects.inv does not build properly, and intersphinx cannot find these classe
    (
        'py:class',
        'h.Vector',
    ),
    ('py:class', 'h.APCount'),
    ('py:class', 'neuron.h.APCount'),
    ('py:class', 'h.Section'),
    ('py:class', 'neuron.h.Section'),
    ('py:class', 'h.NetStim'),
    ('py:class', 'h.NetCon'),
    ('py:class', 'h.IClamp'),
    ('py:class', 'h.trainIClamp'),  # added from a pyfibers mod file, so no object.inv
    ('py:class', 'neuron.h.Vector'),
    ('py:class', 'nd_line'),  # no sphinx docs for this package
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
    "neuron": ("https://www.neuronsimulator.org/en/latest", None),
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
html_favicon = '_static/favicon.ico'

# Piccolo theme: link each page to the matching path on GitHub ("View source" / repo icon).
# See https://piccolo-theme.readthedocs.io/en/latest/configuration.html
html_theme_options = {
    "source_url": "https://github.com/wmglab-duke/pyfibers",
    "source_icon": "github",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".


html_show_copyright = True

html_show_sphinx = False


# -- Options for extensions -------------------------------------------------

myst_heading_anchors = 4
myst_enable_extensions = ["colon_fence"]

# default_dark_mode = False
# Add the extension

# Make sure the target is unique
autosectionlabel_prefix_document = True

# Comment out line below to not generate autosummary files
autosummary_generate = True

# Do not evaluate default values of type hints
autodoc_preserve_defaults = True

bibtex_bibfiles = ['refs.bib']

bibtex_reference_style = 'author_year'

# mystnb
nb_render_markdown_format = "myst"
nb_execution_raise_on_error = True
nb_execution_show_tb = True
nb_output_stderr = 'remove-warn'
nb_execution_timeout = 600
nb_execution_mode = "cache"
nb_merge_streams = True
suppress_warnings = ["mystnb.stderr"]

# linkcheck
# Run with: sphinx-build -b linkcheck docs/source docs/build
linkcheck_timeout = 15
linkcheck_retries = 2
linkcheck_workers = 5
linkcheck_anchors = True

# Any redirect (request URL vs final response URL) counts as OK; linkcheck still
# validates the final response (status, body, anchors as configured).
linkcheck_allowed_redirects = {r".*": r".*"}

# Unreachable to bots, anti-bot interstitials, or DOIs that consistently 403 in CI.
linkcheck_ignore = [
    r'^https://validate\.perfdrive\.com/.*',
    r'^https://doi\.org/10\.1088/1741-2552/aa6a5f$',
    r'^https://doi\.org/10\.1162/neco\.1997\.9\.6\.1179$',
    # Journal of Neurophysiology / APS: resolver and journal site often 403 for automated checks.
    r'^https://doi\.org/10\.1152/jn\.',
    r'^https://journals\.physiology\.org/doi/',
    # Self-references to hosted docs can 404/redirect during CI preview builds.
    r'^https://wmglab-duke\.github\.io/pyfibers/?',
    r'^https://wmglab-duke\.github\.io/pyfibers/.*',
    # AIP publishing
    r"https://doi\.org/10\.1063/.*",
]

# latex
latex_engine = 'xelatex'
