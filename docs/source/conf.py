# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../ailever/'))
sys.path.insert(0, os.path.abspath('../../ailever/machine/'))
sys.path.insert(0, os.path.abspath('../../ailever/machine/RL/'))
sys.path.insert(0, os.path.abspath('../../ailever/forecast/'))
sys.path.insert(0, os.path.abspath('../../ailever/forecast/STOCK/'))
sys.path.insert(0, os.path.abspath('../../ailever/captioning/'))
sys.path.insert(0, os.path.abspath('../../ailever/language/'))
sys.path.insert(0, os.path.abspath('../../ailever/detection/'))
sys.path.insert(0, os.path.abspath('../../ailever/analysis/'))
sys.path.insert(0, os.path.abspath('../../ailever/utils/'))



# -- Project information -----------------------------------------------------

project = 'ailever'
copyright = '2020, ailever'
author = 'ailever'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "recommonmark",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

master_doc = 'index'

