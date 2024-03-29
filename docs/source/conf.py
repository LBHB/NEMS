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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'NEMS'
copyright = '2023, Jacob Pennington'
author = 'Jacob Pennington'

# The full version, including alpha/beta/rc tags
release = '0.0.3'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'nbsphinx',
    'sphinx_copybutton',
]

autodoc_default_options = {
    'member-order': 'bysource',
    'special-members': True,
    'private-members': True,
    'undoc-members': True,
    'exclude-members': '__weakref__, __dict__, __module__',
}
autoclass_content = "both"
autosummary_generate: True

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
html_theme = 'sphinx_book_theme'

html_theme_options = {
  "show_toc_level": 2,
  "repository_url": "https://github.com/LBHB/NEMS",
  "path_to_docs": "{./docs}",
  "use_source_button": True,
  "use_repository_button": True,
  "use_issues_button": True,
  "use_fullscreen_button": False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = ['css/theme_override.css']