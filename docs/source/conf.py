# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'rheo'
copyright = '2025, Rheo Development Team'
author = 'Rheo Development Team'
release = '1.0.0'
version = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.ifconfig',
]

templates_path = ['../_templates']
exclude_patterns = []

# Napoleon settings for Google and NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['../_static']
html_logo = None
html_favicon = None

# Theme options
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_css_files = []

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': '',
    'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto, manual, or own class]).
latex_documents = [
    ('index', 'rheo.tex', 'rheo Documentation',
     'Rheo Development Team', 'manual'),
]

# -- Options for manual page output ------------------------------------------
man_pages = [
    ('index', 'rheo', 'rheo Documentation',
     ['Rheo Development Team'], 1)
]

# -- Options for Texinfo output ----------------------------------------------
texinfo_documents = [
    ('index', 'rheo', 'rheo Documentation',
     'Rheo Development Team', 'rheo', 'JAX-powered unified rheology package',
     'Miscellaneous'),
]

# -- Extension configuration -------------------------------------------------
todo_include_todos = True
