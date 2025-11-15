# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "rheojax"
copyright = "2025, Wei Chen"
author = "Wei Chen"
release = "0.2.2"
version = "0.2.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.ifconfig",
    "sphinx.ext.autosummary",  # Auto-generate API docs
]

# Optional extensions (graceful fallback if not installed)

# Add copy button to code blocks
try:
    import sphinx_copybutton  # noqa: F401

    extensions.append("sphinx_copybutton")
except ImportError:
    pass

# Add markdown support via myst-parser
try:
    import myst_parser  # noqa: F401

    extensions.append("myst_parser")
except ImportError:
    pass

# Add tabs for multi-language examples
try:
    import sphinx_tabs  # noqa: F401

    extensions.append("sphinx_tabs.tabs")
except ImportError:
    pass

# Add type hints to autodoc
try:
    import sphinx_autodoc_typehints  # noqa: F401

    extensions.append("sphinx_autodoc_typehints")
except ImportError:
    pass

templates_path = ["../_templates"]
exclude_patterns = []

# Suppress warnings for auto-generated files
suppress_warnings = [
    "autodoc.duplicate_object",  # Suppress duplicate object warnings from autosummary
    "autosummary",  # Suppress autosummary warnings
    "toc.not_readable",  # Suppress warnings about auto-generated files not in toctree
    "ref.doc",  # Suppress missing document warnings
]

# MyST parser settings (if available)
try:
    import myst_parser  # noqa: F401

    myst_enable_extensions = [
        "colon_fence",  # ::: style fences
        "deflist",  # Definition lists
        "dollarmath",  # $math$ and $$math$$
        "fieldlist",  # Field lists
        "html_admonition",  # HTML-style admonitions
        "html_image",  # HTML images
        "linkify",  # Auto-detect URLs
        "replacements",  # Text replacements
        "smartquotes",  # Smart quotes
        "substitution",  # Substitutions
        "tasklist",  # Task lists
    ]
    myst_heading_anchors = 3  # Auto-generate heading anchors
except ImportError:
    pass

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
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "numpyro": ("https://num.pyro.ai/en/stable/", None),
    "arviz": ("https://python.arviz.org/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_static_path = ['../_static']  # Commented out - directory does not exist
html_logo = None
html_favicon = None

# Theme options
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_css_files = []

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": "",
    "figure_align": "htbp",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto, manual, or own class]).
latex_documents = [
    ("index", "rheojax.tex", "rheo Documentation", "Wei Chen", "manual"),
]

# -- Options for manual page output ------------------------------------------
man_pages = [("index", "rheojax", "rheo Documentation", ["Wei Chen"], 1)]

# -- Options for Texinfo output ----------------------------------------------
texinfo_documents = [
    (
        "index",
        "rheojax",
        "rheo Documentation",
        "Wei Chen",
        "rheojax",
        "JAX-powered unified rheology package",
        "Miscellaneous",
    ),
]

# -- Extension configuration -------------------------------------------------
todo_include_todos = True

# Autosummary settings
autosummary_generate = False  # Disabled: using :members: instead
autosummary_imported_members = False

# Copybutton settings (hide prompts)
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_remove_prompts = True

# Autodoc typehints settings
autodoc_typehints = "description"  # Show type hints in description
autodoc_typehints_description_target = "documented"  # Only documented parameters
autodoc_type_aliases = {
    "ArrayLike": "numpy.typing.ArrayLike",
    "NDArray": "numpy.typing.NDArray",
}
