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
release = "0.6.0"
version = "0.6.0"

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
exclude_patterns = [
    "_templates/*",      # Template files not meant for toctree
    "_includes/*",       # Include files meant for .. include::
    "_guides/*",         # Style guides not meant for toctree
]

# Suppress warnings for auto-generated files
suppress_warnings = [
    "autodoc.duplicate_object",  # Suppress duplicate object warnings from autosummary
    "autosummary",  # Suppress autosummary warnings
    "toc.not_readable",  # Suppress warnings about auto-generated files not in toctree
    "toc.not_included",  # Suppress warnings for template/include files not in toctree
    "ref.doc",  # Suppress missing document warnings
    "ref.footnote",  # Suppress warnings for bibliography-style references (not inline-cited)
    "ref.citation",  # Suppress warnings for bibliography-style citations (not inline-cited)
    "docutils",  # Suppress minor docutils warnings from autodoc-processed docstrings
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

html_theme = "furo"
html_static_path = ["_static"]
html_logo = None
html_favicon = None

# Furo theme options for better readability
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
    "light_css_variables": {
        "font-stack": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif",
        "font-stack--monospace": "'JetBrains Mono', 'Fira Code', 'Consolas', monospace",
        "color-brand-primary": "#2962FF",
        "color-brand-content": "#2962FF",
        "color-admonition-title--tip": "#00897B",
        "color-admonition-title-background--tip": "#00897B18",
        "color-admonition-title--note": "#1565C0",
        "color-admonition-title-background--note": "#1565C018",
    },
    "dark_css_variables": {
        "color-brand-primary": "#82B1FF",
        "color-brand-content": "#82B1FF",
    },
}

# Custom CSS for improved table readability and visual hierarchy
html_css_files = ["custom.css"]

# -- Options for LaTeX output ------------------------------------------------
latex_engine = "xelatex"
latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "cmappkg": "",  # cmap is unnecessary for xelatex (causes pdftex detection warning)
    "preamble": "\n".join([
        r"\usepackage{mathtools}",
        # Suppress Underfull/Overfull hbox notices from long API identifiers and
        # auto-generated code blocks (these are typographical, not content errors).
        r"\hbadness=10000",
        r"\hfuzz=\maxdimen",
        r"\vbadness=10000",
        r"\vfuzz=\maxdimen",
        r"\fvset{hfuzz=\maxdimen}",  # Override fancyvrb's hfuzz=2pt in code blocks
        # Redefine \sloppy to not reset \hfuzz (LaTeX's \sloppy sets hfuzz=0.5pt,
        # which \@parboxrestore calls inside every minipage/parbox/table cell)
        r"\DeclareRobustCommand\sloppy{\tolerance 9999\emergencystretch 3em\hfuzz\maxdimen\vfuzz\hfuzz}",
        r"\AtBeginDocument{\hfuzz=\maxdimen\vfuzz=\maxdimen\hbadness=10000\vbadness=10000}",
        # Map Unicode sub/superscripts to LaTeX equivalents (from Python docstrings)
        r"\usepackage{newunicodechar}",
        r"\newunicodechar{ᵖ}{\textsuperscript{p}}",
        r"\newunicodechar{ᵗ}{\textsuperscript{t}}",
        r"\newunicodechar{ˡ}{\textsuperscript{l}}",
        r"\newunicodechar{ᵣ}{\textsubscript{r}}",
        r"\newunicodechar{ₛ}{\textsubscript{s}}",
        r"\newunicodechar{ₘ}{\textsubscript{m}}",
        r"\newunicodechar{ₐ}{\textsubscript{a}}",
        r"\newunicodechar{ₑ}{\textsubscript{e}}",
        r"\newunicodechar{ₓ}{\textsubscript{x}}",
        r"\newunicodechar{ₙ}{\textsubscript{n}}",
        r"\newunicodechar{ₖ}{\textsubscript{k}}",
        r"\newunicodechar{ᵢ}{\textsubscript{i}}",
        r"\newunicodechar{₀}{\textsubscript{0}}",
        r"\newunicodechar{₁}{\textsubscript{1}}",
        r"\newunicodechar{₂}{\textsubscript{2}}",
        # Map Greek letters that appear in text mode to proper LaTeX commands
        r"\newunicodechar{ω}{\ensuremath{\omega}}",
        r"\newunicodechar{α}{\ensuremath{\alpha}}",
        r"\newunicodechar{β}{\ensuremath{\beta}}",
        r"\newunicodechar{γ}{\ensuremath{\gamma}}",
        r"\newunicodechar{δ}{\ensuremath{\delta}}",
        r"\newunicodechar{ε}{\ensuremath{\varepsilon}}",
        r"\newunicodechar{η}{\ensuremath{\eta}}",
        r"\newunicodechar{θ}{\ensuremath{\theta}}",
        r"\newunicodechar{λ}{\ensuremath{\lambda}}",
        r"\newunicodechar{μ}{\ensuremath{\mu}}",
        r"\newunicodechar{ν}{\ensuremath{\nu}}",
        r"\newunicodechar{σ}{\ensuremath{\sigma}}",
        r"\newunicodechar{τ}{\ensuremath{\tau}}",
        r"\newunicodechar{φ}{\ensuremath{\varphi}}",
        r"\newunicodechar{Γ}{\ensuremath{\Gamma}}",
        r"\newunicodechar{Σ}{\ensuremath{\Sigma}}",
        # Font shape substitution: small-caps italic → italic fallback
        r"\makeatletter",
        r"\DeclareFontShape{TU}{FreeSerif(0)}{m}{scit}{<->ssub*FreeSerif(0)/m/it}{}",
        r"\makeatother",
        # Handle math symbols in PDF bookmarks
        r"\hypersetup{pdfencoding=auto, psdextra}",
        r"\usepackage{bookmark}",
        r"\pdfstringdefDisableCommands{%",
        r"  \def\ensuremath#1{#1}%",
        r"  \def\varepsilon{epsilon}%",
        r"  \def\dot#1{#1}%",
        r"  \let\phi\relax%",
        r"}",
        # Suppress hyperref warnings for math tokens in section title bookmarks
        # (section titles with :math: generate $, _, ^ which are invalid in PDF strings;
        # hyperref degrades gracefully by removing them — the warnings are informational)
        r"\makeatletter",
        r"\renewcommand*\HyPsd@Warning[1]{}",
        r"\makeatother",
    ]),
    "figure_align": "htbp",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto, manual, or own class]).
latex_documents = [
    ("index", "rheojax.tex", "RheoJAX Documentation", "Wei Chen", "manual"),
]

# -- Options for manual page output ------------------------------------------
man_pages = [("index", "rheojax", "RheoJAX Documentation", ["Wei Chen"], 1)]

# -- Options for Texinfo output ----------------------------------------------
texinfo_documents = [
    (
        "index",
        "rheojax",
        "RheoJAX Documentation",
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
