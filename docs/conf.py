import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "saddlepoint-expansions"
author = "Simon A. Broda and Raymond Kan"
copyright = "2026, Simon A. Broda and Raymond Kan"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "myst_parser",
]
myst_enable_extensions = ["dollarmath", "amsmath"]
autodoc_member_order = "bysource"
autodoc_typehints = "none"
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
html_theme = "furo"
html_title = "saddlepoint-expansions"
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build"]
