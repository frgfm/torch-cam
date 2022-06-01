# Copyright (C) 2020-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

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
from datetime import datetime

import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath("../.."))
import torchcam

# -- Project information -----------------------------------------------------

master_doc = "index"
project = "torchcam"
copyright = f"2020-{datetime.now().year}, François-Guillaume Fernandez"
author = "François-Guillaume Fernandez"

# The full version, including alpha/beta/rc tags
version = torchcam.__version__
release = torchcam.__version__ + "-git"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinxemoji.sphinxemoji",  # cf. https://sphinxemojicodes.readthedocs.io/en/stable/
    "sphinx_copybutton",
    "recommonmark",
    "sphinx_markdown_tables",
]

napoleon_use_ivar = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["notebooks/*.rst"]


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "friendly"
pygments_dark_style = "monokai"
highlight_language = "python3"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-sidebar-background": "#082747",
        "color-sidebar-background-border": "#082747",
        "color-sidebar-caption-text": "white",
        "color-sidebar-link-text--top-level": "white",
        "color-sidebar-link-text": "white",
        "sidebar-caption-font-size": "normal",
        "color-sidebar-item-background--hover": " #5dade2",
    },
    "dark_css_variables": {
        "color-sidebar-background": "#1a1c1e",
        "color-sidebar-background-border": "#1a1c1e",
        "color-sidebar-caption-text": "white",
        "color-sidebar-link-text--top-level": "white",
    },
}

# html_logo = "_static/images/logo.png"
# html_favicon = "_static/images/favicon.ico"
html_title = "TorchCAM documentation"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Add googleanalytics id
# ref: https://github.com/orenhecht/googleanalytics/blob/master/sphinxcontrib/googleanalytics.py
def add_ga_javascript(app, pagename, templatename, context, doctree):

    metatags = context.get("metatags", "")
    metatags += """
    <!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id={0}"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());
  gtag('config', '{0}');
</script>
    """.format(
        app.config.googleanalytics_id
    )
    context["metatags"] = metatags


def setup(app):
    app.add_config_value("googleanalytics_id", "UA-148140560-4", "html")
    app.add_css_file("css/custom_theme.css")
    app.add_js_file("js/custom.js")
    app.connect("html-page-context", add_ga_javascript)
