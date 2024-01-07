# Copyright (C) 2020-2024, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

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
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, Path().cwd().parent.parent)
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

html_title = "TorchCAM"
# html_logo = "_static/images/logo.png"
# html_favicon = "_static/images/favicon.ico"
language = "en"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/frgfm/torch-cam",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
    "source_repository": "https://github.com/frgfm/torch-cam/",
    "source_branch": "main",
    "source_directory": "docs/source/",
}


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
    """.format(app.config.googleanalytics_id)
    context["metatags"] = metatags


def setup(app):
    app.add_config_value("googleanalytics_id", "UA-148140560-4", "html")
    app.add_css_file("css/custom_theme.css")
    app.add_js_file("js/custom.js")
    app.connect("html-page-context", add_ga_javascript)
