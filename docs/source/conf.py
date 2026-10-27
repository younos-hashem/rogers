# -- Project information -----------------------------------------------------
project = 'rogers'
copyright = '2026, Younos Hashem'
author = 'Younos Hashem'
release = '1.0'

# -- General configuration ---------------------------------------------------
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.napoleon',
              'sphinx.ext.mathjax']

autosummary_generate = True # turn on extension
napoleon_google_docstring = True
napoleone_numpy_docstring = False
default_role = "code" # make `single backticks` appear as code
autodoc_typehints = 'description' # put type hints in desc, not signature
html_favicon = '_static/favicon.png'

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'
html_static_path = ['_static']
