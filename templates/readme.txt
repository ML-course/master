Copy files to nbconvert template directory:
- HTML: .../anaconda3/lib/python3.6/site-packages/nbconvert/templates/html
- Latex: .../anaconda3/lib/python3.6/site-packages/nbconvert/templates/latex

Create Slides:
jupyter nbconvert --to slides --template jads.tpl filename.ipynb --SlidesExporter.reveal_theme=serif --post serve

Export PDF: add '?print-pdf' to url and remove '#'

Create Latex PDF:
jupyter nbconvert --to pdf --template jads.tplx filename.ipynb


JupyterLab setup:

* Jupyter Widgets

    jupyter labextension install @jupyter-widgets/jupyterlab-manager

