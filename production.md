# Production
These are instructions on how to generate the slides and book. As a student, you don't need this. You can simply use the pre-compiled materials.

## Python environment
Create a virtual environment and install the necessary requirements. Tested for Python 3.10 (newer versions up to 3.12 are probably ok).

```
conda create -n mlcourse python=3.10
conda activate mlcourse
pip install -U -r requirements.txt
```

## Generating slides
### Interactive slides
To generate interactive slides (as shown in the videos and lectures), you'll need to install the notebook extensions `rise` and `hide_input_all`.

```
conda install -c conda-forge rise
conda install -c conda-forge jupyter_contrib_nbextensions
jupyter nbextension enable hide_input_all
```

You'll need to launch `jupyter notebook` since rise is not yet fully supported in jupyterlab. You'll see two new icons when you open a notebook. To run the slides interactively:

- Set `interactive = True` in the first cell (if it isn't already)
- Run all cells ('Run' > 'Run all cells') so that all interaction widgets are loaded
- Hide the input code by clicking the 'eye' icon (toggles `hide_all_code`)
- Start the slideshow by clicking the 'chart' icon (toggles `rise`)

### Static slides
You can generate slides from the notebooks using `nbconvert`. First, set `interactive = False` in the first cell of the notebook and rerun the notebooks to generate static versions of the interactive visualizations.

```
jupyter nbconvert --to slides --template reveal --SlidesExporter.reveal_theme=simple --no-input --post serve <NotebookName>
```

To generate PDF handouts, remove `#/` from the url and add `?print-pdf`, then print as PDF. 

Sidenote: Some PDF readers (e.g. Preview) sometimes show grey boxes around code examples, others (e.g. Acrobat Reader, Chrome) do not. Must be an artifact of some PDF readers.

## Generating the online book
To generate the online books, you'll need `jupyter-book`.

```
pip install jupyter-book
```

The configuration file is `_config.yml` and the table of contents is in `_toc.yml`. The cover page is defined in `intro.md`.

To create the book itself, run this from the parent directory of the `master` repo:

```
jupyter-book build master
```

To push the rendered book to GitHub, run:

```
pip install ghp-import
ghp-import -n -p -f _build/html
```

#### Customization
I used a few tweaks to the slide theme by adding these to the css in `custom_reveal.css` in the `nbconvert` reveal template. Copy the styles from slides_html/custom.css to this file.

There also seems to be a bug in reveal that outputs text for hidden interactive elements. I added the following to `index.html.j2` of the same template:

```
<script>
var els = document.getElementsByTagName("pre");
for (var i = 0; i < els.length; ++i) {
    var el = els[i];
    if (el.textContent.indexOf("interactive") > -1) {
      el.style.display = "none";
    }
}
</script>
```
