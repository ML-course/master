# Production
These are instructions on how to generate the slides and book. As a student, you don't need this. You can simply use the pre-compiled materials.

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


## Generating slides
### Interactive slides
To generate interactive slides (as shown in the videos and lectures), you'll need to install the notebook extensions `rise` and `hide_input_all`.

```
conda install -c conda-forge rise
conda install -c conda-forge jupyter_nbextensions_configurator
jupyter nbextension enable rise
jupyter nbextension enable hide_input_all
```

You'll need to launch `jupyter notebook` since rise is not yet supported in jupyterlab. You'll see two new icons when you open a notebook. One will start the slideshow, and the other will hide the code. You'll need to set `interactive = True` in the first cell and then run all cells before starting the slideshow to allow the interactions.

### Static slides
You can generate slides from the notebooks using `nbconvert`. First, set `interactive = False` in the first cell of the notebook and rerun the notebooks to generate static versions of the interactive visualizations.

```
jupyter nbconvert --to slides --template reveal --SlidesExporter.reveal_theme=simple --no-input --post serve <NotebookName>
```

To generate PDF handouts, remove `#/` from the url and add `?print-pdf`, then print as PDF. 

Sidenote: Some PDF readers (e.g. Preview) sometimes show grey boxes around code examples, others (e.g. Acrobat Reader, Chrome) do not. Must be an artifact of some PDF readers.

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
