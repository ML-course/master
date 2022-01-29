# An Open Machine Learning Course

Jupyter notebooks for teaching machine learning. Based on scikit-learn and Keras, with OpenML used to experiment more extensively on many datasets.

[Course website](https://ml-course.github.io/)

## Sources
### Practice-oriented materials
We use many code examples from the following excellent books. We urge you to read them for a more complete coverage of machine learning in Python:

[Introduction to Machine Learning with Python](http://shop.oreilly.com/product/0636920030515.do>) by [Andreas Mueller](http://amueller.io) and [Sarah Guido](https://twitter.com/sarah_guido). Focussing entirely on scikit-learn, and written by one of its core developers, this book offers clear guidance on how to do machine learning with Python.

[Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python) by François Chollet. Written by the author of the Keras library, this book offers a clear explanation of deep learning with practical examples.

[Python machine learning](https://www.amazon.com/Python-Machine-Learning-Sebastian-Raschka/dp/1783555130/ref=sr_1_1?ie=UTF8&qid=1472342570&sr=8-1&keywords=sebastian+raschka) by Sebastian Raschka. One of the classic textbooks on how to do machine learning with Python.

[Python for Data Analysis](http://shop.oreilly.com/product/0636920023784.do) by Wes McKinney. A more introductory and broader text on doing data science with Python.

### Theory-oriented materials
For a deeper understanding of machine learning techniques, we can recommend the following books:

"[Mathematics for Machine Learning](https://mml-book.github.io/book/mml-book.pdf)" by Marc Deisenroth, A. Aldo Faisal and Cheng Soon Ong. This provides the basics of linear algebra, geometry, probabilities, and continuous optimization, and how they are used in several machine learning algorithms. The PDF is available for free.

"[The Elements of Statistical Learning: Data Mining, Inference, and Prediction. (2nd edition)](https://statweb.stanford.edu/~tibs/ElemStatLearn/)" by Trevor Hastie, Robert Tibshirani, Jerome Friedman. One of the key references of the field. Great coverage of linear models, regularization, kernel methods, model evaluation, ensembles, neural nets, unsupervised learning. The PDF is available for free.  

"[Deep Learning](http://www.deeplearningbook.org/)" by Ian Goodfellow, Yoshua Bengio, Aaron Courville. The current reference for deep learning. Chapters can be downloaded from the website.

"[An Introduction to Statistical Learning (with Applications in R)](http://www-bcf.usc.edu/~gareth/ISL/)" by Gareth James, Daniela Witten, Trevor Hastie and Robert Tibshirani. More introductory version of the above book, with many code examples in R. The PDF is also available for free. (Note that we won't be using R in the main course materials, but the examples are still very useful).

"[Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/)" by Carl Edward Rasmussen and Christopher K. I. Williams. The reference for Bayesian Inference. Also see [David MacKay's book](http://www.inference.phy.cam.ac.uk/itila/book.html) for additional insights. Also see [this course by Neil Lawrence](http://inverseprobability.com/mlai2015/) for a great introduction to Gaussian Processes, all from first principles.

## Generating slides
These instructions are for teachers. As a student, you can simply run the notebooks in jupyter lab and/or use the pre-compiled slides and videos.

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

Caveat: nbconvert doesn't properly hide the output of interactive cells. You'll have to hack the css by looking for a file called `custom_reveal.css` in your `nbconvert` installation and add the line `display: none !important;` in class `.reveal pre`.



