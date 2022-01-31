#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import HTML
HTML('''<style>html, body{overflow-y: visible !important} .CodeMirror{min-width:105% !important;} .rise-enabled .CodeMirror, .rise-enabled .output_subarea{font-size:140%; line-height:1.2; overflow: visible;} .output_subarea pre{width:110%}</style>''') # For slides


# # Prerequisites

# ## Python
# You first need to set up a Python environment (if you do not have done so already). The easiest way to do this is by installing [Anaconda](https://www.anaconda.com/distribution/#download-section) or [MiniForge](https://github.com/conda-forge/miniforge), which will install Python as well as a set of commonly used packages. We will be using Python 3, so be sure to install the right version. Always install a 64-bit installer (if your machine supports it), and we recommend using Python 3.9 or later.
# 
# If you are completely new to Python, we recommend reading the [Python Data Science Handbook](https://github.com/jakevdp/PythonDataScienceHandbook) or taking an introductory online course, such as the [Definite Guide to Python](https://www.programiz.com/python-programming), the [Whirlwind Tour of Python](https://github.com/jakevdp/WhirlwindTourOfPython), or [this Python Course](https://www.python-course.eu/). If you like a step-by-step approach, try the [DataCamp Intro to Python for Data Science](https://www.datacamp.com/courses/intro-to-python-for-data-science).
# 
# To practice your skills, try some [Hackerrank challenges](https://www.hackerrank.com/domains/python). 

# ### OS specific notes
# 
# **Windows users**: If you are new to Anaconda, read the [starting guide](https://docs.anaconda.com/anaconda/user-guide/getting-started/). You'll probably use the Anaconda Prompt to run any commands or to start Jupyter Lab.
# 
# **Mac users**: You'll probably use your terminal to run any commands or to start Jupyter Lab. Make sure that you have Command Line tools installed. If not, run `xcode-select —install`. You won't need a full XCode installation. 
# 
# #### Apple silicon (M1)
# For those who have a laptop with Apple Silicon (M1), [this guide may be useful](https://betterdatascience.com/install-tensorflow-2-7-on-macbook-pro-m1-pro/) to install a TensorFlow version that will effectively use the GPUs.
# 
# Also run this ([more info](https://github.com/cvxpy/cvxpy/issues/1604)): `conda install -c conda-forge cvxpy`
# 
# 

# ### Virtual environments
# 
# If you already have a custom Python environment set up, possibly using a different Python version, we highly recommend to set up a virtual environment to avoid interference with other projects and classes. This is not strictly needed if you use a fresh Anaconda install, since that will automatically create a new environment on installation.
# 
# #### Using conda
# To create a new conda environment called 'mlcourse' (or whatever you like), run
# ```
# conda create -n mlcourse python=3.9 conda
# ```
# You activate the environment with `conda activate mlcourse` and deacticate it with `conda deactivate`.
# 
# #### Using virtualenv
# To can also use [venv](http://docs.python-guide.org/en/latest/dev/virtualenvs/) if you prefer:
# ```
# pip install virtualenv
# virtualenv mlcourse
# ```
# Activate the environment with `source mlcourse/bin/activate` or `mlcourse\Scripts\activate` on Windows. To deactivate the virtual environment, type `deactivate`.

# ### Installing TensorFlow
# 
# To install *TensorFlow 2* (if you haven't already), follow [these instructions](https://www.tensorflow.org/install/) for your OS (Windows, Mac, Ubuntu). While installation with `conda` is possible, they recommend to install with `pip`, even with an Anaconda setup. We recommend using TensorFlow 2.7 or later.
# 
# 

# ## Course materials on GitHub
# The course materials are available on GitHub, so that you can easily `pull` (download) the latest updates. We recommend [installing git](https://git-scm.com/book/en/v1/Getting-Started-Installing-Git) (if you haven't already), and then 'clone' the repository from the command line (you can also use a [GUI](https://desktop.github.com/))
# 
#     git clone https://github.com/ML-course/master.git
#     
# To download updates, run `git pull`
#     
# For more details on using git, see the [GitHub 10-minute tutorial](https://guides.github.com/activities/hello-world/
# ) and [Git for Ages 4 and up](https://www.youtube.com/watch?v=1ffBJ4sVUb4). We'll use git extensively in the course (e.g., to submit assignments).
# 
# Alternatively, you can download the course [as a .zip file](https://github.com/ML-course/master.git). Click 'Code' and then 'Download ZIP'. Or, download individual files with right-click -> Save Link As...

# ## Installing required packages
# Next, you'll need to install several packages that we'll be using extensively in this course, using pip (the Python Package index).  
# Run the following from the folder where you cloned (or downloaded) the course, or adjust the path to the `requirements.txt` file:
# 
#     pip install --upgrade pip
#     pip install -U -r requirements.txt
# 
# Note: the -U option updates all packages, should you have older versions already installed.

# ## Running the course notebooks
# As our coding environment, we'll be using Jupyter notebooks. They interleave documentation (in markdown) with executable Python code, and they run in your browser. That means that you can easily edit and re-run all the code in this course. If you are new to notebooks, [take this quick tutorial](https://try.jupyter.org/), or [this more detailed one](http://nbviewer.jupyter.org/github/jupyter/notebook/tree/master/docs/source/examples/Notebook/). Optionally, for a more in-depth coverage, [try the DataCamp tutorial](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook#gs.wlHChdo). 
# 
# Run jupyter lab from the folder where you have downloaded (or cloned) the course materials, using the Python environment you created above.
# 
#     jupyter lab
#     
# A browser window should open with all course materials. Open one of the chapters and check if you can execute all code by clicking Cell > Run all.  You can shut down the notebook by typing CTRL-C in your terminal.

# ### An alternative: Google Colab
# [Google Colab](https://colab.research.google.com/) allows you to run notebooks in your browser without any local installation. It also provides (limited) GPU resources. It is a useful alternative in case you encounter issues with you local installation or don't have it available, or to easily use GPUs.
# 
# The [course overview page](https://ml-course.github.io/master/) has buttons to launch all materials in Colab, or you can upload the notebooks to Colab yourself. There are a few lines at the top of the notebook that you need to uncomment to make it run smoothly in Colab.
# 
# 

# In[ ]:




