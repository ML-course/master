#!/usr/bin/env python
# coding: utf-8

# In[1]:


from preamble import *
get_ipython().run_line_magic('matplotlib', 'inline')


# # Lab 1: Machine Learning with Python
# Joaquin Vanschoren, Pieter Gijsbers, Bilge Celik, Prabhant Singh

# ## Overview
# * Why Python?
# * Intro to scikit-learn
# * Exercises

# # Why Python?
# * Many data-heavy applications are now developed in Python
# * Highly readable, less complexity, fast prototyping
# * Easy to offload number crunching to underlying C/Fortran/... 
# * Easy to install and import many rich libraries
#     - numpy: efficient data structures
#     - scipy: fast numerical recipes
#     - matplotlib: high-quality graphs
#     - scikit-learn: machine learning algorithms
#     - tensorflow: neural networks
#     - ...

# <img src="../images/tut_ecosystem.jpg" alt="ml" style="width: 1000px;"/>

# # Numpy, Scipy, Matplotlib
# * See the tutorials (in the course GitHub)
# * Many good tutorials online
#     - [Jake VanderPlas' book and notebooks](https://github.com/jakevdp/PythonDataScienceHandbook)
#     - [J.R. Johansson's notebooks](https://github.com/jrjohansson/scientific-python-lectures)
#     - [DataCamp](https://www.datacamp.com)
#     - ...

# # scikit-learn
# One of the most prominent Python libraries for machine learning:
# 
# * Contains many state-of-the-art machine learning algorithms
# * Builds on numpy (fast), implements advanced techniques
# * Wide range of evaluation measures and techniques
# * Offers [comprehensive documentation](http://scikit-learn.org/stable/documentation) about each algorithm
# * Widely used, and a wealth of [tutorials](http://scikit-learn.org/stable/user_guide.html) and code snippets are available 
# * Works well with numpy, scipy, pandas, matplotlib,...

# ## Algorithms
# See the [Reference](http://scikit-learn.org/dev/modules/classes.html)

# __Supervised learning:__
# 
# * Linear models (Ridge, Lasso, Elastic Net, ...)
# * Support Vector Machines
# * Tree-based methods (Classification/Regression Trees, Random Forests,...)
# * Nearest neighbors
# * Neural networks 
# * Gaussian Processes
# * Feature selection

# __Unsupervised learning:__
#     
# * Clustering (KMeans, ...)
# * Matrix Decomposition (PCA, ...)
# * Manifold Learning (Embeddings)
# * Density estimation
# * Outlier detection

# __Model selection and evaluation:__
# 
# * Cross-validation
# * Grid-search
# * Lots of metrics

# ## Data import
# Multiple options:
# 
# * A few toy datasets are included in `sklearn.datasets`
# * Import [1000s of datasets](http://www.openml.org) via `sklearn.datasets.fetch_openml`
# * You can import data files (CSV) with `pandas` or `numpy`

# In[2]:


from sklearn.datasets import load_iris, fetch_openml
iris_data = load_iris()
dating_data = fetch_openml("SpeedDating")


# These will return a `Bunch` object (similar to a `dict`)

# In[3]:


print("Keys of iris_dataset: {}".format(iris_data.keys()))
print(iris_data['DESCR'][:193] + "\n...")


# * Targets (classes) and features are lists of strings
# * Data and target values are always numeric (ndarrays)

# In[4]:


print("Targets: {}".format(iris_data['target_names']))
print("Features: {}".format(iris_data['feature_names']))
print("Shape of data: {}".format(iris_data['data'].shape))
print("First 5 rows:\n{}".format(iris_data['data'][:5]))
print("Targets:\n{}".format(iris_data['target']))


# ## Building models
# All scikitlearn _estimators_ follow the same interface

# ```python
# class SupervisedEstimator(...):
#     def __init__(self, hyperparam, ...):
# 
#     def fit(self, X, y):   # Fit/model the training data
#         ...                # given data X and targets y
#         return self
#      
#     def predict(self, X):  # Make predictions
#         ...                # on unseen data X  
#         return y_pred
#     
#     def score(self, X, y): # Predict and compare to true
#         ...                # labels y                
#         return score
# ```

# ### Training and testing data
# To evaluate our classifier, we need to test it on unseen data.  
# `train_test_split`: splits data randomly in 75% training and 25% test data.

# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_data['data'], iris_data['target'], 
    random_state=0)
print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))


# We can also choose other ways to split the data. For instance, the following will create a training set of 10% of the data and a test set of 5% of the data. This is useful when dealing with very large datasets. `stratify` defines the target feature to stratify the data (ensure that the class distributions are kept the same).

# In[6]:


X, y = iris_data['data'], iris_data['target']
Xs_train, Xs_test, ys_train, ys_test = train_test_split(X,y, stratify=y, train_size=0.1, test_size=0.05)
print("Xs_train shape: {}".format(Xs_train.shape))
print("Xs_test shape: {}".format(Xs_test.shape))


# ### Looking at your data (with pandas)

# In[7]:


from pandas.plotting import scatter_matrix

# Build a DataFrame with training examples and feature names
iris_df = pd.DataFrame(X_train, 
                       columns=iris_data.feature_names)

# scatter matrix from the dataframe, color by class
sm = scatter_matrix(iris_df, c=y_train, figsize=(8, 8), 
                  marker='o', hist_kwds={'bins': 20}, s=60, 
                  alpha=.8)


# ### Fitting a model

# The first model we'll build is a k-Nearest Neighbor classifier.  
# kNN is included in `sklearn.neighbors`, so let's build our first model

# In[8]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)


# ### Making predictions
# Let's create a new example and ask the kNN model to classify it

# In[9]:


X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(
       iris_data['target_names'][prediction]))


# ### Evaluating the model
# Feeding all test examples to the model yields all predictions

# In[10]:


y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))


# The `score` function computes the percentage of correct predictions
# 
# ``` python
# knn.score(X_test, y_test)
# ```

# In[11]:


print("Score: {:.2f}".format(knn.score(X_test, y_test) ))


# Instead of a single train-test split, we can use `cross_validate` do run a cross-validation. 
# It will return the test scores, as well as the fit and score times, for every fold.
# By default, scikit-learn does a 5-fold cross-validation, hence returning 5 test scores.

# In[20]:


from sklearn.model_selection import cross_validate
xval = cross_validate(knn, X, y, return_train_score=True, n_jobs=-1)
xval


# The mean should give a better performance estimate

# In[18]:


np.mean(xval['test_score'])


# ### Introspecting the model
# Most models allow you to retrieve the trained model parameters, usually called `coef_`

# In[13]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train, y_train)
lr.coef_


# Matching these with the names of the features, we can see which features are primarily used by the model

# In[14]:


d = zip(iris_data.feature_names,lr.coef_)
set(d)


# Please see the course notebooks for more examples on how to analyse models.
