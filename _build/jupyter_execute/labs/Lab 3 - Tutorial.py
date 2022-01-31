#!/usr/bin/env python
# coding: utf-8

# # Lab 3 Tutorial: Model Selection in scikit-learn

# In[1]:


# Global imports and settings
from preamble import *
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Evaluation procedures
# ### Holdout
# The simplest procedure is [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html), which splits arrays or matrices into random train and test subsets.

# In[2]:


from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# create a synthetic dataset
X, y = make_blobs(centers=2, random_state=0)
# split data and labels into a training and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# Instantiate a model and fit it to the training set
model = LogisticRegression().fit(X_train, y_train)
# evaluate the model on the test set
print("Test set score: {:.2f}".format(model.score(X_test, y_test)))


# ### Cross-validation
# - [cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html?highlight=cross%20val%20score#sklearn.model_selection.cross_val_score)
#     - `cv` parameter defines the kind of cross-validation splits, default is 5-fold CV
#     - `scoring` defines the scoring metric. Also see below.
#     - Returns list of all scores. Models are built internally, but not returned
# - [cross_validate](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html?highlight=cross%20validate#sklearn.model_selection.cross_validate)
#     - Similar, but also returns the fit and test times, and allows multiple scoring metrics.

# In[3]:


from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
logreg = LogisticRegression()

scores = cross_val_score(logreg, iris.data, iris.target, cv=5)
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("Variance in cross-validation score: {:.4f}".format(np.var(scores)))


# ### Custom CV splits
# - You can build folds manually with [KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html?highlight=kfold#sklearn.model_selection.KFold) or [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold)
#     - randomizable (`shuffle` parameter)
# - [LeaveOneOut](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html?highlight=leave%20one%20out#sklearn.model_selection.LeaveOneOut) does leave-one-out cross-validation

# In[4]:


from sklearn.model_selection import KFold, StratifiedKFold
kfold = KFold(n_splits=5)
print("Cross-validation scores KFold(n_splits=5):\n{}".format(
      cross_val_score(logreg, iris.data, iris.target, cv=kfold)))
skfold = StratifiedKFold(n_splits=5, shuffle=True)
print("Cross-validation scores StratifiedKFold(n_splits=5, shuffle=True):\n{}".format(
      cross_val_score(logreg, iris.data, iris.target, cv=skfold)))


# In[5]:


from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
scores = cross_val_score(logreg, iris.data, iris.target, cv=loo)
print("Number of cv iterations: ", len(scores))
print("Mean accuracy: {:.2f}".format(scores.mean()))


# ### Shuffle-split
# These shuffle the data before splitting it.
# - `ShuffleSplit` and `StratifiedShuffleSplit` (recommended for classification)
# - `train_size` and `test_size` can be absolute numbers or a percentage of the total dataset

# In[6]:


from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
shuffle_split = StratifiedShuffleSplit(test_size=.5, train_size=.5, n_splits=10)
scores = cross_val_score(logreg, iris.data, iris.target, cv=shuffle_split)
print("Cross-validation scores:\n{}".format(scores))


# #### Grouped cross-validation
# - Add an array with group membership to `cross_val_scores` 
# - Use `GroupKFold` with the number of groups as CV procedure

# In[7]:


from sklearn.model_selection import GroupKFold
# create synthetic dataset
X, y = make_blobs(n_samples=12, random_state=0)
# the first three samples belong to the same group, etc.
groups = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
scores = cross_val_score(logreg, X, y, groups, cv=GroupKFold(n_splits=4))
print("cross_val_score(logreg, X, y, groups, cv=GroupKFold(n_splits=4)")
print("Cross-validation scores :\n{}".format(scores))


# ## Evaluation Metrics

# #### Binary classification
# - [confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html?highlight=confusion%20matrix#sklearn.metrics.confusion_matrix) returns a matrix counting how many test examples are predicted correctly or 'confused' with other metrics.
# - [sklearn.metrics](https://scikit-learn.org/stable/modules/classes.html?highlight=metrics#module-sklearn.metrics) contains implementations many of the metrics discussed in class
#     - They are all implemented so that 'higher is better'. 
# - [accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score) computes accuracy explictly
# - [classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) returns a table of binary measures, per class, and aggregated according to different aggregation functions.

# In[3]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split
    data.data, data.target, stratify=data.target, random_state=0)
lr = LogisticRegression().fit(X_train, y_train)
y_pred = lr.predict(X_test)

print("confusion_matrix(y_test, y_pred): \n", confusion_matrix(y_test, y_pred))
print("accuracy_score(y_test, y_pred): ", accuracy_score(y_test, y_pred))
print("model.score(X_test, y_test): ", lr.score(X_test, y_test))


# In[9]:


plt.rcParams['figure.dpi'] = 100 
print(classification_report(y_test, lr.predict(X_test)))


# You can explictly define the averaging function for class-level metrics 

# In[10]:


pred = lr.predict(X_test)
print("Micro average f1 score: {:.3f}".format(f1_score(y_test, pred, average="micro")))
print("Weighted average f1 score: {:.3f}".format(f1_score(y_test, pred, average="weighted")))
print("Macro average f1 score: {:.3f}".format(f1_score(y_test, pred, average="macro")))


# #### Probabilistic predictions
# To retrieve the uncertainty in the prediction, scikit-learn offers 2 functions. Often, both are available for every learner, but not always.
# 
# - decision_function: returns floating point (-Inf,Inf) value for each prediction
# - predict_proba: returns probability [0,1] for each prediction

# You can also use these to compute any metric with non-standard thresholds

# In[11]:


print("Threshold -0.8")
y_pred_lower_threshold = lr.decision_function(X_test) > -.8
print(classification_report(y_test, y_pred_lower_threshold))  


# #### Uncertainty in multi-class classification
# 
# - `decision_function` and `predict_proba` also work in the multiclass setting
# - always have shape (n_samples, n_classes)
# - Example on the Iris dataset, which has 3 classes:

# In[4]:


from sklearn.datasets import load_iris

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=42)

lr2 = LogisticRegression()
lr2 = lr2.fit(X_train, y_train)

print("Decision function:\n{}".format(lr2.decision_function(X_test)[:6, :]))
# show the first few entries of predict_proba
print("Predicted probabilities:\n{}".format(lr2.predict_proba(X_test)[:6]))


# ### Precision-Recall and ROC curves
# 
# - [precision_recall_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html?highlight=precision_recall_curve) returns all precision and recall values for all possible thresholds
# - [roc_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html?highlight=roc%20curve#sklearn.metrics.roc_curve) does the same for TPR and FPR.
# - The average precision score is returned by the `average_precision_score` measure 
# - The area under the ROC curve is returned by the `roc_auc_score` measure 
#     - Don't use `auc` (this uses a less accurate trapezoidal rule)
#     - Require a decision function or predict_proba.
#     

# In[12]:


from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(
    y_test, lr.decision_function(X_test)) 


# In[13]:


from sklearn.metrics import average_precision_score
ap_pp = average_precision_score(y_test, lr.predict_proba(X_test)[:, 1])
ap_df = average_precision_score(y_test, lr.decision_function(X_test))
print("Average precision of logreg: {:.3f}".format(ap_df))


# In[14]:


from sklearn.metrics import roc_auc_score
rf_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
svc_auc = roc_auc_score(y_test, lr.decision_function(X_test))
print("AUC for Random Forest: {:.3f}".format(rf_auc))
print("AUC for SVC: {:.3f}".format(svc_auc))


# #### Multi-class prediction
# * Build C models, one for every class vs all others
# * Use micro-, macro-, or weighted averaging

# In[15]:


print("Micro average f1 score: {:.3f}".format(f1_score(y_test, pred, average="micro")))
print("Weighted average f1 score: {:.3f}".format(f1_score(y_test, pred, average="weighted")))
print("Macro average f1 score: {:.3f}".format(f1_score(y_test, pred, average="macro")))


# ## Using evaluation metrics in model selection
# 
# - You typically want to use AUC or other relevant measures in `cross_val_score` and `GridSearchCV` instead of the default accuracy.
# - scikit-learn makes this easy through the `scoring` argument
#     - But, you need to need to look the [mapping between the scorer and the metric](http://scikit-learn.org/stable/modules/model_evaluation.html#model-evaluation)

# ![scorers](../images/03_scoring.png)

# Or simply look up like this:

# In[16]:


from sklearn.metrics.scorer import SCORERS
print("Available scorers:\n{}".format(sorted(SCORERS.keys())))


# Cross-validation with AUC

# In[17]:


from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn .svm import SVC
from sklearn.datasets import load_digits
digits = load_digits()

# default scoring for classification is accuracy
print("Default scoring: {}".format(
      cross_val_score(SVC(), digits.data, digits.target == 9)))
# providing scoring="accuracy" doesn't change the results
explicit_accuracy =  cross_val_score(SVC(), digits.data, digits.target == 9, 
                                     scoring="accuracy")
print("Explicit accuracy scoring: {}".format(explicit_accuracy))
roc_auc =  cross_val_score(SVC(), digits.data, digits.target == 9,
                           scoring="roc_auc")
print("AUC scoring: {}".format(roc_auc))


# ## Hyperparameter tuning
# Now that we know how to evaluate models, we can improve them by tuning their hyperparameters

# #### Grid search
# - Create a parameter grid as a dictionary
#     - Keys are parameter names
#     - Values are lists of hyperparameter values

# In[18]:


param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
print("Parameter grid:\n{}".format(param_grid))


# - `GridSearchCV`: like a classifier that uses CV to automatically optimize its hyperparameters internally
#     - Input: (untrained) model, parameter grid, CV procedure
#     - Output: optimized model on given training data
#     - Should only have access to training data

# In[19]:


from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVC
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, random_state=0)
grid_search.fit(X_train, y_train)


# The optimized test score and hyperparameters can easily be retrieved:

# In[20]:


print("Test set score: {:.2f}".format(grid_search.score(X_test, y_test)))


# In[21]:


print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


# In[22]:


print("Best estimator:\n{}".format(grid_search.best_estimator_))


# When hyperparameters depend on other parameters, we can use lists of dictionaries to define the hyperparameter space

# In[23]:


param_grid = [{'kernel': ['rbf'],
               'C': [0.001, 0.01, 0.1, 1, 10, 100],
               'gamma': [0.001, 0.01, 0.1, 1, 10, 100]},
              {'kernel': ['linear'],
               'C': [0.001, 0.01, 0.1, 1, 10, 100]}]
print("List of grids:\n{}".format(param_grid))


# #### Nested cross-validation
# 
# - Nested cross-validation:
#     - Outer loop: split data in training and test sets
#     - Inner loop: run grid search, splitting the training data into train and validation sets
# - Result is a just a list of scores
#     - There will be multiple optimized models and hyperparameter settings (not returned)
# - To apply on future data, we need to train `GridSearchCV` on all data again

# In[24]:


scores = cross_val_score(GridSearchCV(SVC(), param_grid, cv=5),
                         iris.data, iris.target, cv=5)
print("Cross-validation scores: ", scores)
print("Mean cross-validation score: ", scores.mean())


# #### Parallelizing cross-validation and grid-search
# - On a practical note, it is easy to parallellize CV and grid search
# - `cross_val_score` and `GridSearchCV` have a `n_jobs` parameter defining the number of cores it can use.
#     - set it to `n_jobs=-1` to use all available cores.

# ### Random Search
# - `RandomizedSearchCV` works like `GridSearchCV`
# - Has `n_iter` parameter for the number of iterations
# - Search grid can use distributions instead of fixed lists

# In[25]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon

param_grid = {'C': expon(scale=100), 
              'gamma': expon(scale=.1)}
random_search = RandomizedSearchCV(SVC(), param_distributions=param_grid,
                                   n_iter=20)
X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, random_state=0)
random_search.fit(X_train, y_train)

