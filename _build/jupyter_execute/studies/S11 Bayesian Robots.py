#!/usr/bin/env python
# coding: utf-8

# # Bayesian Robots!

# We will use Bayesian optimization to efficiently tune machine learning algorithms for robot movement.
# 
# ### Robot Navigation
# The [Wall robot navigation](https://www.openml.org/d/1497) contains training data for a robot equiped with ultrasound sensors. Based on 24 sensor readings, the robot has to figure out how to move though an office space without hitting walls or other obstacles. The possible actions (classes) are 'Move-Forward', 'Slight-Right-Turn', 'Sharp-Right-Turn', and 'Slight-Left-Turn'.

# In[1]:


# General imports
get_ipython().run_line_magic('matplotlib', 'inline')
import openml as oml

# Download Wall Robot Navigation data from OpenML.
robotnav = oml.datasets.get_dataset(1497)
X, y, cats, attrs = robotnav.get_data(dataset_format='array',
    target=robotnav.default_target_attribute)
labels = ['Move-Forward','Slight-Right-Turn','Sharp-Right-Turn','Slight-Left-Turn']


# ### Visualizing the data
# Let's try to plot the position of the robot and where the walls are according to the sensors.
# We can compute the coordinates of the detected wall using the angle of each sensor and basic
# geometry:  
# 
# $ x_{wall} = x_{robot} + dist * cos(angle_{robot} + angle_{sensor}) $  
# $ y_{wall} = y_{robot} + dist * sin(angle_{robot} + angle_{sensor}) $
# 
# Where $x_{robot}$ is the x-coordinate of the robot, $dist$ is the distance measured by the sensor,
# $angle_{robot}$ is the current direction the robot is facing and $angle_{sensor}$ is the relative
# angle of the specific sensor.
# 
# The dataset and the paper do not give any information on how fast the robot moves and how fast it
# turns, so we have to guess these. It does say that it measures 9 samples per second. After some
# trial and error we get plausible results if we set the speed to 0.1 meter per second and the turning
# rate to about 2 degrees per second. Below is the resulting animation in which the robot is presented
# as a triangle (green when moving and red when turning). The dots are the assumed locations of walls.
# Although the visualization is probably not very precise, we can see that the robot follows the nearest wall.

# In[4]:


import math
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
matplotlib.rcParams['animation.embed_limit'] = 100

fig, ax = plt.subplots()
fig.set_tight_layout(True)
cx, cy = 0, 0 # robot position
angle = 180   # robot direction
ax.clear()
robot = ax.scatter(cx,cy, color='r', s=100, marker=(3, 0, angle+30))

def update(n): 
    global angle, cx, cy, robot
    curr_x, cl = X[n], y[n]
    if cl==0:
        cx+=math.cos(math.radians(angle))*0.012
        cy+=math.sin(math.radians(angle))*0.012
    elif cl==1:
        angle -= 0.02
    elif cl==2:
        angle -= 0.9
    elif cl==3:
        angle += 0.02
    if n%30==0: #speed things up by only plotting every n'th step
        wall_points =np.array([[cx+dist*math.cos(math.radians(angle-180+i*15)),cy+dist*math.sin(math.radians(angle-180+i*15))] 
                               for i, dist in enumerate(curr_x) if dist<2 ])
        ax.clear()
        ax.set_xlim(-10,5)
        ax.set_ylim(-2,10)
        #robot.remove();
        robot = ax.scatter(cx,cy, color=('g' if cl==0 else 'r'), s=100, marker=(3, 0, angle+30))
        ax.scatter(wall_points[:,0],wall_points[:,1], color='k', s=1)
plt.close(fig)


# In[9]:


anim = FuncAnimation(fig, update, frames=np.arange(0, 5000), interval=1)
HTML(anim.to_jshtml())


# ### Classification models
# We try the following models (for simplicity, we will only tune the 2 most important hyperparameters per model):  
# * Support vector machine, with hyperparameters $C \in [10^{-12},10^{12}]$, $\gamma \in [10^{-12},10^{12}]$ (both log scale), and an RBF kernel.
# * Gradient Boosting, with hyperparameters `learning_rate` $\in [10^{-4},10^{-1}]$ (log scale), `max_depth` $\in [1,5]$, and `n_estimators` fixed at least 1000 (you can increase this if your computing resources allow).
# 
# We want a fast way to optimize and re-optimize the model so that it will keep working well. We will use Bayesian optimization for this.

# In[5]:


import pandas as pd
import matplotlib.ticker as mtick
import seaborn as sns
import imageio as io
import itertools
import os
import time

from mpl_toolkits import mplot3d
from random import randint
from scipy.stats import norm
from matplotlib import colors

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from xgboost.sklearn import XGBModel
from xgboost import XGBClassifier, XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

from IPython.display import Image


# ### Helper functions

# In[6]:


# Random Forest that also returns the standard deviation of predictions
class ProbRandomForestRegressor(RandomForestRegressor):
    """
    A Random Forest regressor that can also returns the standard deviations for all predictions
    """
    def predict(self, X, return_std=True):       
        preds = []
        for pred in self.estimators_:
            preds.append(pred.predict(X))
        if return_std:
            return np.mean(preds, axis=0), np.std(preds, axis=0)
        else:
            return np.mean(preds, axis=0)

# Helper function to compute expected improvement 
def EI(surrogate, X: np.ndarray, curr_best=0.0, balance=0.0, **kwargs):
    """Computes the Expected Improvement
    surrogate, The surrogate model
    X: np.ndarray(N, D), The input points where the acquisition function
        should be evaluated. N configurations with D hyperparameters
    curr_best, The current best performance
    balance, Decrease to focus more on exploration, increase to focus on exploitation (optional)
    Returns
    -------
    np.ndarray(N,1), Expected Improvement of X
    """
    if len(X.shape) == 1:
        X = X[:, np.newaxis]

    m, s = surrogate.predict(X, return_std=True) # mean, stdev

    z = (curr_best - m - balance) / s
    f = (curr_best - m - balance) * norm.cdf(z) + s * norm.pdf(z)

    if np.any(s == 0.0): # uncertainty should never be exactly 0.0
        f[s == 0.0] = 0.0

    return f


# ### Load the data

# In[7]:


# Fetch classification data
robotnav = oml.datasets.get_dataset(1497)
X, y, _, _ = robotnav.get_data(dataset_format='array',
    target=robotnav.default_target_attribute)

# Fetch regression data
robotarm = oml.datasets.get_dataset(189)
Xr, yr, _, _ = robotarm.get_data(dataset_format='dataframe',
    target=robotarm.default_target_attribute)

X.shape, y.shape, Xr.shape, yr.shape


# ## 1. Implementing Bayesian Optimization (60 points) {-}
# * Implement Bayesian optimization using the code above and use it to optimize the hyperparameters stated below for each of the two datasets.
#     - Use the hyperparameters and ranges are defined above. Make sure to sample from a log scale (`numpy.logspace`) whenever the hyperparameters should be varied on a log scale. 
#     - The evaluation measure for classification should be misclassification error (1 - Accuracy), evaluated using 3-fold cross-validation
#     - The evaluation measure for regression should be mean squared error, also evaluated using 3-fold cross-validation
# * Initialize the surrogate model with 10 randomly sampled configurations and visualize the surrogate model.
#     - Hint: Use a 2D slice of each hyperparameter (e.g. $C$=4 and $\gamma$=0.1) to show both the predicted values and the uncertainty.
#     - For simplicity, you can build a separate surrogate model for each algorithm and each dataset (4 models in total) 
# * Visualize the resulting acquisition function, either as 2D slices (or, more difficult, as a 3D surface)
# * Visualize 3 more iterations, each time visualizing the surrogate model and acquisition function
# * Run the Bayesian optimization for at least 30 iterations, report the optimal configuration and show the final surrogate model (2D slices or 3D surface).
# * Interpret and explain the results. Does Bayesian optimization efficiently find good configurations? Do you notice any
# differences between the different models and the different datasets. Explain the results as well as you can.

# ### Class to perform the Bayesian Optimization
# You can supply your own surrogate and objective function to the model. The hyperparameter combinations we would like to run on the objective model are stored in `self.hyperparam_obj`. After the objective model calculated the losses we create a grid of 30 by 30 of all hyperparameter combinations. This is stored in `self.hyperparams_sur`. Afterwards we predict the loss for all these values and create a 3D plot to show the surrogate model and acquisition function.

# In[12]:


class BayesianOptimization:
    def __init__(self, 
               surrogate_model, 
               objective_model, 
               acquisition_function,
               hyperparam_space,
               X_obj,
               y_obj):
        
        self.surrogate_model = surrogate_model
        self.objective_model = objective_model
        self.hyperparam_names = [*hyperparam_space.keys()]
        self.hyperparam_space = hyperparam_space
        self.acquisition_function = acquisition_function
        self.X_obj=X_obj
        self.y_obj=y_obj
        self.highest_EI=0
        self.model_name = type(objective_model()).__name__
        self.surrogate_model_name = type(surrogate_model).__name__
        self.time_surrogate=[]

        # Dataframe to store all hyperparameters and their predictions
        # of the objective function
        self.hyperparams_obj = pd.DataFrame(
          columns=[*self.hyperparam_names, 'loss'])

        # Dataframe to store all hyperparameters and their predictions
        # of the surrogate model
        self.hyperparams_sur = pd.DataFrame(
          columns=[*self.hyperparam_names, 'pred', 'sigma', 'improvement'])
        
    def optimal_hyperparams(self, rows=1):
        """
        Returns dataframe with the optimal hyperparameters and the corresponding loss score. 
        """
        return self.hyperparams_obj.sort_values('loss')[:rows]
    
    def hyperparam_points(self, hyperparam_name, size, unique=False, discrete=True):
        """
        Create a population for a certain hyperparameter
        (E.g. 1000 log spaced points to sample from)
        """
        # Get the required settings
        settings = self.hyperparam_space[hyperparam_name]
        
        # Log spaced
        if settings['spacing'] == 'log':
            return np.logspace(settings['min'], settings['max'], size)
        
        # Linear spaced
        if settings['spacing'] == 'lin':
            # Only unique values (E.g. 1,2,3,4,5)
            if unique and discrete:
                return np.arange(settings['min'], settings['max']+1, 1).astype(int)
            
            # Only discrete values (E.g. 1,1,1,1,1,1,2,2,2,2,2)
            if discrete:
                return np.round(np.linspace(settings['min'], settings['max'], size)).astype(int)
            
            # Continues values (E.g. 1.0, 1.1, 1.2, 1.3, 1.4)
            return np.linspace(settings['min'], settings['max'], size)
    
    def hyperparam_cartesian(self, size):
        """
        Create a grid of the two hyperparameters.
        """
        points = []

        for hyperparam_name in self.hyperparam_names:
            settings = self.hyperparam_space[hyperparam_name]
            
            # Check if the sampling needs to be discrete
            discrete = ((settings['spacing'] == 'lin') & settings['discrete'])
            
            # Get the population of points
            points.append(self.hyperparam_points(hyperparam_name, size, unique=True, discrete=discrete))

        # Create the cartesian product of the two hyperparams
        cartesian = np.array(np.meshgrid(*points)).T.reshape(-1,2)

        # Store the values
        for idx, hyperparam_name in enumerate(self.hyperparam_names):
                
            self.hyperparams_sur[hyperparam_name] = cartesian[:,idx]

        return self.hyperparams_sur
      
    def hyperparam_sampling(self, n_samples=10):
        """
        Create the initial hyperparameter samples to start the optimization.
        """
        for hyperparam_name in self.hyperparam_names:
            settings = self.hyperparam_space[hyperparam_name]
            
            # Check if the sampling needs to be discrete
            discrete = ((settings['spacing'] == 'lin') & settings['discrete'])
            
            # Get the population of points
            points = self.hyperparam_points(hyperparam_name, 1000, discrete=discrete)

            # Take n samples from the population
            sample = points[np.random.choice(1000, n_samples)]

            # Store the results
            self.hyperparams_obj[hyperparam_name] = sample

        return self.hyperparams_obj
    
    def 
    
    (self, **static_hyperparams):
        """
        Predict all the hyperparameter combinations that do not have a prediction yet
        """
        for idx, hyperparams in (self.hyperparams_obj[self.hyperparams_obj['loss'].isnull()]
                                 .drop('loss', axis=1)
                                 .to_dict('index')
                                 .items()):
            
            # Setup the classifier
            clf = self.objective_model(**hyperparams, **static_hyperparams)

            # Get the mean accuracy over all three folds
            scores = cross_val_score(clf, self.X_obj, self.y_obj, cv=3, n_jobs=-1, 
                                       scoring="accuracy")

            # Calculate the mean 1 - acc loss
            loss = np.mean(1 - np.array(scores))

            # Store the loss
            self.hyperparams_obj.loc[idx, 'loss'] = loss
      
    def regressor_predict(self, **static_hyperparams):
        """
        Predict all the hyperparameter combinations that do not have a prediction yet
        """
        for idx, hyperparams in (self.hyperparams_obj[self.hyperparams_obj['loss'].isnull()]
                                 .drop('loss', axis=1)
                                 .to_dict('index')
                                 .items()):
            
            # Setup the classifier
            clf = self.objective_model(**hyperparams, **static_hyperparams)

            # Get the mean accuracy over all three folds
            scores = cross_val_score(clf, self.X_obj, self.y_obj, cv=3, n_jobs=-1, 
                                       scoring="neg_mean_squared_error")

            # Calculate 
            loss = np.mean(np.array(scores) * -1)

            # Store the loss
            self.hyperparams_obj.loc[idx, 'loss'] = loss
      
    def surrogate_predict(self, **static_hyperparams):
        """
        Predict all the hyperparameter combinations using the surrogate model
        """
        # The train set will be the hyperparameters that have been
        # predicted by the objective model
        X_sur = self.hyperparams_sur[list(self.hyperparam_names)]
        X_train = self.hyperparams_obj[list(self.hyperparam_names)]
        y_train = self.hyperparams_obj['loss']
        
        # Record the start time
        start = time.time()
        
        # Fit the model on the objective model losses
        self.surrogate_model.fit(X_train, y_train)

        # Predict all points using the surrogate model
        y_pred, sigma = self.surrogate_model.predict(
          X_sur, 
          return_std=True)
        
        # Record the end time
        end = time.time()
        self.time_surrogate.append(end-start)
        
        self.hyperparams_sur.loc[:, 'pred'] = y_pred
        self.hyperparams_sur.loc[:, 'sigma'] = sigma

        # Calculate the expected improvement
        expected_improvement = self.acquisition_function(
          self.surrogate_model, 
          X_sur)
        
        # Store the max expected improvement for creating the graphs
        if np.max(expected_improvement) > self.highest_EI:
            self.highest_EI = np.max(expected_improvement)
        
        self.hyperparams_sur.loc[:, 'improvement'] = expected_improvement
  
    def next_sample(self):
        """
        Function that picks the next best hyperparameters to look at
        """
        # Get the index of the hyperparameters that show the highest expected improvement
        # We shuffle the dataframe to prevent it from picking the same values over and over again
        parameters_idx = self.hyperparams_sur['improvement'].sample(frac=1).idxmax()

        # Get the values of hyperparameters associated with highest expected improvement
        opt_params = self.hyperparams_sur.loc[parameters_idx, list(self.hyperparam_names)]

        # Add it to the objective model hyperparameters to be predicted in the next run
        self.hyperparams_obj = self.hyperparams_obj.append(opt_params, ignore_index=True)
        
        # Change the data types to fix a bug with XGBoost
        self.force_dtypes()
  
    def force_dtypes(self):
        """
        Helper fucntion to cast the column of the hyperparams objective function to the right types
        """
        for hyperparam_name in self.hyperparam_names:
            setting = self.hyperparam_space[hyperparam_name]
            
            if (setting['spacing'] == 'lin') and setting['discrete']:
                self.hyperparams_obj[hyperparam_name] = self.hyperparams_obj[hyperparam_name].astype(int)
        
    def plot_surrogate(self, label='', show_confidence=False, show_plot=True, store_plot=False):
        """
        Function to create a 3d plot of the surrogate model and the acquisition 
        """

        # Create the two figures
        fig = plt.figure(figsize=(16, 6))
        ax_sur = fig.add_subplot(1, 2, 1, projection='3d')
        ax_acq = fig.add_subplot(1, 2, 2, projection='3d')

        # Get the z values
        z_loss = self.hyperparams_sur['pred']
        z_loss_obj = self.hyperparams_obj['loss'].astype(float) + 0.03
        z_acq = self.hyperparams_sur['improvement']

        first_param = self.hyperparam_space[self.hyperparam_names[0]]
        second_param = self.hyperparam_space[self.hyperparam_names[1]]

        x_plot = self.hyperparams_sur[self.hyperparam_names[0]]
        y_plot = self.hyperparams_sur[self.hyperparam_names[1]]

        x_plot_obj = self.hyperparams_obj[self.hyperparam_names[0]]
        y_plot_obj = self.hyperparams_obj[self.hyperparam_names[1]]

        # Space the points logaritmic if necessary
        if first_param['spacing'] == 'log':
            x_plot = np.log10(x_plot)
            x_plot_obj = np.log10(x_plot_obj)
        if second_param['spacing'] == 'log':
            y_plot = np.log10(y_plot)
            y_plot_obj = np.log10(y_plot_obj)

        # The maximum loss of the surrogate function
        loss_max = self.hyperparams_sur['pred'].max()
        loss_max += (loss_max * 0.1)

        # Show the plots
        ax_sur.plot_trisurf(x_plot, y_plot, z_loss, cmap='plasma', zorder=1, alpha=0.5)

        for index in range(len(x_plot_obj)):
            first_hyperparam = x_plot_obj[index]
            second_hyperparam = y_plot_obj[index]
            loss = z_loss_obj[index]

            ax_sur.plot(
                [first_hyperparam,first_hyperparam],
                [second_hyperparam,second_hyperparam],
                [0,loss_max*0.05],
                color = 'black', linewidth=1.2, zorder=0)

        ax_acq.plot_trisurf(x_plot, y_plot, z_acq, cmap='plasma')

        if first_param['spacing'] == 'log':
            min, max = first_param['min'], first_param['max']
            plt.xticks(np.linspace(min,max,5), np.logspace(min,max,5))
            ax_sur.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
            ax_acq.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

        ax_sur.set_zlim(0, loss_max)
        ax_acq.set_zlim(0, self.highest_EI)

        # Labels
        ax_sur.set_title('Hyper-parameter loss function for {} {}'.format(
            self.model_name, label), pad=30, fontsize=12)
        ax_acq.set_title('Hyper-parameter expected improvement function for {} {}'.format(
            self.model_name, label), pad=30, fontsize=12)

        ax_sur.set_xlabel(self.hyperparam_names[0], fontsize=10)
        ax_sur.set_ylabel(self.hyperparam_names[1], fontsize=10)
        ax_sur.set_zlabel('loss', fontsize=10)
        ax_acq.set_xlabel(self.hyperparam_names[0], fontsize=10)
        ax_acq.set_ylabel(self.hyperparam_names[1], fontsize=10)
        ax_acq.set_zlabel('Expected Improvement', fontsize=10)

        # Give the axis labels a bit more space
        ax_sur.xaxis.labelpad, ax_sur.yaxis.labelpad, ax_sur.zaxis.labelpad = 10, 10, 10
        ax_acq.xaxis.labelpad, ax_acq.yaxis.labelpad, ax_acq.zaxis.labelpad = 10, 10, 10

        # Alter font sizes
        plt.rcParams.update({'font.size': 14})
        ax_sur.tick_params(axis='both', which='major', labelsize=10)
        ax_sur.tick_params(axis='both', which='minor', labelsize=8)
        ax_acq.tick_params(axis='both', which='major', labelsize=10)
        ax_acq.tick_params(axis='both', which='minor', labelsize=8)

        plt.tight_layout()

        if store_plot:
            plt.savefig('.gif_images/{}_it_{}.png'.format(self.model_name, len(self.hyperparams_obj)))
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
            
    def generate_gif(self):
        """
        Function to generate a gif based on the loss plots of the model
        """
        png_dir = '.gif_images/'
        images = []
        
        for file_name in sorted(os.listdir(png_dir)):
            if file_name.startswith(self.model_name):
                file_path = os.path.join(png_dir, file_name)
                images.append(io.imread(file_path))
                gif_filename = '{}{}.gif'.format(png_dir, self.model_name)
                io.mimsave(gif_filename, images, duration=.5)
                
        return Image(filename=gif_filename)
    
    def plot_loss(self):
        """
        Plots the minimum loss over 30 iterations
        """
        # List for the lowest loss score observed per iteration
        min_points = []

        # Appends the lowest loss from the initial 10 samples
        min_points.append(np.min(self.hyperparams_obj[:10]['loss']))

        # Iteratively appends the lowest loss
        for iteration in range(len(self.hyperparams_obj[10:])):
            min_points.append(np.min(self.hyperparams_obj[:iteration+11]['loss']))
        
        # Lowest loss score overall
        min_loss = np.min(min_points)
        
        # Show plot of minimum loss over the iterations
        fig, ax = plt.subplots(figsize=(20,6))

        textstr = 'minimum loss = {:.5f}'.format(min_loss)

        plt.plot(range(0, len(min_points)), min_points, '--', range(0, len(min_points)), min_points, 'bo')
        plt.title('Minimum loss over 30 iterations for {}'.format(self.model_name))
        plt.xlabel('Iterations')
        plt.ylabel('Loss')

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        textstr = 'minimum loss = {:.5f}'.format(min_loss)

        # place a text box in upper right in axes coords
        ax.text(0.80, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props) 

        plt.show()


# ### 10 random samples

# #### SVM Classification

# In[ ]:


fig, axs = plt.subplots(2, 2)

np.logspace(settings['min'], settings['max'], size)

def initialize:
    print()
    


def update(n): 
    ax.scatter(wall_points[:,0],wall_points[:,1], color='k', s=1)
plt.close(fig)


# In[ ]:


anim = FuncAnimation(fig, update, frames=np.arange(0, 30), interval=100)
HTML(anim.to_jshtml())


# In[13]:


hyperparam_space = {
  'C': np.logspace(-12, 12, num=1000, base=2.0),
  'gamma': np.logspace(-12, 12, num=1000, base=2.0)
}

print(hyperparam_space['C'][0:10])

svm = BayesianOptimization(
  surrogate_model=ProbRandomForestRegressor(n_estimators=100, n_jobs=-1),
  objective_model=SVC,
  acquisition_function=EI,
  hyperparam_space=hyperparam_space,
  X_obj=X,
  y_obj=y
)

svm.hyperparam_sampling(n_samples=10)
svm.hyperparam_cartesian(size=40)

svm.

()
svm.surrogate_predict()

svm.plot_surrogate('(10 Samples)', store_plot=False);


# In[32]:


hyperparam_space = {
  'C': np.logspace(-12, 12, num=1000),
  'gamma': np.logspace(-12, 12, num=1000)
}            

metadata = pd.DataFrame(columns=[*hyperparam_space.keys(), 'loss', 'pred', 'sigma', 'EI'])
for hp in hyperparam_space.keys():
    metadata[hp] = np.random.choice(hyperparam_space[hp],size=10)


# In[53]:


metadata[hyperparam_space.keys()].iloc[0]


# In[55]:


for i in range(metadata.shape[0]):
    clf = SVC(**metadata[hyperparam_space.keys()].iloc[i])
    scores = cross_val_score(clf, X, y, cv=3, n_jobs=-1, scoring="accuracy")
    metadata.at[i, 'loss'] = np.mean(1 - np.array(scores))

metadata


# In[ ]:


sample = points[np.random.choice(1000, n_samples)]


# The black bars indicate the hyperparameter configurations where the objective model was tested.

# #### XG Boost Classification

# In[8]:


xgclas_param_template = {
  'learning_rate': { 'spacing': 'log', 'min': -4, 'max': -1, 'discrete': False },
  'max_depth': { 'spacing': 'lin', 'min': 1, 'max': 6, 'discrete': True }
}

xgclas = BayesianOptimization(
  surrogate_model=ProbRandomForestRegressor(n_estimators=100, n_jobs=-1),
  objective_model=XGBClassifier,
  acquisition_function=EI,
  hyperparam_names=('learning_rate', 'max_depth'),
  hyperparam_space=xgclas_param_template,
  X_obj=X,
  y_obj=y
)

xgclas.hyperparam_sampling(n_samples=10)
xgclas.hyperparam_cartesian(size=40)

xgclas.classifier_predict(n_estimators=1000, n_jobs=-1)
xgclas.surrogate_predict()

xgclas.plot_surrogate(label='(10 Samples)', store_plot=True)


# #### XG Boost Regression

# In[9]:


xgreg_param_template = {
  'learning_rate': { 'spacing': 'log', 'min': -4, 'max': -1, 'discrete': False },
  'max_depth': { 'spacing': 'lin', 'min': 1, 'max': 6, 'discrete': True }
}

xgreg = BayesianOptimization(
  surrogate_model=ProbRandomForestRegressor(n_estimators=100, n_jobs=-1),
  objective_model=XGBRegressor,
    
  acquisition_function=EI,
  hyperparam_names=('learning_rate', 'max_depth'),
  hyperparam_space=xgreg_param_template,
  X_obj=Xr,
  y_obj=yr
)

xgreg.hyperparam_sampling(n_samples=10)
xgreg.hyperparam_cartesian(size=40)

xgreg.regressor_predict(n_estimators=1000, n_jobs=-1)
xgreg.surrogate_predict()

xgreg.plot_surrogate(label='(10 Samples)', store_plot=True)


# #### ElasticNet Regression

# In[10]:


elas_param_template = {
  'alpha': { 'spacing': 'log', 'min': -12, 'max': 12, 'discrete': False },
  'l1_ratio': { 'spacing': 'lin', 'min': 1, 'max': 2, 'discrete': False }
}

elas = BayesianOptimization(
  surrogate_model=ProbRandomForestRegressor(n_estimators=100, n_jobs=-1),
  objective_model=ElasticNet,
  acquisition_function=EI,
  hyperparam_names=('alpha', 'l1_ratio'),
  hyperparam_space=elas_param_template,
  X_obj=Xr,
  y_obj=yr
)

elas.hyperparam_sampling(n_samples=10)
elas.hyperparam_cartesian(size=40)

elas.regressor_predict()
elas.surrogate_predict()

elas.plot_surrogate(label='(10 Samples)', store_plot=True)


# Visualize 3 more iterations, each time visualizing the surrogate model and acquisition function

# ### Visualize 3 iterations

# #### SVM Classification

# In[11]:


svm.next_sample()

for idx in range(3):
    svm.classifier_predict()
    svm.surrogate_predict()
    svm.plot_surrogate('(Iteration {})'.format(idx+2), store_plot=True)
    svm.next_sample()


# #### XG Boost Classification

# In[12]:


xgclas.next_sample()

for idx in range(3):
    xgclas.classifier_predict(n_estimators=1000, n_jobs=-1)
    xgclas.surrogate_predict()
    xgclas.plot_surrogate('(Iteration {})'.format(idx+2), store_plot=True)
    xgclas.next_sample()


# #### XG Boost Regression

# In[13]:


xgreg.next_sample()

for idx in range(3):
    xgreg.regressor_predict(n_estimators=1000, n_jobs=-1)
    xgreg.surrogate_predict()
    xgreg.plot_surrogate('(Iteration {})'.format(idx+2), store_plot=True)
    xgreg.next_sample()


# #### ElasticNet Regression

# In[14]:


elas.next_sample()

for idx in range(3):
    elas.regressor_predict()
    elas.surrogate_predict()
    elas.plot_surrogate('(Iteration {})'.format(idx+2), store_plot=True)
    elas.next_sample()


# ### 30 Iterations

# #### SVM Classification

# In[15]:


for idx in range(27):
    svm.classifier_predict()
    svm.surrogate_predict()
    svm.next_sample()
    svm.plot_surrogate('(Iteration {})'.format(idx+5), show_plot=False, store_plot=True)

svm.generate_gif()


# In[16]:


svm.plot_loss()


# #### SVM optimal hyperparameters

# In[17]:


svm.optimal_hyperparams()


# #### XG Boost Classification

# In[18]:


for idx in range(27):
    xgclas.classifier_predict(n_estimators=1000, n_jobs=-1)
    xgclas.surrogate_predict()
    xgclas.next_sample()
    xgclas.plot_surrogate('(Iteration {})'.format(idx+5), show_plot=False, store_plot=True)
    
xgclas.generate_gif()


# In[19]:


xgclas.plot_loss()


# #### XG Boost Classification optimal hyperparameters

# In[20]:


xgclas.optimal_hyperparams()


# #### XG Boost Regression

# In[21]:


for idx in range(27):
    xgreg.regressor_predict(n_estimators=1000, n_jobs=-1)
    xgreg.surrogate_predict()
    xgreg.next_sample()
    xgreg.plot_surrogate('(Iteration {})'.format(idx+5), show_plot=False, store_plot=True)
    
xgreg.generate_gif()


# In[22]:


xgreg.plot_loss()


# #### XG Boost Regression optimal hyperparameters

# In[23]:


xgreg.optimal_hyperparams()


# #### ElasticNet Regression

# In[24]:


for idx in range(27):
    elas.regressor_predict()
    elas.surrogate_predict()
    elas.next_sample()
    elas.plot_surrogate('(Iteration {})'.format(idx+5), show_plot=False, store_plot=True)
    
elas.generate_gif()


# In[25]:


elas.plot_loss()


# #### ElasticNet optimal hyperparameters

# In[26]:


elas.optimal_hyperparams()


# ### Q1: Discussion & Interpretation 
# 
# Before optimizing the objective model the loss surface is unknown. The real loss surface, in our case, is the same as running a 40x40 grid search over all hyperparameter configurations. However, the computations will be very expensive to run. It order to save on computation cost we apply Bayesian optimization. Bayesian optimization prevents you from searching in locations were the loss is expected to be bad and will probably not improve. This is in stark contrast to grid search, where you need to check every possible hyperparameter conbination. The difference with random search is that random search does not place a prior on the loss function.
# 
# The more complex the loss surface, the longer it will take the Bayesian optimization to find the best hyperparameters. It will need to conduct more exploration to find possible improvements, before it can exploit. As we can see in the surrogate-loss graphs SVM has a more complex loss function than the other models, this results in the optimizer finding better hyperparameter combinations as it explores and exploits. (You can see this effect in the iterations-loss graph)
# 
# Especially for simple loss surfaces an optimal hyperparameter setting can be found among the first 10 random samples. Imagine a flat surface (hyperparameters have no influence on the loss), after one random sample you will have found the optimal hyperparameters (you do not need to explore). For the other models we observe a simpler surface then for SVM and therefore there is not much change in the minimum loss (After the initial 10 samples). 
# 
# The efficiency of Bayesian optimization will become more apparent when increasing the amount of hyperparameters to tune. Adding a parameter to your grid search will add an entire additional dimension to the search space. However, with Bayesian optimization you might only need to run it for a few more iterations, depending on the interaction between hyperparameters. 
# 
# The efficiency of Bayesian optimization is also better when you have a model that needs a lot of tuning to achieve better results. As you can see all models except SVM already perform good out of the box. This means our optimization will not improve our models much. 
# 
# Our conclusion is that Bayesian optimization is not that effective in our situation, since for almost every model the optimal hyperparameters were found in the initial 10 random samples. However, when increasing the amount of hyperparameters to tune, Bayesian optimization will become more efficient.
# 
# (Bayesian optimization works well for both classification and regression, resulting in us not noticing any big differences between the two datasets.)

# #### Note
# We decided not to plot the uncertainty interval as it made our plots convoluted and a bit difficult to grasp.

# --------------------

# ## 2. Warm-starting Bayesian Optimization (20 points) {-}
# 
# * Oh no! 6 of the sensors in the first dataset (robot navigation) suddenly broke. You need to quickly retrain the model but
# there is no time for a complete re-optimization.
# * Revisit question 1, but additionally keep a list of the 10 best hyperparameter configurations while you run Bayesian optimization.
# * Randomly remove 6 columns from the dataset (or remove them manually as long as they are not adjacent) to simulate the broken sensors.
# * Re-run the Bayesian optimization (only for the first dataset), but now start from the 10 best configurations (for each classifier) rather than 10
# random ones.
# * Visualize the surrogate model (as before) at the initial state, and at 3 subsequent iterations.
# * Interpret and discuss the results. Did the warm-start help? Could you find a good model after a few iterations? 
# Explain the benefits of this approach over starting from scratch or using a random search.

# ### 10 best hyperparameter configurations
# #### SVM Classification

# In[27]:


# get 10 best hyperparameter configurations from question 1
svm_best_params = svm.optimal_hyperparams(rows=10)
svm_best_params.reset_index(drop=True, inplace=True)

# set loss to none
svm_best_params['loss'] = None
svm_best_params


# ##### XG Boost Classification

# In[28]:


# get 10 best hyperparameter configurations from question 1
xgclas_best_params = xgclas.optimal_hyperparams(rows=10)
xgclas_best_params.reset_index(drop=True, inplace=True)

# set loss to none
xgclas_best_params['loss'] = None
xgclas_best_params


# #### Randomly remove 6 columns from the dataset

# In[29]:


# Set seed for reproducability
np.random.seed(0)

# Randomly remove 6 columns, i.e., randomly pick 18 columns to keep 
kept_columns = np.random.choice(23, 18, replace=False)
X_broken = X[:, kept_columns]
X_broken.shape


# #### Re-run the Bayesian optimization, starting from the 10 best configurations and visualize surrogate model at initial state
# ##### SVM Classification

# In[30]:


svm_broken = BayesianOptimization(
    surrogate_model=ProbRandomForestRegressor(n_estimators=100, n_jobs=-1),
    objective_model=SVC,
    acquisition_function=EI,
    hyperparam_names=('C', 'gamma'),
    hyperparam_space=svm_param_template,
    X_obj=X_broken,
    y_obj=y
)

svm_broken.hyperparams_obj = svm_best_params
svm_broken.hyperparam_cartesian(size=30)

svm_broken.classifier_predict()
svm_broken.surrogate_predict()

svm_broken.plot_surrogate('(10 Samples)')


# ##### XG Boost Classification

# In[31]:


xgclas_broken = BayesianOptimization(
    surrogate_model=ProbRandomForestRegressor(n_estimators=100, n_jobs=-1),
    objective_model=XGBClassifier,
    acquisition_function=EI,
    hyperparam_names=('learning_rate', 'max_depth'),
    hyperparam_space=xgclas_param_template,
    X_obj=X_broken,
    y_obj=y
)

xgclas_broken.hyperparams_obj = xgclas_best_params
xgclas_broken.hyperparam_cartesian(size=30)

xgclas_broken.classifier_predict()
xgclas_broken.surrogate_predict()

xgclas_broken.plot_surrogate(label='(10 Samples)')


# #### Visualize the surrogate model at 3 subsequent iterations
# ##### SVM Classification

# In[32]:


svm_broken.next_sample()

for idx in range(3):
    svm_broken.classifier_predict()
    svm_broken.surrogate_predict()
    svm_broken.plot_surrogate('(Iteration {})'.format(idx+1))
    svm_broken.next_sample()


# #### SVM warm-start optimal hyperparameters

# In[33]:


svm_broken.optimal_hyperparams()


# ##### XG Boost Classification

# In[34]:


xgclas_broken.next_sample()

for idx in range(3):
    xgclas_broken.classifier_predict()
    xgclas_broken.surrogate_predict()
    xgclas_broken.plot_surrogate('(Iteration {})'.format(idx+1))
    xgclas_broken.next_sample()


# #### XG Boost classification warm-start optimal hyperparameters

# In[35]:


xgclas_broken.optimal_hyperparams()


# ### Q2: Discussion & Interpretation
# 
# The warm start did help with finding a good model in few iterations. From the plots of the XG Boost Classification model, you can see that expected improvement dropped rapidly after only a few iterations, with a low loss and expected improvement after the third iteration. From the plots of the SVM model, we do not observe this phenomenon. We expect that the surrogate model is still adjusting to the new data with six broken columns for the first few iterations. Hence, uncertainty is high and therefore we can see a high expected improvement.  
# 
# The benefits of a warm-start approach over starting from scratch or using a random search are related to the concepts of exploitation versus exploration. Acquisition functions trade off exploitation and exploration, where exploitation means sampling where the surrogate model predicts a low objective loss and exploration means sampling at locations where uncertainty is high. Better initial hyperparameter values, i.e., warm start, encourage Bayesian optimization to prevent exploration and focus on exploitation, hereby finding a good model after fewer iterations.

# ## 3. Gaussian Processes (20 points) {-}
# * Replace the probabilistic Random Forest used above with a Gaussian Process.
# * Repeat the Bayesian Optimization for one of the datasets, again visualizing the surrogate model and the acquisition function.
# * If the surrogate models do not look right, do manual tuning
# - Hint: Try `y_normalize`, regularizing the `alpha` hyperparameter, or trying a different kernel.
# * Interpret and discuss the results. In which ways are the Gaussian Processes better or worse? Consider both accuracy, speed of finding a good configuration, and runtime. Interpret and explain the results as well as you can.

# ### Guassian Process with XGBRegressor

# #### Visualize 10 random samples

# In[36]:


gpr = GaussianProcessRegressor(normalize_y=False, alpha = 1e-3, n_restarts_optimizer=9)

xgreg_gp = BayesianOptimization(
  surrogate_model=gpr,
  objective_model=XGBRegressor,
  acquisition_function=EI,
  hyperparam_names=('learning_rate', 'max_depth'),
  hyperparam_space=xgreg_param_template,
  X_obj=Xr,
  y_obj=yr
)

xgreg_gp.hyperparam_sampling(n_samples=10)
xgreg_gp.hyperparam_cartesian(size=30)

xgreg_gp.regressor_predict(n_estimators=1000, n_jobs=-1)
xgreg_gp.surrogate_predict()

xgreg_gp.plot_surrogate(label='(10 Samples)', store_plot=True)


# ### Visualize 3 iterations

# In[37]:


xgreg_gp.next_sample()

for idx in range(3):
    xgreg_gp.regressor_predict(n_estimators=1000, n_jobs=-1)
    xgreg_gp.surrogate_predict()
    xgreg_gp.plot_surrogate('(Iteration {})'.format(idx+2), store_plot=True)
    xgreg_gp.next_sample()


# ### Visualize 30 iterations with a GIF

# In[38]:


for idx in range(27):
    xgreg_gp.regressor_predict(n_estimators=1000, n_jobs=-1)
    xgreg_gp.surrogate_predict()
    xgreg_gp.next_sample()
    xgreg_gp.plot_surrogate('(Iteration {})'.format(idx+5), show_plot=False, store_plot=True)
    
xgreg_gp.generate_gif()


# #### Plot the minimum loss over 30 iterations with XGBRegressor as objective model

# In[39]:


xgreg.plot_loss()
xgreg_gp.plot_loss()


# #### Runtime comparison with different surrogate models for XGBRegressor as objective model

# In[40]:


def compare_time_sur(model1, model2):
    """
    Compares running time of different surrogate models
    """
    plt.plot(range(1, len(model1.time_surrogate)+1), model1.time_surrogate, label=type(model1.surrogate_model).__name__)
    plt.plot(range(1, len(model2.time_surrogate)+1), model2.time_surrogate, label=type(model2.surrogate_model).__name__)
    plt.title('Comparison surrogate model runtime of {} vs {} for objective model {}'.
              format(type(model1.surrogate_model).__name__, 
                     type(model2.surrogate_model).__name__,
                     model1.model_name),
             fontsize=10)
    plt.xlabel('iteration nr')
    plt.ylabel('runtime (s)')
    plt.legend(fontsize=10)
    return plt.show()


# In[52]:


compare_time_sur(xgreg, xgreg_gp)


# #### Compare optimal hyperparameters for XGBRegressor with GaussianProcessRegressor vs ProbRandomForestRegressor

# In[42]:


print('Optimal hyperparameters for {} with {} as surrogate'.format(xgreg.model_name, xgreg.surrogate_model_name))
xgreg.optimal_hyperparams()


# In[43]:


print('Optimal hyperparameters for {} with {} as surrogate'.format(xgreg_gp.model_name, xgreg_gp.surrogate_model_name))
xgreg_gp.optimal_hyperparams()


# ### Guassian Process with ElasticNet Regression

# ### Visualize 10 random samples

# In[44]:


gpr = GaussianProcessRegressor(normalize_y=False, alpha = 1e-3, n_restarts_optimizer=9)

elas_gpr = BayesianOptimization(
  surrogate_model=gpr,
  objective_model=ElasticNet,
  acquisition_function=EI,
  hyperparam_names=('alpha', 'l1_ratio'),
  hyperparam_space=elas_param_template,
  X_obj=Xr,
  y_obj=yr
)

elas_gpr.hyperparam_sampling(n_samples=10)
elas_gpr.hyperparam_cartesian(size=30)

elas_gpr.regressor_predict()
elas_gpr.surrogate_predict()

elas_gpr.plot_surrogate(label='(10 Samples)', store_plot = True)


# ### Visualize 3 iterations

# In[45]:


elas_gpr.next_sample()

for idx in range(3):
    elas_gpr.regressor_predict()
    elas_gpr.surrogate_predict()
    elas_gpr.plot_surrogate('(Iteration {})'.format(idx+2), store_plot=True)
    elas_gpr.next_sample()


# ### Visualize 30 iterations with GIF

# In[46]:


for idx in range(27):
    elas_gpr.regressor_predict()
    elas_gpr.surrogate_predict()
    elas_gpr.next_sample()
    elas_gpr.plot_surrogate('(Iteration {})'.format(idx+5), show_plot=False, store_plot=True)
    
elas_gpr.generate_gif()


# #### Compare loss xgboost regression RFR & GP

# In[47]:


elas.plot_loss()
elas_gpr.plot_loss()


# #### Compare times xgboost regression RFR & GP

# In[48]:


compare_time_sur(elas, elas_gpr)


# #### Compare optimal hyperparameters for ElasticNet with GaussianProcessRegressor vs ProbRandomForestRegressor

# In[49]:


print('Optimal hyperparameters for {} with {} as surrogate'.format(elas.model_name, elas.surrogate_model_name))
elas.optimal_hyperparams()


# In[50]:


print('Optimal hyperparameters for {} with {} as surrogate'.format(elas_gpr.model_name, elas_gpr.surrogate_model_name))
elas_gpr.optimal_hyperparams()


# ### Q3: Discussion & Interpretation
# 
# * Tuning the GaussianProcessRegressor:
#     - __y-norm__: Did not visibly affect the surface or performance of the surrogate model in this particular situation. Scikit learn suggests to set y-norm to True if the target values’ mean is expected to differ considerable from zero.
#     - __alpha__: Alpha is the regularization parameter and had clear effect on performance of the surrogate model. For the XGBRegressor the default value of alpha (1e-10) led to an error in the code. The Scikit learn documentation suggests this could be the result of numerical issues during the fitting process. With alpha = 1e-3, the surface of the surrogate model behaved as expected.
#     - __n_restarts_optimizer__: Again, did not seem to have an effect on the performance of the surrogate model in this particular situation.
#     - __kernel__: Highly affected the performance of the surrogate model. Default kernel was used this situation as it resulted in a nice surface.
#     
# 
# * First of all, it immediately stands out that the different surrogate models behave quite differently. This is particularly true in the first scenario where XGBRegressor is the objective model. Here, the ProbRandomForestRegressor (PRFR) results in a blocky, slide-looking surface for which the max_depth hyperparameter seems to have little influence on the loss. In contrast, the GaussianProcessRegressor (GPR) is a smooth, wavy-looking surface where both hyperparameters clearly affect the loss. Despite the different surfaces, both surrogate models result in similar values for the optimal hyperparameters. Hence the type of surrogate model does not seem to be an important factor for overall accuracy for this dataset. 
# 
# * We can draw a similar conclusion looking at the plot that tracks the minimum observed loss over all the iterations of the bayesian optimization process. While there might be a minor improvement over the 30 iterations, generally all surrogate models tend to find a near optimal configuration with just 10 initial random samples. Note the small scale for the y-axis of the minimum loss graphs, this makes an improvement look deceptively large even though it only represents a small change. 
# 
# * Lastly, there is only a marginal difference in the runtime between PRFR and GPR as can be seen in the plots above. The runtime reflects the time it takes each iteration to fit the surrogate model and to make predictions on the given sample. Particularly in contrast to the running times of the objective models, both surrogate models are much more efficient in locating an ideal hyperparameter configuration. 
# 
# * Overall we can conclude that PRFR and GPR are equally as good in terms of surrogate models. The only noticable distinction between the two is that GPR requires more manual tuning and has smoother output surfaces. 
