#!/usr/bin/env python
# coding: utf-8

# # Lab 7: Neural networks 
# 
# In this lab we will build dense neural networks on the MNIST dataset.
# 
# Make sure you read the tutorial for this lab first.

# ## Load the data and create train-test splits

# In[28]:


# Global imports and settings
from preamble import *
import tensorflow.keras as keras
print("Using Keras",keras.__version__)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[29]:


# Download MNIST data. Takes a while the first time.
mnist = oml.datasets.get_dataset(554)
X, y, _, _ = mnist.get_data(target=mnist.default_target_attribute, dataset_format='array');
X = X.reshape(70000, 28, 28)

# Take some random examples
from random import randint
fig, axes = plt.subplots(1, 5,  figsize=(10, 5))
for i in range(5):
    n = randint(0,70000)
    axes[i].imshow(X[n], cmap=plt.cm.gray_r)
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    axes[i].set_xlabel("{}".format(y[n]))
plt.show();


# In[30]:


# For MNIST, there exists a predefined stratified train-test split of 60000-10000. We therefore don't shuffle or stratify here.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=60000, random_state=0)


# ## Exercise 1: Preprocessing
# * Normalize the data: map each feature value from its current representation (an integer between 0 and 255) to a floating-point value between 0 and 1.0. 
# * Store the floating-point values in `x_train_normalized` and `x_test_normalized`.
# * Map the class label to a on-hot-encoded value. Store in `y_train_encoded` and `y_test_encoded`.

# ## Exercise 2: Create a deep neural net model
# 
# Implement a `create_model` function which defines the topography of the deep neural net, specifying the following:
# 
# * The number of layers in the deep neural net: Use 2 dense layers for now.
# * The number of nodes in each layer: these are parameters of your function.
# * Any regularization layers. Add at least one dropout layer.
# * The optimizer and learning rate. Make the learning rate a parameter of your function as well.
# 
# Consider:
# * What should be the shape of the input layer?
# * Which activation function you will need for the last layer, since this is a 10-class classification problem?

# In[54]:


### Create and compile a 'deep' neural net
def create_model(layer_1_units=32, layer_2_units=10, learning_rate=0.001, dropout_rate=0.3):
    pass


# ## Exercise 3: Create a training function
# Implement a `train_model` function which trains and evaluates a given model.
# It should do a train-validation split and report the train and validation loss and accuracy, and return the training history.

# In[40]:


def train_model(model, X, y, validation_split=0.1, epochs=10, batch_size=None):
    """
    model: the model to train
    X, y: the training data and labels
    validation_split: the percentage of data set aside for the validation set
    epochs: the number of epochs to train for
    batch_size: the batch size for minibatch SGD
    """
    pass


# ## Exercise 4: Evaluate the model
# 
# Train the model with a learning rate of 0.003, 50 epochs, batch size 4000, and a validation set that is 20% of the total training data.
# Use default settings otherwise. Plot the learning curve of the loss, validation loss, accuracy, and validation accuracy. Finally, report the performance on the test set.
# 
# Feel free to use the plotting function below, or implement the callback from the tutorial to see results in real time.

# In[65]:


# Helper plotting function
#
# history: the history object returned by the fit function
# list_of_metrics: the metrics to plot
def plot_curve(history, list_of_metrics):
    
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")

    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    for m in list_of_metrics:
        x = hist[m]
        plt.plot(epochs[1:], x[1:], label=m, lw=2)

    plt.legend()


# ## Exercise 5: Optimize the model
# 
# Try to optimize the model, either manually or with a tuning method. At least optimize the following:
# * the number of hidden layers 
# * the number of nodes in each layer
# * the amount of dropout layers and the dropout rate
# 
# Try to reach at least 96% accuracy against the test set.
