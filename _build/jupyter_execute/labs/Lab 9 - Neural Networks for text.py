#!/usr/bin/env python
# coding: utf-8

# # Lab 9: Neural Networks for text

# In[7]:


# Global imports and settings
from preamble import *
import tensorflow.keras as keras
print("Using Keras",keras.__version__)
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.dpi'] = 125 # Use 300 for PDF, 100 for slides
# InteractiveShell.ast_node_interactivity = "all"
HTML('''<style>html, body{overflow-y: visible !important} .CodeMirror{min-width:105% !important;} .rise-enabled .CodeMirror, .rise-enabled .output_subarea{font-size:140%; line-height:1.2; overflow: visible;} .output_subarea pre{width:110%}</style>''') # For slides


# Before you start, read the Tutorial for this lab ('Deep Learning with Python')

# ## Exercise 1: Sentiment Analysis
# * Take the IMDB dataset from keras.datasets with 10000 words and the default train-test-split

# In[8]:


from tensorflow.keras.datasets import imdb
# Download IMDB data with 10000 most frequent words
word_index = imdb.get_word_index()
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

for i in [0,5,10]:
    print("Review {}:".format(i),' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[i]][0:20]))


# * Vectorize the reviews using one-hot-encoding (see tutorial for helper code) 

# * Build a network of 2 _Dense_ layers with 16 nodes each and the _ReLU_ activation function.
# * Use cross-entropy as the loss function, RMSprop as the optimizer, and accuracy as the evaluation metric.

# * Plot the learning curves, using the first 10000 samples as the validation set and the rest as the training set.
# * Use 20 epochs and a batch size of 512

# * Retrain the model, this time using early stopping to stop training at the optimal time
# * Evaluate on the test set and report the accuracy

# * Try to manually improve the score and explain what you observe. E.g. you could:
#     - Try 3 hidden layers
#     - Change to a higher learning rate (e.g. 0.4)
#     - Try another optimizer (e.g. Adagrad)
#     - Use more or fewer hidden units (e.g. 64)
#     - `tanh` activation instead of `ReLU`

# * Tune the results by doing a grid search for the most interesting hyperparameters
#     * Tune the learning rate between 0.001 and 1
#     * Tune the number of epochs between 1 and 20
#     * Use only 3-4 values for each

# ## Exercise 2: Topic classification
# * Take the Reuters dataset from keras.datasets with 10000 words and the default train-test-split

# In[64]:


from keras.datasets import reuters

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

for i in [0,5,10]:
    print("News wire {}:".format(i),
          ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[i]]))
    # Note that our indices were offset by 3


# * Vectorize the data and the labels using one-hot-encoding

# * Build a network with 2 dense layers of 64 nodes each
# * Make sensible choices about the activation functions, loss, ...

# * Take a validation set from the first 1000 points of the training set
# * Fit the model with 20 epochs and a batch size of 512
# * Plot the learning curves

# * Create an information bottleneck: rebuild the model, but now use only 4 hidden units in the second layer. Evaluate the model. Does it still perform well?

# ## Exercise 3: Regularization
# * Go back to the IMDB dataset
# * Retrain with only 4 units per layer
# * Plot the results. What do you observe?

# * Use 16 hidden nodes in the layers again, but now add weight regularization. Use L2 loss with alpha=0.001. What do you observe?

# * Add a drop out layer after every dense layer. Use a dropout rate of 0.5. What do you observe?

# ## Exercise 4: Word embeddings
# 
# * Instead of one-hot-encoding, use a word embedding of length 300
# * Only add an output layer after the Embedding layer.
# * Train the embedding as well as you can (takes time!)
#     * Evaluate as before. Does it perform better?
# * Import a GloVe embedding pretrained om Wikipedia
#     * Evaluate as before. Does it perform better?
