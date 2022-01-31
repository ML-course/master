#!/usr/bin/env python
# coding: utf-8

# # Python for scientific computing
# Python has extensive packages to help with data analysis:
# 
# * numpy: matrices, linear algebra, Fourier transform, pseudorandom number generators
# * scipy: advanced linear algebra and maths, signal processing, statistics
# * pandas: DataFrames, data wrangling and analysis
# * matplotlib: visualizations such as line charts, histograms, scatter plots. 

# In[2]:


# Global imports and settings
from preamble import *
get_ipython().run_line_magic('matplotlib', 'inline')
InteractiveShell.ast_node_interactivity = "all"
HTML('''<style>html, body{overflow-y: visible !important} .CodeMirror{min-width:105% !important;} .rise-enabled .CodeMirror, .rise-enabled .output_subarea{font-size:150%; line-height:1.2; overflow: visible;} .output_subarea pre{width:100%}</style>''') # For slides


# ## NumPy
# NumPy is the fundamental package required for high performance scientific computing in Python. It provides:
# 
# * `ndarray`: fast and space-efficient n-dimensional numeric array with vectorized arithmetic operations
# * Functions for fast operations on arrays without having to write loops
# * Linear algebra, random number generation, Fourier transform
# * Integrating code written in C, C++, and Fortran (for faster operations)
# 
# `pandas` provides a richer, simpler interface to many operations. We'll focus on using `ndarrays` here because they are heavily used in scikit-learn.

# ### ndarrays
# There are several ways to create numpy arrays.

# In[3]:


# Convert normal Python array to 1-dimensional numpy array
np.array((1, 2, 53))


# In[4]:


# Convert sequences of sequences of sequences ... to n-dim array
np.array([(1.5, 2, 3), (4, 5, 6)])


# In[5]:


# Define element type at creation time
np.array([[1, 2], [3, 4]], dtype=complex)


# Useful properties of ndarrays:

# In[6]:


my_array = np.array([[1, 0, 3], [0, 1, 2]])
my_array.ndim     # number of dimensions (axes), also called the rank
my_array.shape    # a matrix with n rows and m columns has shape (n,m)
my_array.size     # the total number of elements of the array
my_array.dtype    # type of the elements in the array
my_array.itemsize # the size in bytes of each element of the array


# Quick array creation.  
# It is cheaper to create an array with placeholders than extending it later.

# In[7]:


np.ones(3) # Default type is float64
np.zeros([2, 2]) 
np.empty([2, 2]) # Fills the array with whatever sits in memory
np.random.random((2,3))
np.random.randint(5, size=(2, 4))


# Create sequences of numbers

# In[8]:


np.linspace(0, 1, num=6)   # Linearly distributed numbers between 0 and 1
np.arange(0, 1, step=0.3)  # Fixed step size
np.arange(12).reshape(3,4) # Create and reshape
np.eye(4)                  # Identity matrix


# ### Basic Operations
# Arithmetic operators on arrays apply elementwise. A new array is created and filled with the result. Some operations, such as += and *=, act in place to modify an existing array rather than create a new one.

# In[9]:


a = np.array([20, 30, 40, 50])
b = np.arange(4)
a, b    # Just printing
a-b
b**2
a > 32
a += 1
a


# The product operator * operates elementwise.  
# The matrix product can be performed using dot() 

# In[10]:


A, B = np.array([[1,1], [0,1]]), np.array([[2,0], [3,4]]) # assign multiple variables in one line
A
B
A * B
np.dot(A, B)


# Upcasting: Operations with arrays of different types choose the more general/precise one.

# In[11]:


a = np.ones(3, dtype=np.int) # initialize to integers
b = np.linspace(0, np.pi, 3) # default type is float
a.dtype, b.dtype, (a + b).dtype


# ndarrays have most unary operations (max,min,sum,...) built in

# In[12]:


a = np.random.random((2,3))
a
a.sum(), a.min(), a.max()


# By specifying the axis parameter you can apply an operation along a specified axis of an array

# In[13]:


b = np.arange(12).reshape(3,4)
b
b.sum()
b.sum(axis=0) 
b.sum(axis=1) 


# ### Universal Functions
# 
# NumPy provides familiar mathematical functions such as sin, cos, exp, sqrt, floor,... In NumPy, these are called "universal functions" (ufunc), and operate elementwise on an array, producing an array as output. 

# In[14]:


np.sin(np.arange(0, 10))


# ### Shape Manipulation
# Transpose, flatten, reshape,...

# In[15]:


a = np.floor(10*np.random.random((3,4)))
a
a.transpose()
b = a.ravel() # flatten array
b
b.reshape(3, -1) # reshape in 3 rows (and as many columns as needed)


# Arrays can be split and stacked together

# In[16]:


a = np.floor(10*np.random.random((2,6)))
a
b, c = np.hsplit(a, 2) # Idem: vsplit for vertical splits 
b
c
np.hstack((b, c)) # Idenm: vstack for vertical stacks


# ### Indexing and Slicing
# 
# Arrays can be indexed and sliced using [start:stop:stepsize]. Defaults are [0:ndim:1]

# In[17]:


a = np.arange(10)**2
a


# In[18]:


a[2]


# In[19]:


a[3:10:2]


# In[20]:


a[::-1] # Defaults are used if indices not stated


# In[21]:


a[::2]


# For multi-dimensional arrays, axes are comma-separated: [x,y,z]. 

# In[22]:


b = np.arange(16).reshape(4,4)
b
b[2,3] # row 2, column 3


# In[23]:


b[0:3,1] # Values 0 to 3 in column 1 
b[ : ,1] # The whole column 1 


# In[24]:


b[1:3, : ] # Rows 1:3, all columns


# In[25]:


# Return the last row
b[-1]   


# Note: dots (...) represent as many colons (:) as needed
# * x[1,2,...] = x[1,2,:,:,:]
# * x[...,3] = x[:,:,:,:,3]
# * x[4,...,5,:] = x[4,:,:,5,:]

# Arrays can also be indexed by arrays of integers and booleans.

# In[26]:


a = np.arange(12)**2         
i = np.array([ 1,1,3,8,5 ])
a
a[i]


# A matrix of indices returns a matrix with the corresponding values.

# In[27]:


j = np.array([[ 3, 4], [9, 7]])
a[j]


# With boolean indices we explicitly choose which items in the array we want and which ones we don't.

# In[28]:


a = np.arange(12).reshape(3,4)
a
a[np.array([False,True,True]), :]
b = a > 4
b
a[b]


# ### Iterating 
# Iterating is done with respect to the first axis:

# In[29]:


for row in b:
    print(row)


# Operations on each element can be done by flattening the array (or nested loops)

# In[30]:


for element in b.flat: # flat returns an iterator 
    print(element) 


# ### Copies and Views (or: how to shoot yourself in a foot)
# Assigning an array to another variable does NOT create a copy

# In[31]:


a = np.arange(12)
b = a
a


# In[32]:


b[0] = -100
b


# In[33]:


a


# The view() method creates a NEW array object that looks at the same data. 

# In[34]:


a = np.arange(12)
a
c = a.view()
c.resize((2, 6))
c


# In[35]:


a[0] = 123
c # c is also changed now


# Slicing an array returns a view of it.

# In[36]:


c
s = c[ : , 1:3]  
s[:] = 10
s
c 


# The copy() method makes a deep copy of the array and its data. 

# In[37]:


d = a.copy()      
d[0] = -42
d


# In[38]:


a


# ### Numpy: further reading
# 
# * Numpy Tutorial: http://wiki.scipy.org/Tentative_NumPy_Tutorial
# * "Python for Data Analysis" by Wes McKinney (O'Reilly)

# ## SciPy
# SciPy is a collection of packages for scientific computing, among others:
# 
# * scipy.integrate: numerical integration and differential equation solvers
# * scipy.linalg: linear algebra routines and matrix decompositions
# * scipy.optimize: function optimizers (minimizers) and root finding algorithms
# * scipy.signal: signal processing tools
# * scipy.sparse: sparse matrices and sparse linear system solvers
# * scipy.stats: probability distributions, statistical tests, descriptive statistics

# ### Sparse matrices
# Sparse matrices are used in scikit-learn for (large) arrays that contain mostly zeros. You can convert a dense (numpy) matrix to a sparse matrix.

# In[39]:


from scipy import sparse
eye = np.eye(4)
eye
sparse_matrix = sparse.csr_matrix(eye) # Compressed Sparse Row matrix
sparse_matrix
print("{}".format(sparse_matrix))  


# When the data is too large, you can create a sparse matrix by passing the values and coordinates (COO format).

# In[40]:


data = np.ones(4)                         # [1,1,1,1]
row_indices = col_indices = np.arange(4)  # [0,1,2,3]
col_indices = np.arange(4) * 2
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print("{}".format(eye_coo))


# ### Further reading
# Check the [SciPy reference guide](https://docs.scipy.org/doc/scipy/reference/) for tutorials and examples of all SciPy capabilities.

# ## pandas
# pandas is a Python library for data wrangling and analysis. It provides:
# 
# * ``DataFrame``: a table, similar to an R DataFrame that holds any structured data
#     * Every column can have its own data type (strings, dates, floats,...)
# * A great range of methods to apply to this table (sorting, querying, joining,...)
# * Imports data from a wide range of data formats (CSV, Excel) and databases (e.g. SQL)

# ### Series
# A one-dimensional array of data (of any numpy type), with indexed values. It can be created by passing a Python list or dict, a numpy array, a csv file,...

# In[41]:


import pandas as pd
pd.Series([1,3,np.nan]) # Default integers are integers
pd.Series([1,3,5], index=['a','b','c'])
pd.Series({'a' : 1, 'b': 2, 'c': 3 }) # when given a dict, the keys will be used for the index
pd.Series({'a' : 1, 'b': 2, 'c': 3 }, index = ['b', 'c', 'd']) # this will try to match labels with keys


# Functions like a numpy array, however with index labels as indices

# In[42]:


a = pd.Series({'a' : 1, 'b': 2, 'c': 3 })
a
a['b']       # Retrieves a value
a[['a','b']] # and can also be sliced


# numpy array operations on Series preserve the index value

# In[43]:


a
a[a > 1]
a * 2 
np.sqrt(a)


# Operations over multiple Series will align the indices

# In[44]:


a = pd.Series({'John' : 1000, 'Mary': 2000, 'Andre': 3000 })
b = pd.Series({'John' : 100, 'Andre': 200, 'Cecilia': 300 })
a + b


# ### DataFrame
# A DataFrame is a tabular data structure with both a row and a column index. It can be created by passing a dict of arrays, a csv file,...

# In[45]:


data = {'state': ['Ohio', 'Ohio', 'Nevada', 'Nevada'], 'year': [2000, 2001, 2001, 2002],
'pop': [1.5, 1.7, 2.4, 2.9]}
pd.DataFrame(data)
pd.DataFrame(data, columns=['year', 'state', 'pop', 'color']) # Will match indices  


# It can be composed with a numpy array and row and column indices, and decomposed

# In[46]:


dates = pd.date_range('20130101',periods=4)
df = pd.DataFrame(np.random.randn(4,4),index=dates,columns=list('ABCD'))
df


# In[47]:


df.index
df.columns
df.values


# DataFrames can easily read/write data from/to files
# 
# * `read_csv(source)`: load CSV data from file or url
# * `read_table(source, sep=',')`: load delimited data with separator
# * `df.to_csv(target)`: writes the DataFrame to a file

# In[48]:


df.to_csv('data.csv', index=False) # Don't export the row index
dfs = pd.read_csv('data.csv')
dfs
dfs.set_value(0, 'A', 10) # Set value in row 0, column 'A' to '10'
dfs.to_csv('data.csv', index=False)


# ### Simple operations

# In[49]:


df.head() # First 5 rows
df.tail() # Last 5 rows


# In[50]:


# Quick stats
df.describe()


# In[51]:


# Transpose
df.T


# In[52]:


df
df.sort_index(axis=1, ascending=False) # Sort by index labels
df.sort_values(by='B') # Sort by values


# ### Selecting and slicing

# In[53]:


df['A'] # Get single column by label
df.A    # Shorthand 


# In[54]:


df[0:2]          # Get rows by index number
df.iloc[0:2,0:2] # Get rows and columns by index number
df['20130102':'20130103']                # or row label  
df.loc['20130101':'20130103', ['A','B']] # or row and column label
df.ix[0:2, ['A','B']]   # allows mixing integers and labels 


# query() retrieves data matching a boolean expression

# In[55]:


df
df.query('A > -0.4') # Identical to df[df.A > 0.4]
df.query('A > B')   # Identical to df[df.A > df.B]


# Note: similar to NumPy, indexing and slicing returns a _view_ on the data. Use copy() to make a deep copy.

# ### Operations
# DataFrames offer a [wide range of operations](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html): max, mean, min, sum, std,... 

# In[56]:


df.mean()       # Mean of all values per column
df.mean(axis=1) # Other axis: means per row 


# All of numpy's universal functions also work with dataframes

# In[57]:


np.abs(df)


# Other (custom) functions can be applied with apply(funct)

# In[58]:


df
df.apply(np.max)
df.apply(lambda x: x.max() - x.min())


# Data can be aggregated with groupby()

# In[59]:


df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar'], 'B' : ['one', 'one', 'two', 'three'],
                   'C' : np.random.randn(4), 'D' : np.random.randn(4)})
df
df.groupby('A').sum()
df.groupby(['A','B']).sum()


# ### Data wrangling (some examples)
# Merge: combine two dataframes based on common keys

# In[60]:


df1 = pd.DataFrame({'key': ['b', 'b', 'a'], 'data1': range(3)}) 
df2 = pd.DataFrame({'key': ['a', 'b'], 'data2': range(2)})
df1
df2
pd.merge(df1, df2)


# Append: append one dataframe to another

# In[61]:


df = pd.DataFrame(np.random.randn(2, 4))
df
s = pd.DataFrame(np.random.randn(1,4))
s
df.append(s, ignore_index=True)


# Remove duplicates

# In[62]:


df = pd.DataFrame({'k1': ['one'] * 3, 'k2': [1, 1, 2]})
df
df.drop_duplicates()


# Replace values

# In[63]:


df = pd.DataFrame({'k1': [1, -1], 'k2': [-1, 2]}) # Say that -1 is a sentinel for missing data
df
df.replace(-1, np.nan)


# Discretization and binning

# In[64]:


ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]
cats = pd.cut(ages, bins)
cats.labels
pd.value_counts(cats)


# ### Further reading
# 
# * Pandas docs: http://pandas.pydata.org/pandas-docs/stable/
# * https://bitbucket.org/hrojas/learn-pandas
# * Python for Data Analysis (O'Reilly) by Wes McKinney (the author of pandas)          

# ## matplotlib
# [matplotlib](http://matplotlib.sourceforge.net) is the primary scientific plotting library in Python. It provides:
# 
# * Publication-quality [visualizations](http://matplotlib.org/gallery.html) such as line charts, histograms, and scatter plots.
# * Integration in pandas to make plotting much easier.
# * Interactive plotting in Jupyter notebooks for quick visualizations.
#     * Requires some setup. See preamble and [%matplotlib](http://ipython.readthedocs.io/en/stable/interactive/plotting.html?highlight=matplotlib).
# * Many GUI backends, export to PDF, SVG, JPG, PNG, BMP, GIF, etc.
# * Ecosystem of libraries for more advanced plotting, e.g. [Seaborn](http://seaborn.pydata.org/)

# ### Low-level usage
# `plot()` is the [main function](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot) to generate a plot (but many more exist):
# ```
# plot(x, y)        Plot x vs y, default settings
# plot(x, y, 'bo')  Plot x vs y, blue circle markers
# plot(y, 'r+')     Plot y (x = array 0..N-1), red plusses
# ```
# 
# Every plotting function is completely customizable through a large set of options.

# In[65]:


x = np.linspace(-10, 10, 100) # Sequence for X-axis 
y = np.sin(x) # sine values 
p = plt.plot(x, y, marker="x") # Line plot with marker x


# ### pandas + matplotlib
# pandas DataFrames offer an easier, higher-level interface for matplotlib functions

# In[66]:


df = pd.DataFrame(np.random.randn(500, 4), 
                  columns=['a', 'b', 'c', 'd']) # random 4D data
p = df.cumsum() # Plot cumulative sum of all series
p.plot();


# In[67]:


p = df[:10].plot(kind='bar') # First 10 arrays as bar plots  


# In[68]:


p = df.boxplot() # Boxplot for each of the 4 series


# In[69]:


# Scatter plot using the 4 series for x, y, color, scale 
df[:300].plot(kind='scatter', x='a', y='b', c='c', 
               s=df['d']*50, linewidth='0', cmap='plasma')


# ### Advanced plotting libraries
# Several libraries, such as [Seaborn](http://seaborn.pydata.org/examples/index.html) offer more advanced plots and easier interfaces. 
# ![Seaborn Examples](http://ksopyla.com/wp-content/uploads/2016/11/seaborn_examples.jpg)

# ### Further reading links
# 
# * [Matplotlib examples](http://matplotlib.org/gallery.html)
# * [Plotting with pandas](http://pandas.pydata.org/pandas-docs/version/0.18.1/visualization.html)
# * [Seaborn examples](http://seaborn.pydata.org/examples/index.html)
