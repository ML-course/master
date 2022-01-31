#!/usr/bin/env python
# coding: utf-8

# In[2]:


from IPython.display import HTML
HTML('''<style>html, body{overflow-y: visible !important} .CodeMirror{min-width:105% !important;} .rise-enabled .CodeMirror, .rise-enabled .output_subarea{font-size:140%; line-height:1.2; overflow: visible;} .output_subarea pre{width:110%}</style>''') # For slides


# # Python for data analysis
# For those who are new to using Python for scientific work, we first provide a short introduction to Python and the most useful packages for data analysis.

# ## Python
# Disclaimer: We can only cover some of the basics here. If you are completely new to Python, we recommend to take an introductory online course, such as the [Definite Guide to Python](https://www.programiz.com/python-programming), or the [Whirlwind Tour of Python](https://github.com/jakevdp/WhirlwindTourOfPython). If you like a step-by-step approach, try the [DataCamp Intro to Python for Data Science](https://www.datacamp.com/courses/intro-to-python-for-data-science).
# 
# To practice your skills, try the [Hackerrank challenges](https://www.hackerrank.com/domains/python).

# ### Hello world
# * Printing is done with the print() function. 
# * Everything after # is considered a comment. 
# * You don't need to end commands with ';'.  

# In[3]:


# This is a comment
print("Hello world")
print(5 / 8)
5/8 # This only prints in IPython notebooks and shells.


# _Note: In these notebooks we'll use Python interactively to avoid having to type print() every time._

# ### Basic data types
# Python has all the [basic data types and operations](https://docs.python.org/3/library/stdtypes.htm): int, float, str, bool, None.  
# Variables are __dynamically typed__: you need to give them a value upon creation, and they will have the data type of that value. If you redeclare the same variable, if will have the data type of the new value.  
# You can use type() to get a variable's type. 

# In[4]:


s = 5
type(s)
s > 3 # Booleans: True or False
s = "The answer is "
type(s)


# Python is also __strongly typed__: it won't implicitly change a data type, but throw a TypeError instead. You will have to convert data types explictly, e.g. using str() or int().  
# Exception: Arithmetic operations will convert to the _most general_ type.

# In[5]:


1.0 + 2     # float + int -> float
s + str(42) # string + string
# s + 42    # Bad: string + int


# ### Complex types
# The main complex data types are lists, tuples, sets, and dictionaries (dicts).

# In[6]:


l = [1,2,3,4,5,6]       # list
t = (1,2,3,4,5,6)       # tuple: like a list, but immutable
s = set((1,2,3,4,5,6))  # set: unordered, you need to use add() to add new elements 
d = {2: "a",            # dict: has key - value pairs
     3: "b",
     "foo": "c",
     "bar": "d"}

l  # Note how each of these is printed
t
s 
d


# You can use indices to return a value (except for sets, they are unordered)

# In[7]:


l
l[2]
t
t[2]
d
d[2]
d["foo"]


# You can assign new values to elements, except for tuples

# In[8]:


l
l[2] = 7 # Lists are mutable
l
t[2] = 7 # Tuples are not


# Python allows convenient tuple packing / unpacking

# In[ ]:


b = ("Bob", 19, "CS")      # tuple packing
(name, age, studies) = b   # tuple unpacking
name
age
studies


# ### Strings
# Strings are [quite powerful](https://www.tutorialspoint.com/python/python_strings.htm).  
# They can be used as lists, e.g. retrieve a character by index.  
# They can be formatted with the format operator (%), e.g. %s for strings, %d for decimal integers, %f for floats.

# In[ ]:


s = "The %s is %d" % ('answer', 42)
s
s[0]
s[4:10]
'%.2f' % (3.14159265) # defines number of decimal places in a float


# They also have a format() function for [more complex formatting](https://pyformat.info/)

# In[ ]:


l = [1,2,3,4,5,6]
"{}".format(l) 
"%s" % l       # This is identical
"{first} {last}".format(**{'first': 'Hodor', 
                           'last': 'Hodor!'})  


# ### For loops, If statements
# For-loops and if-then-else statements are written like this.  
# Indentation defines the scope, not brackets.

# In[ ]:


l = [1,2,3]
d = {"foo": "c", "bar": "d"}

for i in l:
    print(i)
    
for k, v in d.items(): # Note how key-value pairs are extracted
    print("%s : %s" % (k,v))
    
if len(l) > 3:
    print('Long list')
else:
    print('Short list')


# ### Functions
# Functions are defined and called like this:

# In[ ]:


def myfunc(a, b):
    return a + b

myfunc(2, 3)


# Function arguments (parameters) can be:
# * variable-length (indicated with \*)
# * a dictionary of keyword arguments (indicated with \*\*). 
# * given a default value, in which case they are not required (but have to come last)

# In[ ]:


def func(*argv, **kwarg):
    print("func argv: %s" % str(argv))
    print("func kwarg: %s" % str(kwarg))
    
func(2, 3, a=4, b=5)

def func(a=2):
    print(a * a)

func(3)
func()


# Functions can have any number of outputs.

# In[ ]:


def func(*argv):
    return sum(argv[0:2]), sum(argv[2:4])

sum1, sum2 = func(2, 3, 4, 5)
sum1, sum2

def squares(limit):
    r = 0
    ret = []
    
    while r < limit:
        ret.append(r**2)
        r += 1
    
    return ret

for i in squares(4):
    print(i)


# Functions can be passed as arguments to other functions

# In[ ]:


def greet(name):
    return "Hello " + name 

def call_func(func):
    other_name = "John"
    return func(other_name)  

call_func(greet)


# Functions can return other functions

# In[ ]:


def compose_greet_func():
    def get_message():
        return "Hello there!"

    return get_message

greet = compose_greet_func()
greet()


# ### Classes
# Classes are defined like this

# In[ ]:


class TestClass(object): # TestClass inherits from object.
    myvar = ""
     
    def __init__(self, myString): # optional constructor, returns nothing
        self.myvar = myString # 'self' is used to store instance properties
    
    def say(self, what): # you need to add self as the first argument 
        return self.myvar + str(what)

a = TestClass("The new answer is ")
a.myvar # You can retrieve all properties of self
a.say(42)


# Static functions need the @staticmethod decorator

# In[ ]:


class TestClass(object):
    myvar = ""
    
    def __init__(self, myString): 
        self.myvar = myString
    
    def say(self, what): # you need to add self as the first argument 
        return self.myvar + str(what)
    
    @staticmethod
    def sayStatic(what): # or declare the function static 
        return "The answer is " + str(what)

a = TestClass("The new answer is ")
a.say(42)
a.sayStatic(42)


# ### Functional Python
# You can write complex procedures in a few elegant lines of code using [built-in functions](https://docs.python.org/2/library/functions.html#map) and libraries such as functools, itertools, operator.

# In[15]:


def square(num):
    return num ** 2

# map(function, iterable) applies a given function to every element of a list
list(map(square, [1,2,3,4]))

# a lambda function is an anonymous function created on the fly
list(map(lambda x: x**2, [1,2,3,4])) 
mydata = list(map(lambda x: x if x>2 else 0, [1,2,3,4])) 
mydata

# reduce(function, iterable ) applies a function with two arguments cumulatively to every element of a list
from functools import reduce
reduce(lambda x,y: x+y, [1,2,3,4]) 
mydata


# In[10]:


# filter(function, iterable)) extracts every element for which the function returns true
list(filter(lambda x: x>2, [1,2,3,4]))

# zip([iterable,...]) returns tuples of corresponding elements of multiple lists
list(zip([1,2,3,4],[5,6,7,8,9]))


# __list comprehensions__ can create lists as follows:  
# 
#     [statement for var in iterable if condition]  
#     
# __generators__ do the same, but are lazy: they don't create the list until it is needed:  
# 
#     (statement for var in list if condition)

# In[11]:


a = [2, 3, 4, 5]

lc = [ x for x in a if x >= 4 ] # List comprehension. Square brackets
lg = ( x for x in a if x >= 4 ) # Generator. Round brackets

a.extend([6,7,8,9])

for i in lc:
    print("%i " % i, end="") # end tells the print function not to end with a newline
print("\n")
for i in lg:
    print("%i " % i, end="")


# __dict comprehensions__ are possible in Python 3:  
# 
#     {key:value for (key,value) in dict.items() if condition}  

# In[12]:


# Quick dictionary creation

numbers = range(10)
{n:n**2 for n in numbers if n%2 == 0}


# In[13]:


# Powerful alternative to replace lambda functions
# Convert Fahrenheit to Celsius
fahrenheit = {'t1': -30,'t2': 0,'t3': 32,'t4': 100}
{k:(float(5)/9)*(v-32) for (k,v) in fahrenheit.items()}

