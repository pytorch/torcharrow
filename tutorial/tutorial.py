#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# coding: utf-8

# # TorchArrow in 10 minutes
#
# TorchArrow is a torch.Tensor-like Python DataFrame library for data preprocessing in deep learning. It supports multiple execution runtimes and Arrow as a common memory format.
#
# (Remark. In case the following looks familiar, it is with gratitude that portions of this tutorial were borrowed and adapted from the 10 Minutes to Pandas (and CuDF) tutorial.)
#
#

# The TorchArrow library consists of 3 parts:
#
#   * *DTypes* define *Schema*, *Fields*, primitive and composite *Types*.
#   * *Columns* defines sequences of strongly typed data with vectorized operations.
#   * *Dataframes*  are sequences of named and typed columns of same length with relational operations.
#
# Let's get started...

# In[1]:


# ## Constructing data: Columns
#
# ### From Pandas to TorchArrow
# To start let's create a Panda series and a TorchArrow column and compare them:

# In[2]:


import pandas as pd
import torcharrow as ta
import torcharrow.dtypes as dt


pd.Series([1, 2, None, 4])


# In Pandas each Series has an index, here depicted as the first column. Note also that the inferred type is float and not int, since in Pandas None implicitly  promotes an int list to a float series.

# TorchArrow has a much more precise type system:

# In[3]:


s = ta.Column([1, 2, None, 4])
s


# TorchArrow creates CPU column by default, which is supported by [Velox](https://github.com/facebookincubator/velox) backend.

# In[4]:


s.device


# TorchArrow infers that that the type is `Int64(nullable=True)`. Of course, we can always get lots of more information from a column: the length, count, null_count determine the total number, the number of non-null, and the number of nulls, respectively.
#
#
#

# In[5]:


len(s), s._count(), s.null_count


# TorchArrow infers Python float as float32 (instead of float64). This follows PyTorch and other deep learning libraries.

# In[6]:


ss = ta.Column([2.718, 3.14, 42.42])
ss


# TorchArrow supports (almost all of Arrow types), including arbitrarily nested structs, maps, lists, and fixed size lists. Here is a non-nullable column of a list of non-nullable strings of arbitrary length.

# In[7]:


sf = ta.Column([["hello", "world"], ["how", "are", "you"]], dtype=dt.List(dt.string))
sf.dtype


# And here is a column of average climate data, one map per continent, with city as key and yearly average min and max temperature:
#

# In[8]:


mf = ta.Column(
    [
        {"helsinki": [-1.3, 21.5], "moscow": [-4.0, 24.3]},
        {"algiers": [11.2, 25.2], "kinshasa": [22.2, 26.8]},
    ]
)
mf


# ### Append and concat

# Columns are immutable. Use `append` to create a new column with a list of values appended.

# In[9]:


sf = sf.append([["I", "am", "fine", "and", "you"]])
sf


# Use `concat` to combine a list of columns.

# In[10]:


# TODO: Fix this!
# sf = sf.concat([ta.Column("I", "am", "fine", "too")]


# ## Constructing data: Dataframes
#
# A Dataframe is just a set of named and strongly typed columns of equal length:

# In[11]:


df = ta.DataFrame(
    {"a": list(range(7)), "b": list(reversed(range(7))), "c": list(range(7))}
)
df


# To access a dataframes columns write:

# In[12]:


df.columns


# Dataframes are also immutable, except you can always add a new column or overwrite an existing column.
#
# When a new column is added, it is appended to the set of existing columns at the end.

# In[13]:


df["d"] = ta.Column(list(range(99, 99 + 7)))
df


# You can also overwrite an existing column.

# In[14]:


df["c"] = df["b"] * 2
df


# Similar to Column, we can also use `append` to create a new DataFrame with a listed of tuples appended.

# In[15]:


df = df.append([(7, 77, 777, 7777), (8, 88, 888, 8888)])
df


# Dataframes can be nested. Here is a Dataframe having sub-dataframes.
#

# In[16]:


df_inner = ta.DataFrame({"b1": [11, 22, 33], "b2": [111, 222, 333]})
df_outer = ta.DataFrame({"a": [1, 2, 3], "b": df_inner})
df_outer


# ## Interop
#
# Coming soon!

# ## Viewing (sorted) data
#
# Take the the top n rows

# In[17]:


df.head(2)


# Or return the last n rows

# In[18]:


df.tail(1)


# or sort the values before hand.

# In[19]:


df.sort(by=["c", "b"]).head(2)


# Sorting can be controlled not only by which columns to sort on, but also whether to sort ascending or descending, and how to deal with nulls, are they listed first or last.
#
# ## Selection using Indices
#
# Torcharrow supports two indices:
#  - Integer indices select rows
#  - String indices select columns
#
# So projecting a single column of a dataframe is simply

# In[20]:


df["a"]


# Selecting a single row uses an integer index. (In TorchArrow everything is zero-based.)

# In[21]:


df[1]


# Selecting a slice keeps the type alive. Here we slice rows:
#

# In[22]:


df[2:6:2]


# But you can also slice columns. The below return all columns after and including 'c'.

# In[23]:


df["c":]


# TorchArrow follows the normal Python semantics for slices: that is a slice interval is closed on the left and open on the right.

# ## Selection by Condition
#
# Selection of a column or dataframe *c* by a condition takes a boolean column *b* of the same length as *c*. If the *i*th row in *b* is true, *c*'s *i*th row is included in the result otherwise it is dropped. Below expression selects the first row, since it is true, and drops all remaining rows, since they are false.
#
#

# In[24]:


df[[True] + [False] * (len(df) - 1)]


# Conditional expressions over vectors return boolean vectors. Conditionals are thus the usual way to write filters.

# In[25]:


b = df["a"] > 4
df[b]


# Torcharrow supports all the usual predicates, like <,==,!=>,>=,<= as well as _in_. The later is denoted by `isin`
#

# In[26]:


df[df["a"].isin([5])]


# ## Missing data
#  Missing data can be filled in via the `fillna` method

# In[27]:


t = s.fillna(999)
t


# Alternatively data that has null data can be dropped:

# In[28]:


s.dropna()


# ## Operators
# Columns and dataframes support all of Python's usual binary operators, like  ==,!=,<=,<,>,>= for equality  and comparison,  +,-,*,,/.//,** for performing arithmetic and &,|,~ for conjunction, disjunction and negation.
#
# The semantics of each operator is given by lifting their scalar operation to vectors and dataframes. So given for instance a scalar comparison operator, in TorchArrow a scalar can be compared to each item in a column, two columns can be compared pointwise, a column can be compared to each column of a dataframe, and two dataframes can be compared by comparing each of their respective columns.
#
# Here are some example expressions:

# In[29]:


u = ta.Column(list(range(5)))
v = -u
w = v + 1
v * w


# In[30]:


uv = ta.DataFrame({"a": u, "b": v})
uu = ta.DataFrame({"a": u, "b": u})
(uv == uu)


# ## Null strictness
#
# The default behavior of torcharrow operators and functions is that *if any argument is null then the result is null*. For instance:

# In[31]:


u = ta.Column([1, None, 3])
v = ta.Column([11, None, None])
u + v


# If null strictness does not work for your code you could call first `fillna` to provide a value that is used instead of null.

# ## Numerical columns and descriptive statistics
# Numerical columns also support lifted operations, for `abs`, `ceil`, `floor`, `round`. Even more excited might be to use their aggregation operators like `count`, `sum`, `prod`, `min`, `max`, or descriptive statistics like `std`, `mean`, `median`, and `mode`. Here is an example ensemble:
#

# In[32]:


(t.min(), t.max(), t.sum(), t.mean())


# The `describe` method puts this nicely together:

# In[33]:


t.describe()


# Sum, prod, min and max are also available as accumulating operators called `cumsum`, `cumprod`, etc.
#
# Boolean vectors are very similar to numerical vector. They offer the aggregation operators `any` and `all`.

# ## String, list and map methods
# Torcharrow provides all of Python's string, list and map processing methods, just lifted to work over columns. Like in Pandas they are all accessible via the `str`, `list` and `map` property, respectively.
#
# ### Strings
# Let's convert a Column of strings to upper case.
#

# In[34]:


s = ta.Column(["what a wonderful world!", "really?"])
s.str.upper()


# We can also split each string to a list of strings with the given delimiter.
#

# In[35]:


ss = s.str.split(sep=" ")
ss


# ### Lists
#
# To operate on a list column use the usual pure list operations, like `len(gth)`, `slice`, `index` and `count`, etc. But there are a couple of additional operations.
#
# For instance to invert the result of a string split operation a list of string column also offers a join operation.
#

# In[36]:


ss.list.join(sep="-")


# In addition lists provide `filter`, `map`, `flatmap` and `reduce` operators, which we will discuss as in more details in functional tools.
#
# ### Maps
#
# Column of type map provide the usual map operations like `len(gth)`, `[]`, `keys` and `values`. Keys and values both return a list column. Key and value columns can be reassembled by calling `mapsto`.

# In[37]:


mf.maps.keys()


# ## Relational tools: Where, select, groupby, join, etc.
#
# TorchArrow also plans to support all relational operators on DataFrame. The following sections discuss what exists today.
#
# ### Where
# The simplest operator is `df.where(p)` which is just another way of writing `df[p]`. (Note: TorchArrow's `where`  is not Pandas' `where`, the latter is a vectorized if-then-else which we call in Torcharrow `ite`.)

# In[38]:


xf = ta.DataFrame({"A": ["a", "b", "a", "b"], "B": [1, 2, 3, 4], "C": [10, 11, 12, 13]})

xf.where(xf["B"] > 2)


# Note that in `xf.where` the predicate `xf['B']>2` refers to self, i.e. `xf`. To access self in an expression TorchArrow introduces the special name `me`. That is, we can also write:
#

# In[39]:


from torcharrow import me


xf.where(me["B"] > 2)


# ### Select
#
# Select is SQL's standard way to define a new set of columns. We use *positional args to keep columns and kwargs to give new bindings. Here is a typical example that keeps all of xf's columns but adds column 'D').
#

# In[40]:


xf.select(*xf.columns, D=me["B"] + me["C"])


# The short form of `*xf.columns` is '\*', so `xf.select('*', D=me['B']+me['C'])` does the same.

# ### Grouping, Join and Tranpose
#
# Coming soon!
#

# ## Functional tools:  map, filter, reduce
#
# Column and dataframe pipelines support map/reduce style programming as well. We first explore column oriented operations.
#
# ###  Map and its variations
#
# `map` maps values of a column according to input correspondence. The input correspondence can be given as a mapping or as a (user-defined-) function (UDF). If the mapping is a dict, then non mapped values become null.
#
#
#

# In[41]:


ta.Column([1, 2, None, 4]).map({1: 111})


# If the mapping is a defaultdict, all values will be mapped as described by the default dict.

# In[42]:


from collections import defaultdict

ta.Column([1, 2, None, 4]).map(defaultdict(lambda: -1, {1: 111}))


# **Handling null.** If the mapping is a function, then it will be applied on all values (including null), unless na_action is `'ignore'`, in which case, null values are passed through.

# In[43]:


def add_ten(num):
    return num + 10


ta.Column([1, 2, None, 4]).map(add_ten, na_action="ignore")


# Note that `.map(add_ten, na_action=None)` would fail with a type error since `addten` is not defined for `None`/null. So if we wanted to pass null to `add_ten` we would have to prepare for it, maybe like so:

# In[44]:


def add_ten_or_0(num):
    return 0 if num is None else num + 10


ta.Column([1, 2, None, 4]).map(add_ten_or_0, na_action=None)


# **Mapping to different types.** If `map` returns a column type that is different from the input column type, then `map` has to specify the returned column type.

# In[45]:


ta.Column([1, 2, 3, 4]).map(str, dtype=dt.string)


# Instead of specifying `dtype` argument, you can also rely on type annotations (both Python annotations and `dtypes` are supported):

# In[46]:


from typing import Optional


def str_only_even(x) -> Optional[str]:
    if x % 2 == 0:
        return str(x)
    return None


ta.Column([1, 2, 3, 4]).map(str_only_even)  # dt.string(nullable=True) is inferred


# **Map over Dataframes** Of course, `map` works over Dataframes, too. In this case the callable gets the whole row as a tuple.

# In[47]:


def add_unary(tup):
    return tup[0] + tup[1]


ta.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]}).map(add_unary, dtype=dt.int64)


# **Multi-parameter functions**. So far all our functions were unary functions. But `map` can be used for n-ary functions, too: simply specify the set of `columns` you want to pass to the nary function.
#

# In[48]:


def add_binary(a, b):
    return a + b


ta.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"], "c": [1, 2, 3]}).map(
    add_binary, columns=["a", "c"], dtype=dt.int64
)


# **Multi-return functions.** Functions that return more than one column can be specified by returning a dataframe  (aka as struct column); providing the  return type is mandatory.

# In[49]:


ta.DataFrame({"a": [17, 29, 30], "b": [3, 5, 11]}).map(
    divmod,
    columns=["a", "b"],
    dtype=dt.Struct([dt.Field("quotient", dt.int64), dt.Field("remainder", dt.int64)]),
)


# **Functions with state**. Functions need sometimes additional precomputed state. We capture the state in a (data)class and use a method as a delegate:
#

# In[50]:


def fib(n):
    if n == 0:
        return 0
    elif n == 1 or n == 2:
        return 1
    else:
        return fib(n - 1) + fib(n - 2)


from dataclasses import dataclass


@dataclass
class State:
    state: int

    def __post_init__(self):
        self.state = fib(self.state)

    def add_fib(self, x):
        return self.state + x


m = State(10)
ta.Column([1, 2, 3]).map(m.add_fib)


# TorchArrow requires that only global functions or methods on class instances can be used as user defined functions. Lambdas, which can can capture arbitrary state and are not inspectable, are not supported.

# ### Filter
#
# `filter` takes a predicate and returns all those rows for which the predicate holds:

# In[51]:


ta.Column([1, 2, 3, 4]).filter(lambda x: x % 2 == 1)


# Instead of the predicate you can pass an iterable of boolean of the same length as the column:

# In[52]:


ta.Column([1, 2, 3, 4]).filter([True, False, True, False])


# If the predicate is an n-ary function, use the  `columns` argument as we have seen for `map`.

# ### Flatmap
#
# `flatmap` combines `map` with `filter`. Each callable can return a list of elements. If that list is empty, flatmap filters, if the returned list is a singleton, flatmap acts like map, if it returns several elements it 'explodes' the input. Here is an example:

# In[53]:


def selfish(words):
    return [words, words] if len(words) >= 1 and words[0] == "I" else []


sf.flatmap(selfish)


# `flatmap` has all the flexibility of `map`, i.e it can take the `ignore`, `dtype` and `column` arguments.

# ### Reduce
# `reduce` is just like Python's `reduce`. Here we compute the product of a column.

# In[54]:


import operator


ta.Column([1, 2, 3, 4]).reduce(operator.mul)


# ## Batch Transform
#

# Batch `transform` is similar to `map`, except the functions takes batch input/output (represented by a Python list, PyTorch tensors, etc).

# In[55]:


from typing import List


def multiple_ten(val: List[int]) -> List[int]:
    return [x * 10 for x in val]


ta.Column([1, 2, 3, 4]).transform(multiple_ten, format="python")


# In[56]:


"End of tutorial"
