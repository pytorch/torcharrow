.. currentmodule:: torcharrow

torcharrow
==========================

The torcharrow package contains data structures for two-dimensional, potentially heterogeneous tabular tabular data,
denoted as dataframe. 
It also defines relational operations over these dataframes. 
Additionally, it provides many utilities for conversion with other formats (especially zero-copy conversion with Arrow arrays),
and other useful utilities.

Creation and Conversion Ops
-----------------------------
.. autosummary::
    :toctree: generated/
    :template: function.rst

    column
    dataframe
    from_arrow
    Column.to_arrow

Mutating Ops
-------------------------
.. autosummary::
    :toctree: generated/
    :template: function.rst

    concat
    if_else

