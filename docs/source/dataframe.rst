.. currentmodule:: torcharrow

torcharrow.DataFrame
==========================

:class:`torcharrow.DataFrame` is a Python DataFrame library (built on the Apache Arrow columnar memory format)
for loading, joining, aggregating, filtering, and otherwise manipulating data.
:class:`torcharrow.DataFrame` also provides a Pandas-like API that naturally fits into the Python ML ecosystem,
and will be familiar to data scientist and ML engineers, so they can use it to express tabular data workflows
in ML, such as feature engineering, training and inference preprocessing.

..
  TODO: Some introduction about __getitem__, i.e. df["a"] = df["b"] + 5, df[df["a"] > 5],


DataFrame Class and General APIs
----------------------------------
.. class:: DataFrame()

.. autoattribute:: DataFrame.columns
.. autoattribute:: DataFrame.dtype
.. autoattribute:: DataFrame.device
.. autoattribute:: DataFrame.length

.. autosummary::
    :toctree: generated
    :nosignatures:

    DataFrame.head
    DataFrame.tail
    DataFrame.describe

    DataFrame.drop
    DataFrame.rename
    DataFrame.reorder

    DataFrame.append
    DataFrame.isin

Functional API
------------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    DataFrame.map
    DataFrame.filter
    DataFrame.flatmap
    DataFrame.transform


Relational API
------------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    DataFrame.select
    DataFrame.where
    DataFrame.sort

Data Cleaning
------------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    DataFrame.fill_null
    DataFrame.drop_null
    DataFrame.drop_duplicates

Conversions
------------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    DataFrame.to_arrow
    DataFrame.to_tensor
    DataFrame.to_pylist
    DataFrame.to_pandas

Statistics
------------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    DataFrame.min
    DataFrame.max
    DataFrame.sum
    DataFrame.mean
    DataFrame.std
    DataFrame.median
    DataFrame.all
    DataFrame.any

Arithmtic Operations
------------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    DataFrame.log
