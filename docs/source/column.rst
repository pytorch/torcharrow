.. currentmodule:: torcharrow

torcharrow.Column
==========================

A :class:`torcharrow.Column` is a 1-dimension torch.Tensor like data structure containing 
elements of a single data type. It also supports non-numeric types such as string, 
list, struct.

UNDER CONSTRUCTION


Column class reference
------------------------------
.. class:: Column()

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    Column.head

    Column.isin
    Column.fill_null
    Column.drop_null


NumericalColumn class reference
-----------------------------------
.. class:: NumericalColumn()

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    NumericalColumn.abs
    NumericalColumn.ceil
    NumericalColumn.floor
    NumericalColumn.round
    NumericalColumn.log

    NumericalColumn.min
    NumericalColumn.max
    NumericalColumn.sum
    NumericalColumn.mean
    NumericalColumn.std

    
UNDER CONSTRUCTION: StringColumn, ListColumn