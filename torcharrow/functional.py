# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from functools import partial
from types import ModuleType
from typing import Dict, List, Optional, Set, Union

from torcharrow.icolumn import Column
from torcharrow.ilist_column import ListColumn
from torcharrow.inumerical_column import NumericalColumn

# In the future, TorchArrow is going to support more device/dispatch_key, such as gpu/libcudf
_device_to_dispatch_key = {"cpu": "velox"}
_column_class_to_dispatch_key = {}

_backend_functional: Dict[str, ModuleType] = {}
_factory_methods: Set[str] = set()


# TODO: prefix all "module private" methods with "_"
def get_backend_functional(dispatch_key):
    backend_functional = _backend_functional.get(dispatch_key)
    if backend_functional is None:
        raise AssertionError(
            f"Functional module for backend {dispatch_key} is not registered"
        )
    return backend_functional


# dispatch the function call based on backend
def _dispatch(op_name: str, *args):
    # Calculate dispatch key based on input
    col_arg: Optional[Column] = next(
        (arg for arg in args if isinstance(arg, Column)), None
    )

    if col_arg is None:
        # TODO: suppoort this to return a constant literal
        raise ValueError("None of the argument is Column")

    if type(col_arg) in _column_class_to_dispatch_key:
        dispatch_key = _column_class_to_dispatch_key[type(col_arg)]
    else:
        device = col_arg.device
        dispatch_key = _device_to_dispatch_key.get(device)

    # dispatch to backend functional namespace
    op = get_backend_functional(dispatch_key).__getattr__(op_name)
    return op(*args)


def create_dispatch_wrapper(op_name: str):
    # TODO: factory dispatch mechanism needs revamp and conslidate with the general constant literal handling
    def factory_dispatch(*args, size=None, device="cpu"):
        if size is None:
            raise AssertionError(
                f"Factory method call {op_name} requires expclit size parameter"
            )

        # We assume that other args for factory functions are non-columns and don't even check args
        if isinstance(size, int):
            dispatch_key = _device_to_dispatch_key[device]
        else:
            # TODO: Support SizeProxy
            raise AssertionError(f"Unsupported size parameter type {type(size)}")

        # dispatch to backend functional namespace
        op = get_backend_functional(dispatch_key).__getattr__(op_name)
        return op(*args, size=size, device=device)

    if op_name in _factory_methods:
        return factory_dispatch
    else:
        return partial(_dispatch, op_name)


def register_dispatch_impl(name: str, module: ModuleType):
    if name in _backend_functional:
        raise AssertionError(
            f"Functional module for backend {name} is already registered"
        )
    _backend_functional[name] = module


def register_column_class_for_dispatch(column_class: type, dispatch_key: str):
    if column_class in _column_class_to_dispatch_key:
        raise AssertionError(
            f"Column class {column_class} is already registered with dispatch key {dispatch_key}"
        )
    _column_class_to_dispatch_key[column_class] = dispatch_key


# TODO: factory dispatch mechanism needs revamp and conslidate with the general constant literal handling
def register_factory_methods(methods):
    _factory_methods.update(methods)


def __getattr__(op_name: str):
    wrapper = create_dispatch_wrapper(op_name)
    setattr(sys.modules["torcharrow.functional"], op_name, wrapper)
    return wrapper


### operations in for recommendation domain
def bucketize(
    value_col: NumericalColumn,
    borders: Union[ListColumn, List[Union[int, float]]],
) -> NumericalColumn:
    """
    Apply bucketization for input feature. This is a common operation in recommendation domain
    to convert dense features into sparse features.

    Parameters
    ----------
    value_col: Numeric column that defines dense feature
    borders: Border values for the discretized sparse features

    Examples
    --------
    >>> import torcharrow as ta
    >>> from torcharrow import functional
    >>> a = ta.column([1, 2, 3, 5, 8, 10, 11])
    >>> functional.bucketize(a, [2, 5, 10])
    0  0
    1  0
    2  1
    3  1
    4  2
    5  2
    6  3
    dtype: Int32(nullable=True), length: 7, null_count: 0
    """
    return _dispatch("bucketize", value_col, borders)


def sigrid_hash(*args):
    """
    TODO: Add docstring
    """
    return _dispatch("sigrid_hash", *args)


def firstx(*args):
    """
    TODO: Add docstring
    """
    return _dispatch("firstx", *args)


def has_id_overlap(*args):
    """
    TODO: Add docstring
    """
    return _dispatch("has_id_overlap", *args)


def id_overlap_count(*args):
    """
    TODO: Add docstring
    """
    return _dispatch("id_overlap_count", *args)


def get_max_count(*args):
    """
    TODO: Add docstring
    """
    return _dispatch("get_max_count", *args)


def get_jaccard_similarity(*args):
    """
    TODO: Add docstring
    """
    return _dispatch("get_jaccard_similarity", *args)


def get_cosine_similarity(*args):
    """
    TODO: Add docstring
    """
    return _dispatch("get_cosine_similarity", *args)


def get_score_sum(*args):
    """
    TODO: Add docstring
    """
    return _dispatch("get_score_sum", *args)


def get_score_min(*args):
    """
    TODO: Add docstring
    """
    return _dispatch("get_score_min", *args)


def get_score_max(*args):
    """
    TODO: Add docstring
    """
    return _dispatch("get_score_max", *args)


### high-level operations
def scale_to_0_1(col: NumericalColumn) -> NumericalColumn:
    """Return the column data scaled to range [0,1].
    If column contains only a single value, then column is scaled with sigmoid function.
    """
    assert isinstance(col, NumericalColumn)
    min_val = col.min()
    max_val = col.max()
    if min_val < max_val:
        return (col - min_val) / (max_val - min_val)
    else:
        # TODO: we should add explicit stub to sigmoid
        return sys.modules["torcharrow.functional"].sigmoid(col)
