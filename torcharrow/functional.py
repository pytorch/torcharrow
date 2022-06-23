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
    # Don't wrap accesses for special module members, like __bases__
    # as it breaks basic Python functionality like
    # `help(torcharrow.functional)`
    if op_name.startswith("__") or op_name == "_fields":
        raise AttributeError
    wrapper = create_dispatch_wrapper(op_name)
    setattr(sys.modules["torcharrow.functional"], op_name, wrapper)
    return wrapper


### operations in for text domain
def add_tokens(
    input_col: Union[ListColumn, List[Union[int, str]]],
    tokens: Union[ListColumn, List[Union[int, str]]],
    begin: bool,
) -> NumericalColumn:
    """
    Append or prepend a list of tokens/indices to a column.
    This is a common operation to add EOS and BOS tokens to text.

    Parameters
    ----------
    input_col: List of input tokens/indices
    tokens: List of tokens/indices to append or prepend
    begin: Boolean to determine whether to prepend or append the tokens/indices

    Examples
    --------
    >>> import torcharrow as ta
    >>> from torcharrow import functional
    >>> a = ta.column([[1, 2], [3, 4, 5]])
    >>> functional.add_tokens(a, [0], begin=True)
    0  [0, 1, 2]
    1  [0, 3, 4, 5]
    dtype: List(Int64(nullable=True), nullable=True), length: 2, null_count: 0
    """
    return _dispatch("add_tokens", input_col, tokens, begin)


# Velox core functions
# Not a comprehensive list yet
def array_constructor(*args) -> ListColumn:
    """
    Construct the array given the input columns.
    All input columns are expected to have the same dtype.

    Example
    --------
    >>> import torcharrow as ta
    >>> from torcharrow import functional
    >>> a = ta.column([1, 2, 3, 10, -1])
    >>> b = ta.column([2, 3, 5, 29, None])
    >>> functional.array_constructor(a, b)
    0  [1, 2]
    1  [2, 3]
    2  [3, 5]
    3  [10, 29]
    4  [-1, None]
    dtype: List(Int64(nullable=True), nullable=True), length: 5, null_count: 0
    """
    return _dispatch("array_constructor", *args)


def array_except(x: ListColumn, y: ListColumn) -> ListColumn:
    """
    Returns the list of the elements in list x but not in list y, without duplicates.

    See Also
    --------
    Velox core function `array_except <https://facebookincubator.github.io/velox/functions/array.html#array_except>`_

    Example
    --------
    >>> x = ta.column([[1, 2, 3], [1, 2, 3], [1, 2, 2], [1, 2, 2], [1, None, None]])
    >>> y = ta.column([[4, 5, 6], [1, 2],    [1, 1, 2], [1, 3, 4], [1, 1, None]])
    >>> functional.array_except(x, y)
    0  [1, 2, 3]
    1  [3]
    2  []
    3  [2]
    4  []
    dtype: List(Int64(nullable=True), nullable=True), length: 5, null_count: 0
    """
    return _dispatch("array_except", x, y)


### operations for recommendation domain
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


def sigrid_hash(value_col: NumericalColumn, salt: int, max_value: int):
    """
    Apply hashing to an index, or a list of indicies. This is a common operation in the
    recommendation domain in order to have valid inputs for shrunken embedding tables.

    Parameters
    ----------
    value_col: Numeric column that defines indicies
    salt: Value used to intialize the random hashing process
    max_value: values will be hashed in the range of [0, max_value)

    Examples
    --------
    >>> import torcharrow as ta
    >>> from torcharrow import functional
    >>> a = ta.column([1, 2, 3, 5, 8, 10, 11])
    >>> functional.sigrid_hash(a, 0, 100)
    0  60
    1  54
    2  54
    3   4
    4  67
    5   2
    6  25
    dtype: Int64(nullable=True), length: 7, null_count: 0
    """
    return _dispatch("sigrid_hash", value_col, salt, max_value)


def firstx(col: ListColumn, num_to_copy: int):
    """
    Returns the first x values of the head of the input column

    Parameters
    ----------
    col: Column that has a list of values
    num_to_copy: Number of elements to return from the head of the list

    Examples
    --------
    >>> import torcharrow as ta
    >>> from torcharrow import functional
    >>> a = ta.column([[1, 2, 3],[5,8],[13]])
    >>> functional.firstx(a, 3)
    0  [1, 2, 3]
    1  [5, 8]
    2  [13]
    dtype: List(Int64(nullable=True), nullable=True), length: 3, null_count: 0
    """
    return _dispatch("firstx", col, num_to_copy)


def has_id_overlap(input_ids: ListColumn, matching_ids: ListColumn):
    """
    Returns 1.0 if the two input columns overlap, otherwise 0.0

    Parameters
    ---------
    input_ids: First list of ids
    matching_ids: Second list of ids

    Examples
    -------
    >>> import torcharrow as ta
    >>> from torcharrow import functional
    >>> input_ids = ta.column([[1, 2, 3],[5,8],[13]])
    >>> matching_ids = ta.column([[1],[2,3],[13]])
    >>> functional.has_id_overlap(input_ids, matching_ids)
    0  1
    1  0
    2  1
    dtype: Float32(nullable=True), length: 3, null_count: 0
    """
    return _dispatch("has_id_overlap", input_ids, matching_ids)


def id_overlap_count(input_ids: ListColumn, matching_ids: ListColumn):
    """
    Returns the number of overlaps between two lists of ids

    Parameters
    ---------
    input_ids: First list of ids
    matching_ids: Second list of ids

    Examples
    -------
    >>> import torcharrow as ta
    >>> from torcharrow import functional
    >>> input_ids = ta.column([[1, 2, 3],[5,8],[13]])
    >>> matching_ids = ta.column([[1],[2,3],[13]])
    >>> functional.id_overlap_count(input_ids, matching_ids)
    0  3
    1  0
    2  1
    dtype: Float32(nullable=True), length: 3, null_count: 0
    """
    return _dispatch("id_overlap_count", input_ids, matching_ids)


def get_max_count(input_ids: ListColumn, matching_ids: ListColumn):
    """
    If there are items that overlap between input_ids and matching_ids
    contribute the maximum number of instances of overlapped ids to
    the max count.

    Parameters
    ---------
    input_ids: First list of ids
    matching_ids: Second list of ids

    Examples
    -------
    >>> import torcharrow as ta
    >>> from torcharrow import functional
    >>> input_ids = ta.column([[1, 1, 2, 3],[5,8],[13]])
    >>> matching_ids = ta.column([[1,2,3],[2,3],[13,13,13,13,13]])
    >>> functional.get_max_count(input_ids, matching_ids)
    0  4
    1  0
    2  5
    dtype: Float32(nullable=True), length: 3, null_count: 0
    """
    return _dispatch("get_max_count", input_ids, matching_ids)


def get_jaccard_similarity(input_ids: ListColumn, matching_ids: ListColumn):
    """
    Return the jaccard_similarity between input_ids and matching_ids.
    The jaccard similarity is |input_ids.intersect(matching_ids)|/|input_ids.union(matching_ids)|

    Parameters
    ---------
    input_ids: First list of ids
    matching_ids: Second list of ids

    Examples
    -------
    >>> import torcharrow as ta
    >>> from torcharrow import functional
    >>> input_ids = ta.column([[1, 1, 2, 3],[5,8],[13]])
    >>> matching_ids = ta.column([[1,2,3],[2,3],[13,13,13,13,13]])
    >>> functional.get_jaccard_similarity(input_ids, matching_ids)
    0  0.75
    1  0
    2  0.2
    dtype: Float32(nullable=True), length: 3, null_count: 0
    """
    return _dispatch("get_jaccard_similarity", input_ids, matching_ids)


def get_cosine_similarity(
    input_ids: ListColumn,
    input_id_scores: ListColumn,
    matching_ids: ListColumn,
    matching_id_scores: ListColumn,
):
    """
    Return the cosine between the vector defined by input_ids weighted by input_id_scores and
    the vector defined by matching_ids weighted by matching_id_scores

    Parameters
    ---------
    input_ids: First list of ids
    input_ids_scores: scores (weights) of input_ids
    matching_ids: Second list of ids
    matching_ids_scores: scores (weights) of matching_ids

    Examples
    --------
    >>> import torcharrow as ta
    >>> from torcharrow import functional
    >>> input_ids = ta.column([[1, 2, 3]])
    >>> input_id_scores = ta.column([[1.0,2.0,3.0]])
    >>> matching_ids = ta.column([[1,2,3]])
    >>> matching_id_scores = ta.column([[5.0,4.0,3.0]])
    >>> functional.get_cosine_similarity(input_ids, input_id_scores, matching_ids, matching_id_scores)
    0  0.831522
    dtype: Float32(nullable=True), length: 1, null_count: 0
    """
    return _dispatch(
        "get_cosine_similarity",
        input_ids,
        input_id_scores,
        matching_ids,
        matching_id_scores,
    )


def get_score_sum(
    input_ids: ListColumn,
    input_id_scores: ListColumn,
    matching_ids: ListColumn,
    matching_id_scores: ListColumn,
):
    """
    Return the sum of all the scores in matching_id_scores that has a corresponding id in matching_ids that is also in input_ids.
    Parameters
    ---------
    input_ids: First list of ids
    input_ids_scores: scores (weights) of input_ids
    matching_ids: Second list of ids
    matching_ids_scores: scores (weights) of matching_ids

    Examples
    --------
    >>> import torcharrow as ta
    >>> from torcharrow import functional
    >>> input_ids = ta.column([[1, 2, 3]])
    >>> input_id_scores = ta.column([[1.0,2.0,3.0]])
    >>> matching_ids = ta.column([[1,2]])
    >>> matching_id_scores = ta.column([[5.0,4.0]])
    >>> functional.get_score_sum(input_ids, input_id_scores, matching_ids, matching_id_scores)
    0  9
    dtype: Float32(nullable=True), length: 1, null_count: 0
    """
    return _dispatch(
        "get_score_sum", input_ids, input_id_scores, matching_ids, matching_id_scores
    )


def get_score_min(
    input_ids: ListColumn,
    matching_ids: ListColumn,
    matching_id_scores: ListColumn,
):
    """
    Return the min among of all the scores in matching_id_scores that has a corresponding id in matching_ids that is also in input_ids.

    Parameters
    ---------
    input_ids: First list of ids
    matching_ids: Second list of ids
    matching_ids_scores: scores (weights) of matching_ids

    Examples
    --------
    >>> import torcharrow as ta
    >>> from torcharrow import functional
    >>> input_ids = ta.column([[1, 2, 3]])
    >>> matching_ids = ta.column([[1,2]])
    >>> matching_id_scores = ta.column([[5.0,4.0]])
    >>> functional.get_score_min(input_ids, matching_ids, matching_id_scores)
    0  4.0
    dtype: Float32(nullable=True), length: 1, null_count: 0
    """
    return _dispatch("get_score_min", input_ids, matching_ids, matching_id_scores)


def get_score_max(
    input_ids: ListColumn,
    matching_ids: ListColumn,
    matching_id_scores: ListColumn,
):
    """
    Return the min among of all the scores in matching_id_scores that has a corresponding id in matching_ids that is also in input_ids.

    Parameters
    ---------
    input_ids: First list of ids
    matching_ids: Second list of ids
    matching_ids_scores: scores (weights) of matching_ids

    Examples
    --------
    >>> import torcharrow as ta
    >>> from torcharrow import functional
    >>> input_ids = ta.column([[1, 2, 3]])
    >>> matching_ids = ta.column([[1,2]])
    >>> matching_id_scores = ta.column([[5.0,4.0]])
    >>> functional.get_score_min(input_ids, matching_ids, matching_id_scores)
    0  5.0
    dtype: Float32(nullable=True), length: 1, null_count: 0
    """
    return _dispatch("get_score_max", input_ids, matching_ids, matching_id_scores)


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
