# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from types import ModuleType
from typing import Dict, List, Optional, Set, Union

from torcharrow.icolumn import Column
from torcharrow.ilist_column import ListColumn
from torcharrow.inumerical_column import NumericalColumn


class _Functional(ModuleType):
    __file__ = "_functional.py"

    # In the future, TorchArrow is going to support more device/dispatch_key, such as gpu/libcudf
    _device_to_dispatch_key = {"cpu": "velox"}
    _column_class_to_dispatch_key = {}

    def __init__(self):
        super().__init__("torcharrow.functional")
        self._backend_functional: Dict[str, ModuleType] = {}
        self._factory_methods: Set[str] = set()

    # TODO: prefix all private methods with "_"
    def get_backend_functional(self, dispatch_key):
        backend_functional = self._backend_functional.get(dispatch_key)
        if backend_functional is None:
            raise AssertionError(
                f"Functional module for backend {dispatch_key} is not registered"
            )
        return backend_functional

    # dispatch the function call based on backend
    def _dispatch(self, op_name: str, *args):
        # Calculate dispatch key based on input
        col_arg: Optional[Column] = next(
            (arg for arg in args if isinstance(arg, Column)), None
        )

        if col_arg is None:
            # TODO: suppoort this to return a constant literal
            raise ValueError("None of the argument is Column")

        if type(col_arg) in type(self)._column_class_to_dispatch_key:
            dispatch_key = type(self)._column_class_to_dispatch_key[type(col_arg)]
        else:
            device = col_arg.device
            dispatch_key = _Functional._device_to_dispatch_key.get(device)

        # dispatch to backend functional namespace
        op = self.get_backend_functional(dispatch_key).__getattr__(op_name)
        return op(*args)

    def create_dispatch_wrapper(self, op_name: str):
        # TODO: factory dispatch mechanism needs revamp and conslidate with the general constant literal handling
        def factory_dispatch(*args, size=None, device="cpu"):
            if size is None:
                raise AssertionError(
                    f"Factory method call {op_name} requires expclit size parameter"
                )

            # We assume that other args for factory functions are non-columns and don't even check args
            if isinstance(size, int):
                dispatch_key = _Functional._device_to_dispatch_key[device]
            else:
                # TODO: Support SizeProxy
                raise AssertionError(f"Unsupported size parameter type {type(size)}")

            # dispatch to backend functional namespace
            op = self.get_backend_functional(dispatch_key).__getattr__(op_name)
            return op(*args, size=size, device=device)

        if op_name in self._factory_methods:
            return factory_dispatch
        else:
            return partial(self._dispatch, op_name)

    def register_dispatch_impl(self, name: str, module: ModuleType):
        if name in self._backend_functional:
            raise AssertionError(
                f"Functional module for backend {name} is already registered"
            )
        self._backend_functional[name] = module

    @classmethod
    def register_column_class_for_dispatch(cls, column_class: type, dispatch_key: str):
        if column_class in cls._column_class_to_dispatch_key:
            raise AssertionError(
                f"Column class {column_class} is already registered with dispatch key {dispatch_key}"
            )
        cls._column_class_to_dispatch_key[column_class] = dispatch_key

    # TODO: factory dispatch mechanism needs revamp and conslidate with the general constant literal handling
    def register_factory_methods(self, methods):
        self._factory_methods.update(methods)

    # pyre-fixme[14]: `__getattr__` overrides method defined in `ModuleType`
    #  inconsistently.
    def __getattr__(self, op_name: str):
        wrapper = self.create_dispatch_wrapper(op_name)
        setattr(self, op_name, wrapper)
        return wrapper

    ### operations in for recommendation domain
    def bucketize(
        self,
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
        return self._dispatch("bucketize", value_col, borders)

    def sigrid_hash(self, *args):
        """
        TODO: Add docstring
        """
        return self._dispatch("sigrid_hash", *args)

    def firstx(self, *args):
        """
        TODO: Add docstring
        """
        return self._dispatch("firstx", *args)

    def has_id_overlap(self, *args):
        """
        TODO: Add docstring
        """
        return self._dispatch("has_id_overlap", *args)

    def id_overlap_count(self, *args):
        """
        TODO: Add docstring
        """
        return self._dispatch("id_overlap_count", *args)

    def get_max_count(self, *args):
        """
        TODO: Add docstring
        """
        return self._dispatch("get_max_count", *args)

    def get_jaccard_similarity(self, *args):
        """
        TODO: Add docstring
        """
        return self._dispatch("get_jaccard_similarity", *args)

    def get_cosine_similarity(self, *args):
        """
        TODO: Add docstring
        """
        return self._dispatch("get_cosine_similarity", *args)

    def get_score_sum(self, *args):
        """
        TODO: Add docstring
        """
        return self._dispatch("get_score_sum", *args)

    def get_score_min(self, *args):
        """
        TODO: Add docstring
        """
        return self._dispatch("get_score_min", *args)

    def get_score_max(self, *args):
        """
        TODO: Add docstring
        """
        return self._dispatch("get_score_max", *args)

    ### high-level operations
    def scale_to_0_1(self, col: NumericalColumn) -> NumericalColumn:
        """Return the column data scaled to range [0,1].
        If column contains only a single value, then column is scaled with sigmoid function.
        """
        assert isinstance(col, NumericalColumn)
        min_val = col.min()
        max_val = col.max()
        if min_val < max_val:
            return (col - min_val) / (max_val - min_val)
        else:
            return self.sigmoid(col)


functional = _Functional()
