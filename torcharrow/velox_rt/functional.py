# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import types

from functools import partial

import torcharrow._torcharrow as ta
from torcharrow import functional as global_functional

from .column import ColumnCpuMixin


class VeloxFunctional(types.ModuleType):
    def __init__(self):
        super().__init__("torcharrow.velox_rt.functional")
        self._populate_udfs()

    @staticmethod
    def _dispatch(op_name, *args):
        wrapped_args = []

        first_col = next((arg for arg in args if isinstance(arg, ColumnCpuMixin)), None)
        if first_col is None:
            raise AssertionError("None of the argument is Column")
        length = len(first_col)

        for arg in args:
            if isinstance(arg, ColumnCpuMixin):
                wrapped_args.append(arg._data)
            else:
                # constant value
                wrapped_args.append(ta.ConstantColumn(arg, length))

        result_col = ta.generic_udf_dispatch(op_name, *wrapped_args)
        # Generic dispatch always assumes nullable
        result_dtype = result_col.dtype().with_null(True)

        return ColumnCpuMixin._from_velox(
            first_col.device, result_dtype, result_col, True
        )

    @staticmethod
    def create_dispatch_wrapper(op_name: str):
        def factory_dispatch(*args, size=None, device="cpu"):
            if size is None:
                raise AssertionError(
                    f"Factory method call {op_name} requires expclit size parameter"
                )

            wrapped_args = []
            for arg in args:
                # For factory dispatch, assume each arg is constant
                wrapped_args.append(ta.ConstantColumn(arg, size))
            wrapped_args.append(size)

            result_col = ta.factory_udf_dispatch(op_name, *wrapped_args)
            # Generic dispatch always assumes nullable
            result_dtype = result_col.dtype().with_null(True)

            return ColumnCpuMixin._from_velox(device, result_dtype, result_col, True)

        if op_name in global_functional._factory_methods:
            return factory_dispatch
        else:
            return partial(VeloxFunctional._dispatch, op_name)

    def _if(self, *args):
        # "if" is a keyword in Python, has to alias the dispatch stub
        # TODO: rename this stub name into if_else once tracing backend supports cusotmized dispatch stub
        return self._dispatch("if", *args)

    # TODO: automtically populate it
    # pyre-fixme[14]: `__getattr__` overrides method defined in `ModuleType`
    #  inconsistently.
    def __getattr__(self, op_name: str):
        dispatch_wrapper = self.create_dispatch_wrapper(op_name)
        setattr(self, op_name, dispatch_wrapper)
        return dispatch_wrapper

    def _populate_udfs(self):
        # TODO: implement this
        pass


velox_functional = VeloxFunctional()
global_functional.register_dispatch_impl("velox", velox_functional)
global_functional.register_factory_methods(["rand"])
