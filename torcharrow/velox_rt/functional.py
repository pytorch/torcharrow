# Copyright (c) Facebook, Inc. and its affiliates.
import types

import torcharrow._torcharrow as ta
from torcharrow.functional import _Functional
from torcharrow.functional import functional as functional_registry
from torcharrow.scope import Scope

from .column import ColumnFromVelox


class VeloxFunctional(types.ModuleType):
    def __init__(self):
        super().__init__("torcharrow.velox_rt.functional")
        self._populate_udfs()

    @staticmethod
    def create_dispatch_wrapper(op_name: str):
        def dispatch(*args):
            wrapped_args = []

            first_col = next(
                (arg for arg in args if isinstance(arg, ColumnFromVelox)), None
            )
            if first_col is None:
                raise AssertionError("None of the argument is Column")
            length = len(first_col)

            for arg in args:
                if isinstance(arg, ColumnFromVelox):
                    wrapped_args.append(arg._data)
                else:
                    # constant value
                    wrapped_args.append(ta.ConstantColumn(arg, length))

            result_col = ta.generic_udf_dispatch(op_name, *wrapped_args)
            # Generic dispatch always assumes nullable
            result_dtype = result_col.dtype().with_null(True)

            return ColumnFromVelox.from_velox(
                first_col.device, result_dtype, result_col, True
            )

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

            return ColumnFromVelox.from_velox(device, result_dtype, result_col, True)

        if op_name in functional_registry._factory_methods:
            return factory_dispatch
        else:
            return dispatch

    # TODO: automtically populate it
    def __getattr__(self, op_name: str):
        dispatch_wrapper = self.create_dispatch_wrapper(op_name)
        setattr(self, op_name, dispatch_wrapper)
        return dispatch_wrapper

    def _populate_udfs(self):
        # TODO: implement this
        pass


velox_functional = VeloxFunctional()
functional_registry.register_dispatch_impl("velox", velox_functional)
functional_registry.register_factory_methods(["rand"])
