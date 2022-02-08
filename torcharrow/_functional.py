# Copyright (c) Facebook, Inc. and its affiliates.
from types import ModuleType
from typing import Dict, Set

from torcharrow.icolumn import IColumn


class _Functional(ModuleType):
    __file__ = "_functional.py"

    # In the future, TorchArrow is going to support more device/dispatch_key, such as gpu/libcudf
    _device_to_dispatch_key = {"cpu": "velox"}
    _column_class_to_dispatch_key = {}

    def __init__(self):
        super().__init__("torcharrow.functional")
        self._backend_functional: Dict[str, ModuleType] = {}
        self._factory_methods: Set[str] = set()

    # TODO: prefix all methods with "_"
    def get_backend_functional(self, dispatch_key):
        backend_functional = self._backend_functional.get(dispatch_key)
        if backend_functional is None:
            raise AssertionError(
                f"Functional module for backend {dispatch_key} is not registered"
            )
        return backend_functional

    # dispatch the function call based on backend
    def create_dispatch_wrapper(self, op_name: str):
        def dispatch(*args):
            # Caculate dispatch key based on input
            col_arg: IColumn = next(
                (arg for arg in args if isinstance(arg, IColumn)), None
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
            return dispatch

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

    def __getattr__(self, op_name: str):
        wrapper = self.create_dispatch_wrapper(op_name)
        setattr(self, op_name, wrapper)
        return wrapper


functional = _Functional()
