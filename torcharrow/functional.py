# Copyright (c) Facebook, Inc. and its affiliates.
from types import ModuleType
from typing import Dict, Set

from torcharrow.icolumn import IColumn

from .scope import Scope


class _Functional(ModuleType):
    def __init__(self):
        super().__init__("torcharrow.functional")
        self._backend_functional: Dict[str, ModuleType] = {}
        self._factory_methods: Set[str] = set()

    def get_backend_functional(self, backend):
        backend_functional = self._backend_functional.get(backend)
        if backend_functional is None:
            raise AssertionError(
                f"Functional module for backend {backend} is not registered"
            )
        return backend_functional

    # dispatch the function call based on backend
    def create_dispatch_wrapper(self, op_name: str):
        def dispatch(*args):
            device = next(
                (arg.device for arg in args if isinstance(arg, IColumn)), None
            )
            if device is None:
                raise ValueError("None of the argument is Column")
            backend = _Functional.device_to_backend[device]

            # dispatch to backend functional namespace
            op = self.get_backend_functional(backend).__getattr__(op_name)
            return op(*args)

        def factory_dispatch(*args, size=None, device="cpu"):
            if size is None:
                raise AssertionError(
                    f"Factory method call {op_name} requires expclit size parameter"
                )

            # We assume that other args for factory functions are non-columns and don't even check args
            if isinstance(size, int):
                backend = _Functional.device_to_backend[device]
            else:
                # TODO: Support SizeProxy
                raise AssertionError(f"Unsupported size parameter type {type(size)}")

            # dispatch to backend functional namespace
            op = self.get_backend_functional(backend).__getattr__(op_name)
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

    # TODO: Perhaps this should be part of dispatch backend registration
    # (i.e. registered with register_dispatch_impl)
    def register_factory_methods(self, methods):
        self._factory_methods.update(methods)

    def __getattr__(self, op_name: str):
        wrapper = self.create_dispatch_wrapper(op_name)
        setattr(self, op_name, wrapper)
        return wrapper

    # In the future, TorchArrow is going to support more device/backend, such as gpu/libcudf
    device_to_backend = {"cpu": "velox"}


functional = _Functional()
