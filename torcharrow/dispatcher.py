# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, ClassVar, Dict, Generic, Tuple, TypeVar

# ---------------------------------------------------------------------------
# column factory (class methods only!)

Device = str  # one of cpu, gpu
Typecode = str  # one of dtype.typecode


T = TypeVar("T")


class _Dispatcher(Generic[T]):
    def __init__(self):
        # append only, registering is idempotent
        self._calls: Dict[T, Callable] = {}

    def register(self, key: T, call: Callable):
        if key in self._calls:
            if call == self._calls[key]:
                return
            else:
                raise ValueError("keys for calls can only be registered once")
        self._calls[key] = call

    def lookup(self, key: T) -> Callable:
        return self._calls[key]


Dispatcher: _Dispatcher[Tuple[Typecode, Device]] = _Dispatcher()
