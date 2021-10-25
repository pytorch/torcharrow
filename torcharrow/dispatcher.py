# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple, Callable, Dict, ClassVar

# ---------------------------------------------------------------------------
# column factory (class methods only!)

Device = str  # one of cpu, gpu
Typecode = str  # one of dtype.typecode


class Dispatcher:

    # singelton, append only, registering is idempotent
    _calls: ClassVar[Dict[Tuple[Typecode, Device], Callable]] = {}

    @classmethod
    def register(cls, key: Tuple[Typecode, Device], call: Callable):
        # key is tuple: (device,typecode)
        if key in Dispatcher._calls:
            if call == Dispatcher._calls[key]:
                return
            else:
                raise ValueError("keys for calls can only be registered once")
        Dispatcher._calls[key] = call

    @classmethod
    def lookup(cls, key: Tuple[Typecode, Device]) -> Callable:
        return Dispatcher._calls[key]
