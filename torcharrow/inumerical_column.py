# Copyright (c) Facebook, Inc. and its affiliates.
import torcharrow.dtypes as dt
from torcharrow.dispatcher import Device

from .icolumn import IColumn
from .scope import Scope


class INumericalColumn(IColumn):
    """Abstract Numerical Column"""

    # private
    def __init__(self, device, dtype):
        assert dt.is_boolean_or_numerical(dtype)
        super().__init__(device, dtype)

    # Note all numerical column implementations inherit from INumericalColumn

    def to(self, device: Device):
        from .velox_rt import NumericalColumnCpu

        if self.device == device:
            return self
        elif isinstance(self, NumericalColumnCpu):
            return Scope.default._FullColumn(
                self._data, self.dtype, device=device, mask=self._mask
            )
        else:
            raise AssertionError("unexpected case")
