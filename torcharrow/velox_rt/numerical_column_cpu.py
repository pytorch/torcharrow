# Copyright (c) Facebook, Inc. and its affiliates.
import math
import operator
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torcharrow as ta

# pyre-fixme[21]: Could not find module `torcharrow._torcharrow`.
import torcharrow._torcharrow as velox
import torcharrow.dtypes as dt
import torcharrow.pytorch as pytorch
from torcharrow._functional import functional
from torcharrow.dispatcher import Dispatcher
from torcharrow.expression import expression
from torcharrow.icolumn import IColumn
from torcharrow.inumerical_column import INumericalColumn
from torcharrow.trace import trace, traceproperty

from .column import ColumnFromVelox
from .typing import get_velox_type

# ------------------------------------------------------------------------------


class NumericalColumnCpu(ColumnFromVelox, INumericalColumn):
    """A Numerical Column on Velox backend"""

    # private
    # pyre-fixme[11]: Annotation `BaseColumn` is not defined as a type.
    def __init__(self, device, dtype, data: velox.BaseColumn):
        assert dt.is_boolean_or_numerical(dtype)
        INumericalColumn.__init__(self, device, dtype)
        self._data = data

        # TODO: Deprecate _finalized since Velox Column doesn't have "Builder" mode
        self._finalized = False

    # Any _empty must be followed by a _finalize; no other ops are allowed during this time
    @staticmethod
    def _empty(device, dtype):
        return NumericalColumnCpu(device, dtype, velox.Column(get_velox_type(dtype)))

    @staticmethod
    def _from_pysequence(
        device: str, data: Sequence[Union[int, float, bool]], dtype: dt.DType
    ):
        # pyre-fixme[16]: Module `torcharrow` has no attribute `_torcharrow`.
        velox_column = velox.Column(get_velox_type(dtype), data)
        return ColumnFromVelox._from_velox(
            device,
            dtype,
            velox_column,
            True,
        )

    @staticmethod
    def _from_arrow(device: str, array, dtype: dt.DType):
        import pyarrow as pa
        from pyarrow.cffi import ffi

        assert isinstance(array, pa.Array)

        c_schema = ffi.new("struct ArrowSchema*")
        ptr_schema = int(ffi.cast("uintptr_t", c_schema))
        c_array = ffi.new("struct ArrowArray*")
        ptr_array = int(ffi.cast("uintptr_t", c_array))
        # pyre-fixme[16]: `Array` has no attribute `_export_to_c`.
        array._export_to_c(ptr_array, ptr_schema)

        # pyre-fixme[16]: Module `torcharrow` has no attribute `_torcharrow`.
        velox_column = velox._import_from_arrow(
            get_velox_type(dtype), ptr_array, ptr_schema
        )

        # Make sure the ownership of c_schema and c_array have been transferred
        # to velox_column
        assert c_schema.release == ffi.NULL and c_array.release == ffi.NULL

        return ColumnFromVelox._from_velox(
            device,
            dtype,
            velox_column,
            True,
        )

    def _append_null(self):
        if self._finalized:
            raise AttributeError("It is already finalized.")
        self._data.append_null()

    def _append_value(self, value):
        if self._finalized:
            raise AttributeError("It is already finalized.")
        self._data.append(value)

    def _finalize(self):
        self._finalized = True
        return self

    def _valid_mask(self, ct):
        raise np.full((ct,), False, dtype=np.bool8)

    def __len__(self):
        return len(self._data)

    @property
    def null_count(self):
        return self._data.get_null_count()

    def _getdata(self, i):
        if i < 0:
            i += len(self._data)
        if self._data.is_null_at(i):
            return self.dtype.default
        else:
            return self._data[i]

    def _getmask(self, i):
        if i < 0:
            i += len(self._data)
        return self._data.is_null_at(i)

    # if-then-else ---------------------------------------------------------------

    def _if_else(self, then_, else_):
        """Vectorized if-then-else"""
        self._prototype_support_warning("_if_else")

        if not dt.is_boolean(self.dtype):
            raise TypeError("condition must be a boolean vector")
        if not isinstance(then_, IColumn):
            then_ = ta.Column(then_)
        if not isinstance(else_, IColumn):
            else_ = ta.Column(else_)
        lub = dt.common_dtype(then_.dtype, else_.dtype)

        if lub is None or dt.is_void(lub):
            raise TypeError(
                "then and else branches must have compatible types, got {then_.dtype} and {else_.dtype}, respectively"
            )
        if isinstance(then_, NumericalColumnCpu) and isinstance(
            else_, NumericalColumnCpu
        ):
            col = velox.Column(get_velox_type(lub))
            for i in range(len(self)):
                if self._getmask(i):
                    col.append_null()
                else:
                    col.append(
                        then_._getdata(i) if self._getdata(i) else else_._getdata(i)
                    )
            return ColumnFromVelox._from_velox(self.device, lub, col, True)

        else:
            # refer back to default handling...
            return INumericalColumn._if_else(self, then_, else_)

    # sorting, top-k, unique---------------------------------------------------

    @trace
    @expression
    def sort(
        self,
        columns: Optional[List[str]] = None,
        ascending=True,
        na_position="last",
    ):
        self._prototype_support_warning("sort")

        if columns is not None:
            raise TypeError("sort on numerical column can't have 'columns' parameter")
        res = []
        none_count = 0
        for i in range(len(self)):
            if self._getmask(i):
                none_count += 1
            else:
                res.append(self._getdata(i))
        res.sort(reverse=not ascending)

        # pyre-fixme[16]: Module `torcharrow` has no attribute `_torcharrow`.
        col = velox.Column(get_velox_type(self.dtype))
        if na_position == "first":
            for i in range(none_count):
                col.append_null()
        for value in res:
            col.append(value)
        if na_position == "last":
            for i in range(none_count):
                col.append_null()

        return ColumnFromVelox._from_velox(self.device, self.dtype, col, True)

    @trace
    @expression
    def _nlargest(
        self,
        n=5,
        columns: Optional[List[str]] = None,
        keep="first",
    ):
        if columns is not None:
            raise TypeError(
                "computing n-largest on numerical column can't have 'columns' parameter"
            )
        return self.sort(columns=None, ascending=False, na_position=keep).head(n)

    @trace
    @expression
    def _nsmallest(self, n=5, columns: Optional[List[str]] = None, keep="first"):
        if columns is not None:
            raise TypeError(
                "computing n-smallest on numerical column can't have 'columns' parameter"
            )

        return self.sort(columns=None, ascending=True, na_position=keep).head(n)

    @trace
    @expression
    def _nunique(self, drop_null=True):
        self._prototype_support_warning("_nunique")

        result = set()
        for i in range(len(self)):
            if self._getmask(i):
                if not drop_null:
                    result.add(None)
            else:
                result.add(self._getdata(i))
        return len(result)

    # operators ---------------------------------------------------------------
    def _checked_binary_op_call(
        self, other: Union[INumericalColumn, int, float, bool], op_name: str
    ) -> INumericalColumn:
        if isinstance(other, NumericalColumnCpu):
            if len(other) != len(self):
                raise TypeError("columns must have equal length")
            result_col = getattr(self._data, op_name)(other._data)
            result_dtype = result_col.dtype().with_null(
                self.dtype.nullable or other.dtype.nullable
            )
        else:
            # other is scalar
            assert (
                isinstance(other, int)
                or isinstance(other, float)
                or isinstance(other, bool)
            )
            result_col = getattr(self._data, op_name)(other)
            result_dtype = result_col.dtype().with_null(self.dtype.nullable)

        res = NumericalColumnCpu(self.device, result_dtype, result_col)
        res._finalized = True
        return res

    def _checked_comparison_op_call(
        self,
        other: Union[INumericalColumn, List[int], List[float], int, float],
        op_name: str,
    ) -> INumericalColumn:
        if isinstance(other, list):
            # Reuse the fromlist construction path
            other = ta.Column(other)
        return self._checked_binary_op_call(other, op_name)

    def _checked_arithmetic_op_call(
        self, other: Union[int, float, bool], op_name: str, fallback_py_op: Callable
    ) -> INumericalColumn:
        def should_use_py_impl(
            self, other: Union[INumericalColumn, int, float, bool]
        ) -> bool:
            # Arithmetic operations and bitwise operations are not supported in Velox
            # for boolean type, so let's fall back to Pyhton implementation when both
            # operands are boolean
            # TODO: Support native Velox execution for boolean ops
            if dt.is_boolean(self.dtype):
                if isinstance(other, NumericalColumnCpu) and dt.is_boolean(other.dtype):
                    return True
                # TODO
                # After we match PyTorch semantic to promote boolean type to integer
                # when other is an integer scalar, we should return False for that case
                elif not isinstance(other, NumericalColumnCpu):
                    return True
            return False

        if should_use_py_impl(self, other):
            self._prototype_support_warning(op_name)
            return self._py_arithmetic_op(other, fallback_py_op)

        return self._checked_binary_op_call(other, op_name)

    @trace
    @expression
    def __add__(self, other: Union[INumericalColumn, int, float]) -> INumericalColumn:
        # pyre-fixme[6]: For 1st param expected `Union[bool, float, int]` but got
        #  `Union[INumericalColumn, float, int]`.
        return self._checked_arithmetic_op_call(other, "add", operator.add)

    @trace
    @expression
    def __radd__(self, other: Union[int, float]) -> INumericalColumn:
        return self._checked_arithmetic_op_call(
            other, "radd", IColumn._swap(operator.add)
        )

    @trace
    @expression
    def __sub__(self, other: Union[INumericalColumn, int, float]) -> INumericalColumn:
        # pyre-fixme[6]: For 1st param expected `Union[bool, float, int]` but got
        #  `Union[INumericalColumn, float, int]`.
        return self._checked_arithmetic_op_call(other, "sub", operator.sub)

    @trace
    @expression
    def __rsub__(self, other: Union[int, float]) -> INumericalColumn:
        return self._checked_arithmetic_op_call(
            other, "rsub", IColumn._swap(operator.sub)
        )

    @trace
    @expression
    def __mul__(self, other: Union[INumericalColumn, int, float]) -> INumericalColumn:
        # pyre-fixme[6]: For 1st param expected `Union[bool, float, int]` but got
        #  `Union[INumericalColumn, float, int]`.
        return self._checked_arithmetic_op_call(other, "mul", operator.mul)

    @trace
    @expression
    def __rmul__(self, other: Union[int, float]) -> INumericalColumn:
        return self._checked_arithmetic_op_call(
            other, "rmul", IColumn._swap(operator.mul)
        )

    @trace
    @expression
    def __floordiv__(self, other):
        """
        Note: if a is integer type, a // 0 will raise ZeroDivisionError.
        otherwise a // 0 will return
            - float("inf") when a > 0
            - float("-inf") when a < 0
            - float("nan") when a == 0
        This behavior is different from __truediv__, but is consistent with Pytorch.
        """
        return self._rethrow_zero_division_error(
            lambda: self._checked_arithmetic_op_call(
                other, "floordiv", operator.floordiv
            )
        )

    @trace
    @expression
    def __rfloordiv__(self, other):
        """
        Note: if a is integer type, a // 0 will raise ZeroDivisionError.
        otherwise a // 0 will return
            - float("inf") when a > 0
            - float("-inf") when a < 0
            - float("nan") when a == 0
        This behavior is different from __rtruediv__, but is consistent with Pytorch.
        """
        return self._rethrow_zero_division_error(
            lambda: self._checked_arithmetic_op_call(
                other, "rfloordiv", IColumn._swap(operator.floordiv)
            )
        )

    @trace
    @expression
    def __truediv__(self, other):
        """
        Note: divide by zero will return
            - float("inf") when a > 0
            - float("-inf") when a < 0
            - float("nan") when a == 0
        instead of raising exceptions.
        """
        # Result of velox division of integers is integer, to achieve consistent python truediv behavior,
        # here we multiply self by 1.0 to force convert it to real type.
        # TODO: use type cast once https://github.com/facebookresearch/torcharrow/issues/143 completed,
        # and ensure cast performance is better than current.
        return (self * 1.0)._checked_arithmetic_op_call(
            other, "truediv", operator.truediv
        )

    @trace
    @expression
    def __rtruediv__(self, other):
        """
        Note: divide by zero will return
            - float("inf") when a > 0
            - float("-inf") when a < 0
            - float("nan") when a == 0
        instead of raising exceptions.
        """
        # Result of velox division of integers is integer, to achieve consistent python truediv behavior,
        # here we multiply self by 1.0 to force convert it to real type.
        # TODO: use type cast once https://github.com/facebookresearch/torcharrow/issues/143 completed,
        # and ensure cast performance is better than current.
        return (self * 1.0)._checked_arithmetic_op_call(
            other, "rtruediv", IColumn._swap(operator.truediv)
        )

    @trace
    @expression
    def __mod__(self, other: Union[INumericalColumn, int, float]) -> INumericalColumn:
        """
        Note: if a is integer type, a % 0 will raise ZeroDivisionError.
              if a is float type, a % 0 will be float('nan')
        """
        return self._rethrow_zero_division_error(
            # pyre-fixme[6]: For 1st param expected `Union[bool, float, int]` but
            #  got `Union[INumericalColumn, float, int]`.
            lambda: self._checked_arithmetic_op_call(other, "mod", operator.mod)
        )

    @trace
    @expression
    def __rmod__(self, other: Union[int, float]) -> INumericalColumn:
        """
        Note: if a is integer type, a % 0 will raise ZeroDivisionError.
              if a is float type, a % 0 will be float('nan')
        """
        return self._rethrow_zero_division_error(
            lambda: self._checked_arithmetic_op_call(
                other, "rmod", IColumn._swap(operator.mod)
            )
        )

    @trace
    @expression
    def __pow__(self, other):
        return self._checked_arithmetic_op_call(other, "pow", operator.pow)

    @trace
    @expression
    def __rpow__(self, other):
        return self._checked_arithmetic_op_call(
            other, "rpow", IColumn._swap(operator.pow)
        )

    @trace
    @expression
    def __eq__(
        self, other: Union[INumericalColumn, List[int], List[float], int, float]
    ):
        return self._checked_comparison_op_call(other, "eq")

    @trace
    @expression
    def __ne__(
        self, other: Union[INumericalColumn, List[int], List[float], int, float]
    ):
        return self._checked_comparison_op_call(other, "neq")

    @trace
    @expression
    def __lt__(
        self, other: Union[INumericalColumn, List[int], List[float], int, float]
    ):
        return self._checked_comparison_op_call(other, "lt")

    @trace
    @expression
    def __gt__(
        self, other: Union[INumericalColumn, List[int], List[float], int, float]
    ):
        return self._checked_comparison_op_call(other, "gt")

    @trace
    @expression
    def __le__(
        self, other: Union[INumericalColumn, List[int], List[float], int, float]
    ):
        return self._checked_comparison_op_call(other, "lte")

    @trace
    @expression
    def __ge__(
        self, other: Union[INumericalColumn, List[int], List[float], int, float]
    ):
        return self._checked_comparison_op_call(other, "gte")

    @trace
    @expression
    def __and__(self, other: Union[INumericalColumn, int]) -> INumericalColumn:
        # pyre-fixme[6]: For 1st param expected `Union[bool, float, int]` but got
        #  `Union[INumericalColumn, int]`.
        return self._checked_arithmetic_op_call(other, "bitwise_and", operator.__and__)

    @trace
    @expression
    def __rand__(self, other: Union[int]) -> INumericalColumn:
        return self._checked_arithmetic_op_call(
            other, "bitwise_rand", IColumn._swap(operator.__and__)
        )

    @trace
    @expression
    def __or__(self, other: Union[INumericalColumn, int]) -> INumericalColumn:
        # pyre-fixme[6]: For 1st param expected `Union[bool, float, int]` but got
        #  `Union[INumericalColumn, int]`.
        return self._checked_arithmetic_op_call(other, "bitwise_or", operator.__or__)

    @trace
    @expression
    def __ror__(self, other: Union[int]) -> INumericalColumn:
        return self._checked_arithmetic_op_call(
            other, "bitwise_ror", IColumn._swap(operator.__or__)
        )

    @trace
    @expression
    def __xor__(self, other: Union[INumericalColumn, int]) -> INumericalColumn:
        # pyre-fixme[6]: For 1st param expected `Union[bool, float, int]` but got
        #  `Union[INumericalColumn, int]`.
        return self._checked_arithmetic_op_call(other, "bitwise_xor", operator.__xor__)

    @trace
    @expression
    def __rxor__(self, other: Union[int]) -> INumericalColumn:
        return self._checked_arithmetic_op_call(
            other, "bitwise_rxor", IColumn._swap(operator.__xor__)
        )

    @trace
    @expression
    def __invert__(self):
        return ColumnFromVelox._from_velox(
            self.device, self.dtype, self._data.invert(), True
        )

    @trace
    @expression
    def __neg__(self):
        return ColumnFromVelox._from_velox(
            self.device, self.dtype, self._data.neg(), True
        )

    @trace
    @expression
    def __pos__(self):
        return self

    @trace
    @expression
    def isin(self, values):
        self._prototype_support_warning("isin")

        col = velox.Column(get_velox_type(dt.boolean))
        for i in range(len(self)):
            if self._getmask(i):
                col.append(False)
            else:
                col.append(self._getdata(i) in values)
        return ColumnFromVelox._from_velox(
            self.device, dt.Boolean(self.dtype.nullable), col, True
        )

    @trace
    @expression
    def abs(self):
        return ColumnFromVelox._from_velox(
            self.device, self.dtype, self._data.abs(), True
        )

    @trace
    @expression
    def cast(self, dtype: dt.DType):
        if self.null_count != 0 and not dtype.nullable:
            raise ValueError("Cannot cast a column with nulls to a non-nullable type")

        return ColumnFromVelox._from_velox(
            self.device, dtype, self._data.cast(get_velox_type(dtype).kind()), True
        )

    @trace
    @expression
    def ceil(self):
        return ColumnFromVelox._from_velox(
            self.device, self.dtype, self._data.ceil(), True
        )

    @trace
    @expression
    def floor(self):
        return ColumnFromVelox._from_velox(
            self.device, self.dtype, self._data.floor(), True
        )

    @trace
    @expression
    def round(self, decimals=0):
        return functional.torcharrow_round(self, decimals)._with_null(
            self.dtype.nullable
        )

    @trace
    @expression
    def log(self):
        return functional.torcharrow_log(self)._with_null(self.dtype.nullable)

    # data cleaning -----------------------------------------------------------

    @trace
    @expression
    def fill_null(self, fill_value: Union[dt.ScalarTypes, Dict]):
        self._prototype_support_warning("fill_null")

        if not isinstance(fill_value, IColumn._scalar_types):
            raise TypeError(f"fill_null with {type(fill_value)} is not supported")
        if not self.is_nullable:
            return self
        else:
            # pyre-fixme[16]: Module `torcharrow` has no attribute `_torcharrow`.
            col = velox.Column(get_velox_type(self.dtype))
            for i in range(len(self)):
                if self._getmask(i):
                    if isinstance(fill_value, Dict):
                        raise NotImplementedError()
                    else:
                        col.append(fill_value)
                else:
                    col.append(self._getdata(i))
            return ColumnFromVelox._from_velox(self.device, self.dtype, col, True)

    @trace
    @expression
    def drop_null(self, how="any"):
        self._prototype_support_warning("drop_null")

        if not self.is_nullable:
            return self
        else:
            col = velox.Column(get_velox_type(self.dtype))
            for i in range(len(self)):
                if self._getmask(i):
                    pass
                else:
                    col.append(self._getdata(i))
            return ColumnFromVelox._from_velox(self.device, self.dtype, col, True)

    @trace
    @expression
    def drop_duplicates(
        self,
        subset: Optional[List[str]] = None,
    ):
        self._prototype_support_warning("drop_duplicates")

        if subset is not None:
            raise TypeError(f"subset parameter for numerical columns not supported")
        seen = set()
        # pyre-fixme[16]: Module `torcharrow` has no attribute `_torcharrow`.
        col = velox.Column(get_velox_type(self.dtype))
        for i in range(len(self)):
            if self._getmask(i):
                col.append_null()
            else:
                current = self._getdata(i)
                if current not in seen:
                    col.append(current)
                    seen.add(current)
        return ColumnFromVelox._from_velox(self.device, self.dtype, col, True)

    # universal  ---------------------------------------------------------------

    @trace
    @expression
    def all(self):
        self._prototype_support_warning("all")

        for i in range(len(self)):
            if not self._getmask(i):
                value = self._getdata(i)
                if value == False:
                    return False
        return True

    @trace
    @expression
    def any(self):
        self._prototype_support_warning("any")

        for i in range(len(self)):
            if not self._getmask(i):
                value = self._getdata(i)
                if value == True:
                    return True
        return False

    def _accumulate_column(self, func, *, skipna=True, initial=None):
        it = iter(self)
        res = []
        total = initial
        rest_is_null = False
        if initial is None:
            try:
                total = next(it)
            except StopIteration:
                raise ValueError(f"cum[min/max] undefined for empty column.")
        if total is None:
            raise ValueError(f"cum[min/max] undefined for columns with row 0 as null.")

        res.append(total)
        for element in it:
            if rest_is_null:
                res.append(None)
                continue
            if element is None:
                if skipna:
                    res.append(None)
                else:
                    res.append(None)
                    rest_is_null = True
            else:
                total = func(total, element)
                res.append(total)
        return ta.Column(res, self.dtype)

    @trace
    @expression
    def _cummin(self):
        self._prototype_support_warning("_cummin")

        return self._accumulate_column(min, skipna=True, initial=None)

    @trace
    @expression
    def _cummax(self):
        self._prototype_support_warning("_cummax")

        return self._accumulate_column(max, skipna=True, initial=None)

    @trace
    @expression
    def cumsum(self):
        self._prototype_support_warning("cumsum")

        return self._accumulate_column(operator.add, skipna=True, initial=None)

    @trace
    @expression
    def _cumprod(self):
        self._prototype_support_warning("_cumprod")

        return self._accumulate_column(operator.mul, skipna=True, initial=None)

    @trace
    @expression
    def quantile(self, q, interpolation="midpoint"):
        self._prototype_support_warning("quantile")

        if len(self) == 0 or len(q) == 0:
            return []
        out = []
        s = sorted(self)
        for percent in q:
            k = (len(self) - 1) * (percent / 100)
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                out.append(s[int(k)])
                continue
            d0 = s[int(f)] * (c - k)
            d1 = s[int(c)] * (k - f)
            out.append(d0 + d1)
        return out

    # unique and montonic  ----------------------------------------------------

    @property  # type: ignore
    @traceproperty
    def is_unique(self):
        return self._nunique(drop_null=False) == len(self)

    @property  # type: ignore
    @traceproperty
    def is_monotonic_increasing(self):
        self._prototype_support_warning("is_monotonic_increasing")

        first = True
        prev = None
        for i in range(len(self)):
            if not self._getmask(i):
                current = self._getdata(i)
                if not first:
                    if prev > current:
                        return False
                else:
                    first = False
                prev = current
        return True

    @property  # type: ignore
    @traceproperty
    def is_monotonic_decreasing(self):
        self._prototype_support_warning("is_monotonic_decreasing")

        first = True
        prev = None
        for i in range(len(self)):
            if not self._getmask(i):
                current = self._getdata(i)
                if not first:
                    if prev < current:
                        return False
                else:
                    first = False
                prev = current
        return True

    # interop
    def to_arrow(self):
        import pyarrow as pa
        from pyarrow.cffi import ffi
        from torcharrow._interop import _dtype_to_arrowtype

        c_array = ffi.new("struct ArrowArray*")
        ptr_array = int(ffi.cast("uintptr_t", c_array))
        self._data._export_to_arrow(ptr_array)

        return pa.Array._import_from_c(ptr_array, _dtype_to_arrowtype(self.dtype))

    def _to_tensor_default(self):
        pytorch.ensure_available()
        import torch

        torch_dtype = pytorch._dtype_to_pytorch_dtype(self.dtype)

        # TODO: figure out zero copy from Velox vector
        arrow_array = self.to_arrow()
        res = torch.tensor(
            arrow_array.to_numpy(zero_copy_only=False), dtype=torch_dtype
        )
        if not self._dtype.nullable:
            return res

        presence = torch.tensor(
            arrow_array.is_valid().to_numpy(zero_copy_only=False), dtype=torch.bool
        )
        return pytorch.WithPresence(values=res, presence=presence)


# ------------------------------------------------------------------------------
# registering all numeric and boolean types for the factory...
_primitive_types: List[dt.DType] = [
    dt.Int8(),
    dt.Int16(),
    dt.Int32(),
    dt.Int64(),
    dt.Float32(),
    dt.Float64(),
    dt.Boolean(),
]
for t in _primitive_types:
    Dispatcher.register((t.typecode + "_empty", "cpu"), NumericalColumnCpu._empty)
    Dispatcher.register(
        (t.typecode + "_from_pysequence", "cpu"), NumericalColumnCpu._from_pysequence
    )
    Dispatcher.register(
        (t.typecode + "_from_arrow", "cpu"), NumericalColumnCpu._from_arrow
    )
