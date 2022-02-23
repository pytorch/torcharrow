# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import math
import operator
import statistics
from functools import partial
from typing import List, Optional, Union, Callable

import torcharrow.dtypes as dt
from torcharrow.dispatcher import Device

from .expression import expression
from .icolumn import IColumn
from .scope import Scope
from .trace import trace


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
                self._data,
                self.dtype,
                device=device,
                # pyre-fixme[16]: `NumericalColumnCpu` has no attribute `_mask`.
                mask=self._mask,
            )
        else:
            raise AssertionError("unexpected case")

    def log(self):
        """Returns a new column with the natural logarithm of the elements"""
        raise self._not_supported("log")

    @trace
    @expression
    def __mod__(self, other):
        """Vectorized a % b."""
        self._prototype_support_warning("__mod__")
        return self._py_arithmetic_op(other, operator.mod)

    @trace
    @expression
    def __add__(self, other):
        """Vectorized a + b."""
        self._prototype_support_warning("__add__")
        return self._py_arithmetic_op(other, operator.add)

    @trace
    @expression
    def __radd__(self, other):
        """Vectorized b + a."""
        self._prototype_support_warning("__radd__")
        return self._py_arithmetic_op(other, IColumn._swap(operator.add))

    @trace
    @expression
    def __sub__(self, other):
        """Vectorized a - b."""
        self._prototype_support_warning("__sub__")
        return self._py_arithmetic_op(other, operator.sub)

    @trace
    @expression
    def __rsub__(self, other):
        """Vectorized b - a."""
        self._prototype_support_warning("__rsub__")
        return self._py_arithmetic_op(other, IColumn._swap(operator.sub))

    @trace
    @expression
    def __mul__(self, other):
        """Vectorized a * b."""
        self._prototype_support_warning("__mul__")
        return self._py_arithmetic_op(other, operator.mul)

    @trace
    @expression
    def __rmul__(self, other):
        """Vectorized b * a."""
        self._prototype_support_warning("__rmul__")
        return self._py_arithmetic_op(other, IColumn._swap(operator.mul))

    @trace
    @expression
    def __floordiv__(self, other):
        """Vectorized a // b."""
        self._prototype_support_warning("__floordiv__")
        return self._py_arithmetic_op(other, operator.floordiv)

    @trace
    @expression
    def __rfloordiv__(self, other):
        """Vectorized b // a."""
        self._prototype_support_warning("__rfloordiv__")
        return self._py_arithmetic_op(other, IColumn._swap(operator.floordiv))

    @trace
    @expression
    def __truediv__(self, other):
        """Vectorized a / b."""
        self._prototype_support_warning("__truediv__")
        return self._py_arithmetic_op(other, operator.truediv, div="__truediv__")

    @trace
    @expression
    def __rtruediv__(self, other):
        """Vectorized b / a."""
        self._prototype_support_warning("__rtruediv__")
        return self._py_arithmetic_op(
            other, IColumn._swap(operator.truediv), div="__rtruediv__"
        )

    @trace
    @expression
    def __rmod__(self, other):
        """Vectorized b % a."""
        self._prototype_support_warning("__rmod__")
        return self._py_arithmetic_op(other, IColumn._swap(operator.mod))

    @trace
    @expression
    def __pow__(self, other):
        """Vectorized a ** b."""
        self._prototype_support_warning("__pow__")
        return self._py_arithmetic_op(other, operator.pow)

    @trace
    @expression
    def __rpow__(self, other):
        """Vectorized b ** a."""
        self._prototype_support_warning("__rpow__")
        return self._py_arithmetic_op(other, IColumn._swap(operator.pow))

    # describe ----------------------------------------------------------------
    @trace
    @expression
    def describe(
        self,
        percentiles=None,
        include: Union[List[dt.DType], Optional[List[dt.DType]]] = None,
        exclude: Union[List[dt.DType], Optional[List[dt.DType]]] = None,
    ):
        """
        Generate descriptive statistics.

        Parameters
        ----------
        percentiles - array-like, default None
            Defines which percentiles to calculate.  If None, uses [25,50,75].

        Examples
        --------
        >>> import torcharrow
        >>> t = ta.column([1,2,999,4])
        >>> t.describe()
          index  statistic      value
        -------  -----------  -------
              0  count          4
              1  mean         251.5
              2  std          498.335
              3  min            1
              4  25%            1.5
              5  50%            3
              6  75%          501.5
              7  max          999
        """

        # Not supported: datetime_is_numeric=False,
        if include is not None or exclude is not None:
            raise TypeError(
                f"'include/exclude columns' parameter for '{type(self).__name__}' not supported "
            )
        if percentiles is None:
            percentiles = [25, 50, 75]
        percentiles = sorted(set(percentiles))
        if len(percentiles) > 0:
            if percentiles[0] < 0 or percentiles[-1] > 100:
                raise ValueError("percentiles must be betwen 0 and 100")

        if dt.is_numerical(self.dtype):
            res = Scope._EmptyColumn(
                dt.Struct(
                    [dt.Field("statistic", dt.string), dt.Field("value", dt.float64)]
                )
            )
            res._append(("count", self._count()))
            res._append(("mean", self.mean()))
            res._append(("std", self.std()))
            res._append(("min", self.min()))
            values = self._quantile(percentiles, "midpoint")
            for p, v in zip(percentiles, values):
                res._append((f"{p}%", v))
            res._append(("max", self.max()))
            return res._finalize()
        else:
            raise ValueError(f"describe undefined for {type(self).__name__}.")

    @trace
    @expression
    def abs(self):
        """Absolute value of each element of the series."""
        return self._vectorize(abs, self.dtype)

    @trace
    @expression
    def ceil(self):
        """Rounds each value upward to the smallest integral"""
        return self._vectorize(math.ceil, self.dtype)

    @trace
    @expression
    def floor(self):
        """Rounds each value downward to the largest integral value"""
        return self._vectorize(math.floor, self.dtype)

    @trace
    @expression
    def round(self, decimals=0):
        """Round each value in a data to the given number of decimals.

        decimals : int, default 0
            Number of decimal places to round to. If decimals is negative,
            it specifies the number of positions to the left of the decimal point.
        """
        return self._vectorize(partial(round, ndigits=decimals), self.dtype)

    # cumsum
    @trace
    @expression
    def cumsum(self):
        """Return cumulative sum of the data."""
        self._prototype_support_warning("cumsum")
        self._check(dt.is_numerical, "cumsum")
        return self._accumulate(operator.add)

    @trace
    @expression
    def min(self):
        """
        Return the minimum of the non-null values.

        Examples
        --------
        >>> import torcharrow as ta
        >>> s = ta.column([1,2,None,4])
        >>> s.min()
        1
        """
        import pyarrow.compute as pc

        # TODO: use pc.min once upgrade to later version of PyArrow
        return pc.min_max(self.to_arrow())[0].as_py()

    @trace
    @expression
    def max(self):
        """
        Return the maximum of the non-null values.

        Examples
        --------
        >>> import torcharrow as ta
        >>> s = ta.column([1,2,None,4])
        >>> s.max()
        4
        """
        import pyarrow.compute as pc

        # TODO: use pc.max once upgrade to later version of PyArrow
        return pc.min_max(self.to_arrow())[1].as_py()

    @trace
    @expression
    def sum(self):
        """
        Return the sum of the non-null values.

        Examples
        --------
        >>> import torcharrow as ta
        >>> s = ta.column([1,2,None,4])
        >>> s.sum()
        7
        """
        import pyarrow.compute as pc

        return pc.sum(self.to_arrow()).as_py()

    # cummin/cummax/cumprod
    @trace
    @expression
    def _cummin(self):
        """Return cumulative minimum of the data."""
        self._prototype_support_warning("_cummin")
        return self._accumulate(min)

    @trace
    @expression
    def _cummax(self):
        """Return cumulative maximum of the data."""
        self._prototype_support_warning("_cummax")
        return self._accumulate(max)

    @trace
    @expression
    def _cumprod(self):
        """Return cumulative product of the data."""
        self._prototype_support_warning("_cumprod")
        self._check(dt.is_numerical, "_cumprod")
        return self._accumulate(operator.mul)

    @trace
    @expression
    def mean(self):
        self._check(dt.is_numerical, "mean")
        """
        Return the mean of the non-null values.

        Examples
        --------
        >>> import torcharrow as ta
        >>> s = ta.column([1,2,None,6])
        >>> s.mean()
        3.0
        """
        import pyarrow.compute as pc

        return pc.mean(self.to_arrow()).as_py()

    @trace
    @expression
    def std(self):
        """Return the stddev(s) of the data."""
        import math

        import pyarrow.compute as pc

        # PyArrow's variance/stdev returns 1/N * \sigma (X_i - mu)^2, while
        # we expect unbiased estimation of variance/standard deviation, which is
        # 1/(N-1) * \sigma (X_i - mu)^2
        N = len(self) - self.null_count
        return math.sqrt(pc.variance(self.to_arrow()).as_py() * N / (N - 1))

    @trace
    @expression
    def median(self):
        """Return the median of the values in the data."""
        self._prototype_support_warning("median")

        self._check(dt.is_numerical, "median")
        return statistics.median((float(i) for i in list(self._data_iter())))

    @trace
    @expression
    def mode(self):
        """Return the mode(s) of the data."""
        self._check(dt.is_numerical, "mode")

        import pyarrow as pa

        modes = pa.compute.mode(self.to_arrow())
        if isinstance(modes, pa.StructArray):
            # pyarrow.compute.mode returns StructArray in >=3.0. but StructScalar in 2.0
            assert len(modes) == 1
            modes = modes[0]

        return modes["mode"].as_py()

    def _py_arithmetic_op(self, other, fun, div=""):
        others = None
        other_dtype = None
        if isinstance(other, IColumn):
            others = other._items()
            other_dtype = other.dtype
        else:
            others = itertools.repeat((False, other))
            other_dtype = dt.infer_dtype_from_value(other)

        # TODO: should we just move _py_arithmetic_op to INumericColumn since it only works for boolean/numeric types
        if not dt.is_boolean_or_numerical(self.dtype) or not dt.is_boolean_or_numerical(
            other_dtype
        ):
            raise TypeError(f"{type(self).__name__}.{fun.__name__} is not supported")

        res = []
        if div != "":
            res_dtype = dt.Float64(self.dtype.nullable or other_dtype.nullable)
            for (m, i), (n, j) in zip(self._items(), others):
                # TODO Use error handling to mke this more efficient..
                if m or n:
                    res.append(None)
                elif div == "__truediv__" and j == 0:
                    res.append(None)
                elif div == "__rtruediv__" and i == 0:
                    res.append(None)
                else:
                    res.append(fun(i, j))
        else:
            res_dtype = dt.promote(self.dtype, other_dtype)
            if res_dtype is None:
                raise TypeError(f"{self.dtype} and {other_dtype} are incompatible")
            for (m, i), (n, j) in zip(self._items(), others):
                if m or n:
                    res.append(None)
                else:
                    res.append(fun(i, j))
        return Scope._FromPySequence(res, res_dtype)

    def _is_zero_division_error(self, ex: Exception) -> bool:
        ex_str = str(ex)
        return "division by zero" in ex_str or "Cannot divide by 0" in ex_str

    def _rethrow_zero_division_error(self, func: Callable) -> "INumericalColumn":
        try:
            result = func()
        except Exception as ex:
            # cast velox error to standard ZeroDivisionError
            if self._is_zero_division_error(ex):
                raise ZeroDivisionError
            raise ex
        return result
