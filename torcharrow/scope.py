# Copyright (c) Facebook, Inc. and its affiliates.
import json
import typing as ty
import warnings

import torcharrow.dtypes as dt

from .column_factory import ColumnFactory, Device
from .trace import Trace, trace

# ---------------------------------------------------------------------------
# Scope, pipelines global state...

# helper class


class Counter:
    def __init__(self):
        self._value = 0

    @property
    def value(self):
        return self._value

    def next(self):
        n = self._value
        self._value += 1
        return n


class Scope:

    default: ty.ClassVar["Scope"]

    default_config: ty.Dict = {
        "device": "cpu",
        "tracing": False,
        "types_to_trace": [],
    }

    def __init__(self, config: ty.Union[dict, str, None] = None):
        if config is None:
            self.config = type(self).default_config
        elif isinstance(config, str):
            path = config
            self.config = {**type(self).default_config, **json.load(open(path))}
        elif isinstance(config, dict):
            self.config = {**type(self).default_config, **config}

        self.ct = Counter()
        self.id = "s0"
        self._scope = self

        tracing = ty.cast(bool, self.config["tracing"])
        types = ty.cast(ty.Iterable[ty.Type], self.config["types_to_trace"])
        self.trace = Trace(tracing, tuple(types))

    # device handling --------------------------------------------------------

    @property
    def device(self):
        return self.config["device"]

    # tracing handling --------------------------------------------------------
    @property
    def tracing(self):
        return self.config["tracing"]

    # only one scope --------------------------------------------------------

    def is_same(self, other):
        return id(self) == id(other)

    def check_is_same(self, other):
        if id(self) != id(other) or self.device != other.device:
            raise TypeError("scope and device must be the same")

    def check_are_same(self, others):
        if not all(
            self.is_same(other) and self.device == other.device for other in others
        ):
            raise TypeError("scope and device must be the same")

    # column factory -----------------------------------------------------------

    @staticmethod
    def _require_column_constructors_to_be_registered():
        from .idataframe import DataFrame
        from .ilist_column import IListColumn
        from .imap_column import IMapColumn
        from .istring_column import IStringColumn

        from .velox_rt import NumericalColumnCpu

    # private column/dataframe constructors -----------------------------------
    def _EmptyColumn(self, dtype, device=""):
        """
        Column row builder method

        * This methods return a "column builder" that may not support all the methods of column.
        * Backend may provide more efficient _append_row implementation that doesn't need to copy the whole column per appending.
        * _finalize has to be called before returning the Column to user

        TODO: rename this method to _EmptyColumnBuilder to emphasize it returns a builder
        """
        Scope._require_column_constructors_to_be_registered()

        device = device if device != "" else self.device
        call = ColumnFactory.lookup((dtype.typecode + "_empty", device))

        return call(self, device, dtype)

    def _FullColumn(self, data, dtype, device="", mask=None):
        """
        Column vector builder method -- data is already in right form
        """
        Scope._require_column_constructors_to_be_registered()

        device = device if device != "" else self.device
        call = ColumnFactory.lookup((dtype.typecode + "_full", device))

        return call(self, device, data, dtype, mask)

    def _FromPyList(self, data: ty.List, dtype: dt.DType, device=""):
        """
        Convert from plain Python container (list of scalars or containers).
        """
        Scope._require_column_constructors_to_be_registered()

        device = device if device != "" else self.device
        # TODO: rename the dispatch key to be "_from_python"
        call = ColumnFactory.lookup((dtype.typecode + "_fromlist", device))

        return call(self, device, data, dtype)

    # public column (dataframe) constructors ----------------------------------
    def arrange(
        self,
        start: int,
        stop: int,
        step: int = 1,
        dtype: ty.Optional[dt.DType] = None,
        device: ty.Optional[Device] = None,
    ):
        return self.Column(list(range(start, stop, step)), dtype, device)

    @trace
    def Column(
        self,
        data=None,
        dtype: ty.Optional[dt.DType] = None,
        device: Device = "",
    ):
        """
        Column factory method
        """

        device = self.device if device is None else device

        if (data is None) and (dtype is None):
            raise TypeError(
                f"Column requires data and/or dtype parameter {data} {dtype}"
            )

        if isinstance(data, dt.DType) and isinstance(dtype, dt.DType):
            raise TypeError("Column can only have one dtype parameter")

        if isinstance(data, dt.DType):
            (data, dtype) = (dtype, data)

        # data is Python list
        if isinstance(data, ty.List):
            # TODO: infer the type from the whole list
            dtype = dtype or dt.infer_dtype_from_prefix(data[:7])
            if dtype is None or dt.is_any(dtype):
                raise ValueError("Column cannot infer type from data")
            if dt.contains_tuple(dtype):
                raise TypeError("Cannot infer type from Python tuple")
            return self._FromPyList(data, dtype, device)

        if data is not None:
            warnings.warn(
                "Constructing column from non Python list may result in degenerated performance"
            )
            # TODO: Shall we only allow constructing Column from list?

        # data is already IColumn
        if self._is_column(data):
            raise ValueError("data is already IColumn")

        # dtype given, optional data
        if isinstance(dtype, dt.DType):
            col = self._EmptyColumn(dtype, device)
            if data is not None:
                for i in data:
                    col._append(i)
            return col._finalize()

        # data given, optional column
        if data is not None:
            if isinstance(data, ty.Sequence):
                data = iter(data)
            if isinstance(data, ty.Iterable):
                prefix = []
                for i, v in enumerate(data):
                    prefix.append(v)
                    if i > 5:
                        break
                dtype = dt.infer_dtype_from_prefix(prefix)
                if dtype is None or dt.is_any(dtype):
                    raise ValueError("Column cannot infer type from data")

                if dt.is_tuple(dtype):
                    # TODO fix me
                    raise TypeError(
                        "Column cannot be used to created structs, use Dataframe constructor instead"
                    )
                col = self._EmptyColumn(dtype, device=device)
                # add prefix and ...
                for p in prefix:
                    col._append(p)
                # ... continue enumerate the data
                for _, v in enumerate(data):
                    col._append(v)
                return col._finalize()
            else:
                raise TypeError(
                    f"data parameter of ty.Sequence type expected (got {type(dtype).__name__})"
                )
        else:
            raise AssertionError("unexpected case")

    # public dataframe (column)) constructor
    @trace
    def DataFrame(
        self,
        data=None,  # : DataOrDTypeOrNone = None,
        dtype=None,  # : ty.Optional[dt.DType] = None,
        columns=None,  # : ty.Optional[List[str]] = None,
        device="",
    ):
        """
        Dataframe factory method
        """

        if data is None and dtype is None:
            assert columns is None
            return self._EmptyColumn(dt.Struct([]), device=device)._finalize()

        if data is not None and isinstance(data, dt.DType):
            if dtype is not None and isinstance(dtype, dt.DType):
                raise TypeError("Dataframe can only have one dtype parameter")
            dtype = data
            data = None

        # dtype given, optional data
        if dtype is not None:
            if not dt.is_struct(dtype):
                raise TypeError(
                    f"Dataframe takes a dt.Struct dtype as parameter (got {dtype})"
                )
            dtype = ty.cast(dt.Struct, dtype)
            if data is None:
                return self._EmptyColumn(dtype, device=device)._finalize()
            else:
                if isinstance(data, ty.Sequence):
                    res = self._EmptyColumn(dtype, device=device)
                    for i in data:
                        res._append(i)
                    return res._finalize()
                elif isinstance(data, ty.Mapping):
                    res = {}
                    dtype_fields = {f.name: f.dtype for f in dtype.fields}

                    if len(data) != len(dtype_fields):
                        raise TypeError(
                            f"""dtype provides {len(dtype.fields)} fields: {dtype_fields.keys()}
but data only provides {len(data)} fields: {data.keys()}
"""
                        )

                    for n, c in data.items():
                        if n not in dtype_fields:
                            raise AttributeError(
                                f"Column {n} is present in the data but absent in explicitly provided dtype"
                            )
                        if Scope._is_column(c):
                            if c.dtype != dtype_fields[n]:
                                raise TypeError(
                                    f"Wrong type for column {n}: dtype specifies {dtype_fields[n]} while column of {c.dtype} is provided"
                                )
                        else:
                            c = self.Column(c, dtype_fields[n])
                        res[n] = c

                    return self._FullColumn(res, dtype)

                else:
                    raise TypeError(
                        f"Dataframe does not support constructor for data of type {type(data).__name__}"
                    )

        # data given, optional column
        if data is not None:
            if isinstance(data, ty.Sequence):
                prefix = []
                for i, v in enumerate(data):
                    prefix.append(v)
                    if i > 5:
                        break
                dtype = dt.infer_dtype_from_prefix(prefix)
                if dtype is None or not dt.is_tuple(dtype):
                    raise TypeError("Dataframe cannot infer struct type from data")
                dtype = ty.cast(dt.Tuple, dtype)
                columns = [] if columns is None else columns
                if len(dtype.fields) != len(columns):
                    raise TypeError("Dataframe column length must equal row length")
                dtype = dt.Struct(
                    [dt.Field(n, t) for n, t in zip(columns, dtype.fields)]
                )
                res = self._EmptyColumn(dtype, device=device)
                for i in data:
                    res._append(i)
                return res._finalize()
            elif isinstance(data, ty.Mapping):
                res = {}
                for n, c in data.items():
                    if Scope._is_column(c):
                        res[n] = c
                    elif isinstance(c, ty.Sequence):
                        res[n] = self.Column(c, device=device)
                    else:
                        raise TypeError(
                            f"dataframe does not support constructor for column data of type {type(c).__name__}"
                        )
                return self._FullColumn(
                    res, dtype=dt.Struct([dt.Field(n, c.dtype) for n, c in res.items()])
                )
            elif Scope.is_dataframe(data):
                return data
            else:
                raise TypeError(
                    f"dataframe does not support constructor for data of type {type(data).__name__}"
                )
        else:
            raise AssertionError("unexpected case")

    # interop -----------------------------------------------------------------
    def from_arrow(self, data, dtype: ty.Optional[dt.DType] = None, device=""):
        """
        Convert from arrow array or table
        """
        import pyarrow as pa
        from torcharrow._interop import _arrowtype_to_dtype

        assert isinstance(data, pa.Array) or isinstance(data, pa.Table)

        dtype = dtype or _arrowtype_to_dtype(data.type, data.null_count > 0)
        device = device or self.device

        call = ColumnFactory.lookup((dtype.typecode + "_fromarrow", device))

        return call(self, device, data, dtype)

    Frame = DataFrame

    # helper ------------------------------------------------------------------
    @staticmethod
    def _is_column(c):
        # NOTE: should be isinstance(c, IColumn)
        # But can't do tha due to cyclic reference, so we use ...
        return hasattr(c, "_dtype") and hasattr(c, "_scope") and hasattr(c, "_device")

    @staticmethod
    def _is_dataframe(c):
        # NOTE: should be isinstance(c, DataFrame)
        # But can't do tha due to cyclic reference, so we use ...
        return (
            hasattr(c, "_dtype")
            and hasattr(c, "_scope")
            and hasattr(c, "_device")
            and hasattr(c, "_field_data")
        )


# ------------------------------------------------------------------------------
# registering the default scope
Scope.default = Scope()
