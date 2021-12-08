# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Optional

from .dispatcher import Dispatcher
from .icolumn import IColumn
from .idataframe import DataFrame, IDataFrame
from .scope import Scope


def _from_arrow_array(
    array,  # type: pa.Array
    nullable: Optional[bool] = None,
    device: str = "",
) -> IColumn:
    device = device or Scope.default.device

    import pyarrow as pa
    from torcharrow._interop import _arrowtype_to_dtype

    assert isinstance(array, pa.Array)

    if nullable is None:
        # Using the most narrow type we can, we (i) don't restrict in any
        # way where it can be used (since we can pass a narrower typed
        # non-null column to a function expecting a nullable type, but not
        # vice versa), (ii) when we bring in a stricter type system in Velox
        # to allow functions to only be invokable on non-null types we
        # increase the amount of places we can use the from_arrow result
        nullable = array.null_count > 0
    if not nullable and array.null_count > 0:
        raise RuntimeError("Cannot store nulls in a non-nullable column")
    dtype = _arrowtype_to_dtype(array.type, nullable)

    device = device or Scope.default.device

    call = Dispatcher.lookup((dtype.typecode + "_fromarrow", device))

    return call(device, array, dtype)


def _from_arrow_table(
    table,  # type: pa.Table
    device: str = "",
) -> IDataFrame:
    device = device or Scope.default.device

    import pyarrow as pa

    assert isinstance(table, pa.Table)

    # For now Arrow Table -> TA DataFrame is implemented by decomposing the table
    # into Arrow Arrays and then zero-copy converting the Arrays into TA Columns
    # and finally constructing a TA DataFrame from the Columns. The parent struct
    # Array of the Arrow Table (RecordBatch) is not zero-copy converted into TA
    # DataFrame, which is fine since the heavy part of the entire data is the
    # Columns and we do zero-copy conversion for them. We will be able to do
    # zero-copy conversion for the parent struct Array as well once we support
    # from_arrow for struct type

    # May not be zero-copy here if multiple chunks need to be combined
    table.combine_chunks()

    df_data = {}
    for i in range(0, len(table.schema)):
        field = table.schema.field(i)
        assert len(table[i].chunks) == 1
        df_data[field.name] = _from_arrow_array(
            table[i].chunk(0), field.nullable, device
        )

    return DataFrame(df_data, device=device)
