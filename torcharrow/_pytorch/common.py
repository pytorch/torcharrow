# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Generic, List, Optional, Tuple, TypeVar, Union

import torch  # type: ignore
import torcharrow as ta
import torcharrow.dtypes as dt
from torcharrow import dtypes
from torcharrow.dtypes import DType, is_list, is_map, is_struct
from torcharrow.scope import Scope
from typing_extensions import final

T = TypeVar("T")
KT = TypeVar("KT")


@dataclass
class WithPresence(Generic[T]):
    values: T
    # 1-D bool vector of per-element validity bit
    presence: torch.Tensor


@dataclass
class PackedList(Generic[T]):
    # 1-D int32 tensor
    offsets: torch.Tensor
    # Type hints don't really work because of recursive type remapping
    values: Union[T]


@dataclass
class PackedMap(Generic[KT, T]):
    # 1-D int32 tensor
    offsets: torch.Tensor
    keys: Union[KT]
    values: Union[T]


def infer_dtype_from_tensor(
    data: Union[PackedMap, PackedList, List, torch.Tensor, Tuple]
):
    if isinstance(data, WithPresence):
        t = infer_dtype_from_tensor(data.values)
        if t.nullable:
            raise TypeError("WithPresence structs can't be nested")
        return t.with_null()

    if isinstance(data, torch.Tensor):
        torcharrow_dtype_name = str(data.dtype)
        assert torcharrow_dtype_name.startswith("torch.")
        torcharrow_dtype_name = torcharrow_dtype_name[len("torch.") :]
        if torcharrow_dtype_name == "bool":
            torcharrow_dtype_name = "boolean"
        if not hasattr(dtypes, torcharrow_dtype_name):
            raise TypeError(f"Unexpected dtype for the tensor: {data.dtype}")
        return getattr(dtypes, torcharrow_dtype_name)

    if isinstance(data, PackedList):
        if not isinstance(data.offsets, torch.Tensor) or data.offsets.dtype not in [
            torch.int16,
            torch.int32,
            torch.int64,
        ]:
            raise TypeError(
                "PackedList.offsets is expected to be an integer-valued tensor"
            )
        return dtypes.List(infer_dtype_from_tensor(data.values))

    if isinstance(data, PackedMap):
        if not isinstance(data.offsets, torch.Tensor) or data.offsets.dtype not in [
            torch.int16,
            torch.int32,
            torch.int64,
        ]:
            raise TypeError(
                "PackedMap.offsets is expected to be an integer-valued tensor"
            )
        return dtypes.Map(
            infer_dtype_from_tensor(data.keys), infer_dtype_from_tensor(data.values)
        )

    if isinstance(data, tuple):
        types = [infer_dtype_from_tensor(x) for x in data]
        fields = getattr(data, "_fields", None)
        if fields is not None:
            assert len(fields) == len(types)
            return dtypes.Struct([dtypes.Field(n, t) for n, t in zip(fields, types)])
        else:
            return dtypes.Tuple(types)

    if isinstance(data, list):
        return dtypes.infer_dtype_from_prefix(data)

    raise TypeError(
        f"Can't infer datatype based on torch structure of type {type(data)}"
    )


def from_tensor(
    data: Union[PackedMap, PackedList, List, torch.Tensor, Tuple],
    dtype: Optional[DType] = None,
    device: str = "",
):
    if dtype is None:
        dtype = infer_dtype_from_tensor(data)
    device = device or Scope.default.device
    assert isinstance(dtype, DType)

    if isinstance(data, list):
        # if it's a python list - we're switching to python representation for this subtree. This path is also taken for strings, because they are represented as List[str] in PyTorch
        return ta.column(data, dtype=dtype)

    # handling nullability
    if isinstance(data, WithPresence):
        if not dtype.nullable:
            raise ValueError(
                f"Expected nullable type when the value is pytorch.WithPresence: {dtype}"
            )
        nested = from_tensor(data.values, dtype=dtype.with_null(False))
        # TODO: this implementation is very inefficient, we should wrap the column directly instead of round-tripping through python
        return ta.column(
            [(x if data.presence[i].item() else None) for i, x in enumerate(nested)],
            dtype=dtype,
            device=device,
        )
    if dtype.nullable:
        raise ValueError(
            f"nullable dtype must be represented by pytorch.WithPresence: {dtype}"
        )

    # container types
    if isinstance(data, PackedList):
        if not is_list(dtype):
            raise ValueError(
                f"Expected list type when the value is pytorch.PackedList: {dtype}"
            )
        assert isinstance(dtype, dtypes.List)  # make mypy happy
        nested = list(from_tensor(data.values, dtype=dtype.item_dtype))
        if not isinstance(data.offsets, torch.Tensor) or data.offsets.dtype not in [
            torch.int16,
            torch.int32,
            torch.int64,
        ]:
            raise ValueError(
                "PackedList.offsets is expected to be an integer-valued tensor"
            )
        # TODO: this implementation is very inefficient, we should wrap the column directly instead of round-tripping through python
        offsets = data.offsets.tolist()
        return ta.column(
            [nested[offsets[i] : offsets[i + 1]] for i in range(len(data.offsets) - 1)],
            dtype=dtype,
            device=device,
        )
    if isinstance(data, PackedMap):
        if not is_map(dtype):
            raise ValueError(
                f"Expected map type when the value is pytorch.PackedMap: {dtype}"
            )
        assert isinstance(dtype, dtypes.Map)  # make mypy happy
        nested_keys = list(from_tensor(data.keys, dtype=dtype.key_dtype, device=device))
        nested_values = list(
            from_tensor(data.values, dtype=dtype.item_dtype, device=device)
        )
        if not isinstance(data.offsets, torch.Tensor) or data.offsets.dtype not in [
            torch.int16,
            torch.int32,
            torch.int64,
        ]:
            raise ValueError(
                "PackedMap.offsets is expected to be an integer-valued tensor"
            )
        # TODO: this implementation is very inefficient, we should wrap the column directly instead of round-tripping through python
        offsets = data.offsets.tolist()
        return ta.column(
            [
                OrderedDict(
                    zip(
                        nested_keys[offsets[i] : offsets[i + 1]],
                        nested_values[offsets[i] : offsets[i + 1]],
                    )
                )
                for i in range(len(data.offsets) - 1)
            ],
            dtype=dtype,
            device=device,
        )
    if isinstance(data, tuple):
        # TODO: check that fields of named tuples match?
        if not is_struct(dtype):
            raise ValueError(
                f"Expected map type when the value is pytorch.PackedMap: {dtype}"
            )
        assert isinstance(dtype, dtypes.Struct)  # make mypy happy
        if len(data) != len(dtype.fields):
            raise ValueError(
                f"Tuple has {len(data)} fields when the type has {len(dtype.fields)}: {dtype}"
            )
        nested_fields = OrderedDict(
            (
                dtype.fields[i].name,
                list(from_tensor(data[i], dtype=dtype.fields[i].dtype, device=device)),
            )
            for i in range(len(data))
        )
        # TODO: this implementation is very inefficient, we should wrap the column directly instead of round-tripping through python
        return ta.dataframe(nested_fields, dtype=dtype, device=device)

    # numerics!
    if isinstance(data, torch.Tensor):
        # lazy way of converting types
        torcharrow_dtype_name = str(data.dtype)
        assert torcharrow_dtype_name.startswith("torch.")
        torcharrow_dtype_name = torcharrow_dtype_name[len("torch.") :]
        if torcharrow_dtype_name == "bool":
            torcharrow_dtype_name = "boolean"
        if not hasattr(dtypes, torcharrow_dtype_name):
            raise ValueError(f"Unexpected dtype for the tensor: {data.dtype}")
        if getattr(dtypes, torcharrow_dtype_name) != dtype:
            raise ValueError(
                f"Unexpected dtype {data.dtype} for the tensor, expected {dtype}"
            )
        # TODO: this implementation is very inefficient, we should wrap the column directly instead of round-tripping through python
        return ta.column(data.tolist(), dtype=dtype, device=device)

    raise ValueError(f"Unexpected data in `from_tensor`: {type(data)}")


class TensorConversion(abc.ABC):
    """
    PyTorch Conversion class.

    For built-in PyTorch conversion class, it dispatches to the corresponding internal methods in
    Column. Such as Column._to_tensor_default() or Column._to_tensor_padseq().

    Some PyTorch conversion strategy may be one way (e.g. doesn't support converting from PyTorch representation
    back to DataFrame), in that case only to_tensor needs to be implemented.

    It's also easy to extend with your own Torch conversion class, as long as the
    to_tensor and from_tensor is implemented.
    """

    @abc.abstractmethod
    def to_tensor(self, col) -> torch.Tensor:
        self._not_supported("to_tensor")

    def from_tensor(self, data: torch.Tensor, dtype, device):
        self._not_supported("from_tensor")

    def _not_supported(self, name):
        raise TypeError(f"{name} for type {type(self).__name__} is not supported")


class DefaultTensorConversion(TensorConversion):
    """
    Default Tensor conversion.
    """

    def to_tensor(self, col):
        return col._to_tensor_default()

    def from_tensor(self, data, dtype=None, device=None):
        return from_tensor(data, dtype, device)


@final
# pyre-fixme[39]: `(...) -> Any` is not a valid parent class.
class PadSequence(Callable):
    """
    Pad a batch of variable length numeric lists with ``padding_value``.
    See also https://github.com/pytorch/pytorch/blob/515d9fb2a99586e62cfb941cfc51e86e7d58c1f4/torch/nn/utils/rnn.py#L323-L359
    """

    def __init__(self, *, batch_first: bool = True, padding_value=0.0):
        self.batch_first = batch_first
        self.padding_value = padding_value

    def __call__(self, col) -> torch.Tensor:
        return col._to_tensor_pad_sequence(self.batch_first, self.padding_value)


def _dtype_to_pytorch_dtype(dtype: dt.DType) -> torch.dtype:
    # our names of types conveniently almost match
    # pyre-fixme[16]: `DType` has no attribute `name`.
    torch_dtype_name = "bool" if dtype.name == "boolean" else dtype.name
    if not hasattr(torch, torch_dtype_name):
        raise ValueError(f"Can't convert {dtype} to PyTorch")
    torch_dtype = getattr(torch, torch_dtype_name)

    return torch_dtype
