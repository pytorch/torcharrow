# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Callable

import torch
import torcharrow._torcharrow as _torcharrow
import torcharrow.dtypes as dt
from typing_extensions import final

from .common import _dtype_to_pytorch_dtype


@final
# pyre-fixme[39]: `(...) -> Any` is not a valid parent class.
class Dense(Callable):
    """
    Predefined conversion callable for dense features.

    batch_first: bool, whether keep batch_size as the first dim of output tensor
    with_presense: bool, whether to include vector of per-element validity bit
    """

    def __init__(self, *, batch_first=False, with_presence=False):
        self._batch_first = batch_first
        self._with_presence = with_presence

    def __call__(self, df):
        # Only support CPU dataframe for now
        assert df.device == "cpu"

        if self.with_presence:
            raise NotImplementedError

        if (not dt.is_struct(df.dtype)) or len(df.dtype.fields) == 0:
            raise ValueError(f"Unsupported dtype: {df.dtype}")

        data_tensor = torch.empty(
            (len(df.columns), len(df)),
            dtype=_dtype_to_pytorch_dtype(df.dtype.fields[0].dtype),
        )
        _torcharrow._populate_dense_features_nopresence(
            df._data, data_tensor.data_ptr()
        )

        if self.batch_first:
            # TODO: this would incur an extra copy. We can avoid this copy
            #    by fusing "tranpose-and-copy" when converting Velox vector to tensor.
            data_tensor = data_tensor.transpose(0, 1).contiguous()

        return data_tensor

    @property
    def batch_first(self) -> bool:
        return self._batch_first

    @property
    def with_presence(self) -> bool:
        return self._with_presence


@final
# pyre-fixme[39]: `(...) -> Any` is not a valid parent class.
class IndividualSparse(Callable):
    """
    Predefined conversion for individual (weighted) sparse features.

    Individual sparse conversion:
      From: ARRAY<BIGINT>
      To: Tuple<Tensor, Tensor, Tensor> for (offset, length, value)

    Individual weighted sparse conversion:
      From: ARRAY<ROW<BIGINT, REAL>>
      To: Tuple<Tensor, Tensor, Tensor> for (offset, length, value, weight)

    Args:
        weighted: bool. Whether feature is weighted sparse or not.
    """

    def __init__(
        self,
        *,
        weighted: bool,
    ):
        self._weighted = weighted

    def __call__(self, df):
        # Only support CPU dataframe for now
        assert df.device == "cpu"

        # TODO: Implement OSS Sparse conversions
        raise NotImplementedError

    @property
    def weighted(self) -> bool:
        return self._weighted


class PackOption(Enum):
    """
    Options to pack multiple (weighted) sparse features individually with different reperesentations.

    AS_PYDICT:
        Dict<String, Tuple<Tensor, Tensor, Tensor, (Tensor)>>.
        Key: field name; Value: <offset, length, value, (score)> tuple.

    AS_PYLIST:
        List<[Tensor, Tensor, (Tensor)] x N>.
        Iterates over each child sparse feature, and push length, value, (score) Tensor into the list
    """

    AS_PYDICT = 1
    AS_PYLIST = 2


@final
# pyre-fixme[39]: `(...) -> Any` is not a valid parent class.
class PackedSparse(Callable):
    """
    Predefined conversion for packed (weighted) sparse features
    into individual representation.
    Note: JaggedSparse is the recommended way for multiple (weighted) sparse
    features collation. JaggedSparse will combine multiple features into
    one continuous memory chunk and is GPU/accelerator friendly.

    From: ROW<ARRAY<BIGINT>...> or ROW<ARRAY<ROW<BIGINT, REAL>>...>
    To: Packed tensor representation of each feature alone

    Args:
        weighted: bool. Whether feature is weighted sparse or not.
        pack_option: PackOption. Representation of pack format.
    """

    def __init__(
        self,
        *,
        weighted: bool,
        pack_option: PackOption = PackOption.AS_PYDICT,
    ):
        self._weighted = weighted
        self._pack_option = pack_option

    def __call__(self, df):
        # Only support CPU dataframe for now
        assert df.device == "cpu"

        # TODO: Implement OSS Sparse conversions
        raise NotImplementedError

    @property
    def weighted(self) -> bool:
        return self._weighted

    @property
    def pack_option(self) -> PackOption:
        return self._pack_option


class CombineOption(Enum):
    """
    Options to combine packed (weighted) sparse features with different reperesentations.

    JAGGED:
        Preferred representation option.
        Convert into Tuple<List<str>, List<int>, Tensor, Tensor, (Tensor)>.
        Represent (keys, length per key, length, value, (weight))
        For JaggedTensor layout refer to: https://fburl.com/diffusion/lo7ssa8i

    COMBINED:
        Convert into Tuple<Tensor, Tensor, (Tensor)>.
        packed keys - 2D tensor,
        packed values - concat'ed 1D tensor,
        packed weights - concat'ed 1D tensor

    JAGGED_WITH_OFFSET:
        Convert into Tuple<Tensor, Tensor, (Tensor), List<int>, List<int>>.
        Represents (length, value, (weight), length per key, offset per key)
    """

    JAGGED = 1
    COMBINED = 2
    JAGGED_WITH_OFFSET = 3


@final
# pyre-fixme[39]: `(...) -> Any` is not a valid parent class.
class JaggedSparse(Callable):
    """
    Predefined conversion callable for packed (weighted) sparse features
    into combined representations.

    From: ROW<ARRAY<BIGINT>...> or ROW<ARRAY<ROW<BIGINT, REAL>>...>
    To: Combined tensor representation based on CombineOption

    Args:
       weighted: bool. Whether feature is weighted sparse or not.
       combine_option: CombineOption. Representation of combined format.
    """

    def __init__(
        self,
        *,
        weighted: bool,
        combine_option: CombineOption = CombineOption.JAGGED,
    ):
        self._weighted = weighted
        self._combine_option = combine_option

    def __call__(self, df):
        # Only support CPU dataframe for now
        assert df.device == "cpu"

        # TODO: Implement OSS Sparse conversions
        raise NotImplementedError

    @property
    def weighted(self) -> bool:
        return self._weighted

    @property
    def combine_option(self) -> CombineOption:
        return self._combine_option


@final
# pyre-fixme[39]: `(...) -> Any` is not a valid parent class.
class Embedding(Callable):
    """
    Predefined conversion callable for embedding features.
    """

    def __call__(self, df):
        # Only support CPU dataframe for now
        assert df.device == "cpu"

        # TODO: Implement OSS Embedding conversions
        raise NotImplementedError


@final
# pyre-fixme[39]: `(...) -> Any` is not a valid parent class.
class Scalar(Callable):
    """
    Predefined conversion callable for scalar features.

    with_presense: bool, whether to include vector of per-element validity bit
    """

    def __init__(self, *, with_presence=False):
        self._with_presence = with_presence

    def __call__(self, df):
        # Only support CPU dataframe for now
        assert df.device == "cpu"

        # Only support Scalar_No_Mask format for now
        if self.with_presence:
            raise NotImplementedError

        # TODO: Implement OSS Scalar conversions
        raise NotImplementedError

    @property
    def with_presence(self) -> bool:
        return self._with_presence
