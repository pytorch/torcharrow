# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

import torch

# pyre-fixme[21]: Could not find module `torcharrow._torcharrow`.
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
        self.batch_first = batch_first
        self.with_presence = with_presence

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


@final
# pyre-fixme[39]: `(...) -> Any` is not a valid parent class.
class Sparse(Callable):
    """
    Predefined conversion callable for sparse features.

    is_jagged: bool, whether to output jagged format tensors with keys and key offsets
    is_combined: bool, whether to combine individual features as final output
    as_list: bool, whether to concate individual features to list as final output
    is_legacy: bool, whether to use by legacy HPC (only for combined format)
    """

    def __init__(
        self, *, is_jagged=False, is_combined=False, as_list=False, is_legacy=False
    ):
        if is_legacy and not is_combined:
            raise ValueError("Legacy HPC only supports CombinedSparse format")
        self.is_jagged = is_jagged
        self.is_combined = is_combined
        self.as_list = as_list
        self.is_legacy = is_legacy

    def __call__(self, df):
        # Only support CPU dataframe for now
        assert df.device == "cpu"

        # TODO: Implement OSS Sparse conversions
        raise NotImplementedError


@final
# pyre-fixme[39]: `(...) -> Any` is not a valid parent class.
class WeightedSparse(Callable):
    """
    Predefined conversion callable for weighted sparse features.

    is_jagged: bool, whether to output jagged format tensors with keys and key offsets
    is_combined: bool, whether to combine individual features as final output
    is_legacy: bool, whether to use by legacy HPC (only for combined format)
    """

    def __init__(self, *, is_jagged=False, is_combined=False, is_legacy=False):
        if is_legacy and not is_combined:
            raise ValueError("Legacy HPC only supports CombinedWeightedSparse format")
        self.is_jagged = is_jagged
        self.is_combined = is_combined
        self.is_legacy = is_legacy

    def __call__(self, df):
        # Only support CPU dataframe for now
        assert df.device == "cpu"

        # TODO: Implement OSS Weighted Sparse conversions
        raise NotImplementedError


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
        self.with_presence = with_presence

    def __call__(self, df):
        # Only support CPU dataframe for now
        assert df.device == "cpu"

        # Only support Scalar_No_Mask format for now
        if self.with_presence:
            raise NotImplementedError

        # TODO: Implement OSS Scalar conversions
        raise NotImplementedError
