# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import unittest
from typing import List

import numpy as np
import numpy.testing
import torch
import torcharrow as ta
import torcharrow.dtypes as dt
import torcharrow.pytorch as tap
from torcharrow import functional

try:
    # @manual=//configerator/structs/fblearner/interface:features-py
    import pyarrow.parquet as pq

    HAS_PYARROW_PARQUET = True
except ImportError:
    HAS_PYARROW_PARQUET = False
skipIfNoPyArrowParquet = unittest.skipIf(not HAS_PYARROW_PARQUET, "no PyArrow Parquet")

# Based on https://github.com/facebookresearch/torchrec/blob/9a8a4ad631bbccd7cd8166b7e6d7607e2560d2bd/torchrec/datasets/criteo.py#L37-L46
INT_FEATURE_COUNT = 13
CAT_FEATURE_COUNT = 26
DEFAULT_LABEL_NAME = "label"
DEFAULT_INT_NAMES: List[str] = [f"int_{idx}" for idx in range(INT_FEATURE_COUNT)]
DEFAULT_CAT_NAMES: List[str] = [f"cat_{idx}" for idx in range(CAT_FEATURE_COUNT)]

DTYPE = dt.Struct(
    [
        dt.Field(DEFAULT_LABEL_NAME, dt.int8),
        dt.Field(
            "dense_features",
            dt.Struct(
                [
                    dt.Field(int_name, dt.Int32(nullable=True))
                    for int_name in DEFAULT_INT_NAMES
                ]
            ),
        ),
        dt.Field(
            "sparse_features",
            dt.Struct(
                [
                    dt.Field(cat_name, dt.Int32(nullable=True))
                    for cat_name in DEFAULT_CAT_NAMES
                ]
            ),
        ),
    ]
)

# TODO: allow to use to_tensor with Callable
# TODO: implement conversion in native C++
class _DenseConversion(tap.TensorConversion):
    # pyre-fixme[14]: `to_tensor` overrides method defined in `TensorConversion`
    #  inconsistently.
    def to_tensor(self, df: ta.DataFrame):
        # Default to_tensor, each field is a Tensor
        tensors = df.to_tensor()

        # Stack them into BatchSize * NumFields Tensor
        return torch.cat(
            [column_tensor.values.unsqueeze(0).T for column_tensor in tensors], dim=1
        )


# Based on https://github.com/facebookresearch/torchrec/blob/main/torchrec/datasets/criteo.py#L441-L456
# TODO: this is not a general purpose JaggedTensor conversion -- it leverages the fact that in Criteo preproc, each array is single element
# TODO: implement general purpose jagged sparse tensor conversion in native C++
class _CriteoJaggedTensorConversion(tap.TensorConversion):
    # pyre-fixme[14]: `to_tensor` overrides method defined in `TensorConversion`
    #  inconsistently.
    def to_tensor(self, df: ta.DataFrame):
        # TODO: Implement df.size(), similar to Pandas
        num_arrays = len(df) * len(df.columns)

        keys: List[str] = df.columns
        lengths: torch.Tensor = torch.ones(
            (num_arrays,),
            dtype=torch.int32,
        )
        length_per_key: List[int] = len(df.columns) * [len(df)]

        # Generate values
        tensors = df.to_tensor()
        # Stack them into BatchSize * NumFields Tensor
        packed_tensor = torch.cat(
            [
                # TODO: this ".values.values.values" is horrible for now. It will no longer exist once we have
                # native Jagged Tensor conversion.
                #
                # We should also make torcharrow.functional annotate nullable type in a less pessmistic way.
                column_tensor.values.values.values.unsqueeze(0).T
                for column_tensor in tensors
            ],
            dim=1,
        )
        # Flatten the tensors into one-dimension with columnar ordering
        values = packed_tensor.transpose(1, 0).reshape(-1)

        return (keys, values, lengths, length_per_key)


@skipIfNoPyArrowParquet
class CriteoIntegrationTest(unittest.TestCase):
    NUM_ROWS = 128
    TEMPORARY_PARQUETY_FILE = "_test_criteo.parquet"

    def setUp(self) -> None:
        # Generate some random data
        random.seed(42)

        rows = []
        for _ in range(type(self).NUM_ROWS):
            label = 1 if random.randrange(100) < 4 else 0
            dense_features_struct = tuple(
                random.randrange(1000) if random.randrange(100) < 80 else None
                for j in range(INT_FEATURE_COUNT)
            )
            sparse_features_struct = tuple(
                random.randrange(-(2 ** 31), 2 ** 31)
                if random.randrange(100) < 98
                else None
                for j in range(CAT_FEATURE_COUNT)
            )

            rows.append((label, dense_features_struct, sparse_features_struct))

        self.RAW_ROWS = rows
        df = ta.dataframe(rows, dtype=DTYPE)

        pq.write_table(df.to_arrow(), self.TEMPORARY_PARQUETY_FILE)

    def tearDown(self) -> None:
        os.remove(self.TEMPORARY_PARQUETY_FILE)

    @staticmethod
    def preproc(df: ta.DataFrame) -> ta.DataFrame:
        # 1. fill null values
        df["dense_features"] = df["dense_features"].fill_null(0)
        df["sparse_features"] = df["sparse_features"].fill_null(0)

        # 2. apply log(x+3) on dense features
        df["dense_features"] = (df["dense_features"] + 3).log()

        # 3. Pack each categorical feature as an single element array
        sparse_features = df["sparse_features"]
        for field in sparse_features.columns:
            sparse_features[field] = functional.array_constructor(
                sparse_features[field]
            )

        # FIXME: we shouldn't need to "put back the struct column"
        df["sparse_features"] = sparse_features

        df["label"] = df["label"].cast(dt.int32)

        return df

    def test_criteo_transform(self) -> None:
        # Read data from Parquet file
        table = pq.read_table(type(self).TEMPORARY_PARQUETY_FILE)
        df = ta.from_arrow(table)

        self.assertEqual(df.dtype, DTYPE)
        self.assertEqual(list(df), self.RAW_ROWS)

        # pyre-fixme[6]: For 1st param expected `DataFrame` but got `Union[Column,
        #  DataFrame]`.
        df = type(self).preproc(df)

        # Check result
        self.assertEqual(df["label"].dtype, dt.int32)

        expected_labels = []
        expected_dense_features = []
        expected_sparse_features = []
        for (label, dense_features, sparse_features) in self.RAW_ROWS:
            expected_labels.append(label)
            expected_dense_features.append(
                tuple(
                    np.log(
                        np.array([v or 0 for v in dense_features], dtype=np.float32) + 3
                    )
                )
            )
            expected_sparse_features.append(tuple([v or 0] for v in sparse_features))

        self.assertEqual(list(df["label"]), expected_labels)
        numpy.testing.assert_array_almost_equal(
            np.array(list(df["dense_features"])), np.array(expected_dense_features)
        )
        self.assertEqual(list(df["sparse_features"]), expected_sparse_features)

        # Convert to Tensor
        tensors = df.to_tensor(
            {
                "dense_features": _DenseConversion(),
                "sparse_features": _CriteoJaggedTensorConversion(),
            }
        )
        expected_label_tensor = torch.tensor(expected_labels, dtype=torch.int32)
        self.assertEqual(tensors.label.dtype, torch.int32)
        self.assertTrue(torch.all(tensors.label == expected_label_tensor))

        expected_dense_features_tensor = torch.tensor(
            expected_dense_features, dtype=torch.float32
        )
        self.assertTrue(
            torch.all(tensors.dense_features - expected_dense_features_tensor < 1e-6)
        )

        expected_sparse_features_value_tensor = (
            torch.tensor(expected_sparse_features, dtype=torch.float32)
            .transpose(1, 0)
            .reshape(-1)
        )
        self.assertTrue(
            torch.all(
                tensors.sparse_features[1] == expected_sparse_features_value_tensor
            )
        )


if __name__ == "__main__":
    unittest.main()
