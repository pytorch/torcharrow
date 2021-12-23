# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import torcharrow as ta
import torcharrow.dtypes as dt
import torcharrow.pytorch as tap


class TestInterop(unittest.TestCase):
    def base_test_to_pytorch(self):
        import torch

        df = ta.DataFrame(
            {
                "A": ["a", "b", "c", "d", "e"],
                "B": [[1, 2], [3, None], [4, 5], [6], [7]],
                "N_B": [[1, 2], [3, 4], None, [6], [7]],
                "C": [{1: 11}, {2: 22, 3: 33}, None, {5: 55}, {6: 66}],
                "I": [1, 2, 3, 4, 5],
                "N_I": [1, 2, 3, None, 5],
                "SS": [["a"], ["b", "bb"], ["c"], ["d", None], ["e"]],
                "DSI": [{"a": 1}, {"b": 2, "bb": 22}, {}, {"d": 4}, {}],
                "N_DII": [{1: 11}, {2: 22, 3: 33}, None, {4: 44}, {}],
                # FIXME: https://github.com/facebookresearch/torcharrow/issues/60: Support to_arrow with a list(struct) column
                # "N_ROW": ta.Column(
                #    [
                #        [(1, 1.1)],
                #        [(2, 2.2), (3, 3.3)],
                #        [],
                #        [(4, 4.4), (5, None)],
                #        [(6, 6.6)],
                #    ],
                #    dtype=dt.List(
                #        dt.Struct(
                #            [
                #                dt.Field("i", dt.Int64()),
                #                dt.Field("f", dt.Float32(nullable=True)),
                #            ]
                #        )
                #    ),
                #    device=self.device,
                # ),
            },
            device=self.device,
        )
        p = df["I"][1:4].to_tensor()
        self.assertEqual(p.dtype, torch.int64)
        self.assertEqual(p.tolist(), [2, 3, 4])

        p = df["N_I"][1:4].to_tensor()
        self.assertEqual(p.values.dtype, torch.int64)
        # last value can be anything
        self.assertEqual(p.values.tolist()[:-1], [2, 3])
        self.assertEqual(p.presence.dtype, torch.bool)
        self.assertEqual(p.presence.tolist(), [True, True, False])

        # non nullable list with nullable elements
        p = df["B"][1:4].to_tensor()
        self.assertEqual(p.values.values.dtype, torch.int64)
        self.assertEqual(p.values.presence.dtype, torch.bool)
        self.assertEqual(p.offsets.dtype, torch.int32)
        self.assertEqual(p.values.values.tolist()[0], 3)
        # second value can be anything
        self.assertEqual(p.values.values.tolist()[2:], [4, 5, 6])
        self.assertEqual(p.values.presence.tolist(), [True, False, True, True, True])
        self.assertEqual(p.offsets.tolist(), [0, 2, 4, 5])

        # nullable list with non nullable elements
        p = df["N_B"][1:4].to_tensor()
        self.assertEqual(p.values.values.dtype, torch.int64)
        self.assertEqual(p.presence.dtype, torch.bool)
        self.assertEqual(p.values.offsets.dtype, torch.int32)
        self.assertEqual(p.values.values.tolist(), [3, 4, 6])
        self.assertEqual(p.presence.tolist(), [True, False, True])
        self.assertEqual(p.values.offsets.tolist(), [0, 2, 2, 3])

        # list of strings -> we skip PackedList all together
        p = df["SS"][1:4].to_tensor()
        self.assertEqual(p, [["b", "bb"], ["c"], ["d", None]])

        # map of strings - the keys turns into regular list
        p = df["DSI"][1:4].to_tensor()
        self.assertEqual(p.keys, ["b", "bb", "d"])
        self.assertEqual(p.values.dtype, torch.int64)
        self.assertEqual(p.offsets.dtype, torch.int32)
        self.assertEqual(p.values.tolist(), [2, 22, 4])
        self.assertEqual(p.offsets.tolist(), [0, 2, 2, 3])

        # list of tuples
        # FIXME: https://github.com/facebookresearch/torcharrow/issues/60
        # p = df["N_ROW"][1:4].to_tensor()
        # self.assertEqual(p.offsets.dtype, torch.int32)
        # self.assertEqual(p.offsets.tolist(), [0, 2, 2, 4])
        # self.assertEqual(p.values.i.dtype, torch.int64)
        # self.assertEqual(p.values.i.tolist(), [2, 3, 4, 5])
        # self.assertEqual(p.values.f.presence.dtype, torch.bool)
        # self.assertEqual(p.values.f.presence.tolist(), [True, True, True, False])
        # self.assertEqual(p.values.f.values.dtype, torch.float32)
        # np.testing.assert_almost_equal(p.values.f.values.numpy(), [2.2, 3.3, 4.4, 0.0])

        # Reverse conversion
        p = df.to_tensor()
        df2 = tap.from_tensor(p, dtype=df.dtype, device=self.device)
        self.assertEqual(df.dtype, df2.dtype)
        self.assertEqual(list(df), list(df2))

        # Reverse conversion with type inference
        df3 = tap.from_tensor(p, dtype=df.dtype, device=self.device)
        self.assertEqual(df.dtype, df3.dtype)
        self.assertEqual(list(df), list(df3))

    def base_test_pad_sequence(self):
        import torch

        df = ta.DataFrame(
            {
                "int32": [[11, 12, 13, 14], [21, 22], [31], [41, 42, 43]],
                "int64": [[11, 12, 13, 14], [21, 22], [31], [41, 42, 43]],
                "float32": [
                    [11.5, 12.5, 13.5, 14.5],
                    [21.5, 22.5],
                    [31.5],
                    [41.5, 42.5, 43.5],
                ],
            },
            dtype=dt.Struct(
                [
                    dt.Field("int32", dt.List(dt.int32)),
                    dt.Field("int64", dt.List(dt.int64)),
                    dt.Field("float32", dt.List(dt.float32)),
                ]
            ),
            device=self.device,
        )

        collated_tensors = df.to_tensor(
            {
                "int32": tap.PadSequence(padding_value=-1),
                "int64": tap.PadSequence(padding_value=-2),
                "float32": tap.PadSequence(padding_value=-3),
            }
        )

        # named tuple with 3 fields
        self.assertTrue(isinstance(collated_tensors, tuple))
        self.assertEquals(len(collated_tensors), 3)

        self.assertEquals(collated_tensors.int32.dtype, torch.int32)
        self.assertEquals(
            collated_tensors.int32.tolist(),
            [[11, 12, 13, 14], [21, 22, -1, -1], [31, -1, -1, -1], [41, 42, 43, -1]],
        )

        self.assertEquals(collated_tensors.int64.dtype, torch.int64)
        self.assertEquals(
            collated_tensors.int64.tolist(),
            [[11, 12, 13, 14], [21, 22, -2, -2], [31, -2, -2, -2], [41, 42, 43, -2]],
        )

        self.assertEquals(collated_tensors.float32.dtype, torch.float32)
        self.assertEquals(
            collated_tensors.float32.tolist(),
            [
                [11.5, 12.5, 13.5, 14.5],
                [21.5, 22.5, -3.0, -3.0],
                [31.5, -3.0, -3.0, -3.0],
                [41.5, 42.5, 43.5, -3.0],
            ],
        )

    def base_test_pytorch_transform(self):
        import torch

        df = ta.DataFrame(
            {
                "lst_null": [[1, 2], [3, None], [4, 5], [6]],
                "ids": [[1, 2], [3], [1, 4], [5]],
                "a": [1, 2, 3, 4],
                "b": [10, 20, 30, 40],
            },
            device=self.device,
        )

        from torcharrow.pytorch import WithPresence, PackedList, PackedMap

        def list_plus_one(x: PackedList[WithPresence[torch.Tensor]]):
            return PackedList(
                offsets=x.offsets,
                values=WithPresence(
                    presence=x.values.presence,
                    values=(x.values.values + 1) * x.values.presence,
                ),
            )

        self.assertEqual(
            list(df["lst_null"].transform(list_plus_one, format="torch")),
            [[2, 3], [4, None], [5, 6], [7]],
        )

        # we don't support tensor columns yet, so let's do it a 1d embedding :)
        emb = torch.nn.EmbeddingBag(10, 1, mode="sum", include_last_offset=True)
        emb.weight.data[:] = torch.arange(1, 11).unsqueeze(1)

        def embed(x: PackedList[torch.Tensor]):
            return emb(x.values, x.offsets.to(torch.int64)).squeeze(1)

        self.assertEqual(
            list(df["ids"].transform(embed, dtype=dt.float32, format="torch")),
            [2.0 + 3.0, 4.0, 2.0 + 5.0, 6.0],
        )

        def plus_div(x: torch.Tensor, y: torch.Tensor):
            return torch.add(x, y), torch.div(y, x)

        # TODO: pytorch output type inference
        self.assertEqual(
            list(
                df.transform(
                    plus_div,
                    columns=["a", "b"],
                    dtype=dt.Struct(
                        [dt.Field("sum", dt.int64), dt.Field("ratio", dt.float32)]
                    ),
                    format="torch",
                )
            ),
            [(11, 10.0), (22, 10.0), (33, 10.0), (44, 10.0)],
        )

    # TODO: migrate other tests from https://github.com/facebookresearch/torcharrow/blob/9927a9c428c99ec8c4be3352c318ed2c66d36990/torcharrow/test/test_legacy_interop.py
