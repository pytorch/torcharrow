# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import unittest
from functools import lru_cache
from pathlib import Path

import torcharrow as ta
import torcharrow._torcharrow as _ta
import torcharrow.dtypes as dt
import torcharrow.pytorch as tap
from torcharrow import functional


_ASSET_DIR = (Path(__file__).parent.parent / "asset").resolve()


def get_asset_path(*path_components):
    """Get the path to the file under `test/assets` directory."""
    return str(_ASSET_DIR.joinpath(*path_components))


@lru_cache()
def bytes_to_unicode():
    """
    Original Source: https://github.com/openai/gpt-2/blob/a74da5d99abaaba920de8131d64da2862a8f213b/src/encoder.py#L9

    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


# copied from GPT2BPETokenizer __init__ method
# https://github.com/pytorch/text/blob/27ccd7ece3caf2749b69db22df9ba4442284e1f1/torchtext/transforms.py#L287-L300
def init_bpe_encoder():
    encoder_json_path = get_asset_path("gpt2_bpe_encoder.json")
    vocab_bpe_path = get_asset_path("gpt2_bpe_vocab.bpe")
    _seperator = "\u0001"

    # load bpe encoder and bpe decoder
    with open(encoder_json_path, "r", encoding="utf-8") as f:
        bpe_encoder = json.load(f)
    # load bpe vocab
    with open(vocab_bpe_path, "r", encoding="utf-8") as f:
        bpe_vocab = f.read()
    bpe_merge_ranks = {
        _seperator.join(merge_pair.split()): i
        for i, merge_pair in enumerate(bpe_vocab.split("\n")[1:-1])
    }
    # Caching is enabled in Eager mode
    bpe = _ta.GPT2BPEEncoder(
        bpe_encoder, bpe_merge_ranks, _seperator, bytes_to_unicode(), True
    )
    return bpe


class _TestTextOpsBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (tap.available and _ta.is_built_with_torch()):
            raise unittest.SkipTest("Requires PyTorch")

        cls.tokenizer = init_bpe_encoder()
        cls.base_df_bpe = ta.dataframe(
            {
                "text": ["Hello World!, how are you?", "Respublica superiorem"],
                "labels": [0, 1],
                "tokens": [
                    ["15496", "2159", "28265", "703", "389", "345", "30"],
                    ["4965", "11377", "64", "2208", "72", "29625"],
                ],
            },
            dtype=dt.Struct(
                fields=[
                    dt.Field("text", dt.string),
                    dt.Field("labels", dt.int32),
                    dt.Field("tokens", dt.List(dt.string)),
                ]
            ),
        )

        cls.base_df_vocab = ta.dataframe(
            {
                "text": [["Hello", "world"], ["How", "are", "you!", "OOV"]],
                "indices": [[1, 2], [3, 4, 5, 0]],
            },
            dtype=dt.Struct(
                fields=[
                    dt.Field("text", dt.List(dt.string)),
                    dt.Field("indices", dt.List(dt.int64)),
                ]
            ),
        )

        cls.base_df_add_token = ta.dataframe(
            {
                "text": [["Hello", "world"], ["How", "are", "you!", "OOV"]],
                "indices": [[1, 2], [3, 4, 5, 0]],
            },
            dtype=dt.Struct(
                fields=[
                    dt.Field("text", dt.List(dt.string)),
                    dt.Field("indices", dt.List(dt.int64)),
                ]
            ),
        )
        cls.setUpTestCaseData()

    @classmethod
    def setUpTestCaseData(cls):
        # Override in subclass
        # Python doesn't have native "abstract base test" support.
        # So use unittest.SkipTest to skip in base class: https://stackoverflow.com/a/59561905.
        raise unittest.SkipTest("abstract base test")

    @unittest.skipUnless(
        tap.available and _ta.is_built_with_torch(), "Requires PyTorch"
    )
    def test_bpe_encode(self):
        out_df = functional.bpe_tokenize(self.tokenizer, self.df_bpe["text"])
        self.assertEqual(list(out_df), list(self.df_bpe["tokens"]))

    @unittest.skipUnless(
        tap.available and _ta.is_built_with_torch(), "Requires PyTorch"
    )
    def test_vocab_lookup_indices(self):
        tokens = ["<unk>", "Hello", "world", "How", "are", "you!"]
        vocab = _ta.Vocab(tokens, 0)
        indices = [[1, 2], [3, 4, 5, 0]]
        out_df = functional.lookup_indices(vocab, self.df_vocab["text"])
        self.assertEqual(indices, list(out_df))

    @unittest.skipUnless(
        tap.available and _ta.is_built_with_torch(), "Requires PyTorch"
    )
    def test_add_token(self):
        tokens = ["<unk>", "Hello", "world", "How", "are", "you!"]
        vocab = _ta.Vocab(tokens, 0)
        indices = [[1, 2], [3, 4, 5, 0]]
        out_df = functional.lookup_indices(vocab, self.df_vocab["text"])
        self.assertEqual(indices, list(out_df))


class TestTextOpsCpu(_TestTextOpsBase):
    @classmethod
    def setUpTestCaseData(cls):
        cls.df_bpe = cls.base_df_bpe.copy()
        cls.df_vocab = cls.base_df_vocab.copy()


if __name__ == "__main__":
    unittest.main()
