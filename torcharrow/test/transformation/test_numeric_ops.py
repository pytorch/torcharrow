import unittest

import torcharrow as ta

# TODO: add/migrate more numeric tests
class _TestNumericOpsBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Prepare input data as CPU dataframe
        cls.base_df = ta.DataFrame({"c": [0, 1, 3], "d": [5, 5, 6], "e": [1.0, 1, 7]})

        cls.setUpTestCaseData()

    @classmethod
    def setUpTestCaseData(cls):
        # Override in subclass
        # Python doesn't have native "abstract base test" support.
        # So use unittest.SkipTest to skip in base class: https://stackoverflow.com/a/59561905.
        raise unittest.SkipTest("abstract base test")

    def test_add(self):
        c = type(self).df["c"]
        d = type(self).df["d"]

        self.assertEqual(list(c + 1), [1, 2, 4])
        self.assertEqual(list(1 + c), [1, 2, 4])
        self.assertEqual(list(c + d), [5, 6, 9])


class TestNumericOpsCpu(_TestNumericOpsBase):
    @classmethod
    def setUpTestCaseData(cls):
        cls.df = cls.base_df.copy()


if __name__ == "__main__":
    unittest.main()
