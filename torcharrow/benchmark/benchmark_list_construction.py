# Copyright (c) Facebook, Inc. and its affiliates.
import timeit
from abc import ABC, abstractmethod
from typing import List, Optional

import torcharrow.dtypes as dt
from torcharrow.scope import Scope


class BenchmarkListConstruction:
    def setup(self):
        self.nLoop: int = 300

        self.test_strings: List[str] = [f"short{x}" for x in range(1000)]
        self.test_strings.extend(
            [f"this_is_a_long_string_for_velox_stringview{x}" for x in range(1000)]
        )

    class ListConstructionRunner(ABC):
        @abstractmethod
        def setup(self, test_strings: List[str]):
            pass

        @abstractmethod
        def run(self):
            pass

    # pyre-fixme[13]: Attribute `test_strings` is never initialized.
    class AppendRunner(ListConstructionRunner):
        def __init__(self):
            self.test_strings: List[str]

        def setup(self, test_strings: List[str]):
            self.test_strings = test_strings

        def run(self):
            col = Scope.default._EmptyColumn(dtype=dt.string)
            for s in self.test_strings:
                col._append(s)
            col._finalize()

    # pyre-fixme[13]: Attribute `test_strings` is never initialized.
    class FromlistRunner(ListConstructionRunner):
        def __init__(self):
            self.test_strings: List[str]

        def setup(self, test_strings: List[str]):
            self.test_strings = test_strings

        def run(self):
            col = Scope.default._FromPySequence(self.test_strings, dtype=dt.string)

    def runListConstruction(
        self, runner: ListConstructionRunner, test_strings: Optional[List[str]] = None
    ) -> List[int]:
        # pyre-fixme[16]: `BenchmarkListConstruction` has no attribute `test_strings`.
        test_strings = test_strings or self.test_strings
        # pyre-fixme[7]: Expected `List[int]` but got `List[float]`.
        return timeit.repeat(
            stmt=lambda: runner.run(),
            setup=lambda: runner.setup(test_strings),
            # pyre-fixme[16]: `BenchmarkListConstruction` has no attribute `nLoop`.
            number=self.nLoop,
        )

    def printResult(self, result: List[int]):
        print(
            f"min of {len(result)} repeats is {min(result)} seconds over "
            # pyre-fixme[16]: `BenchmarkListConstruction` has no attribute `nLoop`.
            f"{self.nLoop} loops, or {min(result)/self.nLoop} seconds per loop"
        )

    def run(self):
        self.setup()

        appendResult: List[int] = self.runListConstruction(self.AppendRunner())
        print("Benchmark result of constructing list column by append:")
        self.printResult(appendResult)

        fromlistResult: List[int] = self.runListConstruction(self.FromlistRunner())
        print("Benchmark result of constructing list column by fromlist:")
        self.printResult(fromlistResult)


if __name__ == "__main__":
    benchmarkListConstruction = BenchmarkListConstruction()
    benchmarkListConstruction.run()
