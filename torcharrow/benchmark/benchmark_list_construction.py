# Copyright (c) Facebook, Inc. and its affiliates.
import timeit
from abc import ABC, abstractmethod
from typing import List, Optional

import torcharrow as ta
import torcharrow.dtypes as dt


class BenchmarkListConstruction:
    def setup(self):
        self.nLoop: int = 300
        self.tsCpu: ta.Scope = ta.Scope({"device": "cpu"})

        self.test_strings: List[str] = [f"short{x}" for x in range(1000)]
        self.test_strings.extend(
            [f"this_is_a_long_string_for_velox_stringview{x}" for x in range(1000)]
        )

    class ListConstructionRunner(ABC):
        @abstractmethod
        def setup(self, test_strings: List[str], ts: ta.Scope):
            pass

        @abstractmethod
        def run(self):
            pass

    class AppendRunner(ListConstructionRunner):
        def __init__(self):
            self.test_strings: List[str]
            self.ts: ta.Scope

        def setup(self, test_strings: List[str], ts: ta.Scope):
            self.test_strings = test_strings
            self.ts = ts

        def run(self):
            col = self.ts._EmptyColumn(dtype=dt.string)
            for s in self.test_strings:
                col._append(s)
            col._finalize()

    class FromlistRunner(ListConstructionRunner):
        def __init__(self):
            self.test_strings: List[str]
            self.ts: ta.Scope

        def setup(self, test_strings: List[str], ts: ta.Scope):
            self.test_strings = test_strings
            self.ts = ts

        def run(self):
            col = self.ts._FromPyList(self.test_strings, dtype=dt.string)

    def runListConstruction(
        self, runner: ListConstructionRunner, test_strings: Optional[List[str]] = None
    ) -> List[int]:
        test_strings = test_strings or self.test_strings
        return timeit.repeat(
            stmt=lambda: runner.run(),
            setup=lambda: runner.setup(test_strings, self.tsCpu),
            number=self.nLoop,
        )

    def printResult(self, result: List[int]):
        print(
            f"min of {len(result)} repeats is {min(result)} seconds over "
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
