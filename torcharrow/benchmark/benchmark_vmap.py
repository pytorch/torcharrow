# Copyright (c) Facebook, Inc. and its affiliates.
import timeit

import torcharrow as ta
import torcharrow.dtypes as dt


def prepare_list_int_col():
    elements = []
    for i in range(1, 101):
        element = []
        for j in range(1, i // 10 + 2):
            element.append(i * j)
        elements.append(element)

    return ta.from_pysequence(elements, dtype=dt.List(dt.int64), device="cpu")


def list_map_int(col: ta.IColumn):
    return col.list.map(lambda val: val + 1)


def list_vmap_int(col: ta.IColumn):
    return col.list.vmap(lambda col: col + 1)


def prepare_list_str_col():
    elements = []
    for i in range(1, 101):
        element = []
        for j in range(1, i // 10 + 2):
            element.append(f"str{i}_{j}")
        elements.append(element)

    return ta.from_pysequence(elements, dtype=dt.List(dt.string), device="cpu")


def list_map_str(col: ta.IColumn):
    return col.list.map(lambda val: val + "_1")


def list_vmap_str(col: ta.IColumn):
    return col.list.vmap(lambda col: col + "_1")


if __name__ == "__main__":
    import timeit

    print("Benchmark list[int] column:")
    col1 = prepare_list_int_col()
    print(
        "list.map: " + str(timeit.timeit(stmt=lambda: list_map_int(col1), number=100))
    )
    print(
        "list.vmap: " + str(timeit.timeit(stmt=lambda: list_vmap_int(col1), number=100))
    )

    print("Benchmark list[str] column:")
    col2 = prepare_list_str_col()
    print(
        "list.map: " + str(timeit.timeit(stmt=lambda: list_map_str(col2), number=100))
    )
    print(
        "list.vmap: " + str(timeit.timeit(stmt=lambda: list_vmap_str(col2), number=100))
    )
