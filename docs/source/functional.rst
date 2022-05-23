.. currentmodule:: torcharrow.functional

torcharrow.functional
==========================

Velox Core Functions
------------------------------
`Velox core functions <https://facebookincubator.github.io/velox/functions.html>`_
are included in `torcharrow.functional`.

Here is an example usage of Velox string function `lpad <https://facebookincubator.github.io/velox/functions/string.html#lpad>`_:

    >>> import torcharrow as ta
    >>> from torcharrow import functional
    >>> col = ta.column(["abc", "x", "yz"])
    # Velox's lpad function: https://facebookincubator.github.io/velox/functions/string.html#lpad
    >>> functional.lpad(col, 5, "123")
    0  '12abc'
    1  '1231x'
    2  '123yz'
    dtype: String(nullable=True), length: 3, null_count: 0, device: cpu

Here is another example usage of Velox array function `array\_except <https://facebookincubator.github.io/velox/functions/array.html#array_except>`_:

    >>> col1 = ta.column([[1, 2, 3], [1, 2, 3], [1, 2, 2], [1, 2, 2]])
    >>> col2 = ta.column([[4, 5, 6], [1, 2], [1, 1, 2], [1, 3, 4]])
    # Velox's array_except function: https://facebookincubator.github.io/velox/functions/array.html#array_except
    >>> functional.array_except(col1, col2)
    0  [1, 2, 3]
    1  [3]
    2  []
    3  [2]
    dtype: List(Int64(nullable=True), nullable=True), length: 4, null_count: 0

Recommendation Operations
-----------------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    bucketize
    sigrid_hash
    firstx
    has_id_overlap
    id_overlap_count
    get_max_count
    get_jaccard_similarity
    get_cosine_similarity
    get_score_sum
    get_score_min
    get_score_max

High-level Operations
-----------------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    scale_to_0_1
