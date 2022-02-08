# Copyright (c) Facebook, Inc. and its affiliates.
try:
    import torch  # noqa

    available = True
except ModuleNotFoundError:
    available = False

if available:
    from ._pytorch.common import *  # noqa
    from ._pytorch import rec  # noqa
    from ._pytorch.common import _dtype_to_pytorch_dtype  # noqa


def ensure_available():
    if not available:
        raise ModuleNotFoundError(
            "PyTorch is not installed and conversion functionality is not available"
        )
