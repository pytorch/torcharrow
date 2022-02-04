# Copyright (c) Facebook, Inc. and its affiliates.
try:
    import torch  # noqa

    available = True
except ModuleNotFoundError:
    available = False

if available:
    from ._pytorch.common import *  # noqa


def ensure_available():
    if not available:
        raise ModuleNotFoundError(
            "PyTorch is not installed and conversion functionality is not available"
        )
