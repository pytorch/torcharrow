# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

try:
    import torch  # noqa

    available = True
except ModuleNotFoundError:
    available = False

if available:
    from ._pytorch.common import *  # noqa
    from ._pytorch import rec  # noqa
    from ._pytorch.common import _dtype_to_pytorch_dtype  # noqa


def ensure_available() -> None:
    if not available:
        raise ModuleNotFoundError(
            "PyTorch is not installed and conversion functionality is not available"
        )
