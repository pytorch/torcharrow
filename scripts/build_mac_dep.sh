#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

ROOT_DIR="$( cd -- "$( dirname "$( dirname "${BASH_SOURCE[0]}" )" )" &> /dev/null && pwd )"
SCRIPT_DIR=${ROOT_DIR}/scripts
DEPENDENCY_DIR=${ROOT_DIR}/_build
mkdir -p "${DEPENDENCY_DIR}"

source $SCRIPT_DIR/_setup-macos.sh

for name in "$@"; do
    install_$name
done

