#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

git submodule sync --recursive
git submodule update --init --recursive

echo "MACOSX_DEPLOYMENT_TARGET is $MACOSX_DEPLOYMENT_TARGET"
echo "CPU_TARGET is $CPU_TARGET"
echo "CU_VERSION is $CU_VERSION"
echo "bla is $bla"

# MACOSX_DEPLOYMENT_TARGET=10.15 CPU_TARGET="sse" $PYTHON setup.py install --single-version-externally-managed --record=record.txt
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
