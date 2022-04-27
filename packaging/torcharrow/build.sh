#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

git submodule update --init --recursive
scripts/build_mac_dep.sh ranges_v3 fmt double_conversion folly re2
$PYTHON setup.py install --single-version-externally-managed --record=record.txt