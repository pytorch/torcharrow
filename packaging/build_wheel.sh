#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Reference:
# * https://github.com/pytorch/audio/blob/392a03c86d94d2747e1a0fc270a74c3845535173/packaging/build_wheel.sh
# * https://github.com/pytorch/audio/blob/392a03c86d94d2747e1a0fc270a74c3845535173/packaging/pkg_helpers.bash

set -ex

# Populate build version if necessary, and add version suffix
#
# Inputs:
#   BUILD_VERSION (e.g., 0.2.0 or empty)
#   VERSION_SUFFIX (e.g., +cpu)
#
# Outputs:
#   BUILD_VERSION (e.g., 0.2.0.dev20190807+cpu)
#
# Fill BUILD_VERSION if it doesn't exist already with a nightly string
# Usage: setup_build_version 0.2.0
setup_build_version() {
  if [[ -z "$BUILD_VERSION" ]]; then
    export BUILD_VERSION="$1.dev$(date "+%Y%m%d")$VERSION_SUFFIX"
  else
    export BUILD_VERSION="$BUILD_VERSION$VERSION_SUFFIX"
  fi
}

# Set some useful variables for OS X, if applicable
setup_macos() {
  if [[ "$(uname)" == Darwin ]]; then
    export CC=clang CXX=clang++
  fi
}

# Inputs:
#   PYTHON_VERSION (3.7, 3.8, 3.9)
#
# Outputs:
#   PATH modified to put correct Python version in PATH
setup_wheel_python() {
  if [[ -n "$PYTHON_VERSION" ]]; then
      eval "$(conda shell.bash hook)"
      conda env remove -n "env$PYTHON_VERSION" || true
      conda create -yn "env$PYTHON_VERSION" python="$PYTHON_VERSION"
      conda activate "env$PYTHON_VERSION"
 fi

}

setup_build_version 0.0.4
setup_wheel_python
python setup.py clean
if [[ "$(uname)" == Darwin ]]; then
  setup_macos
  python setup.py bdist_wheel
elif [[ "$(uname)" == Linux ]]; then
  python setup.py bdist_wheel
else
  echo "Unsupported"
  exit 1
fi

