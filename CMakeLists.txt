# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.5)

project(torcharrow)

find_package(Torch REQUIRED)

add_subdirectory(csrc/velox)
