#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

function fix_velox_dylib_paths() {
  velox_so=$1

  # other libs
  libs="libgflags libglog libz libssl libcrypto libbz2 liblz4 libzstd libsodium"

  for libx in ${libs}
  do
    liby=$(otool -L "${velox_so}" | sed -n -e "s/\\(.*${libx}\..*dylib\\).*/\\1/p" | tr -d '[:blank:]')

    install_name_tool -change "${liby}" "@loader_path/../../../${libx}.dylib" "${velox_so}"
  done

  libx=libevent
  # libevent
  liby=$(otool -L "${velox_so}" | sed -n -e "s/\\(.*${libx}-.*dylib\\).*/\\1/p" | tr -d '[:blank:]')
  install_name_tool -change "${liby}" "@loader_path/../../../${libx}.dylib" "${velox_so}"

  # boost libs
  boost_libs="libboost_context libboost_filesystem libboost_atomic libboost_regex libboost_system libboost_thread"

  for libx in ${boost_libs}
  do
    liby=$(otool -L "${velox_so}" | sed -n -e "s/\\(.*${libx}.*dylib\\).*/\\1/p" | tr -d '[:blank:]')

    install_name_tool -change "${liby}" "@loader_path/../../../${libx}.dylib" "${velox_so}"
  done

  libx=libboost_program
  liby=$(otool -L "${velox_so}" | sed -n -e "s/\\(.*${libx}.*dylib\\).*/\\1/p" | tr -d '[:blank:]')
  install_name_tool -change "${liby}" "@loader_path/../../../${libx}_options.dylib" "${velox_so}"
}
