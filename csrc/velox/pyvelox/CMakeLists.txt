# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if(${CREATE_PYVELOX_MODULE})

    # Define our Python module:
    pybind11_add_module(
        _pyvelox
        MODULE
        NO_EXTRAS # TODO: LTO crashes GCC9.2. File issues to pybind11
        pyvelox.cpp
        pyvelox.h
    )

    # Link with Velox:
    target_link_libraries(_pyvelox PRIVATE
            velox_type
            )

    install(
            TARGETS _pyvelox
            LIBRARY DESTINATION .
    )
else()
    add_library(pyvelox pyvelox.cpp pyvelox.h)
    target_link_libraries(
            pyvelox
            velox_type
            pybind11::module)
endif()
