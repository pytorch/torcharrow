#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

ROOT_DIR="$( cd -- "$( dirname "$( dirname "${BASH_SOURCE[0]}" )" )" &> /dev/null && pwd )"
SCRIPT_DIR=${ROOT_DIR}/scripts
DEPENDENCY_DIR=${ROOT_DIR}/_build
mkdir -p "${DEPENDENCY_DIR}"

source $SCRIPT_DIR/_setup-macos.sh

for name in "$@"; do
    install_$name
done

