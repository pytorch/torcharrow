/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// This file is directly copied over from the torchtext repo
// (https://github.com/pytorch/text/blob/main/torchtext/csrc/regex.cpp).
// It will be removed once we have a dynamic registration mechanism for
// torcharrow UDFs from other repos.

#include "regex.h"

namespace facebook::torcharrow::functions {

Regex::Regex(const std::string& re_str) : re_str_(re_str) {
  compiled_pattern_ = new RE2(re_str_);
}

std::string Regex::Sub(std::string str, const std::string& repl) const {
  RE2::GlobalReplace(&str, *compiled_pattern_, repl);
  return str;
}

bool Regex::FindAndConsume(re2::StringPiece* input, std::string* text) const {
  return RE2::FindAndConsume(input, *compiled_pattern_, text);
}
} // namespace facebook::torcharrow::functions
