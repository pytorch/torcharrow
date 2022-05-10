/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// This file is directly copied over from the torchtext repo
// (https://github.com/pytorch/text/blob/main/torchtext/csrc/regex.h).
// It will be removed once we have a dynamic registration mechanism for
// torcharrow UDFs from other repos.

#include <re2/re2.h>
#include <re2/stringpiece.h>
#include <string>

// namespace py = pybind11;

namespace facebook::torcharrow::functions {
struct Regex {
 private:
  RE2* compiled_pattern_;

 public:
  std::string re_str_;

  Regex(const std::string& re_str);
  std::string Sub(std::string str, const std::string& repl) const;
  bool FindAndConsume(re2::StringPiece* input, std::string* text) const;
};

} // namespace facebook::torcharrow::functions
