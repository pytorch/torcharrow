#include "regex.h"

namespace facebook::torcharrow {

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
} // namespace torchtext
