#include <re2/re2.h>
#include <re2/stringpiece.h>
#include <string>

// namespace py = pybind11;

namespace facebook::torcharrow {
struct Regex {
 private:
  RE2* compiled_pattern_;

 public:
  std::string re_str_;

  Regex(const std::string& re_str);
  std::string Sub(std::string str, const std::string& repl) const;
  bool FindAndConsume(re2::StringPiece* input, std::string* text) const;
};

} // namespace torchtext
