// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once

#include "velox/functions/Udf.h"
#include "velox/functions/lib/string/StringCore.h"
#include "velox/functions/lib/string/StringImpl.h"

namespace facebook::torcharrow::functions {

namespace internal {
class Utf8CatUtils {
 public:
  static inline bool isAlpha(utf8proc_category_t category) {
    return category == UTF8PROC_CATEGORY_LU /**< Letter, uppercase */ ||
        category == UTF8PROC_CATEGORY_LL /**< Letter, lowercase */ ||
        category == UTF8PROC_CATEGORY_LT /**< Letter, titlecase */ ||
        category == UTF8PROC_CATEGORY_LM /**< Letter, modifier */ ||
        category == UTF8PROC_CATEGORY_LO /**< Letter, other */;
  }

  static inline bool isNumber(utf8proc_category_t category) {
    return category == UTF8PROC_CATEGORY_ND /**< Number, decimal digit */ ||
        category == UTF8PROC_CATEGORY_NL /**< Number, letter */ ||
        category == UTF8PROC_CATEGORY_NO /**< Number, other */;
  }

  static inline bool isDigit(utf8proc_category_t category) {
    return category == UTF8PROC_CATEGORY_ND /**< Number, decimal digit */ ||
        category == UTF8PROC_CATEGORY_NO /**< Number, other */;
  }

  static inline bool isDecimal(utf8proc_category_t category) {
    return category == UTF8PROC_CATEGORY_ND /**< Number, decimal digit */;
  }

  static inline bool isLower(utf8proc_category_t category) {
    return category == UTF8PROC_CATEGORY_LL /**< Letter, lowercase */;
  }

  static inline bool isUpper(utf8proc_category_t category) {
    return category == UTF8PROC_CATEGORY_LU /**< Letter, uppercase */ ||
        category == UTF8PROC_CATEGORY_LT /**< Letter, titlecase */;
  }

  static inline bool isSpace(utf8proc_category_t category) {
    return category == UTF8PROC_CATEGORY_ZS /** Separator, space **/ ||
        category == UTF8PROC_CATEGORY_ZL /** Separator, line **/ ||
        category == UTF8PROC_CATEGORY_ZP /** Separator, paragraph **/;
  }
};

} // namespace internal

/**
 * torcharrow_isalpha(string) → bool
 * Return True if the string is an alphabetic string, False otherwise.
 *
 * A string is alphabetic if all characters in the string are alphabetic
 * and there is at least one character in the string.
 **/
VELOX_UDF_BEGIN(torcharrow_isalpha)
FOLLY_ALWAYS_INLINE
bool call(bool& result, const arg_type<velox::Varchar>& input) {
  using namespace velox::functions;
  using namespace internal;

  size_t size = input.size();
  if (size == 0) {
    result = false;
    return true;
  }

  size_t index = 0;
  while (index < size) {
    int codePointSize;
    utf8proc_int32_t codePoint =
        utf8proc_codepoint(input.data() + index, codePointSize);

    utf8proc_category_t category =
        static_cast<utf8proc_category_t>(utf8proc_category(codePoint));
    if (!Utf8CatUtils::isAlpha(category)) {
      result = false;
      return true;
    }

    index += codePointSize;
  }
  result = true;
  return true;
}
VELOX_UDF_END();

/**
 * torcharrow_isalnum(string) → bool
 * Return True if all characters in the string are alphanumeric (either
 *alphabets or numbers), False otherwise.
 **/
VELOX_UDF_BEGIN(torcharrow_isalnum)
FOLLY_ALWAYS_INLINE
bool call(bool& result, const arg_type<velox::Varchar>& input) {
  using namespace velox::functions;
  using namespace internal;

  size_t size = input.size();
  if (size == 0) {
    result = false;
    return true;
  }

  size_t index = 0;
  while (index < size) {
    int codePointSize;
    utf8proc_int32_t codePoint =
        utf8proc_codepoint(input.data() + index, codePointSize);

    utf8proc_category_t category =
        static_cast<utf8proc_category_t>(utf8proc_category(codePoint));
    if (!(Utf8CatUtils::isAlpha(category) ||
          Utf8CatUtils::isNumber(category))) {
      result = false;
      return true;
    }

    index += codePointSize;
  }
  result = true;
  return true;
}
VELOX_UDF_END();

/**
 * torcharrow_isdigit(string) → bool
 * Return True if all characters in the string are numeric, False otherwise.
 **/
VELOX_UDF_BEGIN(torcharrow_isdigit)
FOLLY_ALWAYS_INLINE
bool call(bool& result, const arg_type<velox::Varchar>& input) {
  using namespace velox::functions;
  using namespace internal;

  size_t size = input.size();
  if (size == 0) {
    result = false;
    return true;
  }

  size_t index = 0;
  while (index < size) {
    int codePointSize;
    utf8proc_int32_t codePoint =
        utf8proc_codepoint(input.data() + index, codePointSize);

    utf8proc_category_t category =
        static_cast<utf8proc_category_t>(utf8proc_category(codePoint));
    if (!Utf8CatUtils::isDigit(category)) {
      result = false;
      return true;
    }

    index += codePointSize;
  }
  result = true;
  return true;
}
VELOX_UDF_END();

/**
 * torcharrow_isinteger(string) → bool
 * Return True if first character is -/+ or a number,
 * followed by all numbers, False otherwise.
 **/
VELOX_UDF_BEGIN(torcharrow_isinteger)
FOLLY_ALWAYS_INLINE
bool call(bool& result, const arg_type<velox::Varchar>& input) {
  using namespace velox::functions;
  using namespace internal;

  size_t size = input.size();
  if (size == 0) {
    result = false;
    return true;
  }

  bool has_digit = false; //this is needed for the case where the string is "+" or "-"
  size_t index = 0;
  while (index < size) {
    int codePointSize;
    utf8proc_int32_t codePoint =
        utf8proc_codepoint(input.data() + index, codePointSize);

    if (index == 0 && (codePoint == '+' || codePoint == '-')) {
        index += codePointSize;
        continue;
    }

    utf8proc_category_t category =
        static_cast<utf8proc_category_t>(utf8proc_category(codePoint));

    if (Utf8CatUtils::isNumber(category)) {
         has_digit = true;
    }
    else {
         result = false;
         return true;
    }

    index += codePointSize;
  }
  result = has_digit;
  return true;
}
VELOX_UDF_END();

/**
 * torcharrow_isdecimal(string) → bool
 * Return True if the string contains only decimal digit (from 0 to 9), False otherwise.
 *
 * A string is decimal if all characters in the string are decimal digits
 * and there is at least one character in the string.
 **/
VELOX_UDF_BEGIN(torcharrow_isdecimal)
FOLLY_ALWAYS_INLINE
bool call(bool& result, const arg_type<velox::Varchar>& input) {
  using namespace velox::functions;
  using namespace internal;

  size_t size = input.size();
  if (size == 0) {
    result = false;
    return true;
  }

  size_t index = 0;
  while (index < size) {
    int codePointSize;
    utf8proc_int32_t codePoint =
        utf8proc_codepoint(input.data() + index, codePointSize);

    utf8proc_category_t category =
        static_cast<utf8proc_category_t>(utf8proc_category(codePoint));
    if (!Utf8CatUtils::isDecimal(category)) {
      result = false;
      return true;
    }

    index += codePointSize;
  }
  result = true;
  return true;
}
VELOX_UDF_END();

/**
 * torcharrow_islower(string) → bool
 * Return True if the string is in lower case, False otherwise.
 *
 * A string is in lower case if all the alphabetic characters in the string are
 * in lower case and there is at least one alphabetic character in the string.
 **/
VELOX_UDF_BEGIN(torcharrow_islower)
FOLLY_ALWAYS_INLINE
bool call(bool& result, const arg_type<velox::Varchar>& input) {
  using namespace velox::functions;
  using namespace internal;

  size_t size = input.size();
  if (size == 0) {
    result = false;
    return true;
  }

  size_t index = 0;
  bool has_alph_lower = false;
  while (index < size) {
    int codePointSize;
    utf8proc_int32_t codePoint =
        utf8proc_codepoint(input.data() + index, codePointSize);

    utf8proc_category_t category =
        static_cast<utf8proc_category_t>(utf8proc_category(codePoint));
    if (Utf8CatUtils::isUpper(category)) {
      result = false;
      return true;
    } else if (Utf8CatUtils::isLower(category)) {
      has_alph_lower = true;
    } else {
      // Ignore non-alphbetic letters. This behavior is consistent with
      // python str.islower() behavior.
    }

    index += codePointSize;
  }
  result = has_alph_lower;
  return true;
}
VELOX_UDF_END();

/**
 * torcharrow_isupper(string) → bool
 * Return True if the string is in upper case, False otherwise.
 *
 * A string is in upper case if all the alphabetic characters in the string are
 * in upper case and there is at least one alphabetic character in the string.
 **/
VELOX_UDF_BEGIN(torcharrow_isupper)
FOLLY_ALWAYS_INLINE
bool call(bool& result, const arg_type<velox::Varchar>& input) {
  using namespace velox::functions;
  using namespace internal;

  size_t size = input.size();
  if (size == 0) {
    result = false;
    return true;
  }

  size_t index = 0;
  bool has_alph_upper = false;
  while (index < size) {
    int codePointSize;
    utf8proc_int32_t codePoint =
        utf8proc_codepoint(input.data() + index, codePointSize);

    utf8proc_category_t category =
        static_cast<utf8proc_category_t>(utf8proc_category(codePoint));
    if (Utf8CatUtils::isLower(category)) {
      result = false;
      return true;
    } else if (Utf8CatUtils::isUpper(category)) {
      has_alph_upper = true;
    } else {
      // Ignore non-alphbetic letters. This behavior is consistent with
      // python str.isupper() behavior.
    }

    index += codePointSize;
  }
  result = has_alph_upper;
  return true;
}
VELOX_UDF_END();

/**
 * torcharrow_isspace(string) → bool
 * Return True all characters in the string are whitespace
 * , False otherwise.
 **/
VELOX_UDF_BEGIN(torcharrow_isspace)
FOLLY_ALWAYS_INLINE
bool call(bool& result, const arg_type<velox::Varchar>& input) {
  using namespace velox::functions;
  using namespace internal;

  size_t size = input.size();
  if (size == 0) {
    result = false;
    return true;
  }

  // Indicating that NLF-sequences (LF, CRLF, CR, NEL) including
  // HorizontalTab (HT) and FormFeed (FF) are representing a line break,
  // and should be converted to the codepoint for line separation (LS)
  utf8proc_option_t options =
      static_cast<utf8proc_option_t>(UTF8PROC_STRIPCC | UTF8PROC_NLF2LS);

  size_t index = 0;
  while (index < size) {
    int codePointSize;
    utf8proc_int32_t codePoint =
        utf8proc_codepoint(input.data() + index, codePointSize);

    // Convert all NLF-sequences in codepoint to line separation (LS)
    // Reason: No explicit category for characters like \n or \t in utfproc
    utf8proc_normalize_utf32(&codePoint, codePointSize, options);

    utf8proc_category_t category =
        static_cast<utf8proc_category_t>(utf8proc_category(codePoint));
    if (!(Utf8CatUtils::isSpace(category))) {
      result = false;
      return true;
    }

    index += codePointSize;
  }
  result = true;
  return true;
}
VELOX_UDF_END();

/**
 * torcharrow_istitle(string) → bool
 * Return True if each word of the string starts with an
 * upper case letter, False otherwise.
 *
 * A string is a title if each word of the string starts with an upper or title
 * case letter and there is at least one alphabetic character in the string.
 **/
VELOX_UDF_BEGIN(torcharrow_istitle)
FOLLY_ALWAYS_INLINE
bool call(bool& result, const arg_type<velox::Varchar>& input) {
  using namespace velox::functions;
  using namespace internal;

  size_t size = input.size();
  if (size == 0) {
    result = false;
    return true;
  }

  size_t index = 0;
  bool has_alph = false;
  bool has_alph_upper = false;
  while (index < size) {
    int codePointSize;
    utf8proc_int32_t codePoint =
        utf8proc_codepoint(input.data() + index, codePointSize);
    utf8proc_category_t category =
        static_cast<utf8proc_category_t>(utf8proc_category(codePoint));

    if (Utf8CatUtils::isAlpha(category)) {
      has_alph = true;
      if (Utf8CatUtils::isUpper(category)) {
        // False if it has more than 1 upper or title case
        if (has_alph_upper) {
          result = false;
          return true;
        } else {
          has_alph_upper = true;
        }
      } else { // isLower
        if (!has_alph_upper) {
          // False if it only has lower case
          result = false;
          return true;
        }
      }
    } else {
      has_alph_upper = false;
    }

    index += codePointSize;
  }
  result = has_alph;
  return true;
}
VELOX_UDF_END();

/**
 * torcharrow_isnumeric(string) → bool
 * returns True if all the characters are numeric, otherwise False.
 *
 * A string is a numeric if each character of the string is numeric
 **/
VELOX_UDF_BEGIN(torcharrow_isnumeric)
FOLLY_ALWAYS_INLINE
bool call(bool& result, const arg_type<velox::Varchar>& input) {
  using namespace velox::functions;
  using namespace internal;

  size_t size = input.size();
  if (size == 0) {
    result = false;
    return true;
  }

  size_t index = 0;
  while (index < size) {
    int codePointSize;
    utf8proc_int32_t codePoint =
        utf8proc_codepoint(input.data() + index, codePointSize);
    utf8proc_category_t category =
        static_cast<utf8proc_category_t>(utf8proc_category(codePoint));

    if (!Utf8CatUtils::isNumber(category)) {
      result = false;
      return true;
    }

    index += codePointSize;
  }
  result = true;
  return true;
}
VELOX_UDF_END();
} // namespace facebook::torcharrow::functions
