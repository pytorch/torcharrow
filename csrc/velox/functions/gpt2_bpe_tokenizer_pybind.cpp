#include "gpt2_bpe_tokenizer_pybind.h"
#include "regex.h" // @manual

#include <stdexcept>
#include <algorithm>
#include <sstream>
#include <unordered_set>
#include <utility>

namespace facebook::torcharrow
{
  const Regex kGPT2Regex(
      "(\\'s|\\'t|\\'re|\\'ve|\\'m|\\'ll|\\'d| ?\\pL+|"
      " ?\\pN+| ?[^\\s\\v\\pL\\pN]+|[\\s\\v]+)");

  bool is_whitespace(const std::string &input)
  {
    for (const char &c : input)
    {
      if (!isspace(c))
      {
        return false;
      }
    }
    return true;
  }

  template <class Key_, class Value_>
  py::dict _map_to_c10_dict(std::unordered_map<Key_, Value_> m)
  {
    py::dict d;
    for (const auto &item : m)
      d[item.first] = item.second;
    return d;
  }

  template <class Key_, class Value_>
  std::unordered_map<Key_, Value_> _c10_dict_to_map(py::dict d)
  {
    std::unordered_map<Key_, Value_> m;
    for (const auto &item : d)
      m[item.first] = item.second;
    return m;
  }

  std::vector<std::string> gpt2_bpe_pre_tokenizer(std::string input)
  {
    // Python implementation:
    // https://github.com/pytorch/fairseq/blob/main/fairseq/data/encoders/gpt2_bpe_utils.py#L69
    // Original regex contains a negative lookahead pattern, which is not
    // supported in re2. This implementation modifies the original regex in
    // the following two ways:
    // 1. Removes negative lookahead and adds a post-processing step instead.
    // 2. Replace all [\s] occurences with [\s\v] because re2 does not include
    //    vertical tab (\v) in whitespace. PCRE and Python re include \v in \s.
    //
    // Pseudocode of post-processing step:
    // - Loop over all tokens
    // - IF token is all whitespace:
    //   - set prepend_space to False
    //   - IF token is last token, add it to return vector
    //   - ELSE
    //     - If token length is >1, add token[0:len(token) - 1] to return list
    //     - IF token[-1] is space (ascii 32), then carry it over for next token,
    //     set append_space = True
    //     - ELSE make token[-1] its own token and add to return list
    // - ELSE IF prepend_space == True, prepend a space to the token and add to
    // return list
    // - ELSE, add token to return list
    std::string token;
    std::vector<std::string> tokens;
    re2::StringPiece inp(input);
    bool prepend_space = false;
    while (kGPT2Regex.FindAndConsume(&inp, &token))
    {
      if (is_whitespace(token))
      {
        prepend_space = false;
        if (inp.empty())
        { // token is last token
          tokens.push_back(token);
        }
        else
        {
          if (token.length() > 1)
          {
            tokens.push_back(token.substr(0, token.length() - 1));
          }
          if (token[token.length() - 1] == ' ')
          { // last char is space
            prepend_space = true;
          }
          else
          { // push last whitespace char as a token if it is not a space
            tokens.push_back(token.substr(token.length() - 1));
          }
        }
      }
      else if (prepend_space)
      {
        tokens.push_back(" " + token);
        prepend_space = false;
      }
      else
      {
        tokens.push_back(token);
      }
    }
    return tokens;
  }

  std::pair<std::string, std::string> split_tokens(
      std::string s,
      std::string delimiter)
  {
    auto pos = s.find(delimiter);
    if (pos != std::string::npos)
    {
      throw std::runtime_error("Expected `s`to contain `delimiter`");
    };
    return std::make_pair(s.substr(0, pos), s.substr(pos + delimiter.length()));
  }

  int list_str_index(
      std::vector<std::string> list,
      std::string element,
      int start)
  {
    // Equivalent to: list.index(element, start)
    for (std::size_t i = start; i < list.size(); ++i)
    {
      if (list[i] == element)
      {
        return i;
      }
    }
    return -1;
  }

  std::string concatenate_strings(const std::vector<std::string> &list)
  {
    std::string ret = "";
    for (auto s : list)
      ret += s;
    return ret;
  }

  std::vector<std::string> get_pairs(
      std::vector<std::string> token_list,
      const std::string &seperator)
  {
    // For example: ["he", "l", "l", "o"]
    //    ==> ["he\u0001l", "l\u0001l", "l\u0001o"]
    std::unordered_set<std::string> pairs;
    std::vector<std::string> pairs_vec;

    if (token_list.empty())
      return pairs_vec;

    std::string prev_token = token_list[0];
    for (std::size_t i = 1; i < token_list.size(); ++i)
    {
      pairs.insert(prev_token + seperator + token_list[i]);
      prev_token = token_list[i];
    }
    pairs_vec.insert(pairs_vec.end(), pairs.begin(), pairs.end());
    return pairs_vec;
  }

  GPT2BPEEncoder::GPT2BPEEncoder(
      const py::dict &bpe_encoder,
      const py::dict &bpe_merge_ranks,
      const std::string &seperator,
      const py::dict &byte_encoder,
      bool caching_enabled)
      : inf_(bpe_merge_ranks.size() + 1),
        bpe_encoder_(std::move(bpe_encoder)),
        bpe_merge_ranks_(std::move(bpe_merge_ranks)),
        byte_encoder_(std::move(byte_encoder)),
        seperator_(std::move(seperator)),
        caching_enabled_(caching_enabled) {}

  GPT2BPEEncoder::GPT2BPEEncoder(
      const std::unordered_map<std::string, int64_t> &bpe_encoder,
      const std::unordered_map<std::string, int64_t> &bpe_merge_ranks,
      const std::string &seperator,
      const std::unordered_map<int64_t, std::string> &byte_encoder,
      bool caching_enabled)
      : GPT2BPEEncoder(
            _map_to_c10_dict<std::string, int64_t>(bpe_encoder),
            _map_to_c10_dict<std::string, int64_t>(bpe_merge_ranks),
            seperator,
            _map_to_c10_dict<int64_t, std::string>(byte_encoder),
            caching_enabled) {}

  std::vector<std::string> GPT2BPEEncoder::ByteEncode_(std::string token)
  {
    // Equivalent to: (self.byte_encoder[b] for b in token.encode('utf-8')
    std::vector<std::string> encoded;
    for (auto &ch : token)
    {
      encoded.push_back(byte_encoder_[(unsigned char)ch]);
    }
    return encoded;
  }

  int64_t GPT2BPEEncoder::GetBPEMergeRank_(std::string pair)
  {
    if (bpe_merge_ranks_.contains(pair))
    {
      return bpe_merge_ranks_[pair];
    }
    return inf_;
  }

  std::string GPT2BPEEncoder::FindBestPair_(std::vector<std::string> pairs)
  {
    // Equivalent to:
    //    min(pairs, key = lambda pair: self.bpe_merge_ranks.get(pair,
    //    float('inf')))
    auto best_pair_idx = 0;
    auto best_rank = GetBPEMergeRank_(pairs[best_pair_idx]);

    for (std::size_t i = 1; i < pairs.size(); ++i)
    {
      auto rank = GetBPEMergeRank_(pairs[i]);
      if (rank < best_rank)
      {
        best_pair_idx = i;
        best_rank = rank;
      }
    }
    return pairs[best_pair_idx];
  }

  std::vector<std::string> GPT2BPEEncoder::BPE_(
      const std::vector<std::string> &token_list)
  {
    // Given a list of input tokens, keep finding the best bpe merge and
    // generate a new list of tokens until
    //  1) token list size reduced to 1
    //      OR
    //  2) can't find bpe merge
    auto concatenated = concatenate_strings(token_list);
    if (caching_enabled_ && cache_.contains(concatenated))
    {
      return cache_[concatenated];
    }

    std::vector<std::string> tok_list = token_list;
    auto pairs = get_pairs(tok_list, seperator_);
    if (pairs.empty())
    {
      return {concatenated};
    }
    while (true)
    {
      auto bigram = FindBestPair_(pairs);
      if (!bpe_merge_ranks_.contains(bigram))
        break;

      // Finding all indexes that token_list[i] == first and token_list[i+1] ==
      // second. After the loop, new token list will be
      //  1) first + second pair
      //  2) all the other tokens in the original token list
      //
      // For example: first="a" second="w" and token_list =
      // ["a", "w", "some", "a", "w", "e"]
      // Result: new_token_list = ["aw", "some", "aw", "e"]

      auto parts = split_tokens(bigram, seperator_);
      std::vector<std::string> new_token_list;
      std::size_t i = 0;
      while (i < tok_list.size())
      {
        auto j = list_str_index(tok_list, parts.first, i);
        if (j != -1)
        {
          for (int k = i; k < j; k++)
            new_token_list.push_back(tok_list[k]);
          i = j;
        }
        else
        {
          for (std::size_t k = i; k < tok_list.size(); k++)
            new_token_list.push_back(tok_list[k]);
          break;
        }

        if (tok_list[i] == parts.first && i < (tok_list.size() - 1) &&
            tok_list[i + 1] == parts.second)
        {
          new_token_list.push_back(parts.first + parts.second);
          i += 2;
        }
        else
        {
          new_token_list.push_back(tok_list[i]);
          i += 1;
        }
      }

      tok_list = new_token_list;
      if (tok_list.size() == 1)
      {
        break;
      }
      else
      {
        pairs = get_pairs(tok_list, seperator_);
      }
    }

    if (caching_enabled_)
      cache_[concatenated] = tok_list;
    return tok_list;
  }

  std::vector<std::string> GPT2BPEEncoder::PreTokenize_(std::string input)
  {
    return gpt2_bpe_pre_tokenizer(input);
  }

  std::vector<int64_t> GPT2BPEEncoder::Encode(const std::string &text)
  {
    std::vector<int64_t> bpe_token_ids;
    for (const auto &token : PreTokenize_(text))
    {
      auto byte_encoded_token = ByteEncode_(token);
      for (const auto &bpe_token : BPE_(byte_encoded_token))
      {
        bpe_token_ids.push_back(bpe_encoder_[bpe_token]);
      }
    }
    return bpe_token_ids;
  }

  std::unordered_map<std::string, int64_t> GPT2BPEEncoder::GetBPEEncoder() const
  {
    return _c10_dict_to_map(bpe_encoder_);
  }

  std::unordered_map<std::string, int64_t> GPT2BPEEncoder::GetBPEMergeRanks()
      const
  {
    return _c10_dict_to_map(bpe_merge_ranks_);
  }

  std::unordered_map<int64_t, std::string> GPT2BPEEncoder::GetByteEncoder()
      const
  {
    return _c10_dict_to_map(byte_encoder_);
  }

} // namespace torchtext
