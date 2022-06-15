/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "velox/functions/Udf.h"
#include "velox/type/Type.h"

namespace facebook::torcharrow::functions
{
    template <typename T>
    struct add_tokens
    {
        VELOX_DEFINE_FUNCTION_TYPES(T);

        // solutions
        // vector fn: reuse
        // use opaque:
        // input and output are strings (we can optimize using setNoCopy)
        template <typename TInput, typename TOutput, typename TTokens>
        FOLLY_ALWAYS_INLINE void call(
            TOutput &output,
            TInput &input,
            TTokens &tokens,
            bool begin = true)
        {
            //     if (begin)
            //     {
            //         output.reserve(input.size() + 1);
            //         output.copy_from(input);
            //         output.add_item(token);
            //     }
            //     else
            //     {
            //         output.reserve(input.size() + 1);
            //         output.add_item(token);
            //         output.add_items(input);
            //     }
            // }
            output.reserve(input.size() + tokens.size());
            if (typeid(tokens[0]) == typeid(int))
            {
                if (begin)
                {
                    output.add_items(tokens);
                    output.add_items(input);
                }
                else
                {
                    output.add_items(input);
                    output.add_items(tokens);
                }
            }
            else
            {
                if (begin)
                {
                    output.add_items(tokens);
                    // for (auto &token : tokens)
                    // {
                    //     auto item_writer = output.add_item();
                    //     item_writer = copy_from(setNoCopy(token));
                    // }

                    output.add_items(input);
                }
                else
                {
                    output.add_items(input);
                    output.add_items(tokens);
                }
            }
        }
    };

} // namespace facebook::torcharrow::functions
