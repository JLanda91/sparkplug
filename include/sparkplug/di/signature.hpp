// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa


#pragma once

namespace sparkplug::di {
    template<typename InputT, typename ReturnT>
    struct Signature {
        using input_t = InputT;
        using return_t = ReturnT;
    };

}