// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa


#pragma once

#include <stdexcept>
#include <string_view>

namespace sparkplug::util {
    class cuda_error : public std::runtime_error {
    public:
        cudaError_t code;

        explicit cuda_error(cudaError_t err, std::string_view msg = "CUDA Error");
    private:
        static std::string ToString(cudaError_t err, std::string_view msg);
    };
}