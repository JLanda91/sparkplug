// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa


#pragma once

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string_view>

namespace sparkplug::util::cuda {

class cuda_fn_error : public std::runtime_error {
public:
    cudaError_t code;

    explicit cuda_fn_error(cudaError_t err, std::string_view msg = "CUDA Error");

private:
    static std::string ToString(cudaError_t err, std::string_view msg);
};

}