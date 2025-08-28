// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa


#include <sstream>

#include "../include/sparkplug/util/cuda_error.cuh"

namespace sparkplug::util {
    cuda_error::cuda_error(cudaError_t err, std::string_view msg)
        : std::runtime_error(ToString(err, msg))
        , code(err) {}

    std::string cuda_error::ToString(cudaError_t err, std::string_view msg) {
        std::ostringstream oss{};
        oss << msg << ": " << cudaGetErrorString(err) << "(Error code: " << static_cast<std::underlying_type_t<cudaError_t>>(err) << ')';
        return oss.str();
    }

}
