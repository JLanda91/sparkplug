// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa


#include <sstream>

#include <sparkplug/util/cuda/cuda_fn_error.cuh>

namespace sparkplug::util::cuda {
    cuda_fn_error::cuda_fn_error(cudaError_t err, std::string_view msg)
        : std::runtime_error(ToString(err, msg))
        , code(err) {}

    std::string cuda_fn_error::ToString(cudaError_t err, std::string_view msg) {
        std::ostringstream oss{};
        oss << msg << ": " << cudaGetErrorString(err) << "(Error code: " << err << ')';
        return oss.str();
    }

}
