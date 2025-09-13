// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa


#pragma once

#include <cuda_runtime_api.h>

#include <string_view>

#include "cuda_fn_error.cuh"

namespace sparkplug::util::cuda {

template <typename CudaFunction, typename ... Args>
requires std::same_as<std::invoke_result_t<CudaFunction, Args...>, cudaError_t>
void check_cuda_fn_error(std::string_view msg, CudaFunction&& cuda_func, Args&&... args) {
    if (const cudaError_t err = cuda_func(std::forward<Args>(args)...); err != cudaSuccess) {
        throw cuda_fn_error(err, msg);
    }
}

}