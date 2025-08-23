#pragma once

#include <string_view>

#include "cuda_error.cuh"

namespace sparkplug::util {
    template<typename CudaFunction, typename ... Args>
    void cuda_check(std::string_view msg, CudaFunction&& cuda_func, Args&&... args) {
        using cuda_func_return_t = std::invoke_result_t<CudaFunction, Args...>;
        static_assert(std::is_same_v<cudaError_t, cuda_func_return_t>, "cuda_check must be used with cuda functions");

        if (const cudaError_t err = cuda_func(std::forward<Args>(args)...); err != cudaSuccess) {
            throw cuda_error(err, msg);
        }
    }
}