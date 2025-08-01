#pragma once

#include <sparkplug/testing/detail/test_driver.cuh>
#include "cuda_error.cuh"

namespace sparkplug::testing {

namespace detail {
    template<typename CudaFunction, typename ... Args>
    void cuda_check(CudaFunction&& cuda_func, Args&&... args, std::string_view msg) {
        using cuda_func_return_t = std::invoke_result_t<CudaFunction, Args...>;
        static_assert(std::is_same_v<cudaError_t, cuda_func_return_t>, "cuda_check must be used with cuda functions");

        if (const cudaError_t err = cuda_func(std::forward<Args>(args)...); err != cudaSuccess) {
            throw cuda_error(err, msg);
        }
    }
}

template<typename FunctorT, typename InputT, typename ReturnT, typename... MockPtrs>
void launch_test_kernel(const InputT* in, ReturnT* d_out, MockPtrs... mocks)
{
    detail::test_driver<FunctorT> <<<1, 1>>>(in, d_out, mocks...);

    // detail::cuda_check(cudaGetLastError, "Getting CUDA last error");
    // detail::cuda_check(cudaDeviceSynchronize, "Synchronizing CUDA device");
    cudaDeviceSynchronize(); // TODO REVERT WHEN STRUCTURE IS DONE
}

}
