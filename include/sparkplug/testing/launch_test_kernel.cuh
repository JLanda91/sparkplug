#pragma once

#include <sparkplug/testing/detail/test_driver.cuh>
#include <sparkplug/util/cuda_check.cuh>

#include "detail/deduced_signature.hpp"

namespace sparkplug::testing {

template<typename FunctorT, typename InputT, typename ReturnT, typename... MockPtrs>
void launch_test_kernel(const InputT* in, ReturnT* d_out, MockPtrs... mocks)
{
    detail::test_driver<FunctorT> <<<1, 1>>>(in, d_out, mocks...);
    util::cuda_check("Failure in kernel launch", cudaGetLastError);
}

template<typename FunctorT>
void launch_test_kernel_nocreate(const FunctorT* device_func,
                                 const typename detail::DeducedSignature<FunctorT>::type::input_t* device_input,
                                 typename detail::DeducedSignature<FunctorT>::type::return_t* device_output,
                                 const cudaStream_t& stream)
{
    detail::test_driver_nocreate<<<1, 1, 0, stream>>>(device_func, device_input, device_output);
    util::cuda_check("Failure in kernel launch", cudaGetLastError);
}

}
