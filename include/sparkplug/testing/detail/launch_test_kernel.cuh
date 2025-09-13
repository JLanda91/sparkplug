// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa


#pragma once

#include <cuda_runtime.h>

#include <sparkplug/util/cuda/check_cuda_fn_error.cuh>
#include <sparkplug/util/signature.hpp>

namespace sparkplug::testing::detail {

template <util::concepts::Callable Functor>
__global__ void functor_test_driver(const Functor* device_functor,
                                     const typename util::deduced_signature_t<Functor>::arg_type* device_arg,
                                     typename util::deduced_signature_t<Functor>::return_type* device_return)
{
    *device_return = device_functor->operator()(*device_arg);
}

template <util::concepts::Callable Functor>
void launch_functor_test_kernel(const Functor* device_func,
                                 const typename util::deduced_signature_t<Functor>::arg_type* device_arg,
                                 typename util::deduced_signature_t<Functor>::return_type* device_return,
                                 const cudaStream_t& stream)
{
    functor_test_driver<<<1, 1, 0, stream>>>(device_func, device_arg, device_return);
    util::cuda::check_cuda_fn_error("Failure in kernel launch", cudaGetLastError);
}

}
