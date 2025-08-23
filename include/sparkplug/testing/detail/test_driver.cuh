#pragma once

#include "deduced_signature.hpp"

namespace sparkplug::testing::detail {
    template<typename FunctorT, typename InputT, typename ReturnT, typename... MockPtrs>
    __global__ void test_driver(const InputT* in, ReturnT* out, MockPtrs... mocks)
    {
        FunctorT f(mocks...);  // Functor constructor accepts mocks
        *out = f(*in);
    }

    template<typename FunctorT>
    __global__ void test_driver_nocreate(const FunctorT* device_functor,
                                         const typename DeducedSignature<FunctorT>::type::input_t* device_input,
                                         typename DeducedSignature<FunctorT>::type::return_t* device_output)
    {
        *device_output = device_functor->operator()(*device_input);
    }
}
