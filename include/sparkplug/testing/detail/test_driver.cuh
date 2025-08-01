#pragma once

namespace sparkplug::testing::detail {
    template<typename FunctorT, typename InputT, typename ReturnT, typename... MockPtrs>
    __global__ void test_driver(const InputT* in, ReturnT* out, MockPtrs... mocks)
    {
        FunctorT f(mocks...);  // Functor constructor accepts mocks
        *out = f(*in);
    }
}
