// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa


#pragma once

#include <sparkplug/util/pinned_scalar.cuh>
#include <sparkplug/util/cuda_stream.cuh>
#include <sparkplug/util/cuda_check.cuh>

#include "detail/deduced_signature.hpp"

namespace sparkplug::testing {

template<typename Functor>
class FunctorTestEnvironment {
    using Signature = typename detail::DeducedSignature<Functor>::type;
public:
    auto GetFunctorPtr() const {
        return functor_.DevicePtr();
    }

    void SetFunctor(const Functor& f) {
        functor_ = f;
        functor_.ToDevAsync(test_driver_stream_);
    }

    auto GetInputPtr() const {
        return input_.DevicePtr();
    }

    void SetInput(const typename Signature::input_t& arg) {
        input_ = arg;
        input_.ToDevAsync(test_driver_stream_);
    }

    auto GetReturnPtr() {
        return return_.DevicePtr();
    }

    auto GetReturnValue() {
        return_.ToHostAsync(test_driver_stream_);
        test_driver_stream_.Synchronize();
        return *return_.HostPtr();
    }

    auto& TestDriverStream() {
        return test_driver_stream_;
    }

    auto& ProxyStream() {
        return proxy_stream_;
    }

private:
    util::CudaStream test_driver_stream_ = {};
    util::CudaStream proxy_stream_ = {};

    util::PinnedScalar<Functor> functor_ {};
    util::PinnedScalar<typename Signature::input_t> input_ {};
    util::PinnedScalar<typename Signature::return_t> return_ {};
};

namespace detail {
    template<typename Functor>
    inline FunctorTestEnvironment<Functor>* functor_test_env = nullptr;
}

}