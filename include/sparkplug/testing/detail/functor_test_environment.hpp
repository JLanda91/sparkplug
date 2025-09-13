// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa


#pragma once

#include <sparkplug/util/cuda/pinned_scalar.cuh>
#include <sparkplug/util/cuda/stream.cuh>
#include <sparkplug/util/signature.hpp>

namespace sparkplug::testing::detail {

template<util::concepts::Callable Functor>
class FunctorTestEnvironment {
    using signature = util::deduced_signature_t<Functor>;

public:
    auto GetFunctorPtr() const {
        return functor_.DevicePtr();
    }

    template<typename ... Args>
    void EmplaceFunctor(Args&& ... args) {
        functor_.emplace(std::forward<Args>(args)...);
        functor_.ToDevAsync(test_driver_stream_);
    }

    auto GetArgPtr() const {
        return arg_.DevicePtr();
    }

    template<typename ... Args>
    void EmplaceArg(Args&& ... args) {
        arg_.emplace(std::forward<Args>(args)...);
        arg_.ToDevAsync(test_driver_stream_);
    }

    auto GetReturnPtr() {
        return return_.DevicePtr();
    }

    auto& GetReturnValue() {
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
    util::cuda::Stream test_driver_stream_ = {};
    util::cuda::Stream proxy_stream_ = {};

    util::cuda::PinnedScalar<Functor> functor_ {};
    util::cuda::PinnedScalar<typename signature::arg_type> arg_ {};
    util::cuda::PinnedScalar<typename signature::return_type> return_ {};
};

template <util::concepts::Callable Functor>
inline FunctorTestEnvironment<Functor>* functor_test_env = nullptr;


}