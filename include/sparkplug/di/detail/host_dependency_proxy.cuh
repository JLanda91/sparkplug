// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa


#pragma once

#include <cuda_runtime.h>

#include <sparkplug/util/cuda/pinned_scalar.cuh>
#include <sparkplug/util/cuda/stream.cuh>
#include <sparkplug/util/signature.hpp>


namespace sparkplug::di::detail {

enum class ProxyState {
    Idle,
    ArgSetOnDevice,
    ReturnValueSetOnHost,
};

inline constexpr unsigned kDeviceProxyPollIntervalNs = 10'000u;

template<util::concepts::Signature Signature>
struct HostDepedencyProxyFunctor {
    mutable Signature::arg_type arg_{};
    volatile Signature::return_type out_{};
    volatile mutable ProxyState state_ = ProxyState::Idle;

    __device__ Signature::return_type operator()(const Signature::arg_type& arg) const {
        arg_ = arg;
        state_ = ProxyState::ArgSetOnDevice;

        while (state_ != ProxyState::ReturnValueSetOnHost) {
            __nanosleep(kDeviceProxyPollIntervalNs);
        }

        state_ = ProxyState::Idle;
        return out_;
    }
};

template<util::concepts::Dependency Dep>
struct HostDependencyProxy : DependencyProxy<Dep, HostDepedencyProxyFunctor<util::deduced_signature_t<typename Dep::type>>> {

    void PollAndUpdateCallableWithHostReturnValue(util::cuda::Stream& stream) {
        this->callable_.ToHostAsync(stream);
        stream.Synchronize();
        if (auto* callable = this->callable_.HostPtr(); callable->state_ == ProxyState::ArgSetOnDevice) {
            callable->out_ = this->host_dependency_->operator()(callable->arg_);
            callable->state_ = ProxyState::ReturnValueSetOnHost;
            this->callable_.ToDevAsync(stream);
        }
    }
};

}
