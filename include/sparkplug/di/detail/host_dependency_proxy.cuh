// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa


#pragma once

#include <cuda_runtime.h>

#include <sparkplug/util/cuda/pinned_scalar.cuh>
#include <sparkplug/util/cuda/stream.cuh>
#include <sparkplug/util/signature.hpp>

#include "dependency_proxy.hpp"


namespace sparkplug::di::detail {

enum class ProxyState {
    Idle,
    ArgSetOnDevice,
    ReturnValueSetOnHost,
};

inline constexpr unsigned kProxySleepIntervalNs = 10'000u;

template<util::concepts::Signature Signature>
class HostDepedencyProxyFunctor {
public:
    HostDepedencyProxyFunctor(ProxyState* state, Signature::arg_type* arg, Signature::return_type* out) : state_(state), arg_(arg), return_(out)  {}

    __device__ Signature::return_type operator()(const Signature::arg_type& arg) const {
        *arg_ = arg;
        *state_ = ProxyState::ArgSetOnDevice;

        while (*state_ != ProxyState::ReturnValueSetOnHost) {
            __nanosleep(kProxySleepIntervalNs);
        }

        *state_ = ProxyState::Idle;
        return *return_;
    }

private:
    volatile ProxyState* state_;
    Signature::arg_type* arg_;
    volatile Signature::return_type* return_;
};

template<util::concepts::Dependency Dep>
class HostDependencyProxy : public DependencyProxy<HostDependencyProxy<Dep>, Dep, HostDepedencyProxyFunctor<util::deduced_signature_t<typename Dep::type>>> {
public:
    void PopulateDevice(util::cuda::Stream& stream) {
        this->callable_.emplace(state_.DevicePtr(), arg_.DevicePtr(), return_.DevicePtr());
        this->callable_.ToDevAsync(stream);
        this->state_.ToDevAsync(stream);
    }

    void PollAndUpdateWithHostReturnValue(util::cuda::Stream& stream) {
        state_.ToHostAsync(stream);
        stream.Synchronize();
        arg_.ToHostAsync(stream);
        if (*state_.HostPtr() == ProxyState::ArgSetOnDevice) {
            *state_.HostPtr() = ProxyState::ReturnValueSetOnHost;
            stream.Synchronize();
            *return_.HostPtr() = this->host_dependency_->operator()(*arg_.HostPtr());
            return_.ToDevAsync(stream);
            state_.ToDevAsync(stream);
        }
    }

private:
    using signature = util::deduced_signature_t<typename Dep::type>;

    util::cuda::PinnedScalar<ProxyState> state_{std::in_place, ProxyState::Idle};
    util::cuda::PinnedScalar<typename signature::arg_type> arg_{};
    util::cuda::PinnedScalar<typename signature::return_type> return_{};
};

}
