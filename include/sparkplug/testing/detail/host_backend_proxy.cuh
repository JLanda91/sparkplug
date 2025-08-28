// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa


#pragma once

#include <sparkplug/util/pinned_scalar.cuh>
#include <sparkplug/util/cuda_stream.cuh>

#include "deduced_signature.hpp"

namespace sparkplug::testing::detail {
    enum class ProxyState {
        Idle,
        ArgSetOnDevice,
        ReturnValueSetOnHost,
    };

    inline constexpr unsigned kDeviceProxyPollIntervalMs = 1u;

    template<typename T>
    class HostBackendProxy {
    public:
        static_assert(DeducedSignature<T>::value, "Cannot instantiate HostBackendProxy: type has no call operator");
        using signature_t = typename DeducedSignature<T>::type;

        explicit HostBackendProxy(util::CudaStream& stream) : stream_(stream) {}

        struct Callable {
            mutable typename signature_t::input_t arg_{};
            volatile typename signature_t::return_t out_{};
            volatile mutable ProxyState state_ = ProxyState::Idle;

            __device__ typename signature_t::return_t operator()(const typename signature_t::input_t& arg) const {
                arg_ = arg;
                state_ = ProxyState::ArgSetOnDevice;

                while (state_ != ProxyState::ReturnValueSetOnHost) {
                    __nanosleep(kDeviceProxyPollIntervalMs * 1'000'000ull);
                }

                state_ = ProxyState::Idle;
                return out_;
            }
        };

        void SetBackend(T* backend) {
            host_backend_ = backend;
        }

        [[nodiscard]] auto DevicePtr() const  {
            return callable_.DevicePtr();
        }

        [[nodiscard]] auto IsInitialized() const {
            return host_backend_ != nullptr;
        }

        void PollAndSyncHostReturnIfArgSetOnDevice() {
            callable_.ToHostAsync(stream_);
            stream_.Synchronize();
            if (auto* callable = callable_.HostPtr(); callable->state_ == ProxyState::ArgSetOnDevice) {
                callable->out_ = host_backend_->operator()(callable->arg_);
                callable->state_ = ProxyState::ReturnValueSetOnHost;
                callable_.ToDevAsync(stream_);
            }
        }

    private:
        T* host_backend_ = nullptr;
        util::PinnedScalar<Callable> callable_{};
        util::CudaStream& stream_;
    };
}
