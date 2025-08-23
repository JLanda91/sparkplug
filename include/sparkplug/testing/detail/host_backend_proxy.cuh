#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <sparkplug/util/cuda_check.cuh>
#include <sparkplug/util/pinned_host_vector.cuh>

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
        using signature = typename DeducedSignature<T>::type;

        HostBackendProxy() = default;

        struct Callable {
            mutable typename signature::input_t arg_{};
            volatile typename signature::return_t out_{};
            volatile mutable ProxyState state_ = ProxyState::Idle;

            __device__ typename signature::return_t operator()(const typename signature::input_t& arg) const {
                assert(state_ == ProxyState::Idle);

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
            return thrust::raw_pointer_cast(device_callable_.data());
        }

        [[nodiscard]] auto IsInitialized() const {
            return host_backend_ != nullptr;
        }

        void PollAndSyncHostReturnIfArgSetOnDevice(const cudaStream_t& stream) {
            util::cuda_check("Polling: D->H step", cudaMemcpyAsync, thrust::raw_pointer_cast(host_callable_.data()), thrust::raw_pointer_cast(device_callable_.data()), sizeof(Callable), cudaMemcpyDeviceToHost, stream);
            util::cuda_check("", cudaStreamSynchronize, stream);
            if (auto& callable = host_callable_[0]; callable.state_ == ProxyState::ArgSetOnDevice) {
                callable.out_ = host_backend_->operator()(callable.arg_);
                callable.state_ = ProxyState::ReturnValueSetOnHost;
                util::cuda_check("Polling: H->D step", cudaMemcpyAsync, thrust::raw_pointer_cast(device_callable_.data()), thrust::raw_pointer_cast(host_callable_.data()), sizeof(Callable), cudaMemcpyHostToDevice, stream);
                util::cuda_check("", cudaStreamSynchronize, stream);
            }
        }

    private:
        T* host_backend_ = nullptr;
        util::pinned_host_vector<Callable> host_callable_ = util::pinned_host_vector<Callable>(1);
        thrust::device_vector<Callable> device_callable_ = thrust::device_vector<Callable>(1);
    };
}
