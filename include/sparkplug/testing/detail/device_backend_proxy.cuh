// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa


#pragma once

#include <sparkplug/util/cuda_stream.cuh>
#include <sparkplug/util/cuda_check.cuh>
#include <sparkplug/util/pinned_scalar.cuh>

#include "deduced_signature.hpp"

namespace sparkplug::testing::detail {

    template<typename T>
    class DeviceBackendProxy {
    public:
        using signature = typename DeducedSignature<T>::type;
        using Callable = T;

        explicit DeviceBackendProxy(util::CudaStream& stream) : stream_(stream) {}

        void SetBackend(T* backend) {
            host_backend_ = backend;
        }

        [[nodiscard]] auto DevicePtr() const  {
            return callable_.DevicePtr();
        }

        [[nodiscard]] auto IsInitialized() const {
            return host_backend_ != nullptr;
        }

        void PopulateDevice() {
            callable_ = *host_backend_;
            callable_.ToDevAsync(stream_);
        }

    private:
        T* host_backend_ = nullptr;
        util::PinnedScalar<Callable> callable_{};
        util::CudaStream& stream_;
    };
}
