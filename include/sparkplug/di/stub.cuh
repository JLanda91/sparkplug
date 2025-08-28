// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa


#pragma once

namespace sparkplug::di {
    template<typename Signature>
    class Stub {
    public:
        explicit Stub(const typename Signature::return_t& return_value) : return_value_(return_value) {}

        __host__ __device__ typename Signature::return_t operator()([[maybe_unused]] const typename Signature::input_t& arg) const {
            return return_value_;
        }

    private:
        typename Signature::return_t return_value_;
    };

}

#define SPARKPLUG_STUB_HOST_BACKEND(InputT, ReturnT) sparkplug::testing::HostBackend<sparkplug::di::Stub<sparkplug::di::Signature<InputT, ReturnT>>>
#define SPARKPLUG_STUB_DEVICE_BACKEND(InputT, ReturnT) sparkplug::testing::DeviceBackend<sparkplug::di::Stub<sparkplug::di::Signature<InputT, ReturnT>>>