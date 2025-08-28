// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa


#pragma once

#include <gmock/gmock.h>

namespace sparkplug::di {
template<typename Signature>
struct HostMock {
    typename Signature::return_t operator()(const typename Signature::input_t& arg) const {
        return Call(arg);
    }
    MOCK_METHOD(typename Signature::return_t, Call, (const typename Signature::input_t&), (const));
};

template<typename Signature>
using NiceHostMock = ::testing::NiceMock<HostMock<Signature>>;

template<typename Signature>
using NaggyHostMock = ::testing::NaggyMock<HostMock<Signature>>;

template<typename Signature>
using StrictHostMock = ::testing::StrictMock<HostMock<Signature>>;

}

#define SPARKPLUG_NICE_MOCK_HOST_BACKEND(InputT, ReturnT) sparkplug::testing::HostBackend<sparkplug::di::NiceHostMock<sparkplug::di::Signature<InputT, ReturnT>>>
#define SPARKPLUG_NAGGY_MOCK_HOST_BACKEND(InputT, ReturnT) sparkplug::testing::HostBackend<sparkplug::di::NaggyHostMock<sparkplug::di::Signature<InputT, ReturnT>>>
#define SPARKPLUG_STRICT_MOCK_HOST_BACKEND(InputT, ReturnT) sparkplug::testing::HostBackend<sparkplug::di::StrictHostMock<sparkplug::di::Signature<InputT, ReturnT>>>
