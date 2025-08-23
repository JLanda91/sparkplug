#pragma once

#include <gmock/gmock.h>

namespace sparkplug::mocking {
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

#define SPARKPLUG_NICE_HOST_MOCK(InputT, ReturnT) sparkplug::mocking::NiceHostMock<sparkplug::mocking::Signature<InputT, ReturnT>>
#define SPARKPLUG_NAGGY_HOST_MOCK(InputT, ReturnT) sparkplug::mocking::NaggyHostMock<sparkplug::mocking::Signature<InputT, ReturnT>>
#define SPARKPLUG_STRICT_HOST_MOCK(InputT, ReturnT) sparkplug::mocking::StrictHostMock<sparkplug::mocking::Signature<InputT, ReturnT>>
