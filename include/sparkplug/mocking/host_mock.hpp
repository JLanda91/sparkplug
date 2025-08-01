#pragma once

#include <gmock/gmock.h>

namespace sparkplug::mocking {
template<typename Signature>
class HostMock {
public:
    typename Signature::return_t operator()(const typename Signature::input_t& arg) const {
        return Call(arg);
    }
private:
    MOCK_METHOD(typename Signature::return_t, Call, (const typename Signature::input_t&), (const));
};

template<typename Signature>
using NiceHostMock = ::testing::NiceMock<HostMock<Signature>>;

template<typename Signature>
using NaggyHostMock = ::testing::NaggyMock<HostMock<Signature>>;

template<typename Signature>
using StrictHostMock = ::testing::StrictMock<HostMock<Signature>>;
}
