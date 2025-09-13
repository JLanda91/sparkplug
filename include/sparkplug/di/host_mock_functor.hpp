// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa


#pragma once

#include <gmock/gmock.h>

#include <sparkplug/util/concepts/signature.hpp>

#include "dependency.hpp"

namespace sparkplug::di {

template<util::concepts::Signature Signature>
struct HostMockFunctor {
    Signature::return_type operator()(const Signature::arg_type& arg) const {
        return Call(arg);
    }
    MOCK_METHOD(typename Signature::return_type, Call, (const typename Signature::arg_type&), (const));
};

template<util::concepts::Signature Signature>
using host_nice_gmock_functor_dependency = host_dependency<::testing::NiceMock<HostMockFunctor<Signature>>>;

template<util::concepts::Signature Signature>
using host_naggy_gmock_functor_dependency = host_dependency<::testing::NaggyMock<HostMockFunctor<Signature>>>;

template<util::concepts::Signature Signature>
using host_strict_gmock_functor_dependency = host_dependency<::testing::StrictMock<HostMockFunctor<Signature>>>;

}
