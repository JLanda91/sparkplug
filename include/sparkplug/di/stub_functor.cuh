// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa


#pragma once

#include <sparkplug/util/concepts/signature.hpp>

#include "dependency.hpp"

namespace sparkplug::di {

template<util::concepts::Signature Signature>
class StubFunctor {
public:
    explicit StubFunctor(const Signature::return_type& return_value) : return_value_(return_value) {}

    __host__ __device__ Signature::return_type operator()([[maybe_unused]] const Signature::arg_type& arg) const {
        return return_value_;
    }

private:
    Signature::return_type return_value_;
};

template<util::concepts::Signature Signature>
using host_stub_functor_dependency = host_dependency<StubFunctor<Signature>>;

template<util::concepts::Signature Signature>
using device_stub_functor_dependency = device_dependency<StubFunctor<Signature>>;

}
