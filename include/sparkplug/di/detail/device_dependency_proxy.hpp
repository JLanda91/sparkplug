// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa


#pragma once

#include <sparkplug/util/cuda/stream.cuh>
#include <sparkplug/util/cuda/pinned_scalar.cuh>
#include <sparkplug/util/concepts/dependency.hpp>

#include "dependency_proxy.hpp"


namespace sparkplug::di::detail {

template<util::concepts::Dependency Dep>
struct DeviceDependencyProxy : DependencyProxy<Dep, typename Dep::type> {

    void PopulateDevice(util::cuda::Stream& stream) {
        this->callable_ = *this->host_dependency_;
        this->callable_.ToDevAsync(stream);
    }
};

}
