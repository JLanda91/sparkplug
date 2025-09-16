// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa

#pragma once

#include <sparkplug/util/cuda/pinned_scalar.cuh>
#include <sparkplug/util/concepts/dependency.hpp>
#include <sparkplug/util/concepts/callable.hpp>

namespace sparkplug::di::detail {

template<typename Derived, util::concepts::Dependency Dep, util::concepts::Callable Callable>
class DependencyProxy {
public:
    using dependency = Dep;
    using callable = Callable;

    void PopulateDevice(util::cuda::Stream& stream) {
        static_cast<Derived*>(this)->PopulateDevice(stream);
    }

    void SetDependency(Dep::type* dep) {
        host_dependency_ = dep;
    }

    [[nodiscard]] auto DevicePtr() const  {
        return callable_.DevicePtr();
    }

protected:
    Dep::type* host_dependency_ = nullptr;
    util::cuda::PinnedScalar<Callable> callable_{};
};

}