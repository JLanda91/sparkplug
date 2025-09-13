// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa

#pragma once

#include <sparkplug/util/cuda/pinned_scalar.cuh>
#include <sparkplug/util/concepts/dependency.hpp>
#include <sparkplug/util/concepts/callable.hpp>

namespace sparkplug::di::detail {

template<util::concepts::Dependency Dep, util::concepts::Callable Callable>
class DependencyProxy {
public:
    using dependency = Dep;
    using callable = Callable;
    static constexpr bool is_device_side = Dep::is_device_side;

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