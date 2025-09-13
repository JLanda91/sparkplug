// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa


#pragma once

namespace sparkplug::di {

template<class T, bool IsDeviceSide>
struct Dependency {
    using type = T;
    static constexpr bool is_device_side = IsDeviceSide;
};

template<class T>
using host_dependency = Dependency<T, false>;

template<class T>
using device_dependency = Dependency<T, true>;

}
