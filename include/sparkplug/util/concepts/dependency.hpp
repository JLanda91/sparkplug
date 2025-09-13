// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa

#pragma once

#include <concepts>

namespace sparkplug::util::concepts {

template<typename T>
concept Dependency = requires {
    typename T::type;
    { T::is_device_side } -> std::same_as<const bool&>;
};

}