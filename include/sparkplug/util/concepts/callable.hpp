// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa

#pragma once


namespace sparkplug::util::concepts {

template<typename T>
concept Callable = requires {
    &T::operator();
};

}
