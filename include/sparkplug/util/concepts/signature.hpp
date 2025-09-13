// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa

#pragma once


namespace sparkplug::util::concepts {

template<typename T>
concept Signature = requires {
    typename T::arg_type;
    typename T::return_type;
};

}