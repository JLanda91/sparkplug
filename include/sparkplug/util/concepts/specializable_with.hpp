// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa

#pragma once


namespace sparkplug::util::concepts {

template<template <typename ...> typename FunctorTemplate, typename ... FunctorDeps>
concept TestableFunctorTemplate = requires {
    typename FunctorTemplate<FunctorDeps...>;
    requires std::constructible_from<FunctorTemplate<FunctorDeps...>, FunctorDeps* ...>;
};



}