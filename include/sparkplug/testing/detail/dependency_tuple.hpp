// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa

#pragma once

#include <sparkplug/util/concepts/dependency.hpp>
#include <sparkplug/util/cuda/stream.cuh>

#include <sparkplug/di/detail/device_dependency_proxy.hpp>
#include <sparkplug/di/detail/host_dependency_proxy.cuh>


namespace sparkplug::testing::detail {
    
template<typename Proxy>
void PollAndUpdateCallableWithHostReturnValue(Proxy& proxy, util::cuda::Stream& stream) {
    if constexpr (!Proxy::is_device_side) {
        proxy.PollAndUpdateCallableWithHostReturnValue(stream);
    }
}

template<typename Proxy>
void populate_device_proxy(Proxy& proxy, util::cuda::Stream& stream) {
    if constexpr (Proxy::is_device_side) {
        proxy.PopulateDevice(stream);
    }
}

template <util::concepts::Dependency Dependency>
using proxy = std::conditional_t<Dependency::is_device_side,
                                 di::detail::DeviceDependencyProxy<Dependency>,
                                 di::detail::HostDependencyProxy<Dependency>>;


template <util::concepts::Dependency ... Deps>
class DependencyTuple {
public:
    static constexpr bool has_host_dependencies = !(Deps::is_device_side && ...);
    static constexpr bool has_device_dependencies = (Deps::is_device_side || ...);

    using proxy_tuple = std::tuple<proxy<Deps>...>;

    void PollAndSyncHostProxies(util::cuda::Stream& stream) {
        std::apply([&stream](proxy<Deps>&... ts) {
            ( (PollAndUpdateCallableWithHostReturnValue(ts, stream)), ...);
        }, dependency_proxies);
    }

    void PopulateDeviceProxies(util::cuda::Stream& stream) {
        std::apply([&stream](proxy<Deps>&... ts) {
            ( (populate_device_proxy(ts, stream)), ...);
        }, dependency_proxies);
    }

    void PrepareProxies(Deps::type* ... arg) {
        PrepareProxiesImpl(std::forward_as_tuple(arg...), std::index_sequence_for<Deps...>{});
        is_initialized = true;
    }

    proxy_tuple dependency_proxies{};
    bool is_initialized = false;

private:

    template <typename ArgTuple, std::size_t... Is>
    void PrepareProxiesImpl(ArgTuple&& arg_tuple, std::index_sequence<Is...> ) {
        ((std::get<Is>(dependency_proxies).SetDependency(std::get<Is>(std::forward<ArgTuple>(arg_tuple)))), ...);
    }
};

}