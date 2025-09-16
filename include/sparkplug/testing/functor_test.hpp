// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa


#pragma once

#include <gtest/gtest.h>

#include <thread>
#include <atomic>
#include <chrono>

#include <sparkplug/util/concepts/dependency.hpp>
#include <sparkplug/util/signature.hpp>

#include "detail/functor_test_environment.hpp"
#include "detail/launch_test_kernel.cuh"
#include "detail/dependency_tuple.hpp"

namespace sparkplug::testing {

namespace detail {

template <template <typename...> typename FunctorTemplate, typename Tuple, std::size_t... I>
auto specialized_functor(std::index_sequence<I...>) -> FunctorTemplate<typename std::tuple_element_t<I, Tuple>::callable...>;

template <template <typename...> typename FunctorTemplate, typename Tuple>
using specialized_functor_t = decltype(specialized_functor<FunctorTemplate, Tuple>(std::make_index_sequence<std::tuple_size_v<Tuple>>{}));

inline constexpr unsigned kHostPollSleepIntervalNs = 10'000u;

}

template <template <typename...> typename FunctorTemplate, util::concepts::Dependency ... Deps>
class FunctorTest : public ::testing::Test {
    using dependency_tuple = detail::DependencyTuple<Deps...>;

public:
    using functor = detail::specialized_functor_t<FunctorTemplate, typename dependency_tuple::proxy_tuple>;
    using functor_signature = util::deduced_signature_t<functor>;

    static void SetUpTestSuite() {
        detail::functor_test_env<functor> = new detail::FunctorTestEnvironment<functor>;
    }

    static void TearDownTestSuite() {
        delete detail::functor_test_env<functor>;
        detail::functor_test_env<functor> = nullptr;
        util::cuda::check_cuda_fn_error("Reset Device", cudaDeviceReset);
    }

    void TearDown() override {
        detail::functor_test_env<functor>->ProxyStream().Synchronize();
    }

    void InjectDependencies(Deps::type* ... arg) {
        dependencies_.PrepareProxies(arg...);

        std::apply([](auto const&... elems) {
           detail::functor_test_env<functor>->EmplaceFunctor(elems.DevicePtr()...);
        }, dependencies_.Proxies());
    }

    template<typename ... Args>
    void ConstructArgumentOnDevice(Args&& ... args) {
        detail::functor_test_env<functor>->EmplaceArg(std::forward<Args>(args)...);
    }

    functor_signature::return_type RunOnDevice() {
        if (!dependencies_.IsInitialized()) {
            throw std::runtime_error("Dependencies were not initialized with FunctorTest::InjectDependencies");
        }

        dependencies_.PopulateProxiesOnDevice(detail::functor_test_env<functor>->TestDriverStream());

        if constexpr (dependency_tuple::has_host_dependencies) {
            is_kernel_finished_.store(false);
            host_poller_ = std::thread([this] {
                while(!is_kernel_finished_.load()) {
                    dependencies_.PollAndSyncHostProxies(detail::functor_test_env<functor>->ProxyStream());
                    std::this_thread::sleep_for(std::chrono::nanoseconds(detail::kHostPollSleepIntervalNs));
                }
            });
        }

        detail::launch_functor_test_kernel(
            detail::functor_test_env<functor>->GetFunctorPtr(),
            detail::functor_test_env<functor>->GetArgPtr(),
            detail::functor_test_env<functor>->GetReturnPtr(),
            detail::functor_test_env<functor>->TestDriverStream()
        );

        detail::functor_test_env<functor>->TestDriverStream().Synchronize();

        if constexpr (dependency_tuple::has_host_dependencies) {
            is_kernel_finished_.store(true);
            host_poller_.join();
        }
        return detail::functor_test_env<functor>->GetReturnValue();
    }

private:
    std::atomic<bool> is_kernel_finished_ = false;
    dependency_tuple dependencies_;
    std::thread host_poller_;
};
}
