// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa


#pragma once

#include <gtest/gtest.h>

#include <thread>
#include <atomic>

#include "functor_test_environment.cuh"
#include "launch_test_kernel.cuh"
#include "detail/mapped_proxy.hpp"

namespace sparkplug::testing {

template <template <typename> typename Functor, typename Backend>
class FunctorTest : public ::testing::Test {
    // backend mappings
    using mapped_proxy_t = detail::mapped_proxy_t<Backend>; // EXPAND TO TUPLE FOR MULTIPLE BACKENDS

public:
    // functor mappings
    using functor_t = Functor<typename mapped_proxy_t::Callable>;
    using functor_signature_t = typename detail::DeducedSignature<functor_t>::type;

    static void SetUpTestSuite() {
        detail::functor_test_env<functor_t> = new FunctorTestEnvironment<functor_t>;
    }

    static void TearDownTestSuite() {
        delete detail::functor_test_env<functor_t>;
        detail::functor_test_env<functor_t> = nullptr;
    }

    void PrepareBackend(typename Backend::type* arg) {
        proxied_backend_.SetBackend(arg);
    }

    typename functor_signature_t::return_t RunOnDevice(const typename functor_signature_t::input_t& arg) {
        if (!proxied_backend_.IsInitialized()) {
            ADD_FAILURE() << "Backend is not initialized";
            return typename functor_signature_t::return_t();
        }

        if constexpr (Backend::is_device_side) {
            proxied_backend_.PopulateDevice();
        }

        detail::functor_test_env<functor_t>->SetInput(arg);
        detail::functor_test_env<functor_t>->SetFunctor(functor_t{proxied_backend_.DevicePtr()});

        if constexpr (!Backend::is_device_side) {
            is_kernel_finished_.store(false);
            host_poller_ = std::thread([this] {
                while(!is_kernel_finished_.load()) {
                    proxied_backend_.PollAndSyncHostReturnIfArgSetOnDevice();
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            });
        }

        detail::functor_test_env<functor_t>->ProxyStream().Synchronize();

        launch_test_kernel_nocreate(
            detail::functor_test_env<functor_t>->GetFunctorPtr(),
            detail::functor_test_env<functor_t>->GetInputPtr(),
            detail::functor_test_env<functor_t>->GetReturnPtr(),
            detail::functor_test_env<functor_t>->TestDriverStream()
        );

        detail::functor_test_env<functor_t>->TestDriverStream().Synchronize();

        if constexpr (!Backend::is_device_side) {
            is_kernel_finished_.store(true);
            host_poller_.join();
        }
        return detail::functor_test_env<functor_t>->GetReturnValue();
    }

protected:
    std::atomic<bool> is_kernel_finished_ = false;
    mapped_proxy_t proxied_backend_ = mapped_proxy_t{detail::functor_test_env<functor_t>->ProxyStream()};
    std::thread host_poller_;
};
}
