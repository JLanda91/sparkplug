#pragma once

#include <gtest/gtest.h>

#include <thread>
#include <atomic>

#include <sparkplug/testing/detail/host_backend_proxy.cuh>

#include "functor_test_environment.cuh"
#include "launch_test_kernel.cuh"

namespace sparkplug::testing {

template <template <typename> typename Functor, typename HostBackend>
class FunctorTest : public ::testing::Test {
    // backend mappings
    static_assert(!HostBackend::is_device_side); // REMOVE WHEN BOTH HOST+DEVICE BACKENDS ARE DONE
    using mapped_proxy_t = detail::HostBackendProxy<typename HostBackend::type>; // EXPAND TO TUPLE FOR MULTIPLE BACKENDS

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

    void PrepareBackend(typename HostBackend::type* arg) {
        proxied_backend_.SetBackend(arg);
    }

    typename functor_signature_t::return_t RunOnDevice(const typename functor_signature_t::input_t& arg) {
        if (!proxied_backend_.IsInitialized()) {
            ADD_FAILURE() << "Backend is not initialized";
            return typename functor_signature_t::return_t();
        }

        detail::functor_test_env<functor_t>->SetInput(arg);
        detail::functor_test_env<functor_t>->SetFunctor(functor_t{proxied_backend_.DevicePtr()});

        is_kernel_finished_.store(false);
        std::thread host_poller([this] {
            while(!is_kernel_finished_.load()) {
                proxied_backend_.PollAndSyncHostReturnIfArgSetOnDevice(detail::functor_test_env<functor_t>->ProxyStream());
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });

        launch_test_kernel_nocreate(
            detail::functor_test_env<functor_t>->GetFunctorPtr(),
            detail::functor_test_env<functor_t>->GetInputPtr(),
            detail::functor_test_env<functor_t>->GetReturnPtr(),
            detail::functor_test_env<functor_t>->TestDriverStream()
        );

        util::cuda_check("FunctorTest::RunOnDevice stream sync", cudaStreamSynchronize, detail::functor_test_env<functor_t>->TestDriverStream());
        is_kernel_finished_.store(true);

        host_poller.join();
        return detail::functor_test_env<functor_t>->GetReturnValue();
    }

protected:
    std::atomic<bool> is_kernel_finished_ = false;
    mapped_proxy_t proxied_backend_ = {};
};
}
