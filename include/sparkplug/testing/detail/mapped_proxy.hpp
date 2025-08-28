//  SPDX-License-Identifier: Apache-2.0
//
//  Copyright 2025 Jasper Landa
//

#pragma once

#include "host_backend_proxy.cuh"
#include "device_backend_proxy.cuh"


namespace sparkplug::testing::detail {
    template<typename Backend>
    using mapped_proxy_t = std::conditional_t<Backend::is_device_side, DeviceBackendProxy<typename Backend::type>, HostBackendProxy<typename Backend::type>>;
}