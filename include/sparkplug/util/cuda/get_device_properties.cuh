// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa


#pragma once

#include <cuda_runtime_api.h>

namespace sparkplug::util {

    const cudaDeviceProp& get_device_properties();
}
