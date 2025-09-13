// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa


#include <sparkplug/util/cuda/get_device_properties.cuh>
#include <sparkplug/util/cuda/check_cuda_fn_error.cuh>

namespace sparkplug::util::cuda {

    const cudaDeviceProp& get_device_properties(){
        static const cudaDeviceProp device_properties = [] {
            cudaDeviceProp result{};
            check_cuda_fn_error("Could not obtain Device 0 properties", cudaGetDeviceProperties, &result, 0);
            return result;
        }();
        return device_properties;
    }

}