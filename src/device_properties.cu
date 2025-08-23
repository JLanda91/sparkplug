#include <sparkplug/util/device_properties.cuh>
#include <sparkplug/util/cuda_check.cuh>

namespace sparkplug::util {

    const cudaDeviceProp& get_device_properties(){
        static const cudaDeviceProp device_properties = [] {
            cudaDeviceProp result{};
            cuda_check("Could not obtain Device 0 properties", cudaGetDeviceProperties, &result, 0);
            return result;
        }();
        return device_properties;
    }

}