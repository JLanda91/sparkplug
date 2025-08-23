#pragma once

#include <thrust/host_vector.h>
#include <thrust/system/cuda/memory_resource.h>
#include <thrust/mr/allocator.h>

namespace sparkplug::util {
    template<typename T>
    using pinned_host_vector = thrust::host_vector<T, thrust::mr::stateless_resource_allocator<T, thrust::system::cuda::universal_host_pinned_memory_resource>>;
}