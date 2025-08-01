#include <fmt/format.h>

#include "../include/sparkplug/testing/cuda_error.cuh"

namespace sparkplug::testing {
    cuda_error::cuda_error(cudaError_t err, std::string_view msg)
        : std::runtime_error(fmt::format("{}: {} (Error code: {})", msg, cudaGetErrorString(err), static_cast<std::underlying_type_t<cudaError_t>>(err)))
        , code(err) {}
}
