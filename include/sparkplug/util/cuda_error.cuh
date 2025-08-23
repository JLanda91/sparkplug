#pragma once

#include <stdexcept>
#include <string_view>

namespace sparkplug::util {
    struct cuda_error : std::runtime_error {
        cudaError_t code;

        explicit cuda_error(cudaError_t err, std::string_view msg = "CUDA Error");
    };
}