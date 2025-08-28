// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa


#pragma once

template<typename Backend>
class Factorial {
    const Backend* recurse_backend_ = nullptr;

public:
    Factorial() = default;

    explicit Factorial(const Backend* backend) : recurse_backend_(backend) {}

    __device__ int operator()(int x) const {
        if (x < 0) {
            return 0;
        }
        if (x <= 1) {
            return 1;
        }
        return x * (*recurse_backend_)(x-1);
    }
};
