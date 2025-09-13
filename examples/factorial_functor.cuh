// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa


#pragma once

template<typename Dependency>
struct Factorial {
    const Dependency* dep_ = nullptr;

    __device__ int operator()(int x) const {
        if (x < 0) return 0;
        if (x <= 1) return 1;
        return x * (*dep_)(x-1);
    }
};
