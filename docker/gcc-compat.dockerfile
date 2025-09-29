ARG CUDA_VERSION=12.8.1

# =========================
# Stage 1: Build
# =========================
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++-10 \
    g++-11 \
    g++-12 \
    g++-13  \
    g++-14 \
    cmake \
    git \
    ninja-build \
    curl \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set environment
ENV VCPKG_ROOT=/opt/vcpkg
ENV PATH="$VCPKG_ROOT:$PATH"
ENV NVCOMPILERS=/usr/local/cuda

RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 10 \
 && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11 \
 && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 12 \
 && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 13 \
 && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 14

# Copy sources
WORKDIR /sparkplug
COPY . .

COPY --from=sparkplug-vcpkg /opt/vcpkg /opt/vcpkg

RUN mkdir /sparkplug-compat

RUN for v in 10 11 12 13 14; do \
      echo "=== Building with GCC $v ==="; \
      { update-alternatives --set g++ /usr/bin/g++-"$v" && cmake --preset=gcc-release && cmake --build ./build/gcc-release --target=all_examples; } > /sparkplug-compat/gcc-"$v"-build.log 2>&1 \
      && echo "✅ GCC $v build succeeded" >> /sparkplug-compat/report.txt \
      || echo "❌ GCC $v build failed" >> /sparkplug-compat/report.txt; \
    done

# =========================
# Stage 2: Runtime
# =========================
FROM busybox:uclibc

WORKDIR /sparkplug-compat

COPY --from=build /sparkplug-compat .
