ARG CUDA_VERSION=12.8.1

# =========================
# Stage 1: vcpkg
# =========================
FROM ubuntu:24.04 AS vcpkg

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    zip \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/microsoft/vcpkg.git /opt/vcpkg \
    && /opt/vcpkg/bootstrap-vcpkg.sh -disableMetrics

# =========================
# Stage 2: Build
# =========================
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++-10 gcc-10 \
    g++-11 gcc-11 \
    g++-12 gcc-12 \
    g++-13 gcc-13 \
    g++-14 gcc-14 \
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

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 10 \
 && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11 \
 && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12 \
 && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 13 \
 && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 14 \
 && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 10 \
 && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11 \
 && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 12 \
 && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 13 \
 && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 14

# Copy sources
WORKDIR /sparkplug
COPY examples ./examples/
COPY include ./include/
COPY src ./src/
COPY CMakeLists.txt CMakePresets.json vcpkg.json vcpkg-configuration.json ./
COPY CMakeUserPresetsDocker.json CMakeUserPresets.json

COPY --from=vcpkg /opt/vcpkg /opt/vcpkg

RUN for v in 10 11 12 13 14; do \
      echo "=== Building with GCC $v ==="; \
      mkdir -p ./build/gcc-$v; \
      { cmake --preset=gcc-$v && cmake --build ./build/gcc-$v --target=all_examples; } > build/gcc-$v/build.log 2>&1 \
      && echo "✅ GCC $v build succeeded" >> /sparkplug/compatibility.txt \
      || echo "❌ GCC $v build failed" >> /sparkplug/compatibility.txt; \
    done

# =========================
# Stage 3: Runtime
# =========================
FROM busybox:uclibc

WORKDIR /sparkplug-compat

COPY --from=build /sparkplug/compatibility.txt .
COPY --from=build /sparkplug/build/gcc-10/build.log ./gcc-10-build.log
COPY --from=build /sparkplug/build/gcc-11/build.log ./gcc-11-build.log
COPY --from=build /sparkplug/build/gcc-12/build.log ./gcc-12-build.log
COPY --from=build /sparkplug/build/gcc-13/build.log ./gcc-13-build.log
COPY --from=build /sparkplug/build/gcc-14/build.log ./gcc-14-build.log
