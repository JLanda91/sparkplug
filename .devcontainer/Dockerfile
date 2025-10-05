FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS dev

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
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

COPY --from=sparkplug-vcpkg /opt/vcpkg /opt/vcpkg