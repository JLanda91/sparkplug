# Sparkplug
**Sparkplug** is a testing and mocking framework for CUDA device functors.

If you like this project, please consider [buying me a coffee](https://buymeacoffee.com/jasperlanda).

## Motivation
It isn't at all trivial to perform tests on device annotated functions and classes to the same extent 
with which testing is done on host-side functions and classes. Sparkplug aims to fill this gap. Being built on top of
GoogleTest, Sparkplug provides means of testing device functors by means of a unified API that injects 
dependencies from host or device. This allows for testing device functors with, for example, injected Google mocks. 
Please see the [examples](#examples) below.

## Features
- Implemented in standard C++17.
- Built on top of GoogleTest and CUDA.
- Provides a base test fixture template to test (stateless) functors with a `__device__` annotated call operator 
  where dependencies are injected by means of static polymorphism.
- Provides templates to deduce call operator signatures, launching 1-thread test kernels, specifying host/device 
  backends and their object types.
- Provides basic dependency blocks: host/device stubs and google mocks with call operators, and aliases to their 
  NiceMock, NaggyMock and StrictMock derived classes.
- Provides utilities to get device properties, check CUDA function call return values on errors, RAII CUDA streams etc.

## Getting started

### Prerequisites
To build and install Sparkplug and to try the examples, the following is required to be installed:
- Have `vcpkg` installed.
- Have the CUDA Toolkit or the HPC SDK installed as `nvcc` is required.
- Have a version of `g++` installed with C++17 support.

The project has a `CMakePresets.json` with `gcc-(debug|release)` presets. They are marked as hidden as a user is
encouraged to create a `CMakeUserPresets.json`, inherit from these, and to set/override for example:
- `NVCOMPILERS` to specify the installation path of the CUDA Toolkit or the HPC SDK
- `CMAKE_CUDA_ARCHITECTURES`: to specify the GPU architectures.
- `VCPKG_MANIFEST_INSTALL`: to specify whether all dependencies are installed with every CMake reload.
- `VCPKG_ROOT`: to specify the installation path of `vcpkg`.

Example of a `CMakePresets.jon`:
```json
{
  "version": 3,
  "configurePresets": [
    {
      "name": "my-env",
      "hidden": true,
      "environment": {
        "VCPKG_ROOT": "/home/myname/vcpkg",
        "NVCOMPILERS": "/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/compilers"
      },
      "cacheVariables": {
        "CMAKE_CUDA_ARCHITECTURES": "86",
        "VCPKG_MANIFEST_INSTALL": "OFF"
      }
    },
    {
      "name": "gcc-14-debug",
      "inherits": ["gcc", "my-env"],
      "displayName": "GCC 14 Debug"
    },
    {
      "name": "gcc-14-release",
      "inherits": ["gcc", "my-env"],
      "displayName": "GCC 14 Release"
    }
  ]
}
```

### Building & Installing
Build and install Sparkplug with the user preset from the previous step as follows:

```shell
cmake --preset=<preset>
cmake --build --preset=<preset>
cmake --install --preset=<preset>
```

After building an example test program in the `./example` directory can be run with:

```shell
./build/<preset>/examples/example_<example_name>
```

## Examples

### Using host-side dependencies for device functors
Real power lies in using Google mocks (restricted to host) as injected dependency. For example, a toy example using a 
device 
factorial 
functor which depends on another functor for the recursion step:

``` c++
template<typename T>
class Factorial {
    const T* recurse_functor_ = nullptr;

public:
    Factorial() = default;
    explicit Factorial(const T* recurse_functor) : recurse_functor_(recurse_functor) {}

    __device__ int operator()(int x) const {
        if (x < 0) return 0;
        if (x <= 1) return 1;
        return x * (*recurse_functor_)(x-1);
    }
};
```

can easily be white-box tested in a test file using Sparkplug:

``` c++
using backend_t = SPARKPLUG_STRICT_MOCK_HOST_BACKEND(int, int);
using mock_t = backend_t::type;

namespace {
    struct FactorialTestWithStrictMock : sparkplug::testing::FunctorTest<Factorial, backend_t>{};
}

TEST_F(FactorialTestWithStrictMock, input_le_one) {
    mock_t mock{};
    PrepareBackend(&mock);

    EXPECT_CALL(mock, Call(_)).Times(0);

    ASSERT_THAT(RunOnDevice(-1), Eq(0));
    ASSERT_THAT(RunOnDevice(0), Eq(1));
    ASSERT_THAT(RunOnDevice(1), Eq(1));
}

TEST_F(FactorialTestWithStrictMock, input_gt_one) {
    mock_t mock{};
    PrepareBackend(&mock);

    EXPECT_CALL(mock, Call(4)).WillOnce(Return(24));

    ASSERT_THAT(RunOnDevice(5), Eq(120));
}
```

### Setting up a simple device stub
The same factorial functor could also use a stub on device as a dependency with the same test file layout using:

``` c++
using backend_t = SPARKPLUG_STUB_DEVICE_BACKEND(int, int);
using stub_t = backend_t::type;

namespace {
    struct FactorialTestWithDeviceStub : sparkplug::testing::FunctorTest<Factorial, backend_t>{};
}

TEST_F(FactorialTestWithDeviceStub, input_le_one) {
    stub_t stub{0};
    PrepareBackend(&stub);

    ASSERT_THAT(RunOnDevice(-1), Eq(0));
    ASSERT_THAT(RunOnDevice(0), Eq(1));
    ASSERT_THAT(RunOnDevice(1), Eq(1));
}

TEST_F(FactorialTestWithDeviceStub, input_gt_one) {
    stub_t stub{24};
    PrepareBackend(&stub);

    ASSERT_THAT(RunOnDevice(5), Eq(120));
}
```

## Constraints
Currently, Sparkplug only supports the testing of gpu functors (aka classes with a `__device__` annotated call 
operator).

The `sparkplug::testing::FunctorTest` template only supports functors under test with the following 
requirements:
- They must have one dependency. This will be extended to arbitrarily many.
- They must have a dependency injected with static polymorphism (aka functor templates). This was chosen to prevent 
  users from having to sacrifice on performance by forcing dynamic polymorphism (aka dependencies as pointer to base 
  and vtable indirections), as gpu functor performance is often of vital importance.
- Its call operator and the call operators of the dependency must have a single argument.

As of now, only a `nvcc + g++` CMakePresets exists. More compatibility will be tested in the future, after which presets 
with other host compilers will be added.

## Compatability
Tested with GCC 14 and CUDA 12.8.