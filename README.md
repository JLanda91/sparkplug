# Sparkplug
**Sparkplug** is a testing and mocking framework for CUDA device functors.

If you like this project, please consider [buying me a coffee](https://buymeacoffee.com/jasperlanda).

## Motivation
It isn't at all trivial to perform tests on device annotated functions and classes to the same extent 
with which testing is done on host-side functions and classes. Sparkplug aims to fill this gap. Being built on top of
GoogleTest, Sparkplug provides means of testing device functors by means of a unified API that allows 
dependencies on both host and device side. This allows for testing device functors with, for example, Google mocks on 
host as a dependency. 
Please see the [examples](#examples) below.

## Features
- Implemented in standard C++20.
- Built on top of GoogleTest and CUDA.
- Provides templates for specifying host/device dependency types.
- Provides basic dependency building blocks: host/device stub functors and google mock functors, and aliases to their
  NiceMock, NaggyMock and StrictMock derived classes.
- Provides a base test fixture template `sparkplug::testing::FunctorTest` to test functors with the specified 
  dependencies injected by means of static polymorphism (a.k.a. functors must be templated on their dependency types). 
  This template abstracts away the creation of the functor, argument and return type on the device, as well as bridging 
  calls from the functor to its dependencies.
- Provides utilities to check CUDA function call return values on errors, RAII CUDA streams, get device properties 
  as a singleton etc.

## Getting started

### Prerequisites
To build and install Sparkplug and to try the examples, the following is required to be installed:
- Have `vcpkg` installed.
- Have the CUDA Toolkit or the HPC SDK installed as `nvcc` is required.
- Have a version of `g++` installed with C++20 support.

The project has a `CMakePresets.json` with `gcc-(debug|release)` presets. They are marked as hidden as a user is
encouraged to create a `CMakeUserPresets.json`, inherit from these, and to set/override:
- `NVCOMPILERS` to specify the installation path of the CUDA Toolkit or the HPC SDK
- `CMAKE_CUDA_ARCHITECTURES`: to specify the GPU architectures.
- `VCPKG_MANIFEST_INSTALL`: to specify whether all dependencies are installed with every CMake reload. Should be `ON`
  the first time.
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
      "inherits": ["gcc-debug", "my-env"],
      "displayName": "GCC 14 Debug"
    },
    {
      "name": "gcc-14-release",
      "inherits": ["gcc-release", "my-env"],
      "displayName": "GCC 14 Release"
    }
  ]
}
```

### Building & Installing
Build and install Sparkplug and the example test programs with the user preset from the previous step as follows:

```shell
cmake --preset=<preset>
cmake --build ./build/<preset> --target=sparkplug
sudo cmake --install ./build/<preset>
```

## Examples

Below, some use-cases of Sparkplug are exhibited. Please refer to the `./examples` directory for more use-cases.
After building, an example test program can be run with:

```shell
./build/<preset>/examples/example_<example_name>
```

### Using host-side dependencies to white-box test device functors
Real power lies in using Google mocks (restricted to host) as injected dependency. Sparkplug's 
`sparkplug::testing::FunctorTest` test fixture template abstracts away the means of routing a device functor 
dependency call to a host dependency. For example, a toy example where two dependencies are called conditionally:

``` c++
template<typename Dep1, typename Dep2>
struct MultiDependencyFunctor {
    const Dep1* dep1_ = nullptr;
    const Dep2* dep2_ = nullptr;

    __device__ unsigned long operator()(unsigned n) const {
        if (n == 0) return 1234u;

        unsigned long result = 1;
        if (n < 7) result += (*dep1_)(n);
        if (n > 3) result += (*dep2_)(n);
        return result;
    }
};
```

can easily be white-box tested in a test file using Sparkplug:

```c++
namespace {
using dependency1_t = sparkplug::di::host_strict_gmock_functor_dependency<sparkplug::util::Signature<unsigned long, unsigned>>;
using dependency2_t = sparkplug::di::host_strict_gmock_functor_dependency<sparkplug::util::Signature<unsigned long, unsigned>>;

class MultiDepTestWithStrictMocks : public sparkplug::testing::FunctorTest<MultiDependencyFunctor, dependency1_t, dependency2_t> {
public:
    void SetUp() override {
        InjectDependencies(&mock1_, &mock2_);
    }

protected:
    dependency1_t::type mock1_{};
    dependency2_t::type mock2_{};
};
}

TEST_F(MultiDepTestWithStrictMocks, arg_zero) {
    EXPECT_CALL(mock1_, Call(_)).Times(0);
    EXPECT_CALL(mock2_, Call(_)).Times(0);

    ConstructArgumentOnDevice(0);
    ASSERT_THAT(RunOnDevice(), Eq(1234u));
}

TEST_F(MultiDepTestWithStrictMocks, only_mock1) {
    EXPECT_CALL(mock1_, Call(3)).WillOnce(Return(321ul));
    EXPECT_CALL(mock2_, Call(_)).Times(0);

    ConstructArgumentOnDevice(3);
    ASSERT_THAT(RunOnDevice(), Eq(322ul));
}

TEST_F(MultiDepTestWithStrictMocks, both_mocks) {
    EXPECT_CALL(mock1_, Call(5)).WillOnce(Return(321ul));
    EXPECT_CALL(mock2_, Call(5)).WillOnce(Return(9));

    ConstructArgumentOnDevice(5);
    ASSERT_THAT(RunOnDevice(), Eq(331ul));
}

TEST_F(MultiDepTestWithStrictMocks, only_mock2) {
    EXPECT_CALL(mock1_, Call(_)).Times(0);
    EXPECT_CALL(mock2_, Call(11)).WillOnce(Return(9));

    ConstructArgumentOnDevice(11);
    ASSERT_THAT(RunOnDevice(), Eq(10ul));
}
```

### Setting up a simple device stub
The same functor can also be tested with a mix of host and device dependencies, for example a StrictMock and a stub 
on device.

``` c++
namespace {
using dependency1_t = sparkplug::di::host_strict_gmock_functor_dependency<sparkplug::util::Signature<unsigned long, unsigned>>;
using dependency2_t = sparkplug::di::device_stub_functor_dependency<sparkplug::util::Signature<unsigned long, unsigned>>;

class MultiDepTestWithStrictMockAndDeviceStub : public sparkplug::testing::FunctorTest<MultiDependencyFunctor, dependency1_t, dependency2_t> {
public:
    void SetUp() override {
        InjectDependencies(&mock_, &stub_);
    }

protected:
    dependency1_t::type mock_{};
    dependency2_t::type stub_{9};
};
}

// Similar tests

```

### Bringing your own dependencies
The example above demonstrates the out-of-the-box mocks and stubs that Sparkplug offers. It also allows for users to 
inject their own dependencies. Use the generic dependency templates

```c++
sparkplug::di::host_dependency<T>
sparkplug::di::device_dependency<T>
```

and use them as template parameters for `sparkplug::testing::FunctorTest`. This allows to set up a functor with, for 
example, its production dependencies and do black-box testing. The constraints on dependencies for them to be 
compatible are described below

## Constraints
Sparkplug only supports the testing of gpu functors (aka classes with a `__device__` annotated call 
operator). Currently, dependencies must also be (stateless) functors.

All tests are run on the default device (device 0).

The `sparkplug::testing::FunctorTest` template only supports functors under test with the following 
requirements:
- They must have dependencies injected with static polymorphism (aka functor templates). This was chosen to prevent 
  users from having to sacrifice on performance by forcing dynamic polymorphism (aka dependencies as pointer to base 
  and vtable indirections), as gpu functor performance is often of vital importance.
- The call operator of the functor and the call operators of the dependencies must have a single parameter. Multiple 
  data types can be combined into an aggregate POD for compatibility.

As of now, only base CMake presets for `nvcc` and `gcc` flags are provided. More  compatibility will be 
tested in the future, after which presets with other host compiler flags will be added.

## Compatability
CUDA 12 is the first version with solid C++20 support. Please use a CUDA 12+ compiler.

Tested with GCC 14 and CUDA 12.8.