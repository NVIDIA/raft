# Installation

RAFT currently provides libraries for C++ and Python. The C++ libraries, including the header-only and optional shared library, can be installed with Conda.

Both the C++ and Python APIs require CMake to build from source.

## Table of Contents

- [Install C++ and Python through Conda](#installing-c-and-python-through-conda)
- [Installing Python through Pip](#installing-python-through-pip)
- [Building C++ and Python from source](#building-c-and-python-from-source)
  - [CUDA/GPU requirements](#cudagpu-requirements)
  - [Build dependencies](#build-dependencies)
    - [Required](#required)
    - [Optional](#optional)
    - [Conda environment scripts](#conda-environment-scripts)
  - [Header-only C++](#header-only-c)
  - [C++ shared library](#c-shared-library-optional)
  - [ccache and sccache](#ccache-and-sccache)
  - [C++ tests](#c-tests)
  - [C++ primitives microbenchmarks](#c-primitives-microbenchmarks)
  - [Python libraries](#python-libraries)
- [Using CMake directly](#using-cmake-directly)
- [Build documentation](#build-documentation)
- [Using RAFT in downstream projects](#using-raft-c-in-downstream-projects)
  - [CMake targets](#cmake-targets)

------

## Installing C++ and Python through Conda

The easiest way to install RAFT is through conda and several packages are provided.
- `libraft-headers` C++ headers
- `pylibraft` (optional) Python library
- `raft-dask` (optional) Python library for deployment of multi-node multi-GPU algorithms that use the RAFT `raft::comms` abstraction layer in Dask clusters.

Use the following command, depending on your CUDA version, to install all of the RAFT packages with conda (replace `rapidsai` with `rapidsai-nightly` to install more up-to-date but less stable nightly packages). `mamba` is preferred over the `conda` command.

```bash
# CUDA 13
mamba install -c rapidsai -c conda-forge raft-dask pylibraft cuda-version=13.3

# CUDA 12
mamba install -c rapidsai -c conda-forge raft-dask pylibraft cuda-version=12.9
```

Note that the above commands will also install `libraft-headers` and `libraft`.

You can also install the conda packages individually using the `mamba` command above. For example, if you'd like to install RAFT's headers to use in your project:

```bash
# CUDA 13
mamba install -c rapidsai -c conda-forge libraft-headers cuda-version=13.3

# CUDA 12
mamba install -c rapidsai -c conda-forge libraft-headers cuda-version=12.9
```

## Installing Python through Pip

`pylibraft` and `raft-dask` both have packages that can be [installed through pip](https://rapids.ai/pip.html#install).

```bash
# CUDA 13
pip install pylibraft-cu13
pip install raft-dask-cu13

# CUDA 12
pip install pylibraft-cu12
pip install raft-dask-cu12
```

## Building C++ and Python from source

### CUDA/GPU Requirements
- cmake 3.26.4+
- GCC 9.3+ (9.5.0+ recommended)
- CUDA Toolkit 12.2+
- NVIDIA driver 450.80.02+
- Ampere architecture or better (compute capability >= 8.0)

### Build Dependencies

In addition to the libraries included with cudatoolkit 12.2+, there are some other dependencies below for building RAFT from source. Many of the dependencies are optional and depend only on the primitives being used. All of these can be installed with cmake or [rapids-cpm](https://github.com/rapidsai/rapids-cmake#cpm) and many of them can be installed with [conda](https://anaconda.org).

#### Required
- [RMM](https://github.com/rapidsai/rmm) corresponding to RAFT version.
- [Thrust](https://github.com/NVIDIA/thrust) v1.17 / [CUB](https://github.com/NVIDIA/cub)

#### Optional
- [NCCL](https://github.com/NVIDIA/nccl) - Used in `raft::comms` API and needed to build `raft-dask`.
- [UCX](https://github.com/openucx/ucx) - Used in `raft::comms` API and needed to build `raft-dask`.
- [Googletest](https://github.com/google/googletest) - Needed to build tests
- [Googlebench](https://github.com/google/benchmark) - Needed to build benchmarks
- [Doxygen](https://github.com/doxygen/doxygen) - Needed to build docs

#### Conda environment scripts

Conda environment scripts are provided for installing the necessary dependencies to build both the C++ and Python libraries from source. It is preferred to use `mamba`, as it provides significant speedup over `conda`:
```bash
mamba env create --name rapids_raft -f conda/environments/all_cuda-133_arch-$(uname -m).yaml
mamba activate rapids_raft
```

All of RAFT's C++ APIs can be used header-only.

The recommended way to build and install RAFT from source is to use the `build.sh` script in the root of the repository. This script can build both the C++ and Python artifacts and provides CMake options for building and installing the headers, tests, benchmarks, and the pre-compiled shared library.

### Header-only C++

`build.sh` uses [rapids-cmake](https://github.com/rapidsai/rapids-cmake), which will automatically download any dependencies which are not already installed.

The following example will download the needed dependencies and install the RAFT headers into `$INSTALL_PREFIX/include/raft`.
```bash
./build.sh libraft
```
The `-n` flag can be passed to just have the build download the needed dependencies. Since RAFT's C++ headers are primarily used during build-time in downstream projects, the dependencies will never be installed by the RAFT build.
```bash
./build.sh libraft -n
```

Once installed, `libraft` headers (and dependencies which were downloaded and installed using `rapids-cmake`) can be uninstalled also using `build.sh`:
```bash
./build.sh libraft --uninstall
```

### C++ Shared Library (optional)

A shared library must be built in order to build `pylibraft`. Pass the `--compile-lib` flag to `build.sh` to build the library:
```bash
./build.sh libraft --compile-lib
```

In above example the shared library is installed by default into `$INSTALL_PREFIX/lib`. To disable this, pass `-n` flag.

Once installed, the shared library, headers (and any dependencies downloaded and installed via `rapids-cmake`) can be uninstalled using `build.sh`:
```bash
./build.sh libraft --uninstall
```


### ccache and sccache

`ccache` and `sccache` can be used to better cache parts of the build when rebuilding frequently, such as when working on a new feature. You can also use `ccache` or `sccache` with `build.sh`:

```bash
./build.sh libraft --cache-tool=ccache
```

### C++ Tests

Compile the tests using the `tests` target in `build.sh`.

```bash
./build.sh libraft tests
```

The tests are broken apart by algorithm category, so you will find several binaries in `cpp/build/` named `*_TEST`.

For example, to run the distance tests:
```bash
./cpp/build/MATRIX_TEST
```

It can take sometime to compile all of the tests. You can build individual tests by providing a semicolon-separated list to the `--limit-tests` option in `build.sh`:

```bash
./build.sh libraft tests -n --limit-tests=CORE_TEST;MATRIX_TEST
```

### C++ Primitives Microbenchmarks

The benchmarks are broken apart by algorithm category, so you will find several binaries in `cpp/build/` named `*_PRIMS_BENCH`.
```bash
./build.sh libraft bench-prims
```

It can take sometime to compile all of the benchmarks. You can build individual benchmarks by providing a semicolon-separated list to the `--limit-bench-prims` option in `build.sh`:

```bash
./build.sh libraft bench-prims -n --limit-bench=MATRIX_PRIMS_BENCH;LINALG_PRIMS_BENCH
```

### Python libraries

The Python libraries can be built and installed using the `build.sh` script:

```bash
# to build pylibraft
./build.sh libraft pylibraft --compile-lib
# to build raft-dask (depends on pylibraft)
./build.sh libraft pylibraft raft-dask --compile-lib
```

`setup.py` can also be used to build the Python libraries manually:

```bash
cd python/raft-dask
python setup.py build_ext --inplace
python setup.py install

cd python/pylibraft
python setup.py build_ext --inplace
python setup.py install
```

Python tests are automatically installed with the corresponding libraries. To run Python tests:
```bash
cd python/raft-dask
py.test -s -v

cd python/pylibraft
py.test -s -v
```

The Python packages can also be uninstalled using the `build.sh` script:
```bash
./build.sh pylibraft raft-dask --uninstall
```

### Using CMake directly

When building RAFT from source, the `build.sh` script offers a nice wrapper around the `cmake` commands to ease the burdens of manually configuring the various available cmake options. When more fine-grained control over the CMake configuration is desired, the `cmake` command can be invoked directly as the below example demonstrates.

The `CMAKE_INSTALL_PREFIX` installs RAFT into a specific location. The example below installs RAFT into the current Conda environment:
```bash
cd cpp
mkdir build
cd build
cmake -D BUILD_TESTS=ON -DRAFT_COMPILE_LIBRARY=ON -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ../
make -j<parallel_level> install
```

RAFT's CMake has the following configurable flags available:

| Flag                            | Possible Values      | Default Value | Behavior                                                                     |
|---------------------------------|----------------------| --- |------------------------------------------------------------------------------|
| BUILD_TESTS                     | ON, OFF              | ON  | Compile Googletests                                                          |
| BUILD_PRIMS_BENCH               | ON, OFF              | OFF | Compile benchmarks                                                           |
| CUDA_ENABLE_KERNELINFO          | ON, OFF              | OFF | Enables `kernelinfo` in nvcc. This is useful for `compute-sanitizer`         |
| CUDA_ENABLE_LINEINFO            | ON, OFF              | OFF | Enable the -lineinfo option for nvcc                                         |
| CUDA_STATIC_MATH_LIBRARIES      | ON, OFF              | OFF | Statically link the CUDA math libraries                                      |
| DETECT_CONDA_ENV                | ON, OFF              | ON  | Enable detection of conda environment for dependencies                       |
| raft_FIND_COMPONENTS            | compiled distributed |     | Configures the optional components as a space-separated list                 |
| RAFT_COMPILE_LIBRARY            | ON, OFF              | ON if either BUILD_TESTS or BUILD_PRIMS_BENCH is ON; otherwise OFF | Compiles all `libraft` shared libraries (these are required for Googletests) |
| RAFT_ENABLE_CUBLAS_DEPENDENCY   | ON, OFF              | ON  | Link against cublas library in `raft::raft`                                  |
| RAFT_ENABLE_CUSOLVER_DEPENDENCY | ON, OFF              | ON  | Link against cusolver library in `raft::raft`                                |
| RAFT_ENABLE_CUSPARSE_DEPENDENCY | ON, OFF              | ON  | Link against cusparse library in `raft::raft`                                |
| RAFT_ENABLE_CUSOLVER_DEPENDENCY | ON, OFF              | ON  | Link against curand library in `raft::raft`                                  |
| RAFT_NVTX                       | ON, OFF              | OFF | Enable NVTX Markers                                                          |

### Build documentation

The documentation requires that the C++ and Python libraries have been built and installed. The following will build the docs along with the C++ and Python packages:

```
./build.sh libraft pylibraft raft-dask docs --compile-lib
```

## Using RAFT C++ in downstream projects

There are a few different strategies for including RAFT in downstream projects, depending on whether the [required build dependencies](#build-dependencies) have already been installed and are available on the `lib` and `include` search paths.

When using the GPU parts of RAFT, you will need to enable CUDA support in your CMake project declaration:
```cmake
project(YOUR_PROJECT VERSION 0.1 LANGUAGES CXX CUDA)
```

Note that some additional compiler flags might need to be added when building against RAFT. For example, if you see an error like this `The experimental flag '--expt-relaxed-constexpr' can be used to allow this.`. The necessary flags can be set with CMake:
```cmake
target_compile_options(your_target_name PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)
```

Further, it's important that the language level be set to at least C++ 20. This can be done with cmake:
```cmake
set_target_properties(your_target_name
PROPERTIES CXX_STANDARD                        20
           CXX_STANDARD_REQUIRED               ON
           CUDA_STANDARD                       20
           CUDA_STANDARD_REQUIRED              ON
           POSITION_INDEPENDENT_CODE           ON
           INTERFACE_POSITION_INDEPENDENT_CODE ON)
```

#### CMake Targets

The `raft::raft` CMake target is made available when including RAFT into your CMake project but additional CMake targets can be made available by adding to the `COMPONENTS` option in CMake's `find_package(raft)` (refer to [CMake docs](https://cmake.org/cmake/help/latest/command/find_package.html#basic-signature) to learn more). The components should be separated by spaces. The `raft::raft` target will always be available. Note that the `distributed` component also exports additional dependencies.

| Component   | Target              | Description                                              | Base Dependencies             |
|-------------|---------------------|----------------------------------------------------------|-------------------------------|
| n/a         | `raft::raft`        | Full RAFT header library                                 | CUDA toolkit, RMM, NVTX, CCCL |
| compiled    | `raft::compiled`    | Pre-compiled template instantiations and runtime library | raft::raft                    |
| distributed | `raft::distributed` | Dependencies for `raft::comms` APIs                      | raft::raft, UCX, NCCL         |
