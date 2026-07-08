# Developer Guide

## General
Please start by reading the [Contributor Guide](contributing.md).

## Performance
1. In performance critical sections of the code, favor `cudaDeviceGetAttribute` over `cudaDeviceGetProperties`. See corresponding CUDA devblog [here](https://devblogs.nvidia.com/cuda-pro-tip-the-fast-way-to-query-device-properties/) to know more.
2. If an algo requires you to launch GPU work in multiple cuda streams, do not create multiple `raft::resources` objects, one for each such work stream. Instead, use the stream pool configured on the given `raft::resources` instance's `raft::resources::get_stream_from_stream_pool()` to pick up the right cuda stream. Refer to the section on [CUDA Resources](#resource-management) and the section on [Threading](#threading-model) for more details. TIP: use `raft::resources::get_stream_pool_size()` to know how many such streams are available at your disposal.


## Local Development

Developing features and fixing bugs for the RAFT library itself is straightforward and only requires building and installing the relevant RAFT artifacts.

The process for working on a CUDA/C++ feature which might span RAFT and one or more consuming libraries can vary slightly depending on whether the consuming project relies on a source build (as outlined in the [BUILD](BUILD.md#install_header_only_cpp) docs). In such a case, the option `CPM_raft_SOURCE=/path/to/raft/source` can be passed to the cmake of the consuming project in order to build the local RAFT from source. The PR with relevant changes to the consuming project can also pin the RAFT version temporarily by explicitly changing the `FORK` and `PINNED_TAG` arguments to the RAFT branch containing their changes when invoking `find_and_configure_raft`.  The pin should be reverted after the changed is merged to the RAFT project and before it is merged to the dependent project(s) downstream.

If building a feature which spans projects and not using the source build in cmake, the RAFT changes (both C++ and Python) will need to be installed into the environment of the consuming project before they can be used. The ideal integration of RAFT into consuming projects will enable both the source build in the consuming project only for this case but also rely on a more stable packaging (such as conda packaging) otherwise.


## Threading Model

With the exception of the `raft::resources`, RAFT algorithms should maintain thread-safety and are, in general,
assumed to be single threaded. This means they should be able to be called from multiple host threads so
long as different instances of `raft::resources` are used.

Exceptions are made for algorithms that can take advantage of multiple CUDA streams within multiple host threads
in order to oversubscribe or increase occupancy on a single GPU. In these cases, the use of multiple host
threads within RAFT algorithms should be used only to maintain concurrency of the underlying CUDA streams.
Multiple host threads should be used sparingly, be bounded, and should steer clear of performing CPU-intensive
computations.

A good example of an acceptable use of host threads within a RAFT algorithm might look like the following

```cpp
#include <raft/core/resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
raft::resources res;

...

sync_stream(res);

...

int n_streams = get_stream_pool_size(res);

#pragma omp parallel for num_threads(n_streams)
for(int i = 0; i < n; i++) {
    int thread_num    = omp_get_thread_num() % n_streams;
    auto s            = get_stream_from_stream_pool(res, thread_num);  // rmm::cuda_stream_view
    ... possible light cpu pre-processing ...
    my_kernel1<<<b, tpb, 0, s>>>(...);
    ...
    ... some possible async d2h / h2d copies ...
    my_kernel2<<<b, tpb, 0, s>>>(...);
    ...
    sync_stream(res, s);
    ... possible light cpu post-processing ...
}
```

In the example above, if there is no CPU pre-processing at the beginning of the for-loop, an event can be registered in
each of the streams within the for-loop to make them wait on the stream from the handle. If there is no CPU post-processing
at the end of each for-loop iteration, `sync_stream(res, s)` can be replaced with a single `sync_stream_pool(res)`
after the for-loop.

To avoid compatibility issues between different threading models, the only threading programming allowed in RAFT is OpenMP.
Though RAFT's build enables OpenMP by default, RAFT algorithms should still function properly even when OpenMP has been
disabled. If the CPU pre- and post-processing were not needed in the example above, OpenMP would not be needed.

The use of threads in third-party libraries is allowed, though they should still avoid depending on a specific OpenMP runtime.

## Public Interface

### General guidelines
Functions exposed via the C++ API must be stateless. Things that are OK to be exposed on the interface:
1. Any [POD](https://en.wikipedia.org/wiki/Passive_data_structure) type. Since `std::is_pod` is deprecated as of C++20, check for [std::is_standard_layout](https://en.cppreference.com/w/cpp/types/is_standard_layout) together with [std::is_trivial](https://en.cppreference.com/w/cpp/types/is_trivial) as a reference for POD types.
2. `raft::resources` - since it stores resource-related state which has nothing to do with model/algo state.
3. Avoid using pointers to POD types (explicitly putting it out, even though it can be considered as a POD) and pass the structures by reference instead.
   Internal to the C++ API, these stateless functions are free to use their own temporary classes, as long as they are not exposed on the interface.
4. Accept single- (`raft::span`) and multi-dimensional views (`raft::mdspan`) and validate their metadata wherever possible.
5. Prefer `std::optional` for any optional arguments (e.g. do not accept `nullptr`)
6. All public APIs should be lightweight wrappers around calls to private APIs inside the `detail` namespace.

### API stability

Since RAFT is a core library with multiple consumers, it's important that the public APIs maintain stability across versions and any changes to them are done with caution, adding new functions and deprecating the old functions over a couple releases as necessary.

### Stateless C++ APIs

Using the IVF-PQ algorithm as an example, the following way of exposing its API would be wrong according to the guidelines in this section, since it exposes a non-POD C++ class object in the C++ API:
```cpp
template <typename value_t, typename idx_t>
class ivf_pq {
  ivf_pq_params params_;
  raft::resources const& res_;

public:
  ivf_pq(raft::resources const& res);
  void train(raft::device_matrix_view<const value_t, idx_t, raft::row_major> dataset);
  void search(raft::device_matrix_view<const value_t, idx_t, raft::row_major> queries,
              raft::device_matrix_view<idx_t, idx_t, raft::row_major> out_inds,
              raft::device_matrix_view<value_t, idx_t, raft::row_major> out_dists);
};
```

An alternative correct way to expose this could be:
```cpp
namespace raft::ivf_pq {

template<typename value_t, typename idx_t>
void ivf_pq_train(raft::resources const& res, const raft::ivf_pq_params& params, raft::ivf_pq_index& index,
raft::device_matrix_view<const value_t, idx_t, raft::row_major> dataset);

template<typename value_t, typename idx_t>
void ivf_pq_search(raft::resources const& res, raft::ivf_pq_params const& params, raft::ivf_pq_index const& index,
raft::device_matrix_view<const value_t, idx_t, raft::row_major> queries,
raft::device_matrix_view<idx_t, idx_t, raft::row_major> out_inds,
raft::device_matrix_view<value_t, idx_t, raft::row_major> out_dists);
}
```

### Other functions on state

These guidelines also mean that it is the responsibility of C++ API to expose methods to load and store (aka marshalling) such a data structure. Further continuing the IVF-PQ example,  the following methods could achieve this:
```cpp
namespace raft::ivf_pq {
   void save(raft::resources const& res, raft::ivf_pq_index const& model, std::ostream& os);
   void load(raft::resources const& res, raft::ivf_pq_index& model, std::istream& is);
}
```


## Coding style

### Code Formatting

#### Using pre-commit hooks

RAFT uses [pre-commit](https://pre-commit.com/) to execute all code linters and formatters. These
tools ensure a consistent code format throughout the project. Using pre-commit ensures that linter
versions and options are aligned for all developers. Additionally, there is a CI check in place to
enforce that committed code follows our standards.

To use `pre-commit`, install via `conda` or `pip`:

```bash
conda install -c conda-forge pre-commit
```

```bash
pip install pre-commit
```

Then run pre-commit hooks before committing code:

```bash
pre-commit run
```

By default, pre-commit runs on staged files (only changes and additions that will be committed).
To run pre-commit checks on all files, execute:

```bash
pre-commit run --all-files
```

Optionally, you may set up the pre-commit hooks to run automatically when you make a git commit. This can be done by running:

```bash
pre-commit install
```

Now code linters and formatters will be run each time you commit changes.

You can skip these checks with `git commit --no-verify` or with the short version `git commit -n`.

#### Summary of pre-commit hooks

The following section describes some of the core pre-commit hooks used by the repository.
See `.pre-commit-config.yaml` for a full list.

C++/CUDA is formatted with [`clang-format`](https://clang.llvm.org/docs/ClangFormat.html).

RAFT relies on `clang-format` to enforce code style across all C++ and CUDA source code. The coding style is based on the [Google style guide](https://google.github.io/styleguide/cppguide.html#Formatting). The only digressions from this style are the following.
1. Do not split empty functions/records/namespaces.
2. Two-space indentation everywhere, including the line continuations.
3. Disable reflowing of comments.
   The reasons behind these deviations from the Google style guide are given in comments [here](https://github.com/rapidsai/raft/blob/main/cpp/.clang-format).

[`doxygen`](https://doxygen.nl/) is used as documentation generator and also as a documentation linter.
In order to run doxygen as a linter on C++/CUDA code, run

```bash
./ci/checks/doxygen.sh
```

Python code runs several linters including [Black](https://black.readthedocs.io/en/stable/),
[isort](https://pycqa.github.io/isort/), and [flake8](https://flake8.pycqa.org/en/latest/).

RAFT also uses [codespell](https://github.com/codespell-project/codespell) to find spelling
mistakes, and this check is run as a pre-commit hook. To apply the suggested spelling fixes,
you can run  `codespell -i 3 -w .` from the repository root directory.
This will bring up an interactive prompt to select which spelling fixes to apply.

### #include style
[include_checker.py](https://github.com/rapidsai/raft/blob/main/cpp/scripts/include_checker.py) is used to enforce the include style as follows:
1. `#include "..."` should be used for referencing local files only. It is acceptable to be used for referencing files in a sub-folder/parent-folder of the same algorithm, but should never be used to include files in other algorithms or between algorithms and the primitives or other dependencies.
2. `#include <...>` should be used for referencing everything else

Manually, run the following to bulk-fix include style issues:
```bash
python ./cpp/scripts/include_checker.py --inplace [cpp/include cpp/tests ... list of folders which you want to fix]
```

### Copyright header
RAPIDS [pre-commit-hooks](https://github.com/rapidsai/pre-commit-hooks) checks the Copyright
header for all git-modified files.

Manually, you can run the following to bulk-fix the header on all files in the repository:
```bash
pre-commit run -a verify-copyright
```
Keep in mind that this only applies to files tracked by git that have been modified.

## Error handling
Call CUDA APIs via the provided helper macros `RAFT_CUDA_TRY`, `RAFT_CUBLAS_TRY` and `RAFT_CUSOLVER_TRY`. These macros take care of checking the return values of the used API calls and generate an exception when the command is not successful. If you need to avoid an exception, e.g. inside a destructor, use `RAFT_CUDA_TRY_NO_THROW`, `RAFT_CUBLAS_TRY_NO_THROW ` and `RAFT_CUSOLVER_TRY_NO_THROW`. These macros log the error but do not throw an exception.

## Logging

### Introduction
Anything and everything about logging is defined inside [logger.hpp](https://github.com/rapidsai/raft/blob/main/cpp/include/raft/core/logger.hpp). RAFT logging is built on top of [rapids-logger](https://github.com/rapidsai/rapids-logger) (which wraps [spdlog](https://github.com/gabime/spdlog) underneath), but this information is transparent to all. The global logger is obtained via `raft::default_logger()`.

### Usage
```cpp
#include <raft/core/logger.hpp>

// Inside your method or function, use any of these macros
RAFT_LOG_TRACE("Hello %s!", "world");
RAFT_LOG_DEBUG("Hello %s!", "world");
RAFT_LOG_INFO("Hello %s!", "world");
RAFT_LOG_WARN("Hello %s!", "world");
RAFT_LOG_ERROR("Hello %s!", "world");
RAFT_LOG_CRITICAL("Hello %s!", "world");
```

### Changing logging level
There are 7 logging levels with each successive level becoming quieter:
1. RAFT_LEVEL_TRACE
2. RAFT_LEVEL_DEBUG
3. RAFT_LEVEL_INFO
4. RAFT_LEVEL_WARN
5. RAFT_LEVEL_ERROR
6. RAFT_LEVEL_CRITICAL
7. RAFT_LEVEL_OFF
   Pass one of these as per your needs into the `set_level()` method as follows:
```cpp
raft::default_logger().set_level(RAFT_LEVEL_WARN);
// From now onwards, this will print only WARN and above kind of messages
```

### Changing logging pattern
Pass the [format string](https://github.com/gabime/spdlog/wiki/3.-Custom-formatting) as follows in order use a different logging pattern than the default.
```cpp
raft::default_logger().set_pattern(YourFavoriteFormat);
```
One can also use the corresponding `get_pattern()` method to know the current format as well.

### Temporarily changing the logging pattern
Sometimes, we need to temporarily change the log pattern (eg: for reporting decision tree structure). This can be achieved in a RAII-like approach as follows:
```cpp
{
  PatternSetter _(MyNewTempFormat);
  // new log format is in effect from here onwards
  doStuff();
  // once the above temporary object goes out-of-scope, the old format will be restored
}
```

### Tips
* Do NOT end your logging messages with a newline! It is automatically added by the logger.
* The `RAFT_LOG_TRACE()` is by default not compiled due to the `RAFT_LOG_ACTIVE_LEVEL` macro setup, for performance reasons. If you need it to be enabled, change this macro accordingly during compilation time

## Common Design Considerations

1. Use the `hpp` extension for files which can be compiled with `gcc` against the CUDA-runtime. Use the `cuh` extension for files which require `nvcc` to be compiled. `hpp` can also be used for functions marked `__host__ __device__` only if proper checks are in place to remove the `__device__` designation when not compiling with `nvcc`.

2. When additional classes, structs, or general POCO types are needed to be used for representing data in the public API, place them in a new file called `<primitive_name>_types.hpp`. This tells users they are safe to expose these types on their own public APIs without bringing in device code. At a minimum, the definitions for these types, at least, should not require `nvcc`. In general, these classes should only store very simple state and should not perform their own computations. Instead, new functions should be exposed on the public API which accept these objects, reading or updating their state as necessary.

3. Documentation for public APIs should be well documented, easy to use, and it is highly preferred that they include usage instructions.

4. Before creating a new primitive, check to see if one exists already. If one exists but the API isn't flexible enough to include your use-case, consider first refactoring the existing primitive. If that is not possible without an extreme number of changes, consider how the public API could be made more flexible. If the new primitive is different enough from all existing primitives, consider whether an existing public API could invoke the new primitive as an option or argument. If the new primitive is different enough from what exists already, add a header for the new public API function to the appropriate subdirectory and namespace.

## Preferred APIs and idioms

When writing new code, re-using functionality, or reviewing changes, prefer:

1. **`raft::resources` over raw handles/streams.** Prefer function overloads that
   take `const raft::resources&` over overloads that take a raw `cudaStream_t` and/or
   library handles (cuBLAS, cuSOLVER, etc.). See [Resource Management](#resource-management)
   and [Asynchronous operations and stream ordering](#asynchronous-operations-and-stream-ordering).

2. **Public API over `detail`.** When re-using functionality or writing tests, call the
   public API rather than functions in the `detail` namespace, unless you are implementing
   that public API. Public APIs are thin wrappers over `detail` (see
   [General guidelines](#general-guidelines)), so tests stay aligned with what users get.

3. **mdarray over raw containers.** Allocate owning data with `raft` mdarray types
   (`raft::device_mdarray`, `raft::host_mdarray`, etc.) instead of `rmm::device_uvector`
   or `std::vector` when representing device/multi-dimensional data, and pass views
   (`raft::mdspan`) across APIs.

4. **Avoid deprecated functions.** Use any reasonable non-deprecated substitute over a
   deprecated function; when adding a replacement, deprecate the old one per
   [API stability](#api-stability).

## Header organization of expensive function templates

RAFT is a heavily templated library. Several core functions are expensive to compile and we want to prevent duplicate compilation of this functionality. To limit build time, RAFT provides a precompiled library (libraft.so) where expensive function templates are instantiated for the most commonly used template parameters. To prevent (1) accidental instantiation of these templates and (2) unnecessary dependency on the internals of these templates, we use a split header structure and define macros to control template instantiation. This section describes the macros and header structure.

**Macros.** We define the macros `RAFT_COMPILED` and `RAFT_EXPLICIT_INSTANTIATE_ONLY`. The `RAFT_COMPILED` macro is defined by `CMake` when compiling code that (1) is part of `libraft.so` or (2) is linked with `libraft.so`. It indicates that a precompiled `libraft.so` is present at runtime.

The `RAFT_EXPLICIT_INSTANTIATE_ONLY` macro is defined by `CMake` during compilation of `libraft.so` itself. When defined, it indicates that implicit instantiations of expensive function templates are forbidden (they result in a compiler error). In the RAFT project, we additionally define this macro during compilation of the tests and benchmarks.

Below, we summarize which combinations of `RAFT_COMPILED` and `RAFT_EXPLICIT_INSTANTIATE_ONLY` are used in practice and what the effect of the combination is.

| RAFT_COMPILED | RAFT_EXPLICIT_INSTANTIATE_ONLY | Which targets                                                                                        |
|---------------|--------------------------------|------------------------------------------------------------------------------------------------------|
| defined       | defined                        | `raft::compiled`, RAFT tests, RAFT benchmarks                                                        |
| defined       |                                | Downstream libraries depending on `libraft` like cuML, cuGraph.                                      |
|               |                                | Downstream libraries depending on `libraft-headers` like cugraph-ops.                                |


| RAFT_COMPILED | RAFT_EXPLICIT_INSTANTIATE_ONLY | Effect                                                                                                |
|---------------|--------------------------------|-------------------------------------------------------------------------------------------------------|
| defined       | defined                        | Templates are precompiled. Compiler error on accidental instantiation of expensive function template. |
| defined       |                                | Templates are precompiled. Implicit instantiation allowed.                                            |
|               |                                | Nothing precompiled. Implicit instantiation allowed.                                                  |
|               | defined                        | Avoid this: nothing precompiled. Compiler error on any instantiation of expensive function template.  |



**Header organization.** Any header file that defines an expensive function template (say `expensive.cuh`) should be split in three parts: `expensive.cuh`, `expensive-inl.cuh`, and `expensive-ext.cuh`. The file `expensive-inl.cuh` ("inl" for "inline") contains the template definitions, i.e., the actual code. The file `expensive.cuh` includes one or both of the other two files, depending on the values of the `RAFT_COMPILED` and `RAFT_EXPLICIT_INSTANTIATE_ONLY` macros. The file `expensive-ext.cuh` contains `extern template` instantiations. In addition, if `RAFT_EXPLICIT_INSTANTIATE_ONLY` is set, it contains template definitions to ensure that a compiler error is raised in case of accidental instantiation.

The dispatching by `expensive.cuh` is performed as follows:
``` c++
#ifndef RAFT_EXPLICIT_INSTANTIATE_ONLY
// If implicit instantiation is allowed, include template definitions.
#include "expensive-inl.cuh"
#endif

#ifdef RAFT_COMPILED
// Include extern template instantiations when RAFT is compiled.
#include "expensive-ext.cuh"
#endif
```

The file `expensive-inl.cuh` is unchanged:
``` c++
namespace raft {
template <typename T>
void expensive(T arg) {
  // .. function body
}
} // namespace raft
```

The file `expensive-ext.cuh` contains the following:
``` c++
#include <raft/util/raft_explicit.cuh> // RAFT_EXPLICIT

#ifdef RAFT_EXPLICIT_INSTANTIATE_ONLY
namespace raft {
// (1) define templates to raise an error in case of accidental instantiation
template <typename T> void expensive(T arg) RAFT_EXPLICIT;
} // namespace raft
#endif //RAFT_EXPLICIT_INSTANTIATE_ONLY

// (2) Provide extern template instantiations.
extern template void raft::expensive<int>(int);
extern template void raft::expensive<float>(float);
```

This header has two responsibilities: (1) define templates to raise an error in case of accidental instantiation and (2) provide `extern template` instantiations.
First, if `RAFT_EXPLICIT_INSTANTIATE_ONLY` is set, `expensive` is defined. This is done for two reasons: (1) to give a definition, because the definition in `expensive-inl.cuh` was skipped and (2) to indicate that the template should be explicitly instantiated by taging it with the `RAFT_EXPLICIT` macro. This macro defines the function body, and it ensures that an informative error message is generated when an implicit instantiation erroneously occurs. Finally, the `extern template` instantiations are listed.

To actually generate the code for the template instances, the file `src/expensive.cu` contains the following. Note that the only difference between the extern template instantiations in `expensive-ext.cuh` and these lines are the removal of the word `extern`:

``` c++
#include <raft/expensive-inl.cuh>

template void raft::expensive<int>(int);
template void raft::expensive<float>(float);
```

**Design considerations**:

1. In the `-ext.cuh` header, do not include implementation headers. Only include function parameter types and types that are used to instantiate the templates. If a primitive takes custom parameter types, define them in a separate header called `<primitive_name>_types.hpp`. (see [Common Design Considerations](https://github.com/rapidsai/raft/blob/7b065aff81a0b1976e2a9e2f3de6690361a1111b/docs/source/developer_guide.md#common-design-considerations)).

2. Keep docstrings in the `-inl.cuh` header, as it is closer to the code. Remove docstrings from template definitions in the `-ext.cuh` header. Make sure to explicitly include public APIs in the RAFT API docs. That is, add `#include <raft/expensive.cuh>` to the docs in `docs/source/cpp_api/expensive.rst` (instead of `#include <raft/expensive-inl.cuh>`).

3. The order of inclusion in `expensive.cuh` is extremely important. If `RAFT_EXPLICIT_INSTANTIATE_ONLY` is not defined, but `RAFT_COMPILED` is defined, then we must include the template definitions before the `extern template` instantiations.

4. If a header file defines multiple expensive templates, it can be that one of them is not instantiated. In this case, **do define** the template with `RAFT_EXPLICIT` in the `-ext` header. This way, when the template is instantiated, the developer gets a helpful error message instead of a confusing "function not found".

This header structure was proposed in [issue #1416](https://github.com/rapidsai/raft/issues/1416), which contains more background on the motivation of this structure and the mechanics of C++ template instantiation.

## Testing

It's important for RAFT to maintain a high test coverage of the public APIs in order to minimize the potential for downstream projects to encounter unexpected build or runtime behavior as a result of changes.

A well-defined public API can help maintain compile-time stability but means more focus should be placed on testing the functional requirements and verifying execution on the various edge cases within RAFT itself. Ideally, bug fixes and new features should be able to be made to RAFT independently of the consuming projects.

## Documentation

Public APIs always require documentation since those will be exposed directly to users. For C++, we use [doxygen](http://www.doxygen.nl) and for Python/cython we use [pydoc](https://docs.python.org/3/library/pydoc.html). In addition to summarizing the purpose of each class / function in the public API, the arguments (and relevant templates) should be documented along with brief usage examples.

## Asynchronous operations and stream ordering
All RAFT algorithms should be as asynchronous as possible avoiding the use of the default stream (aka as NULL or `0` stream). Implementations that require only one CUDA Stream should use the stream from `raft::resources`:

```cpp
#include <raft/core/resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>

void foo(const raft::resources& res, ...)
{
    auto stream = get_cuda_stream(res);  // rmm::cuda_stream_view
}
```
When multiple streams are needed, e.g. to manage a pipeline, use the internal streams available in `raft::resources` (see [Resource Management](#resource-management)). If multiple streams are used all operations still must be ordered according to `raft::resource::get_cuda_stream()` (from `raft/core/resource/cuda_stream.hpp`). Before any operation in any of the internal CUDA streams is started, all previous work in `raft::resource::get_cuda_stream()` must have completed. Any work enqueued in `raft::resource::get_cuda_stream()` after a RAFT function returns should not start before all work enqueued in the internal streams has completed. E.g. if a RAFT algorithm is called like this:
```cpp
#include <raft/core/copy.hpp>           // raft::copy
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <rmm/cuda_stream.hpp>

void foo(raft::device_vector_view<double> srcdata,
         raft::device_vector_view<double> result,
         raft::host_vector_view<const double> h_srcdata,
         raft::host_vector_view<double> h_result)
{
    rmm::cuda_stream stream;  // owns the stream; destroyed automatically
    raft::resources res;
    set_cuda_stream(res, stream.view());

    ...

    raft::copy(res, srcdata, h_srcdata );  // host -> device, ordered on res's stream

    raft::algo(res, dopredict, srcdata, result, ... );

    raft::copy(res, h_result, result );    // device -> host, ordered on res's stream

    ...
}
```
No work in any stream should start in `raft::algo` before the `raft::copy` in `stream` launched before the call to `raft::algo` is done. And all work in all streams used in `raft::algo` should be done before the `raft::copy` in `stream` launched after the call to `raft::algo` starts.

This can be ensured by introducing interstream dependencies with CUDA events and `cudaStreamWaitEvent`. For convenience, the header `raft/core/device_resources.hpp` provides the class `raft::stream_syncer` which lets all `raft::resources` internal CUDA streams wait on `raft::resource::get_cuda_stream()` in its constructor and in its destructor and lets `raft::resource::get_cuda_stream()` wait on all work enqueued in the `raft::resources` internal CUDA streams. The intended use would be to create a `raft::stream_syncer` object as the first thing in an entry function of the public RAFT API:

```cpp
namespace raft {
   void algo(const raft::resources& res, ...)
   {
       raft::stream_syncer _(res);
   }
}
```
This ensures the stream ordering behavior described above.

### Using Thrust
To ensure that thrust algorithms are executed in the intended stream the `thrust::cuda::par` execution policy should be used. To ensure that thrust algorithms allocate temporary memory via the provided device memory allocator, use the `rmm::exec_policy` available in `raft/core/resource/thrust_policy.hpp`, which can be used through `raft::resources`:
```cpp
#include <raft/core/resources.hpp>
#include <raft/core/resource/thrust_policy.hpp>
void foo(const raft::resources& res, ...)
{
    auto execution_policy = get_thrust_policy(res);
    thrust::for_each(execution_policy, ... );
}
```

## Resource Management

Do not create reusable CUDA resources directly in implementations of RAFT algorithms. Instead, use the existing resources in `raft::resources` to avoid constant creation and deletion of reusable resources such as CUDA streams, CUDA events or library handles. Please file a feature request if a resource handle is missing in `raft::resources`.
The resources can be obtained like this
```cpp
#include <raft/core/resources.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
void foo(const raft::resources& res, ...)
{
    cublasHandle_t cublasHandle = get_cublas_handle(res);
    const int num_streams       = get_stream_pool_size(res);
    const int stream_idx        = ...
    auto stream                 = get_stream_from_stream_pool(res, stream_idx);  // rmm::cuda_stream_view
    ...
}
```

The example below shows one way to create `n_stream` number of internal cuda streams with an `rmm::stream_pool` which can later be used by the algos inside RAFT.
```cpp
#include <raft/core/resources.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_pool.hpp>
int main(int argc, char** argv)
{
    int n_streams = argc > 1 ? atoi(argv[1]) : 0;
    raft::resources res;
    set_cuda_stream_pool(res, std::make_shared<rmm::cuda_stream_pool>(n_streams));

    foo(res, ...);
}
```

## Multi-GPU

The multi-GPU paradigm of RAFT is **O**ne **P**rocess per **G**PU (OPG). Each algorithm should be implemented in a way that it can run with a single GPU without any specific dependencies to a particular communication library. A multi-GPU implementation should use the methods offered by the class `raft::comms::comms_t` from `raft/core/comms.hpp` for inter-rank/GPU communication. It is the responsibility of the user of RAFT to create an initialized instance of `raft::comms::comms_t`.

E.g. with a CUDA-aware MPI, a RAFT user could use code like this to inject an initialized instance of `raft::comms::mpi_comms` into a `raft::resources`:

```cpp
#include <mpi.h>
#include <raft/core/resources.hpp>
#include <raft/comms/mpi_comms.hpp>
#include <raft/algo.hpp>
...
int main(int argc, char * argv[])
{
    MPI_Init(&argc, &argv);
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int local_rank = -1;
    {
        MPI_Comm local_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);

        MPI_Comm_rank(local_comm, &local_rank);

        MPI_Comm_free(&local_comm);
    }

    cudaSetDevice(local_rank);

    mpi_comms raft_mpi_comms;
    MPI_Comm_dup(MPI_COMM_WORLD, &raft_mpi_comms);

    {
        raft::resources res;
        initialize_mpi_comms(res, raft_mpi_comms);

        ...

        raft::algo(res, ... );
    }

    MPI_Comm_free(&raft_mpi_comms);

    MPI_Finalize();
    return 0;
}
```

A RAFT developer can assume the following:
* A instance of `raft::comms::comms_t` was correctly initialized.
* All processes that are part of `raft::comms::comms_t` call into the RAFT algorithm cooperatively.

The initialized instance of `raft::comms::comms_t` can be accessed from the `raft::resources` instance:

```cpp
#include <raft/core/resources.hpp>
#include <raft/core/resource/comms.hpp>
void foo(const raft::resources& res, ...)
{
    const raft::comms_t& communicator = get_comms(res);
    const int rank = communicator.get_rank();
    const int size = communicator.get_size();
    ...
}
```
