# Dry Run Protocol

The dry run protocol lets callers estimate an algorithm's memory footprint without executing it. When enabled, the runtime swaps memory resources for lightweight trackers that record every allocation and deallocation, producing peak-usage statistics at the end.

## Using Dry Run Mode

```cpp
#include <raft/core/dry_run_resources.hpp>

raft::resources res;
// auto my_function(const raft::resources& res, my_args...);
auto stats = raft::util::dry_run_execute(res, my_function, my_args...);
// stats.device_global  – peak device memory (bytes)
```

`dry_run_execute` swaps the memory resources, sets the flag, runs the callable, restores everything, and returns a `raft::memory_stats` snapshot of peak allocation usage.

You can also construct `raft::dry_run_resources` directly for finer control (e.g. reading `get_bytes_current()` in addition to `get_bytes_peak()`).

## Three Rules

1. **Allocations must not be guarded.** Every `rmm::device_uvector`, `rmm::device_scalar`, `rmm::device_buffer`, `raft::make_(device|host|pinned|managed)_(mdarray|matrix|vector|scalar)` allocation must execute in both modes so the tracker sees it.

2. **CUDA work must be guarded.** Kernel launches, Thrust algorithms, cuBLAS/cuSOLVER/cuSPARSE compute calls, `cudaMemcpyAsync`, `cudaMemsetAsync`, and `raft::interruptible::synchronize` must not run in dry-run mode.

3. **Every function taking `raft::resources` must be callable in dry-run mode.** If it only delegates to other compliant functions, it needs no guard at all. If it performs raw CUDA work, it must guard that work internally.

## What Needs Guarding

| Must guard | Safe in dry-run (no guard needed) |
|---|---|
| Kernel launches (`<<<>>>`) | Allocations (`rmm::device_uvector`, `make_device_*`, …) |
| `thrust::reduce`, `thrust::for_each`, … | Workspace-size queries (`cub::…(nullptr, &size, …)`, `cusparse…_bufferSize`) |
| cuBLAS / cuSOLVER / cuSPARSE compute calls | cuSPARSE descriptor create/destroy |
| CUB compute calls (second pass) | `resource::sync_stream()` |
| `cudaMemcpyAsync`, `cudaMemsetAsync` | `raft::copy` (takes `raft::resources`) |
| `raft::interruptible::synchronize()` | `raft::linalg::map`, `raft::linalg::reduce`, and other compliant RAFT APIs |

## Patterns

### Basic: allocate, then guard

```cpp
void algo(raft::resources const& handle, int n, cudaStream_t stream)
{
  rmm::device_uvector<float> buf(n, stream);           // tracked
  if (resource::get_dry_run_flag(handle)) { return; }
  kernel<<<grid, block, 0, stream>>>(buf.data(), n);   // skipped in dry-run
}
```

### Workspace-size query before guard

_We assume_ CUB and cuSPARSE workspace queries do not launch device work when the workspace pointer is `nullptr`, so they are safe to run in dry-run mode.

```cpp
size_t ws_bytes = 0;
cub::DeviceRadixSort::SortPairs(nullptr, &ws_bytes, ...);   // query only
rmm::device_uvector<char> workspace(ws_bytes, stream);       // tracked
if (resource::get_dry_run_flag(handle)) { return; }
cub::DeviceRadixSort::SortPairs(workspace.data(), &ws_bytes, ...);  // real work
```

### Guard individual operations (not the whole body)

When cleanup or descriptor destruction must always run, guard each operation instead of returning early.

```cpp
cusparseSpMV_bufferSize(handle, ..., &buf_size);         // safe
rmm::device_uvector<char> tmp(buf_size, stream);         // tracked
if (!is_dry_run) {
  cusparseSpMV(handle, ..., tmp.data());                 // guarded
}
cusparseDestroyDnVec(descr);                             // always runs
```

### Public wrappers: delegate without guards

A wrapper that only calls compliant functions must **not** add an early return—doing so hides allocations made by the callee.

```cpp
// WRONG – hides allocations inside detail::foo
void foo(raft::resources const& handle, ...) {
  if (resource::get_dry_run_flag(handle)) { return; }
  detail::foo(handle, ...);
}

// CORRECT
void foo(raft::resources const& handle, ...) {
  detail::foo(handle, ...);
}
```
