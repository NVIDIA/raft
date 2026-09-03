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

2. **CUDA work must be guarded.** Kernel launches, Thrust algorithms, cuBLAS/cuSOLVER/cuSPARSE compute calls, `cudaMemcpyAsync`, `cudaMemsetAsync`, and `raft::interruptible::synchronize` must not run in dry-run mode. A kernel launched with `raft::launch_kernel` on the handle guards itself — see [Launching kernels](#launching-kernels).

3. **Every function taking `raft::resources` must be callable in dry-run mode.** If it only delegates to other compliant functions, it needs no guard at all. If it performs raw CUDA work, it must guard that work internally.

## What Needs Guarding

| Must guard | Safe in dry-run (no guard needed) |
|---|---|
| Raw kernel launches (`<<<>>>`); `raft::launch_kernel` on a bare stream without `skip_execution` | `raft::launch_kernel(res, …)`, allocations (`rmm::device_uvector`, `make_device_*`, …) |
| `thrust::reduce`, `thrust::for_each`, … | Workspace-size queries (`cub::…(nullptr, &size, …)`, `cusparse…_bufferSize`) |
| cuBLAS / cuSOLVER / cuSPARSE compute calls | cuSPARSE descriptor create/destroy |
| CUB compute calls (second pass) | `resource::sync_stream()`, grid/block sizing and device-metadata queries |
| `cudaMemcpyAsync`, `cudaMemsetAsync` | `raft::copy` (takes `raft::resources`) |
| `raft::interruptible::synchronize()` | `raft::linalg::map`, `raft::linalg::reduce`, and other compliant RAFT APIs |

## Patterns

### Basic: allocate, then guard

Derive the stream (and every other resource) from the `raft::resources` handle
rather than accepting a raw `cudaStream_t`. In dry-run mode the handle owns the
stream that is consistent with the swapped-in tracking resources; passing an
unrelated raw stream breaks that ownership model.

```cpp
void algo(raft::resources const& handle, int n)
{
  auto stream = resource::get_cuda_stream(handle);            // stream owned by handle
  rmm::device_uvector<float> buf(n, stream);                  // tracked
  raft::launch_kernel(handle, grid, block, my_kernel, buf.data(), n);  // skipped in dry-run
}
```

### Launching kernels

`raft::launch_kernel` from `<raft/util/kernel_launch.hpp>` is dry-run compliant
when it is given the handle, because the handle carries the dry-run flag:

```cpp
// COMPLIANT: the handle carries the dry-run flag, so the launch is skipped for you
raft::launch_kernel(handle, grid, block, my_kernel, buf.data(), n);
raft::launch_kernel({handle, smem}, grid, block, my_kernel, buf.data(), n);
```

Such a launch needs **no** guard, and adding one is a defect rather than a
harmless redundancy: a guard or early return placed around it tends to also
swallow the allocations and cleanup that follow, breaking Rule 1.

The stream-taking overloads are **not** dry-run aware, because a bare stream
carries no dry-run state. Their third constructor argument is `skip_execution`:

```cpp
raft::launch_kernel(stream, grid, block, my_kernel, ...);           // WRONG when reachable from resources
raft::launch_kernel({stream, smem}, grid, block, my_kernel, ...);   // WRONG, same reason
raft::launch_kernel({stream, smem, dry_run}, grid, block, ...);     // correct
```

When a launch sits behind an explicit stream, the flag still has to reach it, and
there are only two ways to get it there:

1. change the function to take `raft::resources const&` and launch on it;
2. if the signature must keep a `cudaStream_t` — a deprecated public overload, or
   a leaf where duplicating the handle costs too much — thread a `bool dry_run`
   parameter through it and pass that as `skip_execution`.

Where the function already has the flag and the launch is not the only thing that
must be skipped — a `cudaMemsetAsync`, a CUB call, a device-to-host read — a
single guard over that whole run of device work is simpler than passing
`skip_execution` and guarding the rest separately.

Always convert a raw `kernel<<<>>>` launch to `raft::launch_kernel` first: the
raw syntax cannot express `skip_execution`, and it does not report a failed
launch at the call site.

Two consequences worth knowing:

- A skipped launch is not validated by CUDA, so an invalid grid or block size
  goes unreported in dry-run mode.
- `RAFT_CUDA_TRY(cudaPeekAtLastError())` after `launch_kernel` is unnecessary: a
  failed launch already throws `raft::cuda_error` blaming the call site.

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

## Advanced topic: Probe Memory Semantics

In dry-run mode each memory category (host, device, pinned, managed, workspace,
large-workspace) is backed by a single shared **probe buffer** of only 256 bytes
(`raft::mr::kDryRunProbeSize`). Every logical allocation in that category returns
the *same* probe pointer, and the requested size is merely recorded by the
tracker — it is not physically backed. Consequences:

- **Never dereference dry-run-allocated memory.** Do not read, write, `memcpy`,
  `memset`, or run kernels/Thrust/library compute against a buffer allocated in
  dry-run mode: all such pointers alias the shared probe and are far smaller than
  the requested size (out-of-bounds / data races otherwise).
- **Never use allocated contents for control flow.** Sizes computed from probe
  contents are meaningless.
- **Device-metadata queries are allowed.** Occupancy queries
  (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`, …) and
  `resource::get_device_properties()` do not touch probe memory, so they may run
  in dry-run mode when needed to size a tracked scratch buffer accurately.

## Dry-run Outputs and Data-dependent Sizes

- `make_(host|pinned|managed|device)_scalar(handle, value)` still allocates and
  tracks the scalar in dry-run mode, but **does not write `value`** — the scalar
  is uninitialized and must not be read.
- Scalar-returning metrics/helpers (statistics scores, norms, boolean reductions,
  cardinality outputs, `matrix_wrappers::nrm1`, …) return **non-authoritative
  placeholder values** in dry-run mode. Do not present them as computed results.
- When a real allocation size depends on data that is only produced by skipped
  compute (e.g. a device-side count read back to host, or a min/max label range),
  substitute a **conservative upper bound** in dry-run mode instead of reading the
  uninitialized value.
