/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/error.hpp>
#include <raft/core/logger.hpp>
#include <raft/util/integer_utils.hpp>

#include <cuda/memory_resource>

#include <linux/mman.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>

namespace raft {
namespace mr {

namespace detail {

/** RAII wrapper for a file descriptor. */
struct tmpfile_descriptor {
  explicit tmpfile_descriptor(size_t file_size_bytes) : fd_{std::tmpfile()}
  {
    if (fd_ == nullptr) {
      auto e = errno;
      RAFT_FAIL(
        "raft::mr::tmpfile_descriptor: failed to open a temporary file (std::tmpfile): errno = "
        "%d, %s",
        e,
        strerror(e));
    }
    if (ftruncate(fileno(fd_), file_size_bytes) == -1) {
      auto e = errno;
      RAFT_FAIL(
        "raft::mr::tmpfile_descriptor: failed to call `ftruncate` to allocate memory for a "
        "temporary file. errno = %d, %s",
        e,
        strerror(e));
    }
  }

  tmpfile_descriptor(const tmpfile_descriptor& res)                      = delete;
  auto operator=(const tmpfile_descriptor& other) -> tmpfile_descriptor& = delete;
  tmpfile_descriptor(tmpfile_descriptor&& other) : fd_{std::exchange(other.fd_, nullptr)} {}
  auto operator=(tmpfile_descriptor&& other) -> tmpfile_descriptor&
  {
    std::swap(this->fd_, other.fd_);
    return *this;
  }

  ~tmpfile_descriptor() noexcept
  {
    if (fd_ != nullptr) { std::fclose(fd_); }
  }

  [[nodiscard]] auto value() const -> FILE* { return fd_; }

 private:
  FILE* fd_ = nullptr;
};

}  // namespace detail

/** Default flags for `mmap_memory_resource`. */
constexpr int kMmapDefault = 0x0;
/** Request 2MB huge pages support through `madvise`. */
constexpr int kMmapRequestHugePages = 0x1;
/** Request memory to be backed by a temporary file. */
constexpr int kMmapFileBacked = 0x2;

/**
 * @brief A cuda::mr::synchronous_resource backed by mmap.
 *
 * Host-only; binds to raft::mr::synchronous_host_resource_ref.
 */
class mmap_memory_resource {
 public:
  explicit mmap_memory_resource(int flags = kMmapDefault) noexcept : flags_{flags} {}
  ~mmap_memory_resource() noexcept = default;

  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    if (bytes == 0) { return nullptr; }
    auto prot  = PROT_READ | PROT_WRITE;
    auto flags = MAP_ANONYMOUS | MAP_PRIVATE;
    void* ptr  = nullptr;
    if (flags_ & kMmapFileBacked) {
      // Note, we don't need the file descriptor to live beyond the call to mmap:
      //       according to the POSIX specification, mmap retains its own descriptor.
      detail::tmpfile_descriptor fd{bytes};
      ptr = mmap_verbose(bytes, prot, flags, fileno(fd.value()), 0);
    } else {
      ptr = mmap_verbose(bytes, prot, flags, -1, 0);
    }
    if (flags_ & kMmapRequestHugePages) {
      // Find a page-aligned subrange of the allocated memory to madvise
      auto madvize_start = raft::round_up_safe(reinterpret_cast<uintptr_t>(ptr), kHugePageSize);
      auto madvize_end =
        raft::round_down_safe(reinterpret_cast<uintptr_t>(ptr) + bytes, kHugePageSize);
      auto madvize_size = madvize_end - madvize_start;
      madvise(reinterpret_cast<void*>(madvize_start), madvize_size, MADV_HUGEPAGE);
    }
    return ptr;
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t /*alignment*/ = alignof(std::max_align_t)) noexcept
  {
    if (ptr == nullptr) { return; }
    if (munmap(ptr, bytes) != 0) {
      RAFT_LOG_ERROR("Failed call to raft::mr::mmap_memory_resource::deallocate_sync(%p, %zu): %s",
                     ptr,
                     bytes,
                     strerror(errno));
    }
  }

  [[nodiscard]] bool operator==(mmap_memory_resource const& other) const noexcept
  {
    return flags_ == other.flags_;
  }

  [[nodiscard]] bool operator!=(mmap_memory_resource const& other) const noexcept
  {
    return !(*this == other);
  }

  friend void get_property(mmap_memory_resource const&, cuda::mr::host_accessible) noexcept {}

 private:
  static inline constexpr size_t kHugePageSize = 2ull * 1024ull * 1024ull;
  int flags_{kMmapDefault};

  static inline auto mmap_verbose(size_t length, int prot, int flags, int fd, off_t offset) -> void*
  {
    if (length == 0) { return nullptr; }
    auto ptr = mmap(nullptr, length, prot, flags, fd, offset);
    if (ptr == MAP_FAILED) {
      RAFT_FAIL(
        "Failed call to raft::mr::mmap_memory_resource:mmap(nullptr, %zu, 0x%08x, 0x%08x, %d, "
        "%zd): %s",
        length,
        prot,
        flags,
        fd,
        offset,
        strerror(errno));
    }
    return ptr;
  }
};

}  // namespace mr
}  // namespace raft
