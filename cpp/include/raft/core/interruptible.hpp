/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __RAFT_RT_INTERRUPTIBLE_H
#define __RAFT_RT_INTERRUPTIBLE_H

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/error.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>

namespace RAFT_EXPORT raft {

/**
 * @defgroup interruptible definitions and classes related to the interruptible API
 * @{
 */

/**
 * @brief Exception thrown during `interruptible::synchronize` call when it detects a request
 * to cancel the work performed in this CPU thread.
 */
struct interrupted_exception : public raft::exception {
  using raft::exception::exception;
};

/**
 * @brief Cooperative-style interruptible execution.
 *
 * This class provides facilities for interrupting execution of a C++ thread at designated points
 * in code from outside of the thread. In particular, it provides an interruptible version of the
 * blocking CUDA synchronization function, that allows dropping a long-running GPU work.
 *
 *
 * **Important:** Although CUDA synchronize calls serve as cancellation points, the interruptible
 * machinery has nothing to do with CUDA streams or events. In other words, when you call `cancel`,
 * it’s the CPU waiting function what is interrupted, not the GPU stream work. This means, when the
 * `interrupted_exception` is raised, any unfinished GPU stream work continues to run. It’s the
 * responsibility of the developer then to make sure the unfinished stream work does not affect the
 * program in an undesirable way.
 *
 *
 * What can happen to CUDA stream when the `synchronize` is cancelled? If you catch the
 * `interrupted_exception` immediately, you can safely wait on the stream again.
 * Otherwise, some of the allocated resources may be released before the active kernel finishes
 * using them, which will result in writing into deallocated or reallocated memory and undefined
 * behavior in general. A dead-locked kernel may never finish (or may crash if you’re lucky). In
 * practice, the outcome is usually acceptable for the use case of emergency program interruption
 * (e.g., CTRL+C), but extra effort on the use side is required to allow safe interrupting and
 * resuming of the GPU stream work.
 */
class interruptible {
 public:
  /**
   * @brief Synchronize the CUDA stream, subject to being interrupted by `interruptible::cancel`
   * called on this CPU thread.
   *
   * @param [in] stream a CUDA stream.
   *
   * @throw raft::interrupted_exception if interruptible::cancel() was called on the current CPU
   * thread before the currently captured work has been finished.
   * @throw raft::cuda_error if another CUDA error happens.
   */
  static inline void synchronize(rmm::cuda_stream_view stream)
  {
    get_token()->synchronize_impl(cudaStreamQuery, stream);
  }

  /**
   * @brief Synchronize the CUDA event, subject to being interrupted by `interruptible::cancel`
   * called on this CPU thread.
   *
   * @param [in] event a CUDA event.
   *
   * @throw raft::interrupted_exception if interruptible::cancel() was called on the current CPU
   * thread before the currently captured work has been finished.
   * @throw raft::cuda_error if another CUDA error happens.
   */
  static inline void synchronize(cudaEvent_t event)
  {
    get_token()->synchronize_impl(cudaEventQuery, event);
  }

  /**
   * @brief Check the thread state, whether the thread can continue execution or is interrupted by
   * `interruptible::cancel`.
   *
   * This is a cancellation point for an interruptible thread. It's called in the internals of
   * `interruptible::synchronize` in a loop. If two synchronize calls are far apart, it's
   * recommended to call `interruptible::yield()` in between to make sure the thread does not become
   * unresponsive for too long.
   *
   * Both `yield` and `yield_no_throw` reset the state to non-cancelled after execution.
   *
   * @throw raft::interrupted_exception if interruptible::cancel() was called on the current CPU
   * thread.
   */
  static inline void yield() { get_token()->yield_impl(); }

  /**
   * @brief Check the thread state, whether the thread can continue execution or is interrupted by
   * `interruptible::cancel`.
   *
   * Same as `interruptible::yield`, but does not throw an exception if the thread is cancelled.
   *
   * Both `yield` and `yield_no_throw` reset the state to non-cancelled after execution.
   *
   * @return whether the thread can continue, i.e. `true` means continue, `false` means cancelled.
   */
  static inline auto yield_no_throw() -> bool { return get_token()->yield_no_throw_impl(); }

  /**
   * @brief Get a cancellation token for this CPU thread.
   *
   * @return an object that can be used to cancel the GPU work waited on this CPU thread.
   */
  static inline auto get_token() -> std::shared_ptr<interruptible>
  {
    // NB: using static thread-local storage to keep the token alive once it is initialized
    static thread_local std::shared_ptr<interruptible> s(
      get_token_impl<true>(std::this_thread::get_id()));
    return s;
  }

  /**
   * @brief Get a cancellation token for a CPU thread given by its id.
   *
   * The returned token may live longer than the associated thread. In that case, using its
   * `cancel` method has no effect.
   *
   * @param [in] thread_id an id of a C++ CPU thread.
   * @return an object that can be used to cancel the GPU work waited on the given CPU thread.
   */
  static inline auto get_token(std::thread::id thread_id) -> std::shared_ptr<interruptible>
  {
    return get_token_impl<false>(thread_id);
  }

  /**
   * @brief Cancel any current or next call to `interruptible::synchronize` performed on the
   * CPU thread given by the `thread_id`
   *
   * Note, this function uses a mutex to safely get a cancellation token that may be shared
   * among multiple threads. If you plan to use it from a signal handler, consider the non-static
   * `cancel()` instead.
   *
   * @param [in] thread_id a CPU thread, in which the work should be interrupted.
   */
  static inline void cancel(std::thread::id thread_id) { get_token(thread_id)->cancel(); }

  /**
   * @brief Cancel any current or next call to `interruptible::synchronize` performed on the
   * CPU thread given by this `interruptible` token.
   *
   * Note, this function does not involve thread synchronization/locks and does not throw any
   * exceptions, so it's safe to call from a signal handler.
   */
  inline void cancel() noexcept { continue_.clear(std::memory_order_relaxed); }

  // don't allow the token to leave the shared_ptr
  interruptible(interruptible const&)                    = delete;
  interruptible(interruptible&&)                         = delete;
  auto operator=(interruptible const&) -> interruptible& = delete;
  auto operator=(interruptible&&) -> interruptible&      = delete;

 private:
  /** Global registry of thread-local cancellation stores. */
  using registry_t =
    std::tuple<std::mutex, std::unordered_map<std::thread::id, std::weak_ptr<interruptible>>>;

  /**
   * The registry "garbage collector": a custom deleter for the interruptible tokens that removes
   * the token from the registry, if the registry still exists.
   */
  struct registry_gc_t {
    std::weak_ptr<registry_t> weak_registry;
    std::thread::id thread_id;

    inline void operator()(interruptible* thread_store) const noexcept
    {
      // the deleter kicks in at thread/program exit; in some cases, the registry_ (static variable)
      // may have been destructed by this point of time.
      // Hence, we use a weak pointer to check if the registry still exists.
      auto registry = weak_registry.lock();
      if (registry) {
        std::lock_guard<std::mutex> guard_erase(std::get<0>(*registry));
        auto& map  = std::get<1>(*registry);
        auto found = map.find(thread_id);
        if (found != map.end()) {
          auto stored = found->second.lock();
          // thread_store is not moveable, thus retains its original location.
          // Not equal pointers below imply the new store has been already placed
          // in the registry by the same std::thread::id
          if (!stored || stored.get() == thread_store) { map.erase(found); }
        }
      }
      delete thread_store;
    }
  };

  /**
   * The registry itself is stored in the static memory, in a shared pointer.
   * This is to safely access it from the destructors of the thread-local tokens.
   */
  static inline std::shared_ptr<registry_t> registry_{new registry_t{}};

  /**
   * Create a new interruptible token or get an existing from the global registry_.
   *
   * Presumptions:
   *
   *   1. get_token_impl<true> must be called at most once per thread.
   *   2. When `Claim == true`, thread_id must be equal to std::this_thread::get_id().
   *   3. get_token_impl<false> can be called as many times as needed, producing a valid
   *      token for any input thread_id, independent of whether a C++ thread with this
   *      id exists or not.
   *
   * @tparam Claim whether to bind the token to the given thread.
   * @param [in] thread_id the id of the associated C++ thread.
   * @return new or existing interruptible token.
   */
  template <bool Claim>
  static auto get_token_impl(std::thread::id thread_id) -> std::shared_ptr<interruptible>
  {
    // Make a local copy of the shared pointer to make sure the registry is not destroyed,
    // if, for any reason, this function is called at program exit.
    std::shared_ptr<registry_t> shared_registry = registry_;
    // If the registry is not available, create a lone token that cannot be accessed from
    // the outside of the thread.
    if (!shared_registry) { return std::shared_ptr<interruptible>{new interruptible()}; }
    // Otherwise, proceed with the normal logic
    std::lock_guard<std::mutex> guard_get(std::get<0>(*shared_registry));
    // the following two lines construct an empty shared_ptr if the key does not exist.
    auto& weak_store  = std::get<1>(*shared_registry)[thread_id];
    auto thread_store = weak_store.lock();
    if (!thread_store || (Claim && thread_store->claimed_)) {
      // Create a new thread_store in two cases:
      //  1. It does not exist in the map yet
      //  2. The previous store in the map has not yet been deleted
      thread_store.reset(new interruptible(), registry_gc_t{shared_registry, thread_id});
      std::weak_ptr<interruptible>(thread_store).swap(weak_store);
    }
    // The thread_store is "claimed" by the thread
    if constexpr (Claim) { thread_store->claimed_ = true; }
    return thread_store;
  }

  /**
   * Communicate whether the thread is in a cancelled state or can continue execution.
   *
   * `yield` checks this flag and always resets it to the signalled state; `cancel` clears it.
   * These are the only two places where it's used.
   */
  std::atomic_flag continue_;
  /** This flag is set to true when the created token is placed into a thread-local storage. */
  bool claimed_ = false;

  interruptible() noexcept { yield_no_throw_impl(); }

  void yield_impl()
  {
    if (!yield_no_throw_impl()) {
      throw interrupted_exception("The work in this thread was cancelled.");
    }
  }

  auto yield_no_throw_impl() noexcept -> bool
  {
    return continue_.test_and_set(std::memory_order_relaxed);
  }

  template <typename Query, typename Object>
  inline void synchronize_impl(Query query, Object object)
  {
    cudaError_t query_result;
    while (true) {
      yield_impl();
      query_result = query(object);
      if (query_result != cudaErrorNotReady) { break; }
      std::this_thread::yield();
    }
    RAFT_CUDA_TRY(query_result);
  }
};

/**
 * @} // end doxygen group interruptible
 */

}  // namespace RAFT_EXPORT raft

#endif
