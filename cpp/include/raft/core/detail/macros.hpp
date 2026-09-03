/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifndef _RAFT_HAS_CUDA
#if defined(__CUDACC__)
#define _RAFT_HAS_CUDA __CUDACC__
#endif
#endif

#if defined(_RAFT_HAS_CUDA)
#define CUDA_CONDITION_ELSE_TRUE(condition)  condition
#define CUDA_CONDITION_ELSE_FALSE(condition) condition
#else
#define CUDA_CONDITION_ELSE_TRUE(condition)  true
#define CUDA_CONDITION_ELSE_FALSE(condition) false
#endif

#ifndef _RAFT_HOST_DEVICE
#if defined(_RAFT_HAS_CUDA)
#define _RAFT_DEVICE      __device__
#define _RAFT_HOST        __host__
#define _RAFT_FORCEINLINE __forceinline__
#else
#define _RAFT_DEVICE
#define _RAFT_HOST
#define _RAFT_FORCEINLINE inline
#endif
#endif

#define _RAFT_HOST_DEVICE _RAFT_HOST _RAFT_DEVICE

#ifndef RAFT_INLINE_FUNCTION
#define RAFT_INLINE_FUNCTION _RAFT_HOST_DEVICE _RAFT_FORCEINLINE
#endif

#ifndef RAFT_DEVICE_INLINE_FUNCTION
#define RAFT_DEVICE_INLINE_FUNCTION _RAFT_DEVICE _RAFT_FORCEINLINE
#endif

// The RAFT_INLINE_CONDITIONAL is a conditional inline specifier that removes
// the inline specification when RAFT_COMPILED is defined.
//
// When RAFT_COMPILED is not defined, functions may be defined in multiple
// translation units and we do not want that to lead to linker errors.
//
// When RAFT_COMPILED is defined, this serves two purposes:
//
// 1. It triggers a multiple definition error message when memory_pool-inl.hpp
// (for instance) is accidentally included in multiple translation units.
//
// 2. We function definitions to be non-inline, because non-inline functions
// symbols are always exported in the object symbol table. For inline functions,
// the compiler may elide the external symbol, which results in linker errors.
#ifdef RAFT_COMPILED
#define RAFT_INLINE_CONDITIONAL
#else
#define RAFT_INLINE_CONDITIONAL inline
#endif  // RAFT_COMPILED

// The RAFT_WEAK_FUNCTION specificies that:
//
// 1. A function may be defined in multiple translation units (like inline)
//
// 2. Must still emit an external symbol (unlike inline). This enables declaring
// a function signature in an `-ext` header and defining it in a source file.
//
// From
// https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#Common-Function-Attributes:
//
// "The weak attribute causes a declaration of an external symbol to be emitted
// as a weak symbol rather than a global."
#define RAFT_WEAK_FUNCTION __attribute__((weak))

// Macros for controlling symbol visibility in shared libraries.
// When CXX_VISIBILITY_PRESET is set to hidden (the default for raft_objs),
// only symbols explicitly marked with RAFT_EXPORT will be visible in the
// shared library. RAFT_HIDDEN can be used to explicitly mark symbols as
// hidden (redundant with the default, but useful for clarity).
//
// Hidden visibility also ensures that the linker doesn't de-duplicate the
// symbol across multiple `.so`. This allows multiple libraries to embed raft
// without issue.
#if (defined(__GNUC__) && !defined(__MINGW32__) && !defined(__MINGW64__))
#define RAFT_EXPORT __attribute__((visibility("default")))
#define RAFT_HIDDEN __attribute__((visibility("hidden")))
#else
#define RAFT_EXPORT
#define RAFT_HIDDEN
#endif

// The RAFT_KERNEL specificies that a kernel has hidden visibility
//
// Raft needs to ensure that the visibility of its __global__ function
// templates have hidden visibility ( default is weak visibility).
//
// When kernls have weak visibility it means that if two dynamic libraries
// both contain identical instantiations of a RAFT template, then the linker
// will discard one of the two instantiations and use only one of them.
//
// Do to unique requirements of how the CUDA works this de-deduplication
// can lead to the wrong kernels being called ( SM version being wrong ),
// silently no kernel being called at all, or cuda runtime errors being
// thrown.
//
// https://github.com/rapidsai/raft/issues/1722
#if defined(__CUDACC_RDC__)
#define RAFT_KERNEL RAFT_HIDDEN __global__ void
#elif defined(_RAFT_HAS_CUDA)
#define RAFT_KERNEL static __global__ void
#else
#define RAFT_KERNEL static void
#endif

/**
 * Some macro magic to remove optional parentheses of a macro argument.
 * See https://stackoverflow.com/a/62984543
 */
#ifndef RAFT_DEPAREN_MAGICRAFT_DEPAREN_H1
#define RAFT_DEPAREN(X)      RAFT_DEPAREN_H2(RAFT_DEPAREN_H1 X)
#define RAFT_DEPAREN_H1(...) RAFT_DEPAREN_H1 __VA_ARGS__
#define RAFT_DEPAREN_H2(...) RAFT_DEPAREN_H3(__VA_ARGS__)
#define RAFT_DEPAREN_H3(...) RAFT_DEPAREN_MAGIC##__VA_ARGS__
#define RAFT_DEPAREN_MAGICRAFT_DEPAREN_H1
#endif

#ifndef RAFT_STRINGIFY
#define RAFT_STRINGIFY_DETAIL(...) #__VA_ARGS__
#define RAFT_STRINGIFY(...)        RAFT_STRINGIFY_DETAIL(__VA_ARGS__)
#endif

// Let the compiler check the arguments of a printf-style function against its format string, the
// same way it checks calls to printf itself.
//
// `fmt_index` is the one-based index of the format string parameter and `first_arg_index` the index
// of the first variadic argument (for a non-static member function, `this` is parameter one).
#ifndef RAFT_FORMAT_PRINTF
#if defined(__GNUC__)
#define RAFT_FORMAT_PRINTF(fmt_index, first_arg_index) \
  __attribute__((format(printf, fmt_index, first_arg_index)))
#else
#define RAFT_FORMAT_PRINTF(fmt_index, first_arg_index)
#endif
#endif
