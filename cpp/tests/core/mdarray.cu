/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/device_container_policy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_container_policy.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/managed_mdarray.hpp>
#include <raft/core/managed_mdspan.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/pinned_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resource/managed_memory_resource.hpp>
#include <raft/core/resource/pinned_memory_resource.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/mr/host_device_resource.hpp>
#include <raft/mr/host_memory_resource.hpp>
#include <raft/mr/mmap_memory_resource.hpp>
#include <raft/pmr/resource_adaptor.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/iterator>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/sequence.h>

#include <gtest/gtest.h>

#include <memory_resource>

namespace {
void check_status(int32_t* d_status, rmm::cuda_stream_view stream)
{
  stream.synchronize();
  int32_t h_status{1};
  raft::update_host(&h_status, d_status, 1, stream);
  ASSERT_EQ(h_status, 0);
}

// just simple integration test, main tests are in mdspan ref implementation.
void test_mdspan()
{
  auto stream = rmm::cuda_stream_default;
  rmm::device_uvector<float> a{16ul, stream};
  thrust::sequence(rmm::exec_policy(stream), a.begin(), a.end());
  cuda::std::mdspan<float, cuda::std::extents<int, raft::dynamic_extent, raft::dynamic_extent>>
    span{a.data(), 4, 4};
  thrust::device_vector<int32_t> status(1, 0);
  auto p_status = status.data().get();
  thrust::for_each_n(
    rmm::exec_policy(stream), cuda::make_counting_iterator(0ul), 4, [=] __device__(size_t i) {
      auto v = span(0, i);
      if (v != i) { raft::myAtomicAdd(p_status, 1); }
      auto k = cuda::std::submdspan(span, 0, cuda::std::full_extent);
      if (k(i) != i) { raft::myAtomicAdd(p_status, 1); }
    });
  check_status(p_status, stream);
}
}  // namespace

TEST(MDSpan, Basic) { test_mdspan(); }

namespace raft {
void test_uvector_policy()
{
  auto s = rmm::cuda_stream{};
  device_uvector<float> dvec(10, s);
  auto a  = dvec[2];
  a       = 3;
  float c = a;
  ASSERT_EQ(c, 3);
}

TEST(MDArray, Policy) { test_uvector_policy(); }

void test_mdarray_basic()
{
  using matrix_extent = cuda::std::extents<int, dynamic_extent, dynamic_extent>;
  raft::resources handle;
  auto s = resource::get_cuda_stream(handle);
  {
    /**
     * device policy
     */
    layout_c_contiguous::mapping<matrix_extent> layout{matrix_extent{4, 4}};
    using mdarray_t = device_mdarray<float, matrix_extent, layout_c_contiguous>;
    auto policy     = mdarray_t::container_policy_type{};
    static_assert(
      std::is_same_v<typename decltype(policy)::accessor_type, device_container_policy<float>>);
    device_mdarray<float, matrix_extent, layout_c_contiguous> array{handle, layout, policy};

    array(0, 3) = 1;
    ASSERT_EQ(array(0, 3), 1);
    // non-const access
    auto d_view = array.view();
    static_assert(!decltype(d_view)::accessor_type::is_host_type::value);

    thrust::device_vector<int32_t> status(1, 0);
    auto p_status = status.data().get();
    thrust::for_each_n(rmm::exec_policy(s),
                       cuda::make_counting_iterator(0ul),
                       1,
                       [d_view, p_status] __device__(auto i) {
                         if (d_view(0, 3) != 1) { myAtomicAdd(p_status, 1); }
                         d_view(0, 2) = 3;
                         if (d_view(0, 2) != 3) { myAtomicAdd(p_status, 1); }
                       });
    check_status(p_status, s);

    // const ref access
    auto const& arr = array;
    ASSERT_EQ(arr(0, 3), 1);
    auto const_d_view = arr.view();
    thrust::for_each_n(rmm::exec_policy(s),
                       cuda::make_counting_iterator(0ul),
                       1,
                       [const_d_view, p_status] __device__(auto i) {
                         if (const_d_view(0, 3) != 1) { myAtomicAdd(p_status, 1); }
                       });
    check_status(p_status, s);

    // utilities
    static_assert(array.rank_dynamic() == 2);
    static_assert(array.rank() == 2);
    static_assert(array.is_unique());
    static_assert(array.is_exhaustive());
    static_assert(array.is_strided());

    static_assert(!std::is_nothrow_default_constructible<mdarray_t>::value);  // cuda stream
    static_assert(std::is_nothrow_move_constructible<mdarray_t>::value);
    static_assert(std::is_nothrow_move_assignable<mdarray_t>::value);
  }
  {
    /**
     * host policy
     */
    using mdarray_t = host_mdarray<float, matrix_extent, layout_c_contiguous>;

    raft::mr::mmap_memory_resource mmap_mr_default;
    raft::mr::mmap_memory_resource mmap_mr_hugepages{raft::mr::kMmapRequestHugePages};
    raft::mr::mmap_memory_resource mmap_mr_huge_file{raft::mr::kMmapRequestHugePages |
                                                     raft::mr::kMmapFileBacked};
    std::pmr::synchronized_pool_resource pool_mr{std::pmr::get_default_resource()};

    // std::pmr resources: wrap in resource_adaptor, pass to make_host_mdarray
    for (auto* mr :
         {std::pmr::get_default_resource(), static_cast<std::pmr::memory_resource*>(&pool_mr)}) {
      raft::pmr::resource_adaptor adaptor{mr};
      auto array = make_host_mdarray<float>(handle, adaptor, make_extents<uint32_t>(4u, 4u));

      array(0, 3) = 1;
      ASSERT_EQ(array(0, 3), 1);
      auto h_view = array.view();
      static_assert(decltype(h_view)::accessor_type::is_host_type::value);
      thrust::for_each_n(thrust::host, cuda::make_counting_iterator(0ul), 1, [h_view](auto i) {
        ASSERT_EQ(h_view(0, 3), 1);
      });
    }

    // cuda::mr resources (mmap): use make_host_mdarray with raft::mr::host_resource_ref
    for (auto* mr : {&mmap_mr_default, &mmap_mr_hugepages, &mmap_mr_huge_file}) {
      auto array = make_host_mdarray<float>(handle, *mr, make_extents<uint32_t>(16u));
      array(3)   = 1;
      ASSERT_EQ(array(3), 1);
    }

    static_assert(!std::is_nothrow_default_constructible<mdarray_t>::value);
    static_assert(std::is_nothrow_move_constructible<mdarray_t>::value);
    static_assert(std::is_nothrow_move_assignable<mdarray_t>::value);
    static_assert(!std::is_copy_constructible<mdarray_t>::value);
    static_assert(!std::is_copy_assignable<mdarray_t>::value);
  }
  {
    /**
     * static extent
     */
    using static_extent = cuda::std::extents<int, 16, 16>;
    layout_c_contiguous::mapping<static_extent> layout{static_extent{}};
    using mdarray_t = device_mdarray<float, static_extent, layout_c_contiguous>;
    mdarray_t::container_policy_type policy{};
    device_mdarray<float, static_extent, layout_c_contiguous> array{handle, layout, policy};

    static_assert(array.rank_dynamic() == 0);
    static_assert(array.rank() == 2);
    static_assert(array.is_unique());
    static_assert(array.is_exhaustive());
    static_assert(array.is_strided());

    array(0, 3) = 1;
    ASSERT_EQ(array(0, 3), 1);

    auto const& ref = array;
    ASSERT_EQ(ref(0, 3), 1);
  }
}

TEST(MDArray, Basic) { test_mdarray_basic(); }

template <typename BasicMDarray, typename PolicyFn, typename ThrustPolicy>
void test_mdarray_copy_move(ThrustPolicy exec, PolicyFn make_policy)
{
  raft::resources handle;
  using matrix_extent = cuda::std::extents<size_t, dynamic_extent, dynamic_extent>;
  layout_c_contiguous::mapping<matrix_extent> layout{matrix_extent{4, 4}};

  using mdarray_t = BasicMDarray;
  using policy_t  = typename mdarray_t::container_policy_type;
  auto policy     = make_policy();

  mdarray_t arr_origin{handle, layout, policy};
  thrust::sequence(exec, arr_origin.data_handle(), arr_origin.data_handle() + arr_origin.size());

  auto check_eq = [](auto const& l, auto const& r) {
    ASSERT_EQ(l.extents(), r.extents());
    for (size_t i = 0; i < l.view().extent(0); ++i) {
      for (size_t j = 0; j < l.view().extent(1); ++j) {
        ASSERT_EQ(l(i, j), r(i, j));
      }
    }
  };

  {
    // move ctor
    auto policy = make_policy();
    mdarray_t arr{handle, layout, policy};
    thrust::sequence(exec, arr.data_handle(), arr.data_handle() + arr.size());
    mdarray_t arr_move_construct{std::move(arr)};
    ASSERT_EQ(arr.data_handle(), nullptr);
    check_eq(arr_origin, arr_move_construct);
  }

  {
    // move assign
    auto policy = make_policy();
    mdarray_t arr{handle, layout, policy};
    thrust::sequence(exec, arr.data_handle(), arr.data_handle() + arr.size());
    mdarray_t arr_move_assign{handle, layout, policy};
    arr_move_assign = std::move(arr);
    check_eq(arr_origin, arr_move_assign);
  }
}

TEST(MDArray, Move)
{
  using matrix_extent = cuda::std::extents<size_t, dynamic_extent, dynamic_extent>;
  using d_matrix_t    = device_mdarray<float, matrix_extent>;
  using policy_t      = typename d_matrix_t::container_policy_type;
  raft::resources handle;
  auto s = resource::get_cuda_stream(handle);
  test_mdarray_copy_move<d_matrix_t>(rmm::exec_policy(s), []() { return policy_t{}; });

  using h_matrix_t = host_mdarray<float, matrix_extent>;
  test_mdarray_copy_move<h_matrix_t>(thrust::host, []() { return host_container_policy<float>{}; });

  {
    d_matrix_t arr{handle};
    policy_t policy;
    matrix_extent extents{3, 3};
    d_matrix_t::layout_type::mapping<matrix_extent> layout{extents};
    d_matrix_t non_dft{handle, layout, policy};

    d_matrix_t arr_orig = std::move(arr);
    arr                 = std::move(non_dft);
    ASSERT_NE(arr.data_handle(), arr_orig.data_handle());
    ASSERT_EQ(arr_orig.extent(0), 0);
    ASSERT_EQ(arr.extent(0), 3);
  }
  {
    h_matrix_t arr(handle);
    using h_policy_t = typename h_matrix_t::container_policy_type;
    h_policy_t policy;
    matrix_extent extents{3, 3};
    h_matrix_t::layout_type::mapping<matrix_extent> layout{extents};
    h_matrix_t non_dft{handle, layout, policy};

    h_matrix_t arr_orig = std::move(arr);
    arr                 = std::move(non_dft);
    ASSERT_NE(arr.data_handle(), arr_orig.data_handle());
    ASSERT_EQ(arr_orig.extent(0), 0);
    ASSERT_EQ(arr.extent(0), 3);
  }
}

namespace {
void test_factory_methods()
{
  size_t n{100};
  rmm::device_uvector<float> d_vec(n, rmm::cuda_stream_default);
  {
    auto d_matrix = make_device_matrix_view(d_vec.data(), static_cast<int>(d_vec.size() / 2), 2);
    ASSERT_EQ(d_matrix.extent(0), n / 2);
    ASSERT_EQ(d_matrix.extent(1), 2);
    ASSERT_EQ(d_matrix.data_handle(), d_vec.data());
  }
  {
    auto const& vec_ref = d_vec;
    auto d_matrix = make_device_matrix_view(vec_ref.data(), static_cast<int>(d_vec.size() / 2), 2);
    ASSERT_EQ(d_matrix.extent(0), n / 2);
    ASSERT_EQ(d_matrix.extent(1), 2);
    ASSERT_EQ(d_matrix.data_handle(), d_vec.data());
  }

  std::vector<float> h_vec(n);
  {
    auto h_matrix = make_host_matrix_view(h_vec.data(), static_cast<int>(h_vec.size() / 2), 2);
    ASSERT_EQ(h_matrix.extent(0), n / 2);
    ASSERT_EQ(h_matrix.extent(1), 2);
    ASSERT_EQ(h_matrix.data_handle(), h_vec.data());
    h_matrix(0, 0) = 13;
    ASSERT_EQ(h_matrix(0, 0), 13);
  }
  {
    auto const& vec_ref = h_vec;
    auto h_matrix = make_host_matrix_view(vec_ref.data(), static_cast<int>(d_vec.size() / 2), 2);
    ASSERT_EQ(h_matrix.extent(0), n / 2);
    ASSERT_EQ(h_matrix.extent(1), 2);
    ASSERT_EQ(h_matrix.data_handle(), h_vec.data());
    // const, cannot assign
    // h_matrix(0, 0) = 13;
    ASSERT_EQ(h_matrix(0, 0), 13);
  }

  {
    // host mdarray
    auto h_matrix = make_host_matrix<float>(n, n);
    ASSERT_EQ(h_matrix.extent(0), n);
    ASSERT_EQ(h_matrix.extent(1), n);
    static_assert(h_matrix.rank() == 2);

    auto h_vec = make_host_vector<float>(n);
    static_assert(h_vec.rank() == 1);
    ASSERT_EQ(h_vec.extent(0), n);
  }
  {
    raft::resources handle;
    raft::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        auto d_matrix = make_device_matrix<float>(h, n, n);
        ASSERT_EQ(d_matrix.extent(0), n);
        ASSERT_EQ(d_matrix.extent(1), n);
        static_assert(d_matrix.rank() == 2);

        auto d_vec = make_device_vector<float>(h, n);
        static_assert(d_vec.rank() == 1);
        ASSERT_EQ(d_vec.extent(0), n);
      },
      raft::alloc_behavior::ARGUMENT_DRIVEN,
      n * (n + 1) * sizeof(float));
  }

  {
    raft::resources handle;
    // device scalar
    auto d_scalar = make_device_scalar<double>(handle, 17.0);
    static_assert(d_scalar.rank() == 1);
    static_assert(d_scalar.rank_dynamic() == 0);
    ASSERT_EQ(d_scalar(0), 17.0);

    auto view = d_scalar.view();
    thrust::device_vector<int32_t> status(1, 0);
    auto p_status = status.data().get();
    thrust::for_each_n(rmm::exec_policy(resource::get_cuda_stream(handle)),
                       cuda::make_counting_iterator(0),
                       1,
                       [=] __device__(auto i) {
                         if (view(i) != 17.0) { myAtomicAdd(p_status, 1); }
                       });
    check_status(p_status, resource::get_cuda_stream(handle));
  }
  {
    // host scalar
    raft::resources handle;

    auto h_scalar = make_host_scalar<double>(handle, 17.0);
    static_assert(h_scalar.rank() == 1);
    static_assert(h_scalar.rank_dynamic() == 0);
    ASSERT_EQ(h_scalar(0), 17.0);
    ASSERT_EQ(h_scalar.view()(0), 17.0);

    auto view = make_host_scalar_view(h_scalar.data_handle());
    ASSERT_EQ(view(0), 17.0);
  }

  // managed mdarray (sync allocation, no stream)
  {
    raft::resources handle;
    auto m_vec = make_managed_vector<float>(handle, 10);
    ASSERT_EQ(m_vec.extent(0), 10);
    m_vec(0) = 42.0f;
    ASSERT_EQ(m_vec(0), 42.0f);

    auto mdv = make_managed_mdspan(m_vec.data_handle(), raft::vector_extent<int>{10});
    static_assert(decltype(mdv)::accessor_type::is_managed_accessible, "Not managed mdspan");
    ASSERT_EQ(mdv.size(), 10);
  }

  // pinned mdarray (sync allocation, no stream)
  {
    raft::resources handle;
    auto p_vec = make_pinned_vector<float>(handle, 10);
    ASSERT_EQ(p_vec.extent(0), 10);
    p_vec(0) = 7.0f;
    ASSERT_EQ(p_vec(0), 7.0f);
  }

  // managed memory resource: get/set with default
  {
    raft::resources handle;
    auto ref  = raft::resource::get_managed_memory_resource_ref(handle);
    void* ptr = ref.allocate_sync(256);
    ASSERT_NE(ptr, nullptr);
    ref.deallocate_sync(ptr, 256);
  }

  // managed memory resource: set custom, allocate through mdarray
  {
    raft::resources handle;
    raft::resource::set_managed_memory_resource(handle, cuda::mr::legacy_managed_memory_resource{});
    auto m_vec = make_managed_vector<float>(handle, 10);
    m_vec(0)   = 99.0f;
    ASSERT_EQ(m_vec(0), 99.0f);
  }

  // pinned memory resource: get/set with default
  {
    raft::resources handle;
    auto ref  = raft::resource::get_pinned_memory_resource_ref(handle);
    void* ptr = ref.allocate_sync(256);
    ASSERT_NE(ptr, nullptr);
    ref.deallocate_sync(ptr, 256);
  }

  // pinned memory resource: set custom, allocate through mdarray
  {
    raft::resources handle;
    raft::resource::set_pinned_memory_resource(handle, cuda::mr::legacy_pinned_memory_resource{});
    auto p_vec = make_pinned_vector<float>(handle, 10);
    p_vec(0)   = 55.0f;
    ASSERT_EQ(p_vec(0), 55.0f);
  }

  // shared semantics: two resources objects share the same MR
  {
    raft::resources handle1;
    raft::resource::set_managed_memory_resource(handle1,
                                                cuda::mr::legacy_managed_memory_resource{});
    raft::resources handle2{handle1};
    auto ref1 = raft::resource::get_managed_memory_resource_ref(handle1);
    auto ref2 = raft::resource::get_managed_memory_resource_ref(handle2);
    ASSERT_EQ(ref1, ref2);
  }

  // get_default_host_resource: default ref is usable
  {
    auto ref  = raft::mr::get_default_host_resource();
    void* ptr = ref.allocate_sync(128);
    ASSERT_NE(ptr, nullptr);
    ref.deallocate_sync(ptr, 128);
  }

  // set_default_host_resource: install mmap, allocate through default policy
  {
    raft::mr::mmap_memory_resource mmap_mr;
    raft::mr::set_default_host_resource(mmap_mr);

    auto ref  = raft::mr::get_default_host_resource();
    void* ptr = ref.allocate_sync(4096);
    ASSERT_NE(ptr, nullptr);
    ref.deallocate_sync(ptr, 4096);

    // restore default
    raft::mr::set_default_host_resource(raft::mr::new_delete_resource());
  }
}
}  // anonymous namespace

TEST(MDArray, Factory) { test_factory_methods(); }

namespace {
template <typename T, typename Index, typename LayoutPolicy>
void check_matrix_layout(device_matrix_view<T, Index, LayoutPolicy> in)
{
  static_assert(in.rank() == 2);
  static_assert(in.is_exhaustive());

  bool constexpr kIsCContiguous = std::is_same_v<LayoutPolicy, layout_c_contiguous>;
  bool constexpr kIsFContiguous = std::is_same_v<LayoutPolicy, layout_f_contiguous>;
  // only 1 of them is true
  static_assert(kIsCContiguous || kIsFContiguous);
  static_assert(!(kIsCContiguous && kIsFContiguous));
}
}  // anonymous namespace

TEST(MDArray, FuncArg)
{
  raft::resources handle;
  {
    auto d_matrix = make_device_matrix<float>(handle, 10, 10);
    check_matrix_layout(d_matrix.view());
  }
  {
    auto d_matrix = make_device_matrix<float, int, layout_f_contiguous>(handle, 10, 10);
    check_matrix_layout(d_matrix.view());

    auto slice =
      cuda::std::submdspan(d_matrix.view(), cuda::std::tuple{2ul, 4ul}, cuda::std::tuple{2ul, 5ul});
    static_assert(slice.is_strided());
    ASSERT_EQ(slice.extent(0), 2);
    ASSERT_EQ(slice.extent(1), 3);
    // is using device_accessor mixin.
    static_assert(
      std::is_same_v<decltype(slice)::accessor_type, device_matrix_view<float>::accessor_type>);
  }
}

void test_mdspan_layout_right_padded()
{
  {
    // 5x2 example,
    constexpr int n_rows          = 2;
    constexpr int n_cols          = 5;
    constexpr int alignment       = 8;
    constexpr int alignment_bytes = sizeof(int) * alignment;

    int data_row_major[] = {
      1,
      2,
      3,
      4,
      5, /* X  X  X */
      6,
      7,
      8,
      9,
      10 /* X  X  X */
    };
    // manually aligning the above, using -1 as filler
    static constexpr int X = -1;
    int data_padded[]      = {1, 2, 3, 4, 5, X, X, X, 6, 7, 8, 9, 10, X, X, X};

    using extents_type =
      cuda::std::extents<size_t, cuda::std::dynamic_extent, cuda::std::dynamic_extent>;
    using padded_layout_row_major =
      raft::layout_right_padded_impl<detail::padding<int, alignment_bytes>::value>;
    using padded_mdspan    = cuda::std::mdspan<int, extents_type, padded_layout_row_major>;
    using row_major_mdspan = cuda::std::mdspan<int, extents_type, cuda::std::layout_right>;

    padded_layout_row_major::mapping<extents_type> layout{extents_type{n_rows, n_cols}};

    auto padded    = padded_mdspan(data_padded, layout);
    auto row_major = row_major_mdspan(data_row_major, n_rows, n_cols);

    int failures = 0;
    for (int irow = 0; irow < n_rows; ++irow) {
      for (int icol = 0; icol < n_cols; ++icol) {
        if (padded(irow, icol) != row_major(irow, icol)) { ++failures; }
      }
    }
    ASSERT_EQ(failures, 0);
  }
}

TEST(MDSpan, LayoutRightPadded) { test_mdspan_layout_right_padded(); }

void test_mdarray_padding()
{
  using extents_type = cuda::std::extents<size_t, dynamic_extent, dynamic_extent>;
  raft::resources handle;
  auto s = resource::get_cuda_stream(handle);
  {
    constexpr int rows            = 6;
    constexpr int cols            = 7;
    constexpr int alignment       = 5;
    constexpr int alignment_bytes = sizeof(int) * alignment;

    /**
     * padded device array
     */
    using padded_layout_row_major =
      raft::layout_right_padded_impl<detail::padding<float, alignment_bytes>::value>;

    using padded_mdarray_type = device_mdarray<float, extents_type, padded_layout_row_major>;
    padded_layout_row_major::mapping<extents_type> layout(extents_type(rows, cols));

    auto device_policy = padded_mdarray_type::container_policy_type{};
    static_assert(std::is_same_v<typename decltype(device_policy)::accessor_type,
                                 device_container_policy<float>>);
    padded_mdarray_type padded_device_array{handle, layout, device_policy};

    // direct access mdarray
    padded_device_array(0, 3) = 1;
    ASSERT_EQ(padded_device_array(0, 3), 1);

    // non-const access via mdspan
    auto d_view = padded_device_array.view();
    static_assert(!decltype(d_view)::accessor_type::is_host_type::value);

    thrust::device_vector<int32_t> status(1, 0);
    auto p_status = status.data().get();
    thrust::for_each_n(rmm::exec_policy(),
                       cuda::make_counting_iterator(0ul),
                       1,
                       [d_view, p_status] __device__(size_t i) {
                         if (d_view(0, 3) != 1) { myAtomicAdd(p_status, 1); }
                         d_view(0, 2) = 3;
                         if (d_view(0, 2) != 3) { myAtomicAdd(p_status, 1); }
                       });
    check_status(p_status, s);

    // const ref access via mdspan
    auto const& arr = padded_device_array;
    ASSERT_EQ(arr(0, 3), 1);
    auto const_d_view = arr.view();
    thrust::for_each_n(rmm::exec_policy(s),
                       cuda::make_counting_iterator(0ul),
                       1,
                       [const_d_view, p_status] __device__(size_t i) {
                         if (const_d_view(0, 3) != 1) { myAtomicAdd(p_status, 1); }
                       });
    check_status(p_status, s);

    // initialize with sequence
    thrust::for_each_n(
      rmm::exec_policy(s),
      cuda::make_counting_iterator(0ul),
      rows * cols,
      [d_view, rows, cols] __device__(size_t i) { d_view(i / cols, i % cols) = i; });

    // manually create span with layout
    {
      auto data_padded         = padded_device_array.data_handle();
      using padded_mdspan_type = device_mdspan<float, extents_type, padded_layout_row_major>;
      auto padded_span         = padded_mdspan_type(data_padded, layout);
      thrust::for_each_n(rmm::exec_policy(s),
                         cuda::make_counting_iterator(0ul),
                         rows * cols,
                         [padded_span, rows, cols, p_status] __device__(size_t i) {
                           if (padded_span(i / cols, i % cols) != i) myAtomicAdd(p_status, 1);
                         });
      check_status(p_status, s);
    }

    // utilities
    static_assert(padded_device_array.rank_dynamic() == 2);
    static_assert(padded_device_array.rank() == 2);
    static_assert(padded_device_array.is_unique());
    static_assert(padded_device_array.is_strided());

    static_assert(
      !std::is_nothrow_default_constructible<padded_mdarray_type>::value);  // cuda stream
    static_assert(std::is_nothrow_move_constructible<padded_mdarray_type>::value);
    static_assert(std::is_nothrow_move_assignable<padded_mdarray_type>::value);
  }
}

TEST(MDArray, Padding) { test_mdarray_padding(); }

// Test deactivated as submdspan support requires upstream changes
/*void test_submdspan_padding()
{
  using extents_type = stdex::extents<dynamic_extent, dynamic_extent>;
  auto s             = rmm::cuda_stream_default;

  {
    constexpr int rows            = 6;
    constexpr int cols            = 7;
    constexpr int alignment       = 5;
    constexpr int alignment_bytes = sizeof(int) * alignment;

    using layout_padded_general =
      stdex::layout_padded_general<float, stdex::StorageOrderType::row_major_t, alignment_bytes>;
    using padded_mdarray_type = device_mdarray<float, extents_type, layout_padded_general>;
    using padded_mdspan_type  = device_mdspan<float, extents_type, layout_padded_general>;
    layout_padded_general::mapping<extents_type> layout{extents_type{rows, cols}};

    auto device_policy = padded_mdarray_type::container_policy_type{s};
    static_assert(std::is_same_v<typename decltype(device_policy)::accessor_type,
                                 detail::device_container_policy<float>>);
    padded_mdarray_type padded_device_array{handle, layout, device_policy};

    // test status
    thrust::device_vector<int32_t> status(1, 0);
    auto p_status = status.data().get();

    // initialize with sequence
    {
      auto d_view = padded_device_array.view();
      static_assert(std::is_same_v<typename decltype(d_view)::layout_type, layout_padded_general>);
      thrust::for_each_n(
        rmm::exec_policy(s),
        cuda::make_counting_iterator(0ul),
        rows * cols,
        [d_view, rows, cols] __device__(size_t i) { d_view(i / cols, i % cols) = i; });
    }

    // get mdspan manually from raw data
    {
      auto data_padded = padded_device_array.data();
      auto padded_span = padded_mdspan_type(data_padded, layout);
      thrust::for_each_n(rmm::exec_policy(s),
                         cuda::make_counting_iterator(0ul),
                         rows * cols,
                         [padded_span, rows, cols, p_status] __device__(size_t i) {
                           if (padded_span(i / cols, i % cols) != i) myAtomicAdd(p_status, 1);
                         });
      check_status(p_status, s);
    }

    // full subspan
    {
      auto padded_span  = padded_device_array.view();
      auto subspan_full = stdex::submdspan(padded_span, stdex::full_extent, stdex::full_extent);
      thrust::for_each_n(rmm::exec_policy(s),
                         cuda::make_counting_iterator(0ul),
                         cols * rows,
                         [subspan_full, padded_span, rows, cols, p_status] __device__(size_t i) {
                           if (subspan_full(i / cols, i % cols) != padded_span(i / cols, i % cols))
                             myAtomicAdd(p_status, 1);
                         });
      check_status(p_status, s);

      // resulting submdspan should still be padded
      static_assert(
        std::is_same_v<typename decltype(subspan_full)::layout_type, layout_padded_general>);
    }

    // slicing a row
    {
      auto padded_span = padded_device_array.view();
      auto row3        = stdex::submdspan(padded_span, 3, stdex::full_extent);
      thrust::for_each_n(rmm::exec_policy(s),
                         cuda::make_counting_iterator(0ul),
                         cols,
                         [row3, padded_span, p_status] __device__(size_t i) {
                           if (row3(i) != padded_span(3, i)) myAtomicAdd(p_status, 1);
                         });
      check_status(p_status, s);

      // resulting submdspan should still be padded
      static_assert(std::is_same_v<typename decltype(row3)::layout_type, layout_padded_general>);
    }

    // slicing a column
    {
      auto padded_span = padded_device_array.view();
      auto col1        = stdex::submdspan(padded_span, stdex::full_extent, 1);
      thrust::for_each_n(rmm::exec_policy(s),
                         cuda::make_counting_iterator(0ul),
                         rows,
                         [col1, padded_span, p_status] __device__(size_t i) {
                           if (col1(i) != padded_span(i, 1)) myAtomicAdd(p_status, 1);
                         });
      check_status(p_status, s);

      // resulting submdspan is *NOT* padded anymore
      static_assert(std::is_same_v<typename decltype(col1)::layout_type, stdex::layout_stride>);
    }

    // sub-rectangle of 6x7
    {
      auto padded_span = padded_device_array.view();
      auto subspan =
        stdex::submdspan(padded_span, std::make_tuple(1ul, 4ul), std::make_tuple(2ul, 5ul));
      thrust::for_each_n(rmm::exec_policy(s),
                         cuda::make_counting_iterator(0ul),
                         (rows - 1) * (cols - 2),
                         [subspan, rows, cols, padded_span, p_status] __device__(size_t i) {
                           size_t idx = i / (cols - 2);
                           size_t idy = i % (cols - 2);
                           // elements > subspan range can be accessed as well
                           if (subspan(idx, idy) != padded_span(idx + 1, idy + 2))
                             myAtomicAdd(p_status, 1);
                         });
      check_status(p_status, s);

      // resulting submdspan is *NOT* padded anymore
      static_assert(std::is_same_v<typename decltype(subspan)::layout_type, stdex::layout_stride>);
    }

    // sub-rectangle retaining padded layout
    {
      auto padded_span = padded_device_array.view();
      auto subspan =
        stdex::submdspan(padded_span, std::make_tuple(1ul, 4ul), std::make_tuple(2ul, 5ul));
      thrust::for_each_n(rmm::exec_policy(s),
                         cuda::make_counting_iterator(0ul),
                         (rows - 1) * (cols - 2),
                         [subspan, rows, cols, padded_span, p_status] __device__(size_t i) {
                           size_t idx = i / (cols - 2);
                           size_t idy = i % (cols - 2);
                           // elements > subspan range can be accessed as well
                           if (subspan(idx, idy) != padded_span(idx + 1, idy + 2))
                             myAtomicAdd(p_status, 1);
                         });
      check_status(p_status, s);

      // resulting submdspan is *NOT* padded anymore
      static_assert(std::is_same_v<typename decltype(subspan)::layout_type, stdex::layout_stride>);
    }
  }
}

TEST(MDSpan, SubmdspanPadding) { test_submdspan_padding(); }*/

struct TestElement1 {
  int a, b;
};

void test_mdspan_padding_by_type()
{
  using extents_type = cuda::std::extents<size_t, dynamic_extent, dynamic_extent>;
  raft::resources handle;
  auto s = rmm::cuda_stream_default;

  {
    constexpr int rows            = 6;
    constexpr int cols            = 7;
    constexpr int alignment_bytes = 16;

    thrust::device_vector<int32_t> status(1, 0);
    auto p_status = status.data().get();

    // manually check strides for row major (c style) padding
    {
      using padded_layout_row_major = raft::layout_right_padded_impl<
        detail::padding<std::remove_cv_t<std::remove_reference_t<TestElement1>>,
                        alignment_bytes>::value>;

      using padded_mdarray_type =
        device_mdarray<TestElement1, extents_type, padded_layout_row_major>;
      auto device_policy = padded_mdarray_type::container_policy_type{};

      padded_layout_row_major::mapping<extents_type> layout{extents_type{rows, cols}};
      padded_mdarray_type padded_device_array{handle, layout, device_policy};
      int alignment_elements = detail::padding<TestElement1, alignment_bytes>::value;
      auto padded_span       = padded_device_array.view();
      thrust::for_each_n(
        rmm::exec_policy(s),
        cuda::make_counting_iterator(0ul),
        rows * cols,
        [rows, cols, padded_span, alignment_elements, p_status] __device__(size_t i) {
          size_t idx = i / cols;
          size_t idy = i % cols;
          if ((&(padded_span(idx, idy)) - &(padded_span(0, idy))) % alignment_elements != 0)
            myAtomicAdd(p_status, 1);
          if ((&(padded_span(idx, idy)) - &(padded_span(idx, 0))) != idy) myAtomicAdd(p_status, 1);
        });
      check_status(p_status, s);
    }

    // manually check strides for col major (f style) padding
    {
      using padded_layout_col_major = raft::layout_left_padded_impl<
        detail::padding<std::remove_cv_t<std::remove_reference_t<TestElement1>>,
                        alignment_bytes>::value>;
      using padded_mdarray_type =
        device_mdarray<TestElement1, extents_type, padded_layout_col_major>;
      auto device_policy = padded_mdarray_type::container_policy_type{};

      padded_layout_col_major::mapping<extents_type> layout{extents_type{rows, cols}};
      padded_mdarray_type padded_device_array{handle, layout, device_policy};
      int alignment_elements = detail::padding<TestElement1, alignment_bytes>::value;
      auto padded_span       = padded_device_array.view();
      thrust::for_each_n(
        rmm::exec_policy(s),
        cuda::make_counting_iterator(0ul),
        rows * cols,
        [rows, cols, padded_span, alignment_elements, p_status] __device__(size_t i) {
          size_t idx = i / cols;
          size_t idy = i % cols;
          if ((&(padded_span(idx, idy)) - &(padded_span(idx, 0))) % alignment_elements != 0)
            myAtomicAdd(p_status, 1);
          if ((&(padded_span(idx, idy)) - &(padded_span(0, idy))) != idx) myAtomicAdd(p_status, 1);
        });
      check_status(p_status, s);
    }
  }
}

TEST(MDSpan, MDSpanPaddingType) { test_mdspan_padding_by_type(); }

void test_mdspan_aligned_matrix()
{
  using extents_type = cuda::std::extents<size_t, dynamic_extent, dynamic_extent>;
  raft::resources handle;
  constexpr int rows = 2;
  constexpr int cols = 10;

  // manually aligning the above, using -1 as filler
  static constexpr int X = -1;
  long data_padded[]     = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  X, X, X, X, X, X,
                            10, 11, 12, 13, 14, 15, 16, 17, 18, 19, X, X, X, X, X, X};

  auto my_aligned_host_span =
    make_host_aligned_matrix_view<long, int, layout_right_padded<long>>(data_padded, rows, cols);

  int failures = 0;
  for (int irow = 0; irow < rows; ++irow) {
    for (int icol = 0; icol < cols; ++icol) {
      if (my_aligned_host_span(irow, icol) != irow * cols + icol) { ++failures; }
    }
  }
  ASSERT_EQ(failures, 0);

  // now work with device memory
  // use simple 1D array to allocate some space
  auto s          = rmm::cuda_stream_default;
  using extent_1d = cuda::std::extents<size_t, dynamic_extent>;
  layout_c_contiguous::mapping<extent_1d> layout_1d{extent_1d{rows * 32}};
  using mdarray_t    = device_mdarray<long, extent_1d, layout_c_contiguous>;
  auto device_policy = mdarray_t::container_policy_type{};
  mdarray_t device_array_1d{handle, layout_1d, device_policy};

  // direct access mdarray -- initialize with above data
  for (int i = 0; i < 32; ++i) {
    device_array_1d(i) = data_padded[i];
  }

  auto my_aligned_device_span =
    make_device_aligned_matrix_view<long, int, layout_right_padded<long>>(
      device_array_1d.data_handle(), rows, cols);

  thrust::device_vector<int32_t> status(1, 0);
  auto p_status = status.data().get();
  thrust::for_each_n(rmm::exec_policy(s),
                     cuda::make_counting_iterator(0ul),
                     rows * cols,
                     [rows, cols, my_aligned_device_span, p_status] __device__(size_t i) {
                       size_t idx = i / cols;
                       size_t idy = i % cols;
                       if (my_aligned_device_span(idx, idy) != i) myAtomicAdd(p_status, 1);
                     });
  check_status(p_status, s);
}

TEST(MDSpan, MDSpanAlignedMatrix) { test_mdspan_aligned_matrix(); }

namespace {
void test_mdarray_unravel()
{
  {
    uint32_t v{0};
    ASSERT_EQ(detail::native_popc(v), 0);
    ASSERT_EQ(detail::popc(v), 0);
    v = 1;
    ASSERT_EQ(detail::native_popc(v), 1);
    ASSERT_EQ(detail::popc(v), 1);
    v = 0xffffffff;
    ASSERT_EQ(detail::native_popc(v), 32);
    ASSERT_EQ(detail::popc(v), 32);
  }
  {
    uint64_t v{0};
    ASSERT_EQ(detail::native_popc(v), 0);
    ASSERT_EQ(detail::popc(v), 0);
    v = 1;
    ASSERT_EQ(detail::native_popc(v), 1);
    ASSERT_EQ(detail::popc(v), 1);
    v = 0xffffffff;
    ASSERT_EQ(detail::native_popc(v), 32);
    ASSERT_EQ(detail::popc(v), 32);
    v = 0xffffffffffffffff;
    ASSERT_EQ(detail::native_popc(v), 64);
    ASSERT_EQ(detail::popc(v), 64);
  }

  // examples from numpy unravel_index
  {
    auto coord = unravel_index(22, matrix_extent<int>{7, 6}, cuda::std::layout_right{});
    static_assert(std::tuple_size<decltype(coord)>::value == 2);
    ASSERT_EQ(std::get<0>(coord), 3);
    ASSERT_EQ(std::get<1>(coord), 4);
  }
  {
    auto coord = unravel_index(41, matrix_extent<int>{7, 6}, cuda::std::layout_right{});
    static_assert(std::tuple_size<decltype(coord)>::value == 2);
    ASSERT_EQ(std::get<0>(coord), 6);
    ASSERT_EQ(std::get<1>(coord), 5);
  }
  {
    auto coord = unravel_index(37, matrix_extent<int>{7, 6}, cuda::std::layout_right{});
    static_assert(std::tuple_size<decltype(coord)>::value == 2);
    ASSERT_EQ(std::get<0>(coord), 6);
    ASSERT_EQ(std::get<1>(coord), 1);
  }
  // assignment
  {
    auto m   = make_host_matrix<float, size_t>(7, 6);
    auto m_v = m.view();
    for (size_t i = 0; i < m.size(); ++i) {
      auto coord             = unravel_index(i, m.extents(), typename decltype(m)::layout_type{});
      std::apply(m_v, coord) = i;
    }
    for (size_t i = 0; i < m.size(); ++i) {
      auto coord = unravel_index(i, m.extents(), typename decltype(m)::layout_type{});
      ASSERT_EQ(std::apply(m_v, coord), i);
    }
  }

  {
    raft::resources handle;
    auto m   = make_device_matrix<float, size_t>(handle, 7, 6);
    auto m_v = m.view();
    thrust::for_each_n(resource::get_thrust_policy(handle),
                       cuda::make_counting_iterator(0ul),
                       m_v.size(),
                       [=] HD(size_t i) {
                         auto coord =
                           unravel_index(i, m_v.extents(), typename decltype(m_v)::layout_type{});
                         std::apply(m_v, coord) = static_cast<float>(i);
                       });
    thrust::device_vector<int32_t> status(1, 0);
    auto p_status = status.data().get();
    thrust::for_each_n(resource::get_thrust_policy(handle),
                       cuda::make_counting_iterator(0ul),
                       m_v.size(),
                       [=] __device__(size_t i) {
                         auto coord =
                           unravel_index(i, m_v.extents(), typename decltype(m_v)::layout_type{});
                         auto v = std::apply(m_v, coord);
                         if (v != static_cast<float>(i)) { raft::myAtomicAdd(p_status, 1); }
                       });
    check_status(p_status, resource::get_cuda_stream(handle));
  }
}
}  // anonymous namespace

TEST(MDArray, Unravel) { test_mdarray_unravel(); }

}  // namespace raft
