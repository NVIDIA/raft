/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "unary_op.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/kvp.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/map.cuh>
#include <raft/matrix/init.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cuda/iterator>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include <gtest/gtest.h>

namespace raft {
namespace linalg {

/*
 * Padded_float is a 12 byte type that contains a single float. Two integers are
 * used for padding. It is used to test types that are not power-of-two-sized.
 */
struct padded_float {
  float value_;
  int padding1;
  int padding2;

  padded_float() = default;
  constexpr padded_float(const float& x) : value_(x), padding1(0), padding2(0) {}
  constexpr padded_float(const padded_float&)            = default;
  constexpr padded_float& operator=(const padded_float&) = default;
  constexpr float abs() const { return std::abs(value_); }
};

constexpr padded_float operator+(const padded_float& x, const padded_float& y)
{
  return padded_float(x.value_ + y.value_);
}

constexpr padded_float operator-(const padded_float& x, const padded_float& y)
{
  return padded_float(x.value_ - y.value_);
}
constexpr padded_float operator*(const padded_float& x, const padded_float& y)
{
  return padded_float(x.value_ * y.value_);
}
constexpr padded_float operator*(const padded_float& x, const int& scalar)
{
  return padded_float(scalar * x.value_);
}
constexpr bool operator==(const padded_float& x, const padded_float& y)
{
  return x.value_ == y.value_;
}

constexpr bool operator<(const padded_float& x, const padded_float& y)
{
  return x.value_ < y.value_;
}
constexpr bool operator>(const padded_float& x, const padded_float& y)
{
  return x.value_ > y.value_;
}
inline auto operator<<(std::ostream& os, const padded_float& x) -> std::ostream&
{
  os << x.value_;
  return os;
}

/*
 * KeyValuePair type aliases for testing different sizes:
 *   - KVP4:  4-byte KVP (int16_t, int16_t)
 *   - KVP8:  8-byte KVP (int, float)
 *   - KVP16: 16-byte KVP (int64_t, double)
 */
using KVP4  = raft::KeyValuePair<int16_t, int16_t>;
using KVP8  = raft::KeyValuePair<int, float>;
using KVP16 = raft::KeyValuePair<int64_t, double>;

// Type trait to detect KVP
template <typename T>
struct is_kvp : std::false_type {};
template <typename K, typename V>
struct is_kvp<raft::KeyValuePair<K, V>> : std::true_type {};

// Templated KVP add operation for any KeyValuePair type
template <typename KVPType>
struct KVPAddOp {
  KVPType scalar;
  __device__ KVPType operator()(KVPType a, KVPType b, KVPType c) const
  {
    return KVPType{
      static_cast<typename KVPType::Key>(a.key + b.key + c.key + scalar.key),
      static_cast<typename KVPType::Value>(a.value + b.value + c.value + scalar.value)};
  }
};

template <typename InType, typename IdxType, typename OutType>
void mapLaunch(const raft::resources& handle,
               OutType* out,
               const InType* in1,
               const InType* in2,
               const InType* in3,
               InType scalar,
               IdxType len)
{
  auto out_view = raft::make_device_vector_view(out, len);
  auto in1_view = raft::make_device_vector_view(in1, len);
  auto in2_view = raft::make_device_vector_view(in2, len);
  auto in3_view = raft::make_device_vector_view(in3, len);

  if constexpr (is_kvp<InType>::value) {
    map(handle, out_view, KVPAddOp<InType>{scalar}, in1_view, in2_view, in3_view);
  } else {
    map(
      handle,
      out_view,
      [=] __device__(InType a, InType b, InType c) { return a + b + c + scalar; },
      in1_view,
      in2_view,
      in3_view);
  }
}

template <typename InType, typename IdxType = int, typename OutType = InType>
struct MapInputs {
  InType tolerance;
  IdxType len;
  unsigned long long int seed;
  InType scalar;
};

template <typename KVPType>
struct ThrustKVPAdd {
  __host__ __device__ KVPType operator()(const KVPType& a, const KVPType& b) const
  {
    return KVPType{static_cast<typename KVPType::Key>(a.key + b.key),
                   static_cast<typename KVPType::Value>(a.value + b.value)};
  }
};

// Templated thrust functor for KVP scalar addition
template <typename KVPType>
struct ThrustKVPScalarAdd {
  KVPType scalar;
  __host__ __device__ KVPType operator()(const KVPType& a) const
  {
    return KVPType{static_cast<typename KVPType::Key>(a.key + scalar.key),
                   static_cast<typename KVPType::Value>(a.value + scalar.value)};
  }
};

template <typename InType, typename IdxType, typename OutType = InType>
void create_ref(OutType* out_ref,
                const InType* in1,
                const InType* in2,
                const InType* in3,
                InType scalar,
                IdxType len,
                cudaStream_t stream)
{
  if constexpr (is_kvp<InType>::value) {
    auto policy = thrust::cuda::par.on(stream);
    rmm::device_uvector<InType> tmp(len, stream);

    // tmp = in1 + in2
    thrust::transform(policy,
                      thrust::device_pointer_cast(in1),
                      thrust::device_pointer_cast(in1 + len),
                      thrust::device_pointer_cast(in2),
                      thrust::device_pointer_cast(tmp.data()),
                      ThrustKVPAdd<InType>{});

    // out_ref = tmp + in3
    thrust::transform(policy,
                      thrust::device_pointer_cast(tmp.data()),
                      thrust::device_pointer_cast(tmp.data() + len),
                      thrust::device_pointer_cast(in3),
                      thrust::device_pointer_cast(out_ref),
                      ThrustKVPAdd<InType>{});

    // out_ref = out_ref + scalar
    thrust::transform(policy,
                      thrust::device_pointer_cast(out_ref),
                      thrust::device_pointer_cast(out_ref + len),
                      thrust::device_pointer_cast(out_ref),
                      ThrustKVPScalarAdd<InType>{scalar});
  } else {
    rmm::device_uvector<InType> tmp(len, stream);
    eltwiseAdd(tmp.data(), in1, in2, len, stream);
    eltwiseAdd(out_ref, tmp.data(), in3, len, stream);
    scalarAdd(out_ref, out_ref, (OutType)scalar, len, stream);
  }
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
}

template <typename InType, typename IdxType, typename OutType = InType>
class MapTest : public ::testing::TestWithParam<MapInputs<InType, IdxType, OutType>> {
 public:
  MapTest()
    : params(::testing::TestWithParam<MapInputs<InType, IdxType, OutType>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      in1(params.len, stream),
      in2(params.len, stream),
      in3(params.len, stream),
      out_ref(params.len, stream),
      out(params.len, stream)
  {
  }

  void SetUp() override
  {
    raft::random::RngState r(params.seed);

    IdxType len = params.len;
    if constexpr (std::is_floating_point<InType>::value) {
      uniform(handle, r, in1.data(), len, InType(-1.0), InType(1.0));
      uniform(handle, r, in2.data(), len, InType(-1.0), InType(1.0));
      uniform(handle, r, in3.data(), len, InType(-1.0), InType(1.0));
    } else if constexpr (is_kvp<InType>::value) {
      using KeyType   = typename InType::Key;
      using ValueType = typename InType::Value;

      rmm::device_uvector<double> fkey1(params.len, stream);
      rmm::device_uvector<double> fkey2(params.len, stream);
      rmm::device_uvector<double> fkey3(params.len, stream);
      rmm::device_uvector<double> fval1(params.len, stream);
      rmm::device_uvector<double> fval2(params.len, stream);
      rmm::device_uvector<double> fval3(params.len, stream);
      uniform(handle, r, fkey1.data(), len, double(-100.0), double(100.0));
      uniform(handle, r, fkey2.data(), len, double(-100.0), double(100.0));
      uniform(handle, r, fkey3.data(), len, double(-100.0), double(100.0));
      uniform(handle, r, fval1.data(), len, double(-1.0), double(1.0));
      uniform(handle, r, fval2.data(), len, double(-1.0), double(1.0));
      uniform(handle, r, fval3.data(), len, double(-1.0), double(1.0));

      auto fkey1_view = raft::make_device_vector_view<const double>(fkey1.data(), fkey1.size());
      auto fkey2_view = raft::make_device_vector_view<const double>(fkey2.data(), fkey2.size());
      auto fkey3_view = raft::make_device_vector_view<const double>(fkey3.data(), fkey3.size());
      auto fval1_view = raft::make_device_vector_view<const double>(fval1.data(), fval1.size());
      auto fval2_view = raft::make_device_vector_view<const double>(fval2.data(), fval2.size());
      auto fval3_view = raft::make_device_vector_view<const double>(fval3.data(), fval3.size());
      auto in1_view   = raft::make_device_vector_view(in1.data(), in1.size());
      auto in2_view   = raft::make_device_vector_view(in2.data(), in2.size());
      auto in3_view   = raft::make_device_vector_view(in3.data(), in3.size());

      auto make_kvp = [] __device__(double k, double v) {
        return InType{static_cast<KeyType>(k), static_cast<ValueType>(v)};
      };
      raft::linalg::map(handle, in1_view, make_kvp, fkey1_view, fval1_view);
      raft::linalg::map(handle, in2_view, make_kvp, fkey2_view, fval2_view);
      raft::linalg::map(handle, in3_view, make_kvp, fkey3_view, fval3_view);
    } else {
      // First create random float arrays
      rmm::device_uvector<float> fin1(params.len, stream);
      rmm::device_uvector<float> fin2(params.len, stream);
      rmm::device_uvector<float> fin3(params.len, stream);
      uniform(handle, r, fin1.data(), len, float(-1.0), float(1.0));
      uniform(handle, r, fin2.data(), len, float(-1.0), float(1.0));
      uniform(handle, r, fin3.data(), len, float(-1.0), float(1.0));

      // Then pad them
      raft::device_resources handle{stream};
      auto fin1_view = raft::make_device_vector_view(fin1.data(), fin1.size());
      auto fin2_view = raft::make_device_vector_view(fin2.data(), fin2.size());
      auto fin3_view = raft::make_device_vector_view(fin3.data(), fin3.size());
      auto in1_view  = raft::make_device_vector_view(in1.data(), in1.size());
      auto in2_view  = raft::make_device_vector_view(in2.data(), in2.size());
      auto in3_view  = raft::make_device_vector_view(in3.data(), in3.size());

      auto add_padding = [] __device__(float a) { return padded_float(a); };
      raft::linalg::map(handle, in1_view, add_padding, raft::make_const_mdspan(fin1_view));
      raft::linalg::map(handle, in2_view, add_padding, raft::make_const_mdspan(fin2_view));
      raft::linalg::map(handle, in3_view, add_padding, raft::make_const_mdspan(fin3_view));
    }

    create_ref(out_ref.data(), in1.data(), in2.data(), in3.data(), params.scalar, len, stream);
    raft::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        mapLaunch(h, out.data(), in1.data(), in2.data(), in3.data(), params.scalar, len);
      },
      raft::alloc_behavior::NO_ALLOCATIONS);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  MapInputs<InType, IdxType, OutType> params;
  rmm::device_uvector<InType> in1, in2, in3;
  rmm::device_uvector<OutType> out_ref, out;
};

template <typename OutType, typename IdxType>
class MapOffsetTest : public ::testing::TestWithParam<MapInputs<OutType, IdxType, OutType>> {
 public:
  MapOffsetTest()
    : params(::testing::TestWithParam<MapInputs<OutType, IdxType, OutType>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      out_ref(params.len, stream),
      out(params.len, stream)
  {
  }

  // Functor for KVP map_offset test
  struct KVPScaleOp {
    OutType scalar;
    __host__ __device__ OutType operator()(IdxType idx) const
    {
      using KeyType   = typename OutType::Key;
      using ValueType = typename OutType::Value;
      return OutType{static_cast<KeyType>(static_cast<KeyType>(idx) * scalar.key),
                     static_cast<ValueType>(static_cast<ValueType>(idx) * scalar.value)};
    }
  };

 protected:
  void SetUp() override
  {
    IdxType len    = params.len;
    OutType scalar = params.scalar;

    auto out_view = raft::make_device_vector_view(out.data(), len);

    if constexpr (is_kvp<OutType>::value) {
      KVPScaleOp op{scalar};

      // Use thrust to create reference for KVP type
      auto policy = thrust::cuda::par.on(stream);
      thrust::transform(policy,
                        cuda::counting_iterator<IdxType>(0),
                        cuda::counting_iterator<IdxType>(len),
                        thrust::device_pointer_cast(out_ref.data()),
                        op);

      map_offset(handle, out_view, op);
    } else {
      naiveScale(out_ref.data(), (OutType*)nullptr, scalar, len, stream);
      map_offset(handle,
                 out_view,
                 raft::compose_op(raft::cast_op<OutType>(), raft::mul_const_op<OutType>(scalar)));
    }
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  MapInputs<OutType, IdxType, OutType> params;
  rmm::device_uvector<OutType> out_ref, out;
};

#define MAP_TEST(test_type, test_name, inputs)                       \
  typedef RAFT_DEPAREN(test_type) test_name;                         \
  TEST_P(test_name, Result)                                          \
  {                                                                  \
    ASSERT_TRUE(devArrMatch(this->out_ref.data(),                    \
                            this->out.data(),                        \
                            this->params.len,                        \
                            CompareApprox(this->params.tolerance))); \
  }                                                                  \
  INSTANTIATE_TEST_SUITE_P(MapTests, test_name, ::testing::ValuesIn(inputs))

const std::vector<MapInputs<float, int>> inputsf_i32 = {{0.000001f, 1024 * 1024, 1234ULL, 3.2}};
MAP_TEST((MapTest<float, int>), MapTestF_i32, inputsf_i32);
MAP_TEST((MapOffsetTest<float, int>), MapOffsetTestF_i32, inputsf_i32);

const std::vector<MapInputs<float, size_t>> inputsf_i64 = {{0.000001f, 1024 * 1024, 1234ULL, 9.4}};
MAP_TEST((MapTest<float, size_t>), MapTestF_i64, inputsf_i64);
MAP_TEST((MapOffsetTest<float, size_t>), MapOffsetTestF_i64, inputsf_i64);

const std::vector<MapInputs<float, int, double>> inputsf_i32_d = {
  {0.000001f, 1024 * 1024, 1234ULL, 5.9}};
MAP_TEST((MapTest<float, int, double>), MapTestF_i32_D, inputsf_i32_d);

const std::vector<MapInputs<double, int>> inputsd_i32 = {{0.00000001, 1024 * 1024, 1234ULL, 7.5}};
MAP_TEST((MapTest<double, int>), MapTestD_i32, inputsd_i32);
MAP_TEST((MapOffsetTest<double, int>), MapOffsetTestD_i32, inputsd_i32);

const std::vector<MapInputs<double, size_t>> inputsd_i64 = {
  {0.00000001, 1024 * 1024, 1234ULL, 5.2}};
MAP_TEST((MapTest<double, size_t>), MapTestD_i64, inputsd_i64);
MAP_TEST((MapOffsetTest<double, size_t>), MapOffsetTestD_i64, inputsd_i64);

// This comparison structure is necessary, because it is not straight-forward to
// add an overload of std::abs for padded_float.
struct ComparePadded {
  float eps;
  ComparePadded(float eps_) : eps(eps_) {}
  ComparePadded(padded_float eps_) : eps(eps_.value_) {}
  ComparePadded(double eps_) : eps(eps_) {}
  bool operator()(const padded_float& a, const padded_float& b) const
  {
    float diff  = (a - b).abs();
    float m     = std::max(a.abs(), b.abs());
    float ratio = diff > eps ? diff / m : diff;
    return (ratio <= eps);
  }
};

// Use PaddedComparison
#define MAP_TEST_PADDED(test_type, test_name, inputs)                \
  typedef RAFT_DEPAREN(test_type) test_name;                         \
  TEST_P(test_name, Result)                                          \
  {                                                                  \
    ASSERT_TRUE(devArrMatch(this->out_ref.data(),                    \
                            this->out.data(),                        \
                            this->params.len,                        \
                            ComparePadded(this->params.tolerance))); \
  }                                                                  \
  INSTANTIATE_TEST_SUITE_P(MapTests, test_name, ::testing::ValuesIn(inputs))

const std::vector<MapInputs<padded_float, size_t>> inputsd_padded_float = {
  {0.00000001, 1024 * 1024, 1234ULL, 5.2}};
MAP_TEST_PADDED((MapTest<padded_float, size_t>), MapTestD_padded_float, inputsd_padded_float);
MAP_TEST_PADDED((MapOffsetTest<padded_float, size_t>),
                MapOffsetTestD_padded_float,
                inputsd_padded_float);

template <typename KVPType>
struct CompareKVP {
  double eps;
  CompareKVP(double eps_) : eps(eps_) {}
  bool operator()(const KVPType& a, const KVPType& b) const
  {
    if (a.key != b.key) return false;
    double diff = std::abs(static_cast<double>(a.value) - static_cast<double>(b.value));
    double m =
      std::max(std::abs(static_cast<double>(a.value)), std::abs(static_cast<double>(b.value)));
    double ratio = diff > eps ? diff / m : diff;
    return (ratio <= eps);
  }
};

#define MAP_TEST_KVP(test_type, test_name, kvp_type, inputs)                                 \
  typedef RAFT_DEPAREN(test_type) test_name;                                                 \
  TEST_P(test_name, Result)                                                                  \
  {                                                                                          \
    ASSERT_TRUE(                                                                             \
      devArrMatch(this->out_ref.data(),                                                      \
                  this->out.data(),                                                          \
                  this->params.len,                                                          \
                  CompareKVP<kvp_type>(static_cast<double>(this->params.tolerance.value)))); \
  }                                                                                          \
  INSTANTIATE_TEST_SUITE_P(MapTests, test_name, ::testing::ValuesIn(inputs))

const std::vector<MapInputs<KVP4, int>> inputs_kvp4_i32 = {
  {KVP4{0, 0}, 1024, 1234ULL, KVP4{10, 3}}};
MAP_TEST_KVP((MapTest<KVP4, int>), MapTestKVP4_i32, KVP4, inputs_kvp4_i32);
MAP_TEST_KVP((MapOffsetTest<KVP4, int>), MapOffsetTestKVP4_i32, KVP4, inputs_kvp4_i32);

const std::vector<MapInputs<KVP8, int>> inputs_kvp8_i32 = {
  {KVP8{0, 0.000001f}, 1024 * 1024, 1234ULL, KVP8{10, 1.5f}}};
MAP_TEST_KVP((MapTest<KVP8, int>), MapTestKVP8_i32, KVP8, inputs_kvp8_i32);
MAP_TEST_KVP((MapOffsetTest<KVP8, int>), MapOffsetTestKVP8_i32, KVP8, inputs_kvp8_i32);

const std::vector<MapInputs<KVP8, int64_t>> inputs_kvp8_i64 = {
  {KVP8{0, 0.000001f}, 1024 * 1024, 1234ULL, KVP8{5, 2.3f}}};
MAP_TEST_KVP((MapTest<KVP8, int64_t>), MapTestKVP8_i64, KVP8, inputs_kvp8_i64);
MAP_TEST_KVP((MapOffsetTest<KVP8, int64_t>), MapOffsetTestKVP8_i64, KVP8, inputs_kvp8_i64);

// 16-byte K  VP tests (int64_t, double)
const std::vector<MapInputs<KVP16, int>> inputs_kvp16_i32 = {
  {KVP16{0, 0.00000001}, 1024 * 1024, 1234ULL, KVP16{10, 1.5}}};
MAP_TEST_KVP((MapTest<KVP16, int>), MapTestKVP16_i32, KVP16, inputs_kvp16_i32);
MAP_TEST_KVP((MapOffsetTest<KVP16, int>), MapOffsetTestKVP16_i32, KVP16, inputs_kvp16_i32);

}  // namespace linalg
}  // namespace raft
