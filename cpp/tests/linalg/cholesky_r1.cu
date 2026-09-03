/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cusolver_dn_handle.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/cholesky_r1_update.cuh>
#include <raft/linalg/detail/cusolver_wrappers.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <sstream>
#include <vector>
namespace raft {
namespace linalg {

template <typename math_t>
class CholeskyR1Test : public ::testing::Test {
 protected:
  CholeskyR1Test()
    : G(n_rows * n_rows, resource::get_cuda_stream(handle)),
      L(n_rows * n_rows, resource::get_cuda_stream(handle)),
      L_exp(n_rows * n_rows, resource::get_cuda_stream(handle)),
      devInfo(resource::get_cuda_stream(handle)),
      workspace(0, resource::get_cuda_stream(handle))
  {
    raft::update_device(G.data(), G_host, n_rows * n_rows, resource::get_cuda_stream(handle));

    // Allocate workspace
    solver_handle = resource::get_cusolver_dn_handle(handle);
    // TODO: Call from public API when ready
    RAFT_CUSOLVER_TRY(raft::linalg::detail::cusolverDnpotrf_bufferSize(
      solver_handle, CUBLAS_FILL_MODE_LOWER, n_rows, L.data(), n_rows, &Lwork));
    int n_bytes = 0;
    // Initializing in CUBLAS_FILL_MODE_LOWER, because that has larger workspace
    // requirements.
    raft::linalg::choleskyRank1Update(handle,
                                      L.data(),
                                      n_rows,
                                      n_rows,
                                      nullptr,
                                      &n_bytes,
                                      CUBLAS_FILL_MODE_LOWER,
                                      resource::get_cuda_stream(handle).get());
    Lwork = std::max(Lwork * sizeof(math_t), (size_t)n_bytes);
    workspace.resize(Lwork, resource::get_cuda_stream(handle));
  }

  void testR1Update()
  {
    int n = n_rows * n_rows;
    std::vector<cublasFillMode_t> fillmode{CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER};
    for (auto uplo : fillmode) {
      raft::copy(L.data(), G.data(), n, resource::get_cuda_stream(handle));
      for (int rank = 1; rank <= n_rows; rank++) {
        std::stringstream ss;
        ss << "Rank " << rank << ((uplo == CUBLAS_FILL_MODE_LOWER) ? ", lower" : ", upper");
        SCOPED_TRACE(ss.str());

        // Expected solution using Cholesky factorization from scratch
        raft::copy(L_exp.data(), G.data(), n, resource::get_cuda_stream(handle));
        // TODO: Call from public API when ready
        RAFT_CUSOLVER_TRY(
          raft::linalg::detail::cusolverDnpotrf(solver_handle,
                                                uplo,
                                                rank,
                                                L_exp.data(),
                                                n_rows,
                                                (math_t*)workspace.data(),
                                                Lwork,
                                                devInfo.data(),
                                                resource::get_cuda_stream(handle).get()));

        // Incremental Cholesky factorization using rank one updates.
        raft::linalg::choleskyRank1Update(handle,
                                          L.data(),
                                          rank,
                                          n_rows,
                                          workspace.data(),
                                          &Lwork,
                                          uplo,
                                          resource::get_cuda_stream(handle).get());

        ASSERT_TRUE(raft::devArrMatch(L_exp.data(),
                                      L.data(),
                                      n_rows * rank,
                                      raft::CompareApprox<math_t>(3e-3),
                                      resource::get_cuda_stream(handle).get()));
      }
    }
  }

  void testR1Error()
  {
    raft::update_device(G.data(), G2_host, 4, resource::get_cuda_stream(handle));
    std::vector<cublasFillMode_t> fillmode{CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER};
    for (auto uplo : fillmode) {
      raft::copy(L.data(), G.data(), 4, resource::get_cuda_stream(handle));
      ASSERT_NO_THROW(raft::linalg::choleskyRank1Update(handle,
                                                        L.data(),
                                                        1,
                                                        2,
                                                        workspace.data(),
                                                        &Lwork,
                                                        uplo,
                                                        resource::get_cuda_stream(handle).get()));
      ASSERT_THROW(raft::linalg::choleskyRank1Update(handle,
                                                     L.data(),
                                                     2,
                                                     2,
                                                     workspace.data(),
                                                     &Lwork,
                                                     uplo,
                                                     resource::get_cuda_stream(handle).get()),
                   raft::exception);

      math_t eps = std::numeric_limits<math_t>::epsilon();
      ASSERT_NO_THROW(raft::linalg::choleskyRank1Update(handle,
                                                        L.data(),
                                                        2,
                                                        2,
                                                        workspace.data(),
                                                        &Lwork,
                                                        uplo,
                                                        resource::get_cuda_stream(handle).get(),
                                                        eps));
    }
  }

  raft::resources handle;
  cudaStream_t stream;

  cusolverDnHandle_t solver_handle;

  int n_rows = 4;
  int Lwork;
  math_t G_host[16] =  // clang-format off
     {107.,  1393.,  1141.,  91.,
      1393., 21132., 15689., 9539.,
      1141., 15689., 13103., 2889.,
      91.,   9539.,  2889.,  23649.};
                       // clang-format on

  math_t G2_host[4] = {3, 4, 2, 1};

  rmm::device_scalar<int> devInfo;
  rmm::device_uvector<math_t> G;
  rmm::device_uvector<math_t> L_exp;
  rmm::device_uvector<math_t> L;
  rmm::device_uvector<char> workspace;
};

typedef ::testing::Types<float, double> FloatTypes;

TYPED_TEST_CASE(CholeskyR1Test, FloatTypes);

TYPED_TEST(CholeskyR1Test, update) { this->testR1Update(); }
TYPED_TEST(CholeskyR1Test, throwError) { this->testR1Error(); }

};  // namespace linalg
};  // namespace raft
