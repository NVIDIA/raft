#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

# Delete system libnccl.so to ensure the wheel is used
rm -rf /usr/lib64/libnccl*

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
LIBRAFT_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libraft raft --cuda "$RAPIDS_CUDA_VERSION")")
PYLIBRAFT_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python pylibraft raft --stable --cuda "$RAPIDS_CUDA_VERSION")")
RAFT_DASK_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python raft_dask raft --stable --cuda "$RAPIDS_CUDA_VERSION")")

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints test_python "${PIP_CONSTRAINT}"

# notes:
#
#   * echo to expand wildcard before adding `[test]` requires for pip
#   * just providing --constraint="${PIP_CONSTRAINT}" to be explicit, and because
#     that environment variable is ignored if any other --constraint are passed via the CLI
#
rapids-pip-retry install -v \
    --constraint "${PIP_CONSTRAINT}" \
    "${LIBRAFT_WHEELHOUSE}"/libraft*.whl \
    "${PYLIBRAFT_WHEELHOUSE}"/pylibraft*.whl \
    "$(echo "${RAFT_DASK_WHEELHOUSE}"/raft_dask_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]"

test_dir="python/raft-dask/raft_dask/tests"

rapids-logger "pytest raft-dask"
python -m pytest --import-mode=append ${test_dir}

rapids-logger "pytest raft-dask (ucxx only)"
python -m pytest --import-mode=append ${test_dir} --run_ucx
