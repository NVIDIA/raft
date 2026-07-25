#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

LIBRAFT_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libraft raft --cuda "$RAPIDS_CUDA_VERSION")")
PYLIBRAFT_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python pylibraft raft --stable --cuda "$RAPIDS_CUDA_VERSION")")

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints test_python "${PIP_CONSTRAINT}"

# notes:
#
#   * echo to expand wildcard before adding `[test]` requires for pip
#   * just providing --constraint="${PIP_CONSTRAINT}" to be explicit, and because
#     that environment variable is ignored if any other --constraint are passed via the CLI
#
rapids-pip-retry install \
    --constraint "${PIP_CONSTRAINT}" \
    "${LIBRAFT_WHEELHOUSE}"/libraft*.whl \
    "$(echo "${PYLIBRAFT_WHEELHOUSE}"/pylibraft*.whl)[test]"

python -m pytest ./python/pylibraft/pylibraft/tests
