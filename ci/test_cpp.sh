#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking test_cpp.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libraft raft --cuda "$RAPIDS_CUDA_VERSION")")
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

rapids-logger "Generate C++ testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_cpp \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" \
  --prepend-channel "${CPP_CHANNEL}" \
  | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "TEMP: WSL diagnostics"
uname -a
nproc
free -m
cat /sys/fs/cgroup/memory.max || true
cat /proc/sys/vm/overcommit_memory || true
pushd "${CONDA_PREFIX}/bin/gtests/libraft"
rapids-logger "TEMP: InterruptibleOpenMP x50"
./CORE_TEST --gtest_filter='Raft.InterruptibleOpenMP' --gtest_repeat=50 2>&1 | grep -E "FAILED  \]|Which is|OK \]" | sort | uniq -c || true
rapids-logger "TEMP: select_k LargeSize alone"
(while sleep 2; do echo "MEMMON $(date +%T) $(free -m | sed -n 2p)"; done) &
MEMMON_PID=$!
./MATRIX_SELECT_TEST --gtest_filter='SelectK/ReferencedRandomDoubleInt.LargeSize/4*:SelectK/ReferencedRandomDoubleInt.LargeSize/5*' && echo "select_k exit=0" || echo "select_k exit=$?"
kill "${MEMMON_PID}" || true
dmesg 2>&1 | tail -20 || true
popd
if [[ "${RAFT_WSL_DIAG_ONLY:-1}" == "1" ]]; then
    exit 1
fi

export GTEST_OUTPUT=xml:${RAPIDS_TESTS_DIR}/
# Run libraft gtests from libraft-tests package
./ci/run_ctests.sh -j8 && EXITCODE=$? || EXITCODE=$?;

rapids-logger "Test script exiting with value: $EXITCODE"
exit "${EXITCODE}"
