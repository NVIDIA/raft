# raft 26.06.00 (3 Jun 2026)

### 🚨 Breaking Changes
* Unify memory resources by @achirkin in https://github.com/rapidsai/raft/pull/2968
* Migrate RMM usage to CCCL MR design by @bdice in https://github.com/rapidsai/raft/pull/2996
* Predictable raft::resources by @achirkin in https://github.com/rapidsai/raft/pull/3005
### 🐛 Bug Fixes
* Fix gbench 1.9.5 by @aamijar in https://github.com/rapidsai/raft/pull/3000
* Fix eigDC version check for syevd by @lowener in https://github.com/rapidsai/raft/pull/3012
* Fix off-by-one `num_segments` in `matrix::sort_cols_per_row` by @viclafargue in https://github.com/rapidsai/raft/pull/3010
* Revert "Predictable raft::resources (#3005)" by @divyegala in https://github.com/rapidsai/raft/pull/3027
### 📖 Documentation
* Docs improvements by @aamijar in https://github.com/rapidsai/raft/pull/2993
### 🚀 New Features
* Tracking memory resources by @achirkin in https://github.com/rapidsai/raft/pull/2973
* Add Randomized SVDs by @Intron7 in https://github.com/rapidsai/raft/pull/2999
### 🛠️ Improvements
* chore: bump `codespell` version for Python 3.14+ compatibility by @gforsyth in https://github.com/rapidsai/raft/pull/2989
* pre-commit: consolidate duplicate entries, bump some versions by @jameslamb in https://github.com/rapidsai/raft/pull/2991
* update pip devcontainers' base image tags by @trxcllnt in https://github.com/rapidsai/raft/pull/2995
* Update to clang 20.1.8 by @bdice in https://github.com/rapidsai/raft/pull/3001
* fix(ci): resolve all zizmor findings and add zizmor pre-commit checks by @gforsyth in https://github.com/rapidsai/raft/pull/3009
* [REVIEW] Drop extra copy in `CudaRuntimeError` by @jakirkham in https://github.com/rapidsai/raft/pull/3007
* Use `token.rapids.nvidia.com` when issuing S3 bucket creds in devcontainers by @trxcllnt in https://github.com/rapidsai/raft/pull/3008
* Use cudaStream_t instead of cuda_stream_view in pylibraft Cython by @vyasr in https://github.com/rapidsai/raft/pull/3011
* Fix symbol export by @vyasr in https://github.com/rapidsai/raft/pull/3006
* Build and test with CUDA 13.2.0 by @bdice in https://github.com/rapidsai/raft/pull/3013
* fix(ci): add explicit `actions: write` permission for `telemetry-summarize`
 by @gforsyth in https://github.com/rapidsai/raft/pull/3014
* Expose numpy_serializer helpers as public APIs by @julianmi in https://github.com/rapidsai/raft/pull/3003
* Restore resource_cast guard for RAFT memory pool by @bdice in https://github.com/rapidsai/raft/pull/3020
* skip CuPy 14.1.0 by @jameslamb in https://github.com/rapidsai/raft/pull/3045

## New Contributors
* @julianmi made their first contribution in https://github.com/rapidsai/raft/pull/3003
* @Intron7 made their first contribution in https://github.com/rapidsai/raft/pull/2999

**Full Changelog**: https://github.com/rapidsai/raft/compare/v26.06.00a...release/26.06

# raft 26.04.00 (8 Apr 2026)

### 🚨 Breaking Changes
* Rename lanczos by @aamijar in https://github.com/rapidsai/raft/pull/2936
* Remove deprecated headers by @aamijar in https://github.com/rapidsai/raft/pull/2939
### 🐛 Bug Fixes
* Disable lanczos SM tests by @aamijar in https://github.com/rapidsai/raft/pull/2941
* Add `binary_mult` and `const` fixes for `ratio` and `sqrt` apis by @aamijar in https://github.com/rapidsai/raft/pull/2940
* Temporarily increase `max_days_without_success` by @aamijar in https://github.com/rapidsai/raft/pull/2950
* Remove "requires" keyword from raft::linalg::map by @tarang-jain in https://github.com/rapidsai/raft/pull/2949
* Add nvjitlink to cuda-toolkit pip extras for cusparse compatibility by @bdice in https://github.com/rapidsai/raft/pull/2948
* Fix cub missing header by @aamijar in https://github.com/rapidsai/raft/pull/2957
* Fix cub header by @aamijar in https://github.com/rapidsai/raft/pull/2959
* [REVIEW] Fix l_offset overflow in select_k kernels for 32-bit IdxT by @yan-zaretskiy in https://github.com/rapidsai/raft/pull/2974
* Use the new reference-based RMM API to set RMM pools in `device_resources_snmg` by @viclafargue in https://github.com/rapidsai/raft/pull/2972
### 📖 Documentation
* Raft docs cleanup by @aamijar in https://github.com/rapidsai/raft/pull/2935
### 🚀 New Features
* [FEA] Support raft::KeyValuePair Inputs in linalg::map by @tarang-jain in https://github.com/rapidsai/raft/pull/2913
* Backport memory resources PRs by @achirkin in https://github.com/rapidsai/raft/pull/2983
* Move PCA and TSVD from cuml to raft by @aamijar in https://github.com/rapidsai/raft/pull/2952
### 🛠️ Improvements
* Replace `thrust::tuple` with `cuda::std::tuple` by @miscco in https://github.com/rapidsai/raft/pull/2928
* Use verify-hardcoded-version pre-commit hook by @KyleFromNVIDIA in https://github.com/rapidsai/raft/pull/2931
* expand CI-skipping logic, other small build changes by @jameslamb in https://github.com/rapidsai/raft/pull/2933
* remove pip.conf migration code in CI scripts, update CI-skipping rules by @jameslamb in https://github.com/rapidsai/raft/pull/2938
* Drop Python 3.10 support by @gforsyth in https://github.com/rapidsai/raft/pull/2932
* Don't hard-code branch names and UCXX version by @KyleFromNVIDIA in https://github.com/rapidsai/raft/pull/2934
* refactor: build wheels and conda packages using Python limited API by @gforsyth in https://github.com/rapidsai/raft/pull/2943
* remove gitutils by @jameslamb in https://github.com/rapidsai/raft/pull/2946
* Use GHA id-token for `sccache-dist` auth token by @trxcllnt in https://github.com/rapidsai/raft/pull/2947
* Use Specific CCCL Includes by @divyegala in https://github.com/rapidsai/raft/pull/2953
* Drop uses of `thrust/functional.h` by @miscco in https://github.com/rapidsai/raft/pull/2954
* check-nightly-ci: update to new version by @jameslamb in https://github.com/rapidsai/raft/pull/2956
* Drop `thrust::constant_iterator` in favor of `cuda::constant_iterator` by @miscco in https://github.com/rapidsai/raft/pull/2955
* Drop `thrust::counting_iterator` in favor of `cuda::counting_iterator` by @miscco in https://github.com/rapidsai/raft/pull/2958
* check-nightly-ci: remove testing config by @jameslamb in https://github.com/rapidsai/raft/pull/2960
* Revert `max_days_without_success` by @aamijar in https://github.com/rapidsai/raft/pull/2962
* refactor(limited api): add explicit `wheel.py-api` to `pyproject.toml`
 by @gforsyth in https://github.com/rapidsai/raft/pull/2967
* Add support for Python 3.14 by @gforsyth in https://github.com/rapidsai/raft/pull/2966
* Drop uses of `thrust::discard_iterator` in favor of `cuda::discard_iterator` by @miscco in https://github.com/rapidsai/raft/pull/2963
* Update Cython lower bound pin to 3.2.2 by @vyasr in https://github.com/rapidsai/raft/pull/2969
* Remove pytest upper bound pin by @vyasr in https://github.com/rapidsai/raft/pull/2970
* add no_libucx matrix option in dependencies.yaml by @trxcllnt in https://github.com/rapidsai/raft/pull/2975
* fix verify-hardcoded-versions issues by @jameslamb in https://github.com/rapidsai/raft/pull/2980
* build wheels with CUDA 13.0.x, test wheels against mix of CTK versions by @jameslamb in https://github.com/rapidsai/raft/pull/2971

## New Contributors
* @yan-zaretskiy made their first contribution in https://github.com/rapidsai/raft/pull/2974

**Full Changelog**: https://github.com/rapidsai/raft/compare/v26.04.00a...release/26.04

# raft 26.02.00 (4 Feb 2026)

### 🚨 Breaking Changes
* Use CCCL's mdspan implementation by @bdice in https://github.com/rapidsai/raft/pull/2836
* Default to static linking of libcudart by @bdice in https://github.com/rapidsai/raft/pull/2890
* Remove `neighbors/`, `cluster/`, `distance/`, `spatial/`, `sparse/neighbors/` apis by @aamijar in https://github.com/rapidsai/raft/pull/2885
* Remove cutlass and cuco dependencies by @divyegala in https://github.com/rapidsai/raft/pull/2916
### 🐛 Bug Fixes
* Include `<thrust/for_each.h>` where it is used by @bdice in https://github.com/rapidsai/raft/pull/2883
* Include CTest module in CMakeLists.txt by @bdice in https://github.com/rapidsai/raft/pull/2895
* Fix Lanczos Determinism by @aamijar in https://github.com/rapidsai/raft/pull/2894
* Change compile-time assertion to runtime assertion on is_strided by @bdice in https://github.com/rapidsai/raft/pull/2909
* Set memory pool through RMM by @viclafargue in https://github.com/rapidsai/raft/pull/2866
### 📖 Documentation
* New readme image by @aamijar in https://github.com/rapidsai/raft/pull/2907
* Readme improvements by @aamijar in https://github.com/rapidsai/raft/pull/2906
### 🚀 New Features
* Tile Policy for Uint8 Input (Pairwise) by @tarang-jain in https://github.com/rapidsai/raft/pull/2770
* Add copy_vectorized to RAFT by @lowener in https://github.com/rapidsai/raft/pull/2900
### 🛠️ Improvements
* Use strict priority in CI conda tests by @bdice in https://github.com/rapidsai/raft/pull/2879
* Use strict priority in CI conda tests by @bdice in https://github.com/rapidsai/raft/pull/2884
* Remove alpha specs from non-RAPIDS dependencies by @bdice in https://github.com/rapidsai/raft/pull/2886
* Enable merge barriers by @KyleFromNVIDIA in https://github.com/rapidsai/raft/pull/2889
* Fix is_exhaustive, no longer constexpr by @bdice in https://github.com/rapidsai/raft/pull/2888
* Add devcontainer fallback for C++ test location by @bdice in https://github.com/rapidsai/raft/pull/2893
* `eigsh` optional seed by @aamijar in https://github.com/rapidsai/raft/pull/2899
* Empty commit to trigger a build by @bdice in https://github.com/rapidsai/raft/pull/2904
* Update to C++20 by @divyegala in https://github.com/rapidsai/raft/pull/2908
* Use SPDX license identifiers in pyproject.toml, bump build dependency floors by @jameslamb in https://github.com/rapidsai/raft/pull/2910
* Remove `neighbors/detail/faiss_select` by @aamijar in https://github.com/rapidsai/raft/pull/2902
* Remove `sparse/distance` by @aamijar in https://github.com/rapidsai/raft/pull/2905
* Add CUDA 13.1 support by @bdice in https://github.com/rapidsai/raft/pull/2896
* Fix CCCL 3.2 mdspan constexpr issues by @bdice in https://github.com/rapidsai/raft/pull/2911
* build and test against CUDA 13.1.0 by @jameslamb in https://github.com/rapidsai/raft/pull/2912
* Laplacian Kernel for COO inputs by @aamijar in https://github.com/rapidsai/raft/pull/2891
* Empty commit to trigger a build by @jameslamb in https://github.com/rapidsai/raft/pull/2919
* Use main shared-workflows branch by @jameslamb in https://github.com/rapidsai/raft/pull/2921
* Fix update-version.sh incorrectly replacing main() function names by @AyodeAwe in https://github.com/rapidsai/raft/pull/2923
* Lanczos remove dead code by @aamijar in https://github.com/rapidsai/raft/pull/2918
* wheel builds: react to changes in pip's handling of build constraints by @mmccarty in https://github.com/rapidsai/raft/pull/2927
* fix(build): build package on merge to `release/*` branch by @gforsyth in https://github.com/rapidsai/raft/pull/2929

## New Contributors
* @mmccarty made their first contribution in https://github.com/rapidsai/raft/pull/2927

**Full Changelog**: https://github.com/rapidsai/raft/compare/v26.02.00a...release/26.02

# raft 25.12.00 (10 Dec 2025)

### 🚨 Breaking Changes
* More consistent container policies & host memory resource by @achirkin in https://github.com/rapidsai/raft/pull/2835
* Require CUDA 12.2+ by @jakirkham in https://github.com/rapidsai/raft/pull/2850
### 🐛 Bug Fixes
* Correct tagging in the `irecv` function of the STD communicator by @viclafargue in https://github.com/rapidsai/raft/pull/2829
* Fix copyright hook file exclusion by @KyleFromNVIDIA in https://github.com/rapidsai/raft/pull/2840
* Properly guard usage of openmp function calls by @robertmaynard in https://github.com/rapidsai/raft/pull/2839
* Fix reduce mdspan API by @lowener in https://github.com/rapidsai/raft/pull/2853
* Fix for STD comm waitall function by @viclafargue in https://github.com/rapidsai/raft/pull/2852
* Pin Cython pre-3.2.0 and PyTest pre-9 by @jakirkham in https://github.com/rapidsai/raft/pull/2864
* refactored update-version.sh to handle new branching strategy by @rockhowse in https://github.com/rapidsai/raft/pull/2863
* Fix laplacian scaling coefficients by @aamijar in https://github.com/rapidsai/raft/pull/2871
* Revert "Remove Deprecated API (#2813)" by @csadorf in https://github.com/rapidsai/raft/pull/2881
### 📖 Documentation
* Use current system architecture in conda environment creation command by @bdice in https://github.com/rapidsai/raft/pull/2862
### 🚀 New Features
* BENCH_PRIMS: convenience reporting of benchmark parameters and read throughput by @achirkin in https://github.com/rapidsai/raft/pull/2824
### 🛠️ Improvements
* Update to rapids-logger 0.2 by @bdice in https://github.com/rapidsai/raft/pull/2828
* Enable `sccache-dist` connection pool by @trxcllnt in https://github.com/rapidsai/raft/pull/2837
* Use main in RAPIDS_BRANCH by @bdice in https://github.com/rapidsai/raft/pull/2842
* Use main shared-workflows branch by @bdice in https://github.com/rapidsai/raft/pull/2844
* Use SPDX for all copyright headers by @KyleFromNVIDIA in https://github.com/rapidsai/raft/pull/2845
* Use ruff-check, ruff-format instead of black, flake8, isort by @KyleFromNVIDIA in https://github.com/rapidsai/raft/pull/2855
* Remove shims for CCCL < 3.1 compatibility by @bdice in https://github.com/rapidsai/raft/pull/2858
* Always convert warnings to errors by @jakirkham in https://github.com/rapidsai/raft/pull/2857
* Lanczos Solver with COO input and cusparse wrappers by @aamijar in https://github.com/rapidsai/raft/pull/2851
* COO support in sparse matrix utilities by @aamijar in https://github.com/rapidsai/raft/pull/2861
* Update RMM includes from `<rmm/mr/device/*>` to `<rmm/mr/*>` by @bdice in https://github.com/rapidsai/raft/pull/2867
* Use `sccache-dist` build cluster for conda and wheel builds by @trxcllnt in https://github.com/rapidsai/raft/pull/2859
* Remove Deprecated API by @jnke2016 in https://github.com/rapidsai/raft/pull/2813

## New Contributors
* @rockhowse made their first contribution in https://github.com/rapidsai/raft/pull/2863

**Full Changelog**: https://github.com/rapidsai/raft/compare/v25.12.00a...release/25.12

# raft 25.10.00 (8 Oct 2025)

## 🐛 Bug Fixes

- Workaround for an illegal memory access on SM 120 devices ([#2821](https://github.com/rapidsai/raft/pull/2821)) [@achirkin](https://github.com/achirkin)
- Fix sparse select_k: don&#39;t write beyond min(input_len, k) ([#2814](https://github.com/rapidsai/raft/pull/2814)) [@achirkin](https://github.com/achirkin)
- [BUG] Fix compilation error in matrix/detail/gather.cuh ([#2811](https://github.com/rapidsai/raft/pull/2811)) [@enp1s0](https://github.com/enp1s0)
- Fix select_k for negative bfloat16 ([#2799](https://github.com/rapidsai/raft/pull/2799)) [@apivovarov](https://github.com/apivovarov)
- Fix index types for coo kernels ([#2793](https://github.com/rapidsai/raft/pull/2793)) [@aamijar](https://github.com/aamijar)
- Fix the GEMM pointer mode setting ([#2777](https://github.com/rapidsai/raft/pull/2777)) [@achirkin](https://github.com/achirkin)
- Fix `host_vector_policy` issue ([#2739](https://github.com/rapidsai/raft/pull/2739)) [@viclafargue](https://github.com/viclafargue)

## 📖 Documentation

- Fix UCX-Py mention to UCXX in docstring ([#2804](https://github.com/rapidsai/raft/pull/2804)) [@pentschev](https://github.com/pentschev)

## 🚀 New Features

- Update cutlass to a version that supports CUDA 13 ([#2774](https://github.com/rapidsai/raft/pull/2774)) [@robertmaynard](https://github.com/robertmaynard)

## 🛠️ Improvements

- Fix missed deps in `update-version.sh` ([#2826](https://github.com/rapidsai/raft/pull/2826)) [@AyodeAwe](https://github.com/AyodeAwe)
- Empty commit to trigger a build ([#2816](https://github.com/rapidsai/raft/pull/2816)) [@msarahan](https://github.com/msarahan)
- Make warpsort kernels use the IEEE 754 bit representation for ordering ([#2807](https://github.com/rapidsai/raft/pull/2807)) [@achirkin](https://github.com/achirkin)
- Configure repo for automatic release notes generation ([#2806](https://github.com/rapidsai/raft/pull/2806)) [@AyodeAwe](https://github.com/AyodeAwe)
- Support &lt; 2 element arrays in `rand_index`/`adjusted_rand_index` ([#2805](https://github.com/rapidsai/raft/pull/2805)) [@jcrist](https://github.com/jcrist)
- update dependencies: use cuda-toolkit wheels ([#2802](https://github.com/rapidsai/raft/pull/2802)) [@jameslamb](https://github.com/jameslamb)
- Use branch-25.10 again ([#2800](https://github.com/rapidsai/raft/pull/2800)) [@jameslamb](https://github.com/jameslamb)
- Remove CMake find UCX package ([#2798](https://github.com/rapidsai/raft/pull/2798)) [@pentschev](https://github.com/pentschev)
- use dask-cuda[cu12, cu13] extras for wheel dependencies ([#2797](https://github.com/rapidsai/raft/pull/2797)) [@jameslamb](https://github.com/jameslamb)
- Remove UCX-Py ([#2791](https://github.com/rapidsai/raft/pull/2791)) [@pentschev](https://github.com/pentschev)
- Update rapids-dependency-file-generator ([#2790](https://github.com/rapidsai/raft/pull/2790)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Build and test with CUDA 13.0.0 ([#2787](https://github.com/rapidsai/raft/pull/2787)) [@jameslamb](https://github.com/jameslamb)
- Fix template arg passing in `adjusted_rand_index` ([#2785](https://github.com/rapidsai/raft/pull/2785)) [@jinsolp](https://github.com/jinsolp)
- Use build cluster in devcontainers ([#2781](https://github.com/rapidsai/raft/pull/2781)) [@trxcllnt](https://github.com/trxcllnt)
- Use rapids_cuda_enable_fatbin_compression ([#2780](https://github.com/rapidsai/raft/pull/2780)) [@robertmaynard](https://github.com/robertmaynard)
- Increase Dask tests verbosity in CI ([#2779](https://github.com/rapidsai/raft/pull/2779)) [@pentschev](https://github.com/pentschev)
- Update rapids_config to handle user defined branch name ([#2778](https://github.com/rapidsai/raft/pull/2778)) [@robertmaynard](https://github.com/robertmaynard)
- [REVIEW] Fix: skip default_allocation_limit() if unnecessary ([#2775](https://github.com/rapidsai/raft/pull/2775)) [@i-Pear](https://github.com/i-Pear)
- Update rapids-build-backend to 0.4.1 ([#2773](https://github.com/rapidsai/raft/pull/2773)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- ci(labeler): update labeler action to [@v5 ([#2772](https://github.com/rapidsai/raft/pull/2772)) @gforsyth](https://github.com/v5 ([#2772](https://github.com/rapidsai/raft/pull/2772)) @gforsyth)
- Register bfloat16/bfloat162 in util/vectorized.cuh ([#2769](https://github.com/rapidsai/raft/pull/2769)) [@apivovarov](https://github.com/apivovarov)
- Use mdspan::index_type to Only Instantiate Specific Kernels ([#2767](https://github.com/rapidsai/raft/pull/2767)) [@tarang-jain](https://github.com/tarang-jain)
- Allow latest OS in devcontainers ([#2759](https://github.com/rapidsai/raft/pull/2759)) [@bdice](https://github.com/bdice)
- Update build infra to support new branching strategy ([#2751](https://github.com/rapidsai/raft/pull/2751)) [@robertmaynard](https://github.com/robertmaynard)
- Use GCC 14 in conda builds. ([#2708](https://github.com/rapidsai/raft/pull/2708)) [@vyasr](https://github.com/vyasr)

# raft 25.08.00 (6 Aug 2025)

## 🚨 Breaking Changes

- `MatrixLinewiseOp` compile-time-invocation ([#2701](https://github.com/rapidsai/raft/pull/2701)) [@aamijar](https://github.com/aamijar)
- Remove CUDA 11 from dependencies.yaml ([#2695](https://github.com/rapidsai/raft/pull/2695)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- stop uploading packages to downloads.rapids.ai ([#2688](https://github.com/rapidsai/raft/pull/2688)) [@jameslamb](https://github.com/jameslamb)
- Reduce instantiations of `Reduction` kernels ([#2679](https://github.com/rapidsai/raft/pull/2679)) [@divyegala](https://github.com/divyegala)

## 🐛 Bug Fixes

- Fix stream sync for Copy2DAsync test ([#2744](https://github.com/rapidsai/raft/pull/2744)) [@lowener](https://github.com/lowener)
- Several small fixes to make Raft compile with LLVM. ([#2735](https://github.com/rapidsai/raft/pull/2735)) [@vitor1001](https://github.com/vitor1001)
- Add missing header ([#2734](https://github.com/rapidsai/raft/pull/2734)) [@vitor1001](https://github.com/vitor1001)
- [REVIEW] Fix static initialization order fiasco in `lanczos.cu` ([#2733](https://github.com/rapidsai/raft/pull/2733)) [@legrosbuffle](https://github.com/legrosbuffle)
- [REVIEW] Fix assertion in `fill_indices_by_rows_kernel`. ([#2732](https://github.com/rapidsai/raft/pull/2732)) [@legrosbuffle](https://github.com/legrosbuffle)
- libucx: consider post-releases in wheel builds ([#2729](https://github.com/rapidsai/raft/pull/2729)) [@jameslamb](https://github.com/jameslamb)
- Fix laplacian cast ([#2725](https://github.com/rapidsai/raft/pull/2725)) [@aamijar](https://github.com/aamijar)
- Fix excess_subsample ([#2723](https://github.com/rapidsai/raft/pull/2723)) [@mfoerste4](https://github.com/mfoerste4)
- Fix the constructor for `coordinate_structure` for non-zero `nnz`. ([#2717](https://github.com/rapidsai/raft/pull/2717)) [@legrosbuffle](https://github.com/legrosbuffle)
- [REVIEW] Fix compile error when using `mdbuffer` with all-static extents. ([#2716](https://github.com/rapidsai/raft/pull/2716)) [@legrosbuffle](https://github.com/legrosbuffle)
- Fix unsafe cast `coo_remove_scalar` ([#2713](https://github.com/rapidsai/raft/pull/2713)) [@aamijar](https://github.com/aamijar)
- Fix laplacian self-loops ([#2712](https://github.com/rapidsai/raft/pull/2712)) [@aamijar](https://github.com/aamijar)
- [REVIEW] Fix a few memory leaks. ([#2710](https://github.com/rapidsai/raft/pull/2710)) [@legrosbuffle](https://github.com/legrosbuffle)
- Fix MST bug for graph with identical edge weights ([#2707](https://github.com/rapidsai/raft/pull/2707)) [@jnke2016](https://github.com/jnke2016)
- Missed update accounting for reduction related APIs ([#2704](https://github.com/rapidsai/raft/pull/2704)) [@divyegala](https://github.com/divyegala)
- Adding GH_TOKEN pass-through to summarize job ([#2702](https://github.com/rapidsai/raft/pull/2702)) [@msarahan](https://github.com/msarahan)
- Work around Cython ctypedef bug ([#2686](https://github.com/rapidsai/raft/pull/2686)) [@vyasr](https://github.com/vyasr)

## 📖 Documentation

- add docs on CI workflow inputs ([#2728](https://github.com/rapidsai/raft/pull/2728)) [@jameslamb](https://github.com/jameslamb)

## 🛠️ Improvements

- An additional small change to remove cuda 11 stuff ([#2763](https://github.com/rapidsai/raft/pull/2763)) [@cjnolet](https://github.com/cjnolet)
- Removing CUDA 11 from docs and code ([#2757](https://github.com/rapidsai/raft/pull/2757)) [@cjnolet](https://github.com/cjnolet)
- fix(docker): use versioned `-latest` tag for all `rapidsai` images ([#2745](https://github.com/rapidsai/raft/pull/2745)) [@gforsyth](https://github.com/gforsyth)
- Update protocol name for UCX-Py tests ([#2743](https://github.com/rapidsai/raft/pull/2743)) [@pentschev](https://github.com/pentschev)
- Remove sphinx upper bound ([#2742](https://github.com/rapidsai/raft/pull/2742)) [@bdice](https://github.com/bdice)
- remove cuspatial references, avoid triggering tests on clang-format config changes ([#2740](https://github.com/rapidsai/raft/pull/2740)) [@jameslamb](https://github.com/jameslamb)
- MST Edge Case ([#2736](https://github.com/rapidsai/raft/pull/2736)) [@tarang-jain](https://github.com/tarang-jain)
- Add missing `#include &lt;cassert&gt;` in `cpp/include/raft/core/math.hpp` ([#2730](https://github.com/rapidsai/raft/pull/2730)) [@trxcllnt](https://github.com/trxcllnt)
- Update leftover CUDA 12.8 to 12.9 in docs ([#2724](https://github.com/rapidsai/raft/pull/2724)) [@jakirkham](https://github.com/jakirkham)
- Fix docs lanczos solver ([#2722](https://github.com/rapidsai/raft/pull/2722)) [@aamijar](https://github.com/aamijar)
- Use CUDA 12.9 in Conda, Devcontainers, Spark, GHA, etc. ([#2721](https://github.com/rapidsai/raft/pull/2721)) [@jakirkham](https://github.com/jakirkham)
- Remove nvidia and dask channels ([#2720](https://github.com/rapidsai/raft/pull/2720)) [@vyasr](https://github.com/vyasr)
- [REVIEW] Fix compile error of `abs_op` when compiling with `clang` ([#2718](https://github.com/rapidsai/raft/pull/2718)) [@legrosbuffle](https://github.com/legrosbuffle)
- Avoid using internal method std::experimental::details::alignTo(). ([#2714](https://github.com/rapidsai/raft/pull/2714)) [@vitor1001](https://github.com/vitor1001)
- refactor(shellcheck): fix all remaining warnings/errors ([#2703](https://github.com/rapidsai/raft/pull/2703)) [@gforsyth](https://github.com/gforsyth)
- `MatrixLinewiseOp` compile-time-invocation ([#2701](https://github.com/rapidsai/raft/pull/2701)) [@aamijar](https://github.com/aamijar)
- Remove pytest pin ([#2699](https://github.com/rapidsai/raft/pull/2699)) [@vyasr](https://github.com/vyasr)
- Fix several issues that breaks LLVM ([#2698](https://github.com/rapidsai/raft/pull/2698)) [@vitor1001](https://github.com/vitor1001)
- Remove CUDA 11 from dependencies.yaml ([#2695](https://github.com/rapidsai/raft/pull/2695)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Remove CUDA 11 devcontainers and update CI scripts ([#2690](https://github.com/rapidsai/raft/pull/2690)) [@bdice](https://github.com/bdice)
- refactor(rattler): remove cuda11 options and general cleanup ([#2689](https://github.com/rapidsai/raft/pull/2689)) [@gforsyth](https://github.com/gforsyth)
- stop uploading packages to downloads.rapids.ai ([#2688](https://github.com/rapidsai/raft/pull/2688)) [@jameslamb](https://github.com/jameslamb)
- fix(devcontainers): typo in container name ([#2687](https://github.com/rapidsai/raft/pull/2687)) [@gforsyth](https://github.com/gforsyth)
- Reduce instantiations of `Reduction` kernels ([#2679](https://github.com/rapidsai/raft/pull/2679)) [@divyegala](https://github.com/divyegala)
- Forward-merge branch-25.06 into branch-25.08 ([#2675](https://github.com/rapidsai/raft/pull/2675)) [@divyegala](https://github.com/divyegala)
- Add support for F16 in linalg::transpose ([#2672](https://github.com/rapidsai/raft/pull/2672)) [@enp1s0](https://github.com/enp1s0)
- Forward-merge branch-25.06 into branch-25.08 ([#2664](https://github.com/rapidsai/raft/pull/2664)) [@gforsyth](https://github.com/gforsyth)
- Support `coo_matrix` in `coo_symmetrize` and `coo_remove_scalar` ([#2662](https://github.com/rapidsai/raft/pull/2662)) [@aamijar](https://github.com/aamijar)
- Lanczos Solver `which=SA,SM,LA,LM` argument ([#2628](https://github.com/rapidsai/raft/pull/2628)) [@aamijar](https://github.com/aamijar)

# raft 25.06.00 (5 Jun 2025)

## 🚨 Breaking Changes

- Decoupling multi gpu resources from nccl usage ([#2647](https://github.com/rapidsai/raft/pull/2647)) [@jinsolp](https://github.com/jinsolp)

## 🐛 Bug Fixes

- NCCL comm resource fix ([#2692](https://github.com/rapidsai/raft/pull/2692)) [@viclafargue](https://github.com/viclafargue)
- Fix the launch bounds for nn-descent kernel for 1210 and remove nn-descent tests ([#2691](https://github.com/rapidsai/raft/pull/2691)) [@viclafargue](https://github.com/viclafargue)
- Prefer host gather when dataset is available both on host and device ([#2671](https://github.com/rapidsai/raft/pull/2671)) [@tfeher](https://github.com/tfeher)
- Fix warnings treated as errors downstream in cuVS ([#2644](https://github.com/rapidsai/raft/pull/2644)) [@achirkin](https://github.com/achirkin)
- Fix nccl_comm.hpp warning: #83-D: type qualifier specified more than once ([#2643](https://github.com/rapidsai/raft/pull/2643)) [@achirkin](https://github.com/achirkin)
- NVTX: null destination pointer warning-treated-as-error ([#2639](https://github.com/rapidsai/raft/pull/2639)) [@achirkin](https://github.com/achirkin)
- Add UCXX and NCCL to `libraft` conda recipe ([#2636](https://github.com/rapidsai/raft/pull/2636)) [@divyegala](https://github.com/divyegala)
- Fix building cutlass ([#2619](https://github.com/rapidsai/raft/pull/2619)) [@miscco](https://github.com/miscco)
- Fix COO symmetrization ([#2582](https://github.com/rapidsai/raft/pull/2582)) [@viclafargue](https://github.com/viclafargue)

## 🚀 New Features

- [Feat] add `cudaMemcpy2DAsync` wrapper ([#2674](https://github.com/rapidsai/raft/pull/2674)) [@rhdong](https://github.com/rhdong)
- Python wrapper for `device_resources_snmg` ([#2666](https://github.com/rapidsai/raft/pull/2666)) [@jinsolp](https://github.com/jinsolp)
- Laplacian normalization primitives ([#2648](https://github.com/rapidsai/raft/pull/2648)) [@aamijar](https://github.com/aamijar)
- [FEA] Matrix shift rows and columns ([#2634](https://github.com/rapidsai/raft/pull/2634)) [@jinsolp](https://github.com/jinsolp)
- Use NCCL wheels from PyPI for CUDA 12 builds ([#2629](https://github.com/rapidsai/raft/pull/2629)) [@divyegala](https://github.com/divyegala)
- Support strided matrix view as an input to matrix::samples_rows ([#2626](https://github.com/rapidsai/raft/pull/2626)) [@enp1s0](https://github.com/enp1s0)
- [Feat] add support for bm25 and tfidf ([#2567](https://github.com/rapidsai/raft/pull/2567)) [@jperez999](https://github.com/jperez999)

## 🛠️ Improvements

- use &#39;rapids-init-pip&#39; in wheel CI, other CI changes ([#2677](https://github.com/rapidsai/raft/pull/2677)) [@jameslamb](https://github.com/jameslamb)
- Dask 2025.4.1 compatibility ([#2673](https://github.com/rapidsai/raft/pull/2673)) [@TomAugspurger](https://github.com/TomAugspurger)
- Finish CUDA 12.9 migration and use branch-25.06 workflows ([#2669](https://github.com/rapidsai/raft/pull/2669)) [@bdice](https://github.com/bdice)
- Update to clang 20 ([#2665](https://github.com/rapidsai/raft/pull/2665)) [@bdice](https://github.com/bdice)
- Quote head_rev in conda recipes ([#2660](https://github.com/rapidsai/raft/pull/2660)) [@bdice](https://github.com/bdice)
- CUDA 12.9 use updated compression flags ([#2657](https://github.com/rapidsai/raft/pull/2657)) [@robertmaynard](https://github.com/robertmaynard)
- Build and test with CUDA 12.9.0 ([#2655](https://github.com/rapidsai/raft/pull/2655)) [@bdice](https://github.com/bdice)
- Exclude librmm.so from auditwheel ([#2654](https://github.com/rapidsai/raft/pull/2654)) [@bdice](https://github.com/bdice)
- Fix cub include in normalize.cuh ([#2652](https://github.com/rapidsai/raft/pull/2652)) [@lowener](https://github.com/lowener)
- Add support for Python 3.13 ([#2649](https://github.com/rapidsai/raft/pull/2649)) [@gforsyth](https://github.com/gforsyth)
- Decoupling multi gpu resources from nccl usage ([#2647](https://github.com/rapidsai/raft/pull/2647)) [@jinsolp](https://github.com/jinsolp)
- [BUGFIX] Fixed quoting in wheel paths in pylibraft and raft_dask wheel tests ([#2645](https://github.com/rapidsai/raft/pull/2645)) [@VenkateshJaya](https://github.com/VenkateshJaya)
- Download build artifacts from Github for CI ([#2640](https://github.com/rapidsai/raft/pull/2640)) [@VenkateshJaya](https://github.com/VenkateshJaya)
- Limit allowed wheel sizes ([#2638](https://github.com/rapidsai/raft/pull/2638)) [@divyegala](https://github.com/divyegala)
- Remove CUDA whole compilation ODR violations ([#2633](https://github.com/rapidsai/raft/pull/2633)) [@divyegala](https://github.com/divyegala)
- refactor(rattler): enable strict channel priority for builds ([#2632](https://github.com/rapidsai/raft/pull/2632)) [@gforsyth](https://github.com/gforsyth)
- Vendor RAPIDS.cmake ([#2631](https://github.com/rapidsai/raft/pull/2631)) [@bdice](https://github.com/bdice)
- Replace `Thrust` iterator facilities and replace them with `libcu++` ones ([#2627](https://github.com/rapidsai/raft/pull/2627)) [@miscco](https://github.com/miscco)
- Port all conda recipes to `rattler-build` ([#2623](https://github.com/rapidsai/raft/pull/2623)) [@gforsyth](https://github.com/gforsyth)
- Add missing thrust include ([#2618](https://github.com/rapidsai/raft/pull/2618)) [@miscco](https://github.com/miscco)
- Moving wheel builds to specified location and uploading build artifacts to Github ([#2617](https://github.com/rapidsai/raft/pull/2617)) [@VenkateshJaya](https://github.com/VenkateshJaya)
- Fixed pytest marker warnings by removing unused pytest.ini ([#2591](https://github.com/rapidsai/raft/pull/2591)) [@TomAugspurger](https://github.com/TomAugspurger)
- Introduction of the `raft::device_resources_snmg` type ([#2549](https://github.com/rapidsai/raft/pull/2549)) [@viclafargue](https://github.com/viclafargue)
- Create a NCCL sub-communicator using ncclCommSplit ([#2495](https://github.com/rapidsai/raft/pull/2495)) [@seunghwak](https://github.com/seunghwak)

# raft 25.04.00 (9 Apr 2025)

## 🚨 Breaking Changes

- Account for cugraph API breakage ([#2581](https://github.com/rapidsai/raft/pull/2581)) [@divyegala](https://github.com/divyegala)
- Use new rapids-logger library ([#2566](https://github.com/rapidsai/raft/pull/2566)) [@vyasr](https://github.com/vyasr)

## 🐛 Bug Fixes

- Backport build patch fix ([#2620](https://github.com/rapidsai/raft/pull/2620)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Revert &quot;Temporarily increase `max_days_without_success` ([#2602)&quot; (#2613](https://github.com/rapidsai/raft/pull/2602)&quot; (#2613)) [@divyegala](https://github.com/divyegala)
- Relax max duplicates in batched NN Descent ([#2610](https://github.com/rapidsai/raft/pull/2610)) [@jinsolp](https://github.com/jinsolp)
- [Fix] Lanczos solver gemv fix ([#2607](https://github.com/rapidsai/raft/pull/2607)) [@aamijar](https://github.com/aamijar)
- [Fix] `select-k-csr` failure on CUDA11.x + H100 ([#2604](https://github.com/rapidsai/raft/pull/2604)) [@rhdong](https://github.com/rhdong)
- Temporarily increase `max_days_without_success` ([#2602](https://github.com/rapidsai/raft/pull/2602)) [@divyegala](https://github.com/divyegala)
- Swap `blocks` and `threads_per_block` in `compute_graph_laplacian` ([#2597](https://github.com/rapidsai/raft/pull/2597)) [@jcrist](https://github.com/jcrist)
- [BUG] Fix illegal memory access in linalg::reduction ([#2592](https://github.com/rapidsai/raft/pull/2592)) [@enp1s0](https://github.com/enp1s0)
- Require sphinx&lt;8.2.0 ([#2590](https://github.com/rapidsai/raft/pull/2590)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Account for cugraph API breakage ([#2581](https://github.com/rapidsai/raft/pull/2581)) [@divyegala](https://github.com/divyegala)
- `#include &lt;numeric&gt;` for `std::iota` ([#2578](https://github.com/rapidsai/raft/pull/2578)) [@benfred](https://github.com/benfred)
- Fix Laplacian calculation in spectral partitioning ([#2568](https://github.com/rapidsai/raft/pull/2568)) [@wphicks](https://github.com/wphicks)
- Take argument by `const&amp;` as the input range is const ([#2558](https://github.com/rapidsai/raft/pull/2558)) [@miscco](https://github.com/miscco)
- Allow some of the sparse utility functions to handle larger matrices ([#2541](https://github.com/rapidsai/raft/pull/2541)) [@viclafargue](https://github.com/viclafargue)

## 🛠️ Improvements

- ci: pre-filter 11.4 jobs before they are enabled in shared workflows ([#2608](https://github.com/rapidsai/raft/pull/2608)) [@gforsyth](https://github.com/gforsyth)
- Use conda-build instead of conda-mambabuild ([#2595](https://github.com/rapidsai/raft/pull/2595)) [@bdice](https://github.com/bdice)
- Replace `cub::Sum` and `cub::Max` with `cuda::std::plus` and `cuda::maximum` ([#2594](https://github.com/rapidsai/raft/pull/2594)) [@miscco](https://github.com/miscco)
- Update all `conda_build_config.yaml`s RAPIDS UCX version ([#2589](https://github.com/rapidsai/raft/pull/2589)) [@jakirkham](https://github.com/jakirkham)
- Drop `cub::TransformInputIterator` in favor of `thrust::transform_iterator` ([#2588](https://github.com/rapidsai/raft/pull/2588)) [@miscco](https://github.com/miscco)
- Consolidate more Conda solves in CI ([#2587](https://github.com/rapidsai/raft/pull/2587)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Fix duplicate indices in batch NN Descent ([#2586](https://github.com/rapidsai/raft/pull/2586)) [@jinsolp](https://github.com/jinsolp)
- Require CMake 3.30.4 ([#2584](https://github.com/rapidsai/raft/pull/2584)) [@robertmaynard](https://github.com/robertmaynard)
- Create Conda CI test env in one step ([#2580](https://github.com/rapidsai/raft/pull/2580)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Use shared-workflows branch-25.04 ([#2576](https://github.com/rapidsai/raft/pull/2576)) [@bdice](https://github.com/bdice)
- Add `shellcheck` to pre-commit and fix warnings ([#2575](https://github.com/rapidsai/raft/pull/2575)) [@gforsyth](https://github.com/gforsyth)
- Add build_type input field for `test.yaml` ([#2573](https://github.com/rapidsai/raft/pull/2573)) [@gforsyth](https://github.com/gforsyth)
- Use `rapids-pip-retry` in CI jobs that might need retries ([#2571](https://github.com/rapidsai/raft/pull/2571)) [@gforsyth](https://github.com/gforsyth)
- Avoid limited memory adaptor issue in balanced KMeans ([#2570](https://github.com/rapidsai/raft/pull/2570)) [@csadorf](https://github.com/csadorf)
- update telemetry and retarget 25.04 ([#2569](https://github.com/rapidsai/raft/pull/2569)) [@msarahan](https://github.com/msarahan)
- Use new rapids-logger library ([#2566](https://github.com/rapidsai/raft/pull/2566)) [@vyasr](https://github.com/vyasr)
- disallow fallback to Make in Python builds ([#2563](https://github.com/rapidsai/raft/pull/2563)) [@jameslamb](https://github.com/jameslamb)
- Forward-merge branch-25.02 into branch-25.04 ([#2561](https://github.com/rapidsai/raft/pull/2561)) [@bdice](https://github.com/bdice)
- Migrate to NVKS for amd64 CI runners ([#2559](https://github.com/rapidsai/raft/pull/2559)) [@bdice](https://github.com/bdice)
- Add `verify-codeowners` hook ([#2557](https://github.com/rapidsai/raft/pull/2557)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)

# raft 25.02.00 (13 Feb 2025)

## 🚨 Breaking Changes

- Update pip devcontainers to UCX 1.18 ([#2550](https://github.com/rapidsai/raft/pull/2550)) [@jameslamb](https://github.com/jameslamb)
- Switch over to rapids-logger ([#2530](https://github.com/rapidsai/raft/pull/2530)) [@vyasr](https://github.com/vyasr)
- Adapt to rmm logger changes ([#2513](https://github.com/rapidsai/raft/pull/2513)) [@vyasr](https://github.com/vyasr)

## 🐛 Bug Fixes

- Rename test to tests. ([#2546](https://github.com/rapidsai/raft/pull/2546)) [@bdice](https://github.com/bdice)
- Fix bit order of RMAT Rectangular Generator to match expectation ([#2542](https://github.com/rapidsai/raft/pull/2542)) [@mfoerste4](https://github.com/mfoerste4)
- Fix broken link to python doc ([#2537](https://github.com/rapidsai/raft/pull/2537)) [@lowener](https://github.com/lowener)
- Fix lanczos solver integer overflow ([#2536](https://github.com/rapidsai/raft/pull/2536)) [@viclafargue](https://github.com/viclafargue)
- Fix rnd bit generation in rmat_rectangular_kernel ([#2524](https://github.com/rapidsai/raft/pull/2524)) [@tfeher](https://github.com/tfeher)

## 📖 Documentation

- Fix docs builds ([#2562](https://github.com/rapidsai/raft/pull/2562)) [@bdice](https://github.com/bdice)
- [DOC] Fix sample codes ([#2518](https://github.com/rapidsai/raft/pull/2518)) [@enp1s0](https://github.com/enp1s0)

## 🚀 New Features

- Add cuda 12.8 support ([#2551](https://github.com/rapidsai/raft/pull/2551)) [@robertmaynard](https://github.com/robertmaynard)
- Add support for different data type of bitset ([#2535](https://github.com/rapidsai/raft/pull/2535)) [@lowener](https://github.com/lowener)
- [Feat] Support `bitset_to_csr` ([#2523](https://github.com/rapidsai/raft/pull/2523)) [@rhdong](https://github.com/rhdong)
- Remove upper bounds on cuda-python to allow 12.6.2 and 11.8.5 ([#2517](https://github.com/rapidsai/raft/pull/2517)) [@bdice](https://github.com/bdice)

## 🛠️ Improvements

- Revert CUDA 12.8 shared workflow branch changes ([#2560](https://github.com/rapidsai/raft/pull/2560)) [@vyasr](https://github.com/vyasr)
- Build and test with CUDA 12.8.0 ([#2555](https://github.com/rapidsai/raft/pull/2555)) [@bdice](https://github.com/bdice)
- Update pip devcontainers to UCX 1.18 ([#2550](https://github.com/rapidsai/raft/pull/2550)) [@jameslamb](https://github.com/jameslamb)
- use dynamic CUDA wheels on CUDA 11 ([#2548](https://github.com/rapidsai/raft/pull/2548)) [@jameslamb](https://github.com/jameslamb)
- Normalize whitespace ([#2547](https://github.com/rapidsai/raft/pull/2547)) [@bdice](https://github.com/bdice)
- Use cuda.bindings layout. ([#2545](https://github.com/rapidsai/raft/pull/2545)) [@bdice](https://github.com/bdice)
- Revert &quot;Introduction of the `raft::device_resources_snmg` type ([#2487)&quot; (#2543](https://github.com/rapidsai/raft/pull/2487)&quot; (#2543)) [@cjnolet](https://github.com/cjnolet)
- Add missing `#include &lt;cstdint&gt;` ([#2540](https://github.com/rapidsai/raft/pull/2540)) [@jakirkham](https://github.com/jakirkham)
- Use GCC 13 in CUDA 12 conda builds. ([#2539](https://github.com/rapidsai/raft/pull/2539)) [@bdice](https://github.com/bdice)
- Use rapids-cmake for the logger ([#2534](https://github.com/rapidsai/raft/pull/2534)) [@vyasr](https://github.com/vyasr)
- Check if nightlies have succeeded recently enough ([#2533](https://github.com/rapidsai/raft/pull/2533)) [@vyasr](https://github.com/vyasr)
- remove unused &#39;joblib&#39; and &#39;numba&#39; dependencies, other packaging cleanup ([#2532](https://github.com/rapidsai/raft/pull/2532)) [@jameslamb](https://github.com/jameslamb)
- introduce libraft wheels ([#2531](https://github.com/rapidsai/raft/pull/2531)) [@jameslamb](https://github.com/jameslamb)
- Switch over to rapids-logger ([#2530](https://github.com/rapidsai/raft/pull/2530)) [@vyasr](https://github.com/vyasr)
- reduce duplication, removed unused things in dependencies.yaml ([#2529](https://github.com/rapidsai/raft/pull/2529)) [@jameslamb](https://github.com/jameslamb)
- Update cuda-python lower bounds to 12.6.2 / 11.8.5 ([#2522](https://github.com/rapidsai/raft/pull/2522)) [@bdice](https://github.com/bdice)
- [Opt] Optimizing the performance of `bitmap_to_csr` ([#2516](https://github.com/rapidsai/raft/pull/2516)) [@rhdong](https://github.com/rhdong)
- prefer system install of UCX in devcontainers, update outdated RAPIDS references ([#2514](https://github.com/rapidsai/raft/pull/2514)) [@jameslamb](https://github.com/jameslamb)
- Adapt to rmm logger changes ([#2513](https://github.com/rapidsai/raft/pull/2513)) [@vyasr](https://github.com/vyasr)
- Require approval to run CI on draft PRs ([#2512](https://github.com/rapidsai/raft/pull/2512)) [@bdice](https://github.com/bdice)
- Shrink wheel size limit following removal of vector search APIs. ([#2509](https://github.com/rapidsai/raft/pull/2509)) [@bdice](https://github.com/bdice)
- Forward-merge branch-24.12 to branch-25.02 ([#2508](https://github.com/rapidsai/raft/pull/2508)) [@bdice](https://github.com/bdice)
- Introduction of the `raft::device_resources_snmg` type ([#2487](https://github.com/rapidsai/raft/pull/2487)) [@viclafargue](https://github.com/viclafargue)
- Add breaking change workflow trigger ([#2482](https://github.com/rapidsai/raft/pull/2482)) [@AyodeAwe](https://github.com/AyodeAwe)
- Remove &#39;sample&#39; parameter from stats::mean API ([#2389](https://github.com/rapidsai/raft/pull/2389)) [@mfoerste4](https://github.com/mfoerste4)

# raft 24.12.00 (11 Dec 2024)

## 🚨 Breaking Changes

- Do not initialize the pinned mdarray at construction time ([#2478](https://github.com/rapidsai/raft/pull/2478)) [@achirkin](https://github.com/achirkin)

## 🐛 Bug Fixes

- Skip gtests for new lanczos solver when CUDA version is 11.4 or below. ([#2520](https://github.com/rapidsai/raft/pull/2520)) [@cjnolet](https://github.com/cjnolet)
- Switch `assert` to `static_assert` ([#2510](https://github.com/rapidsai/raft/pull/2510)) [@divyegala](https://github.com/divyegala)
- Revert use of new Lanczos solver in spectral clustering ([#2507](https://github.com/rapidsai/raft/pull/2507)) [@lowener](https://github.com/lowener)
- Put a ceiling on cuda-python ([#2486](https://github.com/rapidsai/raft/pull/2486)) [@bdice](https://github.com/bdice)
- Don&#39;t presume pointers location infers usability. ([#2480](https://github.com/rapidsai/raft/pull/2480)) [@robertmaynard](https://github.com/robertmaynard)
- Use Python for sccache hit rate computation. ([#2474](https://github.com/rapidsai/raft/pull/2474)) [@bdice](https://github.com/bdice)
- Allow compilation with CUDA 12.6.1 ([#2469](https://github.com/rapidsai/raft/pull/2469)) [@robertmaynard](https://github.com/robertmaynard)

## 🚀 New Features

- [FEA] Lanczos solver v2 ([#2481](https://github.com/rapidsai/raft/pull/2481)) [@lowener](https://github.com/lowener)

## 🛠️ Improvements

- Skip gtests for Rmat Lanczos tests with cuda &lt;= 11.4 ([#2525](https://github.com/rapidsai/raft/pull/2525)) [@benfred](https://github.com/benfred)
- Upgrade to latest cutlass version ([#2503](https://github.com/rapidsai/raft/pull/2503)) [@vyasr](https://github.com/vyasr)
- Removing some left over places where implicit instantiations were being ignored in headers ([#2501](https://github.com/rapidsai/raft/pull/2501)) [@cjnolet](https://github.com/cjnolet)
- Remove leftover template project code. ([#2500](https://github.com/rapidsai/raft/pull/2500)) [@bdice](https://github.com/bdice)
- 2412 remove libraft vss instantiations ([#2498](https://github.com/rapidsai/raft/pull/2498)) [@cjnolet](https://github.com/cjnolet)
- Remove raft-ann-bench ([#2497](https://github.com/rapidsai/raft/pull/2497)) [@cjnolet](https://github.com/cjnolet)
- Pin FAISS Version for raft-ann-bench ([#2496](https://github.com/rapidsai/raft/pull/2496)) [@tarang-jain](https://github.com/tarang-jain)
- enforce wheel size limits and README formatting in CI, put a ceiling on Cython dependency ([#2490](https://github.com/rapidsai/raft/pull/2490)) [@jameslamb](https://github.com/jameslamb)
- Do not initialize the pinned mdarray at construction time ([#2478](https://github.com/rapidsai/raft/pull/2478)) [@achirkin](https://github.com/achirkin)
- Use environment variables in cache hit rate computation. ([#2475](https://github.com/rapidsai/raft/pull/2475)) [@bdice](https://github.com/bdice)
- devcontainer: replace `VAULT_HOST` with `AWS_ROLE_ARN` ([#2472](https://github.com/rapidsai/raft/pull/2472)) [@jjacobelli](https://github.com/jjacobelli)
- print sccache stats in builds ([#2470](https://github.com/rapidsai/raft/pull/2470)) [@jameslamb](https://github.com/jameslamb)
- make package installations in CI stricter ([#2467](https://github.com/rapidsai/raft/pull/2467)) [@jameslamb](https://github.com/jameslamb)
- Prune workflows based on changed files ([#2466](https://github.com/rapidsai/raft/pull/2466)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Merge branch-24.10 into branch-24.12 ([#2461](https://github.com/rapidsai/raft/pull/2461)) [@jameslamb](https://github.com/jameslamb)
- Update all rmm imports to use pylibrmm/librmm ([#2451](https://github.com/rapidsai/raft/pull/2451)) [@Matt711](https://github.com/Matt711)

# raft 24.10.00 (9 Oct 2024)

## 🚨 Breaking Changes

- [Feat] add `repeat`, `sparsity`,  `eval_n_elements` APIs to `bitset` ([#2439](https://github.com/rapidsai/raft/pull/2439)) [@rhdong](https://github.com/rhdong)

## 🐛 Bug Fixes

- Disable NN Descent Batch tests temporarily ([#2453](https://github.com/rapidsai/raft/pull/2453)) [@divyegala](https://github.com/divyegala)
- Fix sed syntax in `update-version.sh` ([#2441](https://github.com/rapidsai/raft/pull/2441)) [@raydouglass](https://github.com/raydouglass)
- Use runtime check of cudart version for eig ([#2430](https://github.com/rapidsai/raft/pull/2430)) [@lowener](https://github.com/lowener)
- [BUG] Fix bitset function visibility ([#2429](https://github.com/rapidsai/raft/pull/2429)) [@lowener](https://github.com/lowener)
- Exclude any kernel symbol that uses cutlass ([#2425](https://github.com/rapidsai/raft/pull/2425)) [@robertmaynard](https://github.com/robertmaynard)

## 🚀 New Features

- [Feat] add `repeat`, `sparsity`,  `eval_n_elements` APIs to `bitset` ([#2439](https://github.com/rapidsai/raft/pull/2439)) [@rhdong](https://github.com/rhdong)
- [Opt] Enforce the UT Coverity and add benchmark for `transpose` ([#2438](https://github.com/rapidsai/raft/pull/2438)) [@rhdong](https://github.com/rhdong)
- [FEA] Support for half-float mixed precise in brute-force ([#2382](https://github.com/rapidsai/raft/pull/2382)) [@rhdong](https://github.com/rhdong)

## 🛠️ Improvements

- bump NCCL floor to 2.19 ([#2458](https://github.com/rapidsai/raft/pull/2458)) [@jameslamb](https://github.com/jameslamb)
- Deprecating vector search APIs and updating README accordingly ([#2448](https://github.com/rapidsai/raft/pull/2448)) [@cjnolet](https://github.com/cjnolet)
- Update update-version.sh to use packaging lib ([#2447](https://github.com/rapidsai/raft/pull/2447)) [@AyodeAwe](https://github.com/AyodeAwe)
- Switch traceback to `native` ([#2446](https://github.com/rapidsai/raft/pull/2446)) [@galipremsagar](https://github.com/galipremsagar)
- bump NCCL floor to 2.18.1.1 ([#2443](https://github.com/rapidsai/raft/pull/2443)) [@jameslamb](https://github.com/jameslamb)
- Add missing `cuda_suffixed: true` ([#2440](https://github.com/rapidsai/raft/pull/2440)) [@trxcllnt](https://github.com/trxcllnt)
- Use CI workflow branch &#39;branch-24.10&#39; again ([#2437](https://github.com/rapidsai/raft/pull/2437)) [@jameslamb](https://github.com/jameslamb)
- Update to flake8 7.1.1. ([#2435](https://github.com/rapidsai/raft/pull/2435)) [@bdice](https://github.com/bdice)
- Update fmt (to 11.0.2) and spdlog (to 1.14.1). ([#2433](https://github.com/rapidsai/raft/pull/2433)) [@jameslamb](https://github.com/jameslamb)
- Allow coo_sort to work on int64_t indices ([#2432](https://github.com/rapidsai/raft/pull/2432)) [@benfred](https://github.com/benfred)
- Adding NCCL clique to the RAFT handle ([#2431](https://github.com/rapidsai/raft/pull/2431)) [@viclafargue](https://github.com/viclafargue)
- Add support for Python 3.12 ([#2428](https://github.com/rapidsai/raft/pull/2428)) [@jameslamb](https://github.com/jameslamb)
- Update rapidsai/pre-commit-hooks ([#2420](https://github.com/rapidsai/raft/pull/2420)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Drop Python 3.9 support ([#2417](https://github.com/rapidsai/raft/pull/2417)) [@jameslamb](https://github.com/jameslamb)
- Use CUDA math wheels ([#2415](https://github.com/rapidsai/raft/pull/2415)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Remove NumPy &lt;2 pin ([#2414](https://github.com/rapidsai/raft/pull/2414)) [@seberg](https://github.com/seberg)
- Update pre-commit hooks ([#2409](https://github.com/rapidsai/raft/pull/2409)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Improve update-version.sh ([#2408](https://github.com/rapidsai/raft/pull/2408)) [@bdice](https://github.com/bdice)
- Use tool.scikit-build.cmake.version, set scikit-build-core minimum-version ([#2406](https://github.com/rapidsai/raft/pull/2406)) [@jameslamb](https://github.com/jameslamb)
- [FEA] Batching NN Descent ([#2403](https://github.com/rapidsai/raft/pull/2403)) [@jinsolp](https://github.com/jinsolp)
- Update pip devcontainers to UCX v1.17.0 ([#2401](https://github.com/rapidsai/raft/pull/2401)) [@jameslamb](https://github.com/jameslamb)
- Merge branch-24.08 into branch-24.10 ([#2397](https://github.com/rapidsai/raft/pull/2397)) [@jameslamb](https://github.com/jameslamb)

# raft 24.08.00 (7 Aug 2024)

## 🚨 Breaking Changes

- [Refactor] move `popc` to under util ([#2394](https://github.com/rapidsai/raft/pull/2394)) [@rhdong](https://github.com/rhdong)
- [Opt] Expose the `detail::popc` as public API ([#2346](https://github.com/rapidsai/raft/pull/2346)) [@rhdong](https://github.com/rhdong)

## 🐛 Bug Fixes

- Add timeout to UCXX generic operations ([#2398](https://github.com/rapidsai/raft/pull/2398)) [@pentschev](https://github.com/pentschev)
- [Fix] bitmap set/test issue ([#2371](https://github.com/rapidsai/raft/pull/2371)) [@rhdong](https://github.com/rhdong)
- Fix 0 recall issue in `raft_cagra_hnswlib` ANN benchmark ([#2369](https://github.com/rapidsai/raft/pull/2369)) [@divyegala](https://github.com/divyegala)
- Fix `ef` setting in HNSW wrapper ([#2367](https://github.com/rapidsai/raft/pull/2367)) [@divyegala](https://github.com/divyegala)
- Fix cagra graph opt bug ([#2365](https://github.com/rapidsai/raft/pull/2365)) [@enp1s0](https://github.com/enp1s0)
- Fix a bug where the wrong API is used to free the memory ([#2361](https://github.com/rapidsai/raft/pull/2361)) [@PointKernel](https://github.com/PointKernel)
- Allow anonymous user in devcontainer name ([#2355](https://github.com/rapidsai/raft/pull/2355)) [@bdice](https://github.com/bdice)
- Fix compilation error when _CLK_BREAKDOWN is defined in cagra. ([#2350](https://github.com/rapidsai/raft/pull/2350)) [@jiangyinzuo](https://github.com/jiangyinzuo)
- ensure raft-dask wheel tests install pylibraft wheel from the same CI run, fix wheel dependencies ([#2349](https://github.com/rapidsai/raft/pull/2349)) [@jameslamb](https://github.com/jameslamb)
- Change --config-setting to --config-settings ([#2342](https://github.com/rapidsai/raft/pull/2342)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Add workaround for syevd in CUDA 12.0 ([#2332](https://github.com/rapidsai/raft/pull/2332)) [@lowener](https://github.com/lowener)

## 🚀 New Features

- [FEA] add the support of `masked_matmul` ([#2362](https://github.com/rapidsai/raft/pull/2362)) [@rhdong](https://github.com/rhdong)
- [FEA] Dice Distance for Dense Inputs ([#2359](https://github.com/rapidsai/raft/pull/2359)) [@aamijar](https://github.com/aamijar)
- [Opt] Expose the `detail::popc` as public API ([#2346](https://github.com/rapidsai/raft/pull/2346)) [@rhdong](https://github.com/rhdong)
- Enable distance return for NN Descent ([#2345](https://github.com/rapidsai/raft/pull/2345)) [@jinsolp](https://github.com/jinsolp)

## 🛠️ Improvements

- [Refactor] move `popc` to under util ([#2394](https://github.com/rapidsai/raft/pull/2394)) [@rhdong](https://github.com/rhdong)
- split up CUDA-suffixed dependencies in dependencies.yaml ([#2388](https://github.com/rapidsai/raft/pull/2388)) [@jameslamb](https://github.com/jameslamb)
- Use workflow branch 24.08 again ([#2385](https://github.com/rapidsai/raft/pull/2385)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Add cusparseSpMV_preprocess to cusparse wrapper ([#2384](https://github.com/rapidsai/raft/pull/2384)) [@Kh4ster](https://github.com/Kh4ster)
- Consolidate SUM reductions ([#2381](https://github.com/rapidsai/raft/pull/2381)) [@mfoerste4](https://github.com/mfoerste4)
- Use slicing kernel to copy distances inside NN Descent ([#2380](https://github.com/rapidsai/raft/pull/2380)) [@jinsolp](https://github.com/jinsolp)
- Build and test with CUDA 12.5.1 ([#2378](https://github.com/rapidsai/raft/pull/2378)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Add CUDA_STATIC_MATH_LIBRARIES ([#2376](https://github.com/rapidsai/raft/pull/2376)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- skip CMake 3.30.0 ([#2375](https://github.com/rapidsai/raft/pull/2375)) [@jameslamb](https://github.com/jameslamb)
- Use verify-alpha-spec hook ([#2373](https://github.com/rapidsai/raft/pull/2373)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Binarize Dice Distance for Dense Inputs ([#2370](https://github.com/rapidsai/raft/pull/2370)) [@aamijar](https://github.com/aamijar)
- [FEA] Add distance epilogue for NN Descent ([#2364](https://github.com/rapidsai/raft/pull/2364)) [@jinsolp](https://github.com/jinsolp)
- resolve dependency-file-generator warning, other rapids-build-backend followup ([#2360](https://github.com/rapidsai/raft/pull/2360)) [@jameslamb](https://github.com/jameslamb)
- Remove text builds of documentation ([#2354](https://github.com/rapidsai/raft/pull/2354)) [@vyasr](https://github.com/vyasr)
- Use default init in reduction ([#2351](https://github.com/rapidsai/raft/pull/2351)) [@akifcorduk](https://github.com/akifcorduk)
- ensure update-version.sh preserves alpha spec, add tests on version constants ([#2344](https://github.com/rapidsai/raft/pull/2344)) [@jameslamb](https://github.com/jameslamb)
- remove unnecessary &#39;setuptools&#39; dependencies ([#2343](https://github.com/rapidsai/raft/pull/2343)) [@jameslamb](https://github.com/jameslamb)
- Use rapids-build-backend ([#2331](https://github.com/rapidsai/raft/pull/2331)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Add FAISS with RAFT enabled Benchmarking to raft-ann-bench ([#2026](https://github.com/rapidsai/raft/pull/2026)) [@tarang-jain](https://github.com/tarang-jain)

# raft 24.06.00 (5 Jun 2024)

## 🚨 Breaking Changes

- Rename raft-ann-bench module to raft_ann_bench ([#2333](https://github.com/rapidsai/raft/pull/2333)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Scaling workspace resources ([#2322](https://github.com/rapidsai/raft/pull/2322)) [@achirkin](https://github.com/achirkin)
- [REVIEW] Adjust UCX dependencies ([#2304](https://github.com/rapidsai/raft/pull/2304)) [@pentschev](https://github.com/pentschev)
- Convert device_memory_resource* to device_async_resource_ref ([#2269](https://github.com/rapidsai/raft/pull/2269)) [@harrism](https://github.com/harrism)

## 🐛 Bug Fixes

- Fix import of VERSION file in raft-ann-bench ([#2338](https://github.com/rapidsai/raft/pull/2338)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Rename raft-ann-bench module to raft_ann_bench ([#2333](https://github.com/rapidsai/raft/pull/2333)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Support building faiss main statically ([#2323](https://github.com/rapidsai/raft/pull/2323)) [@robertmaynard](https://github.com/robertmaynard)
- Refactor spectral scale_obs to use existing normalization function ([#2319](https://github.com/rapidsai/raft/pull/2319)) [@ChuckHastings](https://github.com/ChuckHastings)
- Correct initializer list order found by cuvs ([#2317](https://github.com/rapidsai/raft/pull/2317)) [@robertmaynard](https://github.com/robertmaynard)
- ANN_BENCH: enable move semantics for configured_raft_resources ([#2311](https://github.com/rapidsai/raft/pull/2311)) [@achirkin](https://github.com/achirkin)
- Revert &quot;Build C++ wheel ([#2264)&quot; (#2305](https://github.com/rapidsai/raft/pull/2264)&quot; (#2305)) [@vyasr](https://github.com/vyasr)
- Revert &quot;Add `compile-library` by default on pylibraft build&quot; ([#2300](https://github.com/rapidsai/raft/pull/2300)) [@vyasr](https://github.com/vyasr)
- Add VERSION to raft-ann-bench package ([#2299](https://github.com/rapidsai/raft/pull/2299)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Remove nonexistent job from workflow ([#2298](https://github.com/rapidsai/raft/pull/2298)) [@vyasr](https://github.com/vyasr)
- `libucx` should be run dependency of `raft-dask` ([#2296](https://github.com/rapidsai/raft/pull/2296)) [@divyegala](https://github.com/divyegala)
- Fix clang intrinsic warning ([#2292](https://github.com/rapidsai/raft/pull/2292)) [@aaronmondal](https://github.com/aaronmondal)
- Replace too long index file name with hash in ANN bench ([#2280](https://github.com/rapidsai/raft/pull/2280)) [@tfeher](https://github.com/tfeher)
- Fix build command for C++ compilation ([#2270](https://github.com/rapidsai/raft/pull/2270)) [@lowener](https://github.com/lowener)
- Fix a compilation error in CAGRA when enabling log output ([#2262](https://github.com/rapidsai/raft/pull/2262)) [@enp1s0](https://github.com/enp1s0)
- Correct member initialization order ([#2254](https://github.com/rapidsai/raft/pull/2254)) [@robertmaynard](https://github.com/robertmaynard)
- Fix time computation in CAGRA notebook ([#2231](https://github.com/rapidsai/raft/pull/2231)) [@lowener](https://github.com/lowener)

## 📖 Documentation

- Fix citation info ([#2318](https://github.com/rapidsai/raft/pull/2318)) [@enp1s0](https://github.com/enp1s0)

## 🚀 New Features

- Scaling workspace resources ([#2322](https://github.com/rapidsai/raft/pull/2322)) [@achirkin](https://github.com/achirkin)
- ANN_BENCH: AnnGPU::uses_stream() for optional algo GPU sync ([#2314](https://github.com/rapidsai/raft/pull/2314)) [@achirkin](https://github.com/achirkin)
- [FEA] Split Bitset code ([#2295](https://github.com/rapidsai/raft/pull/2295)) [@lowener](https://github.com/lowener)
- [FEA] support of prefiltered brute force ([#2294](https://github.com/rapidsai/raft/pull/2294)) [@rhdong](https://github.com/rhdong)
- Always use a static gtest and gbench ([#2265](https://github.com/rapidsai/raft/pull/2265)) [@robertmaynard](https://github.com/robertmaynard)
- Build C++ wheel ([#2264](https://github.com/rapidsai/raft/pull/2264)) [@vyasr](https://github.com/vyasr)
- InnerProduct Distance Metric for CAGRA search ([#2260](https://github.com/rapidsai/raft/pull/2260)) [@tarang-jain](https://github.com/tarang-jain)
- [FEA] Add support for `select_k` on CSR matrix ([#2140](https://github.com/rapidsai/raft/pull/2140)) [@rhdong](https://github.com/rhdong)

## 🛠️ Improvements

- ANN_BENCH: common AnnBase::index_type ([#2315](https://github.com/rapidsai/raft/pull/2315)) [@achirkin](https://github.com/achirkin)
- ANN_BENCH: split instances of RaftCagra into multiple files ([#2313](https://github.com/rapidsai/raft/pull/2313)) [@achirkin](https://github.com/achirkin)
- ANN_BENCH: a global pool of result buffers across benchmark cases ([#2312](https://github.com/rapidsai/raft/pull/2312)) [@achirkin](https://github.com/achirkin)
- Remove the shared state and the mutex from NVTX internals ([#2310](https://github.com/rapidsai/raft/pull/2310)) [@achirkin](https://github.com/achirkin)
- docs: update README.md ([#2308](https://github.com/rapidsai/raft/pull/2308)) [@eltociear](https://github.com/eltociear)
- [REVIEW] Reenable raft-dask wheel tests requiring UCX-Py ([#2307](https://github.com/rapidsai/raft/pull/2307)) [@pentschev](https://github.com/pentschev)
- [REVIEW] Adjust UCX dependencies ([#2304](https://github.com/rapidsai/raft/pull/2304)) [@pentschev](https://github.com/pentschev)
- Overhaul ops-codeowners ([#2303](https://github.com/rapidsai/raft/pull/2303)) [@raydouglass](https://github.com/raydouglass)
- Make thrust nosync execution policy the default thrust policy ([#2302](https://github.com/rapidsai/raft/pull/2302)) [@abc99lr](https://github.com/abc99lr)
- InnerProduct testing for CAGRA+HNSW ([#2297](https://github.com/rapidsai/raft/pull/2297)) [@divyegala](https://github.com/divyegala)
- Enable warnings as errors for Python tests ([#2288](https://github.com/rapidsai/raft/pull/2288)) [@mroeschke](https://github.com/mroeschke)
- Normalize dataset vectors in the CAGRA InnerProduct tests ([#2287](https://github.com/rapidsai/raft/pull/2287)) [@enp1s0](https://github.com/enp1s0)
- Use dynamic version for raft-ann-bench ([#2285](https://github.com/rapidsai/raft/pull/2285)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Make &#39;librmm&#39; a &#39;host&#39; dependency for conda packages ([#2284](https://github.com/rapidsai/raft/pull/2284)) [@jameslamb](https://github.com/jameslamb)
- Fix comments in cpp/include/raft/neighbors/cagra_serialize.cuh ([#2283](https://github.com/rapidsai/raft/pull/2283)) [@jiangyinzuo](https://github.com/jiangyinzuo)
- Only use functions in the limited API ([#2282](https://github.com/rapidsai/raft/pull/2282)) [@vyasr](https://github.com/vyasr)
- define &#39;ucx&#39; pytest marker ([#2281](https://github.com/rapidsai/raft/pull/2281)) [@jameslamb](https://github.com/jameslamb)
- Migrate to `{{ stdlib(&quot;c&quot;) }}` ([#2278](https://github.com/rapidsai/raft/pull/2278)) [@hcho3](https://github.com/hcho3)
- add --rm and --name to devcontainer run args ([#2275](https://github.com/rapidsai/raft/pull/2275)) [@trxcllnt](https://github.com/trxcllnt)
- Update pip devcontainers to UCX v1.15.0 ([#2274](https://github.com/rapidsai/raft/pull/2274)) [@trxcllnt](https://github.com/trxcllnt)
- `#ifdef` out pragma deprecation warning messages ([#2271](https://github.com/rapidsai/raft/pull/2271)) [@trxcllnt](https://github.com/trxcllnt)
- Convert device_memory_resource* to device_async_resource_ref ([#2269](https://github.com/rapidsai/raft/pull/2269)) [@harrism](https://github.com/harrism)
- Update the developer&#39;s guide with new copyright hook ([#2266](https://github.com/rapidsai/raft/pull/2266)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Improve coalesced reduction performance for tall and thin matrices (up to 2.6x faster) ([#2259](https://github.com/rapidsai/raft/pull/2259)) [@Nyrio](https://github.com/Nyrio)
- Adds missing files to `update-version.sh` ([#2255](https://github.com/rapidsai/raft/pull/2255)) [@AyodeAwe](https://github.com/AyodeAwe)
- Enable all tests for `arm64` jobs ([#2248](https://github.com/rapidsai/raft/pull/2248)) [@galipremsagar](https://github.com/galipremsagar)
- Update nvtx3 link in cmake ([#2246](https://github.com/rapidsai/raft/pull/2246)) [@lowener](https://github.com/lowener)
- Add CAGRA-Q subspace dim = 4 support ([#2244](https://github.com/rapidsai/raft/pull/2244)) [@enp1s0](https://github.com/enp1s0)
- Get rid of `cuco::sentinel` namespace ([#2243](https://github.com/rapidsai/raft/pull/2243)) [@PointKernel](https://github.com/PointKernel)
- Replace usages of raw `get_upstream` with `get_upstream_resource()` ([#2207](https://github.com/rapidsai/raft/pull/2207)) [@miscco](https://github.com/miscco)
- Set the import mode for dask tests ([#2142](https://github.com/rapidsai/raft/pull/2142)) [@vyasr](https://github.com/vyasr)
- Add UCXX support ([#1983](https://github.com/rapidsai/raft/pull/1983)) [@pentschev](https://github.com/pentschev)

# raft 24.04.00 (10 Apr 2024)

## 🐛 Bug Fixes

- Update pre-commit-hooks to v0.0.3 ([#2239](https://github.com/rapidsai/raft/pull/2239)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- MAINT: Simplify NCCL worker rank identification ([#2228](https://github.com/rapidsai/raft/pull/2228)) [@VibhuJawa](https://github.com/VibhuJawa)
- Fix bug in blockRankedReduce ([#2226](https://github.com/rapidsai/raft/pull/2226)) [@akifcorduk](https://github.com/akifcorduk)
- Fix illegal acces mean/stdev, sum add Kahan Summation ([#2223](https://github.com/rapidsai/raft/pull/2223)) [@mfoerste4](https://github.com/mfoerste4)
- Batch cutlass distance kernels along N matrix dim ([#2215](https://github.com/rapidsai/raft/pull/2215)) [@mdoijade](https://github.com/mdoijade)
- Fix out of bounds access in sum kernel ([#2183](https://github.com/rapidsai/raft/pull/2183)) [@tfeher](https://github.com/tfeher)
- Fix ANN bench ground truth generation for k&gt;1024 ([#2180](https://github.com/rapidsai/raft/pull/2180)) [@tfeher](https://github.com/tfeher)
- Fixing cusparse aligned address issue and adding note ([#2179](https://github.com/rapidsai/raft/pull/2179)) [@cjnolet](https://github.com/cjnolet)
- Launch `neighborhood_recall` kernel on CUDA stream ([#2156](https://github.com/rapidsai/raft/pull/2156)) [@divyegala](https://github.com/divyegala)
- Add `compile-library` by default on pylibraft build ([#2090](https://github.com/rapidsai/raft/pull/2090)) [@lowener](https://github.com/lowener)

## 📖 Documentation

- Adding cuVS notice to README and front page of docs. ([#2224](https://github.com/rapidsai/raft/pull/2224)) [@cjnolet](https://github.com/cjnolet)

## 🚀 New Features

- Add CAGRA-Q to ANN benchmarks ([#2233](https://github.com/rapidsai/raft/pull/2233)) [@achirkin](https://github.com/achirkin)
- Add CAGRA-Q build (compression) ([#2213](https://github.com/rapidsai/raft/pull/2213)) [@achirkin](https://github.com/achirkin)
- CAGRA-Q search ([#2206](https://github.com/rapidsai/raft/pull/2206)) [@enp1s0](https://github.com/enp1s0)
- Demangle backtrace symbols on raft error ([#2188](https://github.com/rapidsai/raft/pull/2188)) [@achirkin](https://github.com/achirkin)
- Reapply: Support for fp16 in CAGRA and IVF-PQ ([#2172](https://github.com/rapidsai/raft/pull/2172)) [@achirkin](https://github.com/achirkin)
- Remove supports_streams from custom RAFT memory resources ([#2121](https://github.com/rapidsai/raft/pull/2121)) [@harrism](https://github.com/harrism)
- [FEA] Add support for bitmap_view &amp; the API of `bitmap_to_csr` ([#2109](https://github.com/rapidsai/raft/pull/2109)) [@rhdong](https://github.com/rhdong)

## 🛠️ Improvements

- Use `conda env create --yes` instead of `--force` ([#2247](https://github.com/rapidsai/raft/pull/2247)) [@bdice](https://github.com/bdice)
- Align ucx version pinning with ucx-py/ucxx. ([#2227](https://github.com/rapidsai/raft/pull/2227)) [@bdice](https://github.com/bdice)
- Add upper bound to prevent usage of NumPy 2 ([#2222](https://github.com/rapidsai/raft/pull/2222)) [@bdice](https://github.com/bdice)
- Performance optimization of IVF-flat / select_k ([#2221](https://github.com/rapidsai/raft/pull/2221)) [@mfoerste4](https://github.com/mfoerste4)
- Replace local copyright check with pre-commit-hooks verify-copyright ([#2220](https://github.com/rapidsai/raft/pull/2220)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Remove hard-coding of RAPIDS version where possible ([#2219](https://github.com/rapidsai/raft/pull/2219)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Fix style. ([#2214](https://github.com/rapidsai/raft/pull/2214)) [@bdice](https://github.com/bdice)
- Add explicit instantiations for IVF-PQ search kernels used in tests ([#2212](https://github.com/rapidsai/raft/pull/2212)) [@tfeher](https://github.com/tfeher)
- Improve RBC eps-neighborhood query performance ([#2211](https://github.com/rapidsai/raft/pull/2211)) [@mfoerste4](https://github.com/mfoerste4)
- Add test for spmm ([#2210](https://github.com/rapidsai/raft/pull/2210)) [@mfoerste4](https://github.com/mfoerste4)
- Only install necessary components in conda packages. ([#2209](https://github.com/rapidsai/raft/pull/2209)) [@bdice](https://github.com/bdice)
- Automate C++ include file grouping and ordering using clang-format ([#2202](https://github.com/rapidsai/raft/pull/2202)) [@harrism](https://github.com/harrism)
- Add support for Python 3.11, require NumPy 1.23+ ([#2200](https://github.com/rapidsai/raft/pull/2200)) [@jameslamb](https://github.com/jameslamb)
- Pass `std::optional` instead of `thrust::optional` to RMM ([#2199](https://github.com/rapidsai/raft/pull/2199)) [@trxcllnt](https://github.com/trxcllnt)
- Update devcontainers to CUDA Toolkit 12.2 ([#2192](https://github.com/rapidsai/raft/pull/2192)) [@trxcllnt](https://github.com/trxcllnt)
- target branch-24.04 for GitHub Actions workflows ([#2189](https://github.com/rapidsai/raft/pull/2189)) [@jameslamb](https://github.com/jameslamb)
- Fixing workaround for cuSPARSE bug with correct copy dimensions ([#2185](https://github.com/rapidsai/raft/pull/2185)) [@mfoerste4](https://github.com/mfoerste4)
- Allow topk larger than 1024 in CAGRA ([#2181](https://github.com/rapidsai/raft/pull/2181)) [@benfred](https://github.com/benfred)
- IVF-FLAT support k &gt; 256 ([#2169](https://github.com/rapidsai/raft/pull/2169)) [@mfoerste4](https://github.com/mfoerste4)
- Add environment-agnostic scripts for running ctests and pytests ([#2165](https://github.com/rapidsai/raft/pull/2165)) [@trxcllnt](https://github.com/trxcllnt)
- Ensure that `ctest` is called with `--no-tests=error`. ([#2163](https://github.com/rapidsai/raft/pull/2163)) [@bdice](https://github.com/bdice)
- Update ops-bot.yaml ([#2158](https://github.com/rapidsai/raft/pull/2158)) [@AyodeAwe](https://github.com/AyodeAwe)
- random sampling of dataset rows with improved memory utilization ([#2155](https://github.com/rapidsai/raft/pull/2155)) [@tfeher](https://github.com/tfeher)
- [FIX] Ensure hnswlib can be found from RAFT&#39;s build dir ([#2145](https://github.com/rapidsai/raft/pull/2145)) [@trxcllnt](https://github.com/trxcllnt)
- Improve analysis experience for ANN benchmarks ([#2139](https://github.com/rapidsai/raft/pull/2139)) [@achirkin](https://github.com/achirkin)
- Enable CAGRA index building without adding dataset to the index ([#2126](https://github.com/rapidsai/raft/pull/2126)) [@tfeher](https://github.com/tfeher)
- Add fused cosine 1-NN cutlass based kernel ([#2125](https://github.com/rapidsai/raft/pull/2125)) [@mdoijade](https://github.com/mdoijade)
- Update raft for compatibility with the latest cuco ([#2118](https://github.com/rapidsai/raft/pull/2118)) [@PointKernel](https://github.com/PointKernel)
- Support CUDA 12.2 ([#2092](https://github.com/rapidsai/raft/pull/2092)) [@jameslamb](https://github.com/jameslamb)
- Cache IVF-PQ and select-warpsort kernel launch parameters to reduce latency ([#1786](https://github.com/rapidsai/raft/pull/1786)) [@achirkin](https://github.com/achirkin)

# raft 24.02.00 (12 Feb 2024)

## 🚨 Breaking Changes

- Switch to scikit-build-core ([#2051](https://github.com/rapidsai/raft/pull/2051)) [@vyasr](https://github.com/vyasr)
- Update to CCCL 2.2.0. ([#2049](https://github.com/rapidsai/raft/pull/2049)) [@bdice](https://github.com/bdice)
- Update `raft-ann-bench` output filenames and add features to plotting ([#2043](https://github.com/rapidsai/raft/pull/2043)) [@divyegala](https://github.com/divyegala)
- Remove selection_faiss ([#2027](https://github.com/rapidsai/raft/pull/2027)) [@benfred](https://github.com/benfred)

## 🐛 Bug Fixes

- fix is_row/col_order for strided layouts ([#2173](https://github.com/rapidsai/raft/pull/2173)) [@mfoerste4](https://github.com/mfoerste4)
- Fix failing C++ tests and revert #2097, #2085. ([#2168](https://github.com/rapidsai/raft/pull/2168)) [@cjnolet](https://github.com/cjnolet)
- Exclude tests from builds ([#2162](https://github.com/rapidsai/raft/pull/2162)) [@vyasr](https://github.com/vyasr)
- [HOTFIX] 24.02 Revert Random Sampling ([#2144](https://github.com/rapidsai/raft/pull/2144)) [@cjnolet](https://github.com/cjnolet)
- Pin to pytest 7. ([#2137](https://github.com/rapidsai/raft/pull/2137)) [@bdice](https://github.com/bdice)
- Conditionally include `hnsw` wrapper source in CMake ([#2135](https://github.com/rapidsai/raft/pull/2135)) [@divyegala](https://github.com/divyegala)
- [BUG] Fix `SPMM` strided view ([#2124](https://github.com/rapidsai/raft/pull/2124)) [@lowener](https://github.com/lowener)
- Fixing small bug in CUSPARSE spmm w/ CUDA 12.2 ([#2117](https://github.com/rapidsai/raft/pull/2117)) [@cjnolet](https://github.com/cjnolet)
- [BUG] Fix `num_cta_per_query` div ([#2107](https://github.com/rapidsai/raft/pull/2107)) [@lowener](https://github.com/lowener)
- Remove extraneous host pinnings from libraft-headers-only. ([#2102](https://github.com/rapidsai/raft/pull/2102)) [@bdice](https://github.com/bdice)
- Remove unneeded CI symbol excludes ([#2098](https://github.com/rapidsai/raft/pull/2098)) [@robertmaynard](https://github.com/robertmaynard)
- Properly taking ownership of nccl subcomm (and destroying it) ([#2094](https://github.com/rapidsai/raft/pull/2094)) [@cjnolet](https://github.com/cjnolet)
- Fix `max_queries` for CAGRA ([#2081](https://github.com/rapidsai/raft/pull/2081)) [@lowener](https://github.com/lowener)
- Fix compile failure on RTX 4090 ([#2076](https://github.com/rapidsai/raft/pull/2076)) [@JieFengWang](https://github.com/JieFengWang)
- Fix a crash in FAISS benchmark wrapper introduced in #2021 ([#2062](https://github.com/rapidsai/raft/pull/2062)) [@achirkin](https://github.com/achirkin)
- Correct function that wasn&#39;t returning a value ([#2045](https://github.com/rapidsai/raft/pull/2045)) [@robertmaynard](https://github.com/robertmaynard)
- Fixing small bug in raft-ann-bench ([#2041](https://github.com/rapidsai/raft/pull/2041)) [@cjnolet](https://github.com/cjnolet)
- Make device_resources accessed from device_resources_manager thread-safe ([#2030](https://github.com/rapidsai/raft/pull/2030)) [@wphicks](https://github.com/wphicks)
- Fix ann-bench multithreading ([#2021](https://github.com/rapidsai/raft/pull/2021)) [@achirkin](https://github.com/achirkin)
- Fix `ci/checks/copyright.py` to mirror RAPIDS reference ([#2008](https://github.com/rapidsai/raft/pull/2008)) [@divyegala](https://github.com/divyegala)
- Fix pyproject versions ([#2002](https://github.com/rapidsai/raft/pull/2002)) [@vyasr](https://github.com/vyasr)

## 📖 Documentation

- Adding license info for wiki-all dataset ([#2129](https://github.com/rapidsai/raft/pull/2129)) [@cjnolet](https://github.com/cjnolet)
- [DOC] Documentation updates for release 24.02 ([#2093](https://github.com/rapidsai/raft/pull/2093)) [@cjnolet](https://github.com/cjnolet)
- Fix errors with ingroup exposed by doxygen 1.10 ([#2079](https://github.com/rapidsai/raft/pull/2079)) [@wphicks](https://github.com/wphicks)
- Fix a typo ([#2070](https://github.com/rapidsai/raft/pull/2070)) [@narangvivek10](https://github.com/narangvivek10)
- Add usage example for brute_force::build ([#2029](https://github.com/rapidsai/raft/pull/2029)) [@benfred](https://github.com/benfred)
- Add filtering to vector search tutorial ([#1996](https://github.com/rapidsai/raft/pull/1996)) [@lowener](https://github.com/lowener)

## 🚀 New Features

- Update to use rapids-cmake for all deps ([#2096](https://github.com/rapidsai/raft/pull/2096)) [@robertmaynard](https://github.com/robertmaynard)
- Add IVF-PQ example into the template project ([#2091](https://github.com/rapidsai/raft/pull/2091)) [@achirkin](https://github.com/achirkin)
- Support for fp16 in CAGRA and IVF-PQ ([#2085](https://github.com/rapidsai/raft/pull/2085)) [@achirkin](https://github.com/achirkin)
- Add random subsampling for IVF methods ([#2077](https://github.com/rapidsai/raft/pull/2077)) [@tfeher](https://github.com/tfeher)
- Update `raft-ann-bench` output filenames and add features to plotting ([#2043](https://github.com/rapidsai/raft/pull/2043)) [@divyegala](https://github.com/divyegala)
- Add brute_force index serialization ([#2036](https://github.com/rapidsai/raft/pull/2036)) [@wphicks](https://github.com/wphicks)
- Add eps-neighbor search via RBC ([#2028](https://github.com/rapidsai/raft/pull/2028)) [@mfoerste4](https://github.com/mfoerste4)
- `libraft` and `pylibraft` API for CAGRA build and HNSW search ([#2022](https://github.com/rapidsai/raft/pull/2022)) [@divyegala](https://github.com/divyegala)
- Export Pareto frontier in `raft-ann-bench.data_export` ([#2009](https://github.com/rapidsai/raft/pull/2009)) [@divyegala](https://github.com/divyegala)
- Implement maybe-owning multi-dimensional container (mdbuffer) ([#1999](https://github.com/rapidsai/raft/pull/1999)) [@wphicks](https://github.com/wphicks)
- Add support for 1024+ dim vectors in CAGRA search ([#1994](https://github.com/rapidsai/raft/pull/1994)) [@enp1s0](https://github.com/enp1s0)
- Replace GEMM backend: cublas.gemm -&gt; cublaslt.matmul ([#1736](https://github.com/rapidsai/raft/pull/1736)) [@achirkin](https://github.com/achirkin)

## 🛠️ Improvements

- Remove get_mem_info functions from RAFT custom memory resources ([#2108](https://github.com/rapidsai/raft/pull/2108)) [@harrism](https://github.com/harrism)
- Replace call to mr::get_mem_info() ([#2099](https://github.com/rapidsai/raft/pull/2099)) [@harrism](https://github.com/harrism)
- Allow topk larger than 1024 in CAGRA ([#2097](https://github.com/rapidsai/raft/pull/2097)) [@benfred](https://github.com/benfred)
- Remove usages of rapids-env-update ([#2095](https://github.com/rapidsai/raft/pull/2095)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Provide explicit pool size for pool_memory_resources and clean up includes ([#2088](https://github.com/rapidsai/raft/pull/2088)) [@harrism](https://github.com/harrism)
- refactor CUDA versions in dependencies.yaml ([#2086](https://github.com/rapidsai/raft/pull/2086)) [@jameslamb](https://github.com/jameslamb)
- ANN bench fix latency measurement overhead ([#2084](https://github.com/rapidsai/raft/pull/2084)) [@tfeher](https://github.com/tfeher)
- Remove hardcoded limit in `print_results` function ([#2080](https://github.com/rapidsai/raft/pull/2080)) [@narangvivek10](https://github.com/narangvivek10)
- [FEA] Add support for SDDMM by wrapping the cusparseSDDMM ([#2067) (#2067](https://github.com/rapidsai/raft/pull/2067) (#2067)) [@rhdong](https://github.com/rhdong)
- Benchmark brute force knn ([#2063](https://github.com/rapidsai/raft/pull/2063)) [@benfred](https://github.com/benfred)
- [BUG] fix empty initialization of device_ndarray in pylibraft ([#2061](https://github.com/rapidsai/raft/pull/2061)) [@mfoerste4](https://github.com/mfoerste4)
- Improve parallelism of refine host ([#2059](https://github.com/rapidsai/raft/pull/2059)) [@anaruse](https://github.com/anaruse)
- Subsampling for IVF-PQ codebook generation ([#2052](https://github.com/rapidsai/raft/pull/2052)) [@abc99lr](https://github.com/abc99lr)
- Switch to scikit-build-core ([#2051](https://github.com/rapidsai/raft/pull/2051)) [@vyasr](https://github.com/vyasr)
- Update to CCCL 2.2.0. ([#2049](https://github.com/rapidsai/raft/pull/2049)) [@bdice](https://github.com/bdice)
- Use cuda::proclaim_return_type on device lambda. ([#2048](https://github.com/rapidsai/raft/pull/2048)) [@bdice](https://github.com/bdice)
- Removing code that explicitly compares equality of rmm memory resources ([#2047](https://github.com/rapidsai/raft/pull/2047)) [@cjnolet](https://github.com/cjnolet)
- Add public enum for select-k algorithm selection ([#2046](https://github.com/rapidsai/raft/pull/2046)) [@benfred](https://github.com/benfred)
- Update dependencies.yaml to new pip index ([#2042](https://github.com/rapidsai/raft/pull/2042)) [@vyasr](https://github.com/vyasr)
- Remove RAFT_BUILD_WHEELS and standardize Python builds ([#2040](https://github.com/rapidsai/raft/pull/2040)) [@vyasr](https://github.com/vyasr)
- Fix ucx-py version pinning in dependencies.yaml. ([#2035](https://github.com/rapidsai/raft/pull/2035)) [@bdice](https://github.com/bdice)
- [REVIEW] Fix typos in parameter tuning guide ([#2034](https://github.com/rapidsai/raft/pull/2034)) [@abc99lr](https://github.com/abc99lr)
- Add AIR-Top-k reference ([#2031](https://github.com/rapidsai/raft/pull/2031)) [@tfeher](https://github.com/tfeher)
- Remove selection_faiss ([#2027](https://github.com/rapidsai/raft/pull/2027)) [@benfred](https://github.com/benfred)
- Fixing json parse error in `raft-ann-bench.data_export` ([#2025](https://github.com/rapidsai/raft/pull/2025)) [@cjnolet](https://github.com/cjnolet)
- Updating cagra build constraint ([#2016](https://github.com/rapidsai/raft/pull/2016)) [@cjnolet](https://github.com/cjnolet)
- Update to fmt 10.1.1 and spdlog 1.12.0. ([#1957](https://github.com/rapidsai/raft/pull/1957)) [@bdice](https://github.com/bdice)
- Enable host dataset for IVF-Flat ([#1635](https://github.com/rapidsai/raft/pull/1635)) [@tfeher](https://github.com/tfeher)
- add half/bfloat support to myInf and abs ([#1592](https://github.com/rapidsai/raft/pull/1592)) [@Kh4ster](https://github.com/Kh4ster)

# raft 23.12.00 (6 Dec 2023)

## 🐛 Bug Fixes

- Update actions/labeler to v4 ([#2037](https://github.com/rapidsai/raft/pull/2037)) [@raydouglass](https://github.com/raydouglass)
- pylibraft only depends on numpy at runtime, not build time. ([#2013](https://github.com/rapidsai/raft/pull/2013)) [@bdice](https://github.com/bdice)
- Fixes to update-version.sh ([#1991](https://github.com/rapidsai/raft/pull/1991)) [@raydouglass](https://github.com/raydouglass)
- Adjusting end-to-end start time so it doesn&#39;t include stream creation time ([#1989](https://github.com/rapidsai/raft/pull/1989)) [@cjnolet](https://github.com/cjnolet)
- CAGRA graph optimizer: clamp rev_graph_count ([#1987](https://github.com/rapidsai/raft/pull/1987)) [@tfeher](https://github.com/tfeher)
- Catching conversion errors in data_export instead of fully failing ([#1979](https://github.com/rapidsai/raft/pull/1979)) [@cjnolet](https://github.com/cjnolet)
- Fix syncing mechanism in `raft-ann-bench` C++ search ([#1961](https://github.com/rapidsai/raft/pull/1961)) [@divyegala](https://github.com/divyegala)
- Fixing hnswlib in latency mode ([#1959](https://github.com/rapidsai/raft/pull/1959)) [@cjnolet](https://github.com/cjnolet)
- Fix `ucx-py` alpha version update for `raft-dask` ([#1953](https://github.com/rapidsai/raft/pull/1953)) [@divyegala](https://github.com/divyegala)
- Reduce NN Descent test threshold ([#1946](https://github.com/rapidsai/raft/pull/1946)) [@divyegala](https://github.com/divyegala)
- Fixes to new YAML config `raft-bench-ann` ([#1945](https://github.com/rapidsai/raft/pull/1945)) [@divyegala](https://github.com/divyegala)
- Set RNG seeds in NN Descent to diagnose flaky tests ([#1931](https://github.com/rapidsai/raft/pull/1931)) [@divyegala](https://github.com/divyegala)
- Fix FAISS CPU algorithm names in `raft-ann-bench` ([#1916](https://github.com/rapidsai/raft/pull/1916)) [@divyegala](https://github.com/divyegala)
- Increase iterations in NN Descent tests to avoid flakiness ([#1915](https://github.com/rapidsai/raft/pull/1915)) [@divyegala](https://github.com/divyegala)
- Fix filepath in `raft-ann-bench/split_groundtruth` module ([#1911](https://github.com/rapidsai/raft/pull/1911)) [@divyegala](https://github.com/divyegala)
- Remove dynamic entry-points from raft-ann-bench ([#1910](https://github.com/rapidsai/raft/pull/1910)) [@benfred](https://github.com/benfred)
- Remove unnecessary dataset path check in ANN bench ([#1908](https://github.com/rapidsai/raft/pull/1908)) [@tfeher](https://github.com/tfeher)
- Fixing Googletests and re-enabling in CI ([#1904](https://github.com/rapidsai/raft/pull/1904)) [@cjnolet](https://github.com/cjnolet)
- Fix NN Descent overflows ([#1875](https://github.com/rapidsai/raft/pull/1875)) [@divyegala](https://github.com/divyegala)
- Build fix for CUDA 12.2 ([#1870](https://github.com/rapidsai/raft/pull/1870)) [@benfred](https://github.com/benfred)
- [BUG] Fix a bug in NN descent ([#1869](https://github.com/rapidsai/raft/pull/1869)) [@enp1s0](https://github.com/enp1s0)

## 📖 Documentation

- Brute Force Index documentation fix ([#1944](https://github.com/rapidsai/raft/pull/1944)) [@lowener](https://github.com/lowener)
- Add `wiki_all` dataset config and documentation. ([#1918](https://github.com/rapidsai/raft/pull/1918)) [@cjnolet](https://github.com/cjnolet)
- Updates to raft-ann-bench docs ([#1905](https://github.com/rapidsai/raft/pull/1905)) [@cjnolet](https://github.com/cjnolet)
- End-to-end vector search tutorial in docs ([#1776](https://github.com/rapidsai/raft/pull/1776)) [@cjnolet](https://github.com/cjnolet)

## 🚀 New Features

- Adding `dry-run` option to `raft-ann-bench` ([#1970](https://github.com/rapidsai/raft/pull/1970)) [@cjnolet](https://github.com/cjnolet)
- Add ANN bench scripts to generate ground truth ([#1967](https://github.com/rapidsai/raft/pull/1967)) [@tfeher](https://github.com/tfeher)
- CAGRA build + HNSW search ([#1956](https://github.com/rapidsai/raft/pull/1956)) [@divyegala](https://github.com/divyegala)
- Verify conda-cpp-post-build-checks ([#1935](https://github.com/rapidsai/raft/pull/1935)) [@robertmaynard](https://github.com/robertmaynard)
- Make all cuda kernels have hidden visibility ([#1898](https://github.com/rapidsai/raft/pull/1898)) [@robertmaynard](https://github.com/robertmaynard)
- Update rapids-cmake functions to non-deprecated signatures ([#1884](https://github.com/rapidsai/raft/pull/1884)) [@robertmaynard](https://github.com/robertmaynard)
- [FEA] Helpers for identifying contiguous layouts. ([#1861](https://github.com/rapidsai/raft/pull/1861)) [@trivialfis](https://github.com/trivialfis)
- Add `raft::stats::neighborhood_recall` ([#1860](https://github.com/rapidsai/raft/pull/1860)) [@divyegala](https://github.com/divyegala)
- [FEA] Helpers and CodePacker for IVF-PQ ([#1826](https://github.com/rapidsai/raft/pull/1826)) [@tarang-jain](https://github.com/tarang-jain)

## 🛠️ Improvements

- Pinning fmt and spdlog for raft-ann-bench-cpu ([#2018](https://github.com/rapidsai/raft/pull/2018)) [@cjnolet](https://github.com/cjnolet)
- Build concurrency for nightly and merge triggers ([#2011](https://github.com/rapidsai/raft/pull/2011)) [@bdice](https://github.com/bdice)
- Using `EXPORT_SET` in `rapids_find_package_root` ([#2006](https://github.com/rapidsai/raft/pull/2006)) [@cjnolet](https://github.com/cjnolet)
- Remove static checks for serialization size ([#1997](https://github.com/rapidsai/raft/pull/1997)) [@cjnolet](https://github.com/cjnolet)
- Skipping bad json parse ([#1990](https://github.com/rapidsai/raft/pull/1990)) [@cjnolet](https://github.com/cjnolet)
- Update select-k heuristic ([#1985](https://github.com/rapidsai/raft/pull/1985)) [@benfred](https://github.com/benfred)
- ANN bench: use different offset for each thread ([#1981](https://github.com/rapidsai/raft/pull/1981)) [@tfeher](https://github.com/tfeher)
- Allow `raft-ann-bench/run` to continue after encountering bad YAML configs ([#1980](https://github.com/rapidsai/raft/pull/1980)) [@divyegala](https://github.com/divyegala)
- Add build and search params to `raft-ann-bench.data_export` CSVs ([#1971](https://github.com/rapidsai/raft/pull/1971)) [@divyegala](https://github.com/divyegala)
- Use new `rapids-dask-dependency` metapackage for managing dask versions ([#1968](https://github.com/rapidsai/raft/pull/1968)) [@galipremsagar](https://github.com/galipremsagar)
- Remove unused header ([#1960](https://github.com/rapidsai/raft/pull/1960)) [@wphicks](https://github.com/wphicks)
- Adding pool back in and fixing cagra benchmark params ([#1951](https://github.com/rapidsai/raft/pull/1951)) [@cjnolet](https://github.com/cjnolet)
- Add constraints to `hnswlib` in `raft-bench-ann` ([#1949](https://github.com/rapidsai/raft/pull/1949)) [@divyegala](https://github.com/divyegala)
- Add support for iterating over batches in bfknn ([#1947](https://github.com/rapidsai/raft/pull/1947)) [@benfred](https://github.com/benfred)
- Fix ANN bench latency ([#1940](https://github.com/rapidsai/raft/pull/1940)) [@tfeher](https://github.com/tfeher)
- Add YAML config files to run parameter sweeps for ANN benchmarks ([#1929](https://github.com/rapidsai/raft/pull/1929)) [@divyegala](https://github.com/divyegala)
- Relax ucx pinning ([#1927](https://github.com/rapidsai/raft/pull/1927)) [@vyasr](https://github.com/vyasr)
- Try using contiguous rank to fix cuda_visible_devices ([#1926](https://github.com/rapidsai/raft/pull/1926)) [@VibhuJawa](https://github.com/VibhuJawa)
- Unpin `dask` and `distributed` for `23.12` development ([#1925](https://github.com/rapidsai/raft/pull/1925)) [@galipremsagar](https://github.com/galipremsagar)
- Adding `throughput` and `latency` modes to `raft-ann-bench` ([#1920](https://github.com/rapidsai/raft/pull/1920)) [@cjnolet](https://github.com/cjnolet)
- Providing `aarch64` yaml environment files ([#1914](https://github.com/rapidsai/raft/pull/1914)) [@cjnolet](https://github.com/cjnolet)
- CAGRA ANN bench: parse build options for IVF-PQ build algo ([#1912](https://github.com/rapidsai/raft/pull/1912)) [@tfeher](https://github.com/tfeher)
- Fix python script location in ANN bench description ([#1906](https://github.com/rapidsai/raft/pull/1906)) [@tfeher](https://github.com/tfeher)
- Refactor install/build guide. ([#1899](https://github.com/rapidsai/raft/pull/1899)) [@cjnolet](https://github.com/cjnolet)
- Check return values of raft-ann-bench subprocess calls ([#1897](https://github.com/rapidsai/raft/pull/1897)) [@benfred](https://github.com/benfred)
- ANN bench options to specify CAGRA graph and dataset locations ([#1896](https://github.com/rapidsai/raft/pull/1896)) [@cjnolet](https://github.com/cjnolet)
- Add check-json to pre-commit linters, and fix invalid ann-bench JSON config ([#1894](https://github.com/rapidsai/raft/pull/1894)) [@benfred](https://github.com/benfred)
- Use branch-23.12 workflows. ([#1886](https://github.com/rapidsai/raft/pull/1886)) [@bdice](https://github.com/bdice)
- Setup Consistent Nightly Versions for Pip and Conda ([#1880](https://github.com/rapidsai/raft/pull/1880)) [@divyegala](https://github.com/divyegala)
- Fix and improve one-block radix select ([#1878](https://github.com/rapidsai/raft/pull/1878)) [@yong-wang](https://github.com/yong-wang)
- [FEA] Improvements on bitset class ([#1877](https://github.com/rapidsai/raft/pull/1877)) [@lowener](https://github.com/lowener)
- Branch 23.12 merge 23.10 ([#1873](https://github.com/rapidsai/raft/pull/1873)) [@AyodeAwe](https://github.com/AyodeAwe)
- Branch 23.12 merge 23.10 ([#1868](https://github.com/rapidsai/raft/pull/1868)) [@cjnolet](https://github.com/cjnolet)
- Replace `raft::random` calls to not use deprecated API ([#1867](https://github.com/rapidsai/raft/pull/1867)) [@lowener](https://github.com/lowener)
- raft: Build CUDA 12.0 ARM conda packages. ([#1853](https://github.com/rapidsai/raft/pull/1853)) [@bdice](https://github.com/bdice)
- Documentation for raft ANN benchmark containers. ([#1833](https://github.com/rapidsai/raft/pull/1833)) [@dantegd](https://github.com/dantegd)
- [FEA] Support vector deletion in ANN IVF ([#1831](https://github.com/rapidsai/raft/pull/1831)) [@lowener](https://github.com/lowener)
- Provide a raft::copy overload for mdspan-to-mdspan copies ([#1818](https://github.com/rapidsai/raft/pull/1818)) [@wphicks](https://github.com/wphicks)
- Adding FAISS cpu to `raft-ann-bench` ([#1814](https://github.com/rapidsai/raft/pull/1814)) [@cjnolet](https://github.com/cjnolet)

# raft 23.10.00 (11 Oct 2023)

## 🚨 Breaking Changes

- Change CAGRA auto mode selection ([#1830](https://github.com/rapidsai/raft/pull/1830)) [@enp1s0](https://github.com/enp1s0)
- Update CAGRA serialization ([#1755](https://github.com/rapidsai/raft/pull/1755)) [@benfred](https://github.com/benfred)
- Improvements to ANN Benchmark Python scripts and docs ([#1734](https://github.com/rapidsai/raft/pull/1734)) [@divyegala](https://github.com/divyegala)
- Update to Cython 3.0.0 ([#1688](https://github.com/rapidsai/raft/pull/1688)) [@vyasr](https://github.com/vyasr)
- ANN-benchmarks: switch to use gbench ([#1661](https://github.com/rapidsai/raft/pull/1661)) [@achirkin](https://github.com/achirkin)

## 🐛 Bug Fixes

- [BUG] Fix a bug in the filtering operation in CAGRA multi-kernel ([#1862](https://github.com/rapidsai/raft/pull/1862)) [@enp1s0](https://github.com/enp1s0)
- Fix conf file for benchmarking glove datasets ([#1846](https://github.com/rapidsai/raft/pull/1846)) [@dantegd](https://github.com/dantegd)
- raft-ann-bench package fixes for plotting and conf files ([#1844](https://github.com/rapidsai/raft/pull/1844)) [@dantegd](https://github.com/dantegd)
- Fix update-version.sh for all pyproject.toml files ([#1839](https://github.com/rapidsai/raft/pull/1839)) [@raydouglass](https://github.com/raydouglass)
- Make RMM a run dependency of the raft-ann-bench conda package ([#1838](https://github.com/rapidsai/raft/pull/1838)) [@dantegd](https://github.com/dantegd)
- Printing actual exception in `require base set` ([#1816](https://github.com/rapidsai/raft/pull/1816)) [@cjnolet](https://github.com/cjnolet)
- Adding rmm to `raft-ann-bench` dependencies ([#1815](https://github.com/rapidsai/raft/pull/1815)) [@cjnolet](https://github.com/cjnolet)
- Use `conda mambabuild` not `mamba mambabuild` ([#1812](https://github.com/rapidsai/raft/pull/1812)) [@bdice](https://github.com/bdice)
- Fix `raft-dask` naming in wheel builds ([#1805](https://github.com/rapidsai/raft/pull/1805)) [@divyegala](https://github.com/divyegala)
- neighbors::refine_host: check the dataset bounds ([#1793](https://github.com/rapidsai/raft/pull/1793)) [@achirkin](https://github.com/achirkin)
- [BUG] Fix search parameter check in CAGRA ([#1784](https://github.com/rapidsai/raft/pull/1784)) [@enp1s0](https://github.com/enp1s0)
- IVF-Flat: fix search batching ([#1764](https://github.com/rapidsai/raft/pull/1764)) [@achirkin](https://github.com/achirkin)
- Using expanded distance computations in `pylibraft` ([#1759](https://github.com/rapidsai/raft/pull/1759)) [@cjnolet](https://github.com/cjnolet)
- Fix ann-bench Documentation ([#1754](https://github.com/rapidsai/raft/pull/1754)) [@divyegala](https://github.com/divyegala)
- Make get_cache_idx a weak symbol with dummy template ([#1733](https://github.com/rapidsai/raft/pull/1733)) [@ahendriksen](https://github.com/ahendriksen)
- Fix IVF-PQ fused kernel performance problems ([#1726](https://github.com/rapidsai/raft/pull/1726)) [@achirkin](https://github.com/achirkin)
- Fix build.sh to enable NEIGHBORS_ANN_CAGRA_TEST ([#1724](https://github.com/rapidsai/raft/pull/1724)) [@enp1s0](https://github.com/enp1s0)
- Fix template types for create_descriptor function. ([#1680](https://github.com/rapidsai/raft/pull/1680)) [@csadorf](https://github.com/csadorf)

## 📖 Documentation

- Fix the CAGRA paper citation ([#1788](https://github.com/rapidsai/raft/pull/1788)) [@enp1s0](https://github.com/enp1s0)
- Add citation info for the CAGRA paper preprint ([#1787](https://github.com/rapidsai/raft/pull/1787)) [@enp1s0](https://github.com/enp1s0)
- [DOC] Fix grouping for ANN in C++ doxygen ([#1782](https://github.com/rapidsai/raft/pull/1782)) [@lowener](https://github.com/lowener)
- Update RAFT documentation ([#1717](https://github.com/rapidsai/raft/pull/1717)) [@lowener](https://github.com/lowener)
- Additional polishing of README and docs ([#1713](https://github.com/rapidsai/raft/pull/1713)) [@cjnolet](https://github.com/cjnolet)

## 🚀 New Features

- [FEA] Add `bitset_filter` for CAGRA indices removal ([#1837](https://github.com/rapidsai/raft/pull/1837)) [@lowener](https://github.com/lowener)
- ann-bench: miscellaneous improvements ([#1808](https://github.com/rapidsai/raft/pull/1808)) [@achirkin](https://github.com/achirkin)
- [FEA] Add bitset for ANN pre-filtering and deletion ([#1803](https://github.com/rapidsai/raft/pull/1803)) [@lowener](https://github.com/lowener)
- Adding config files for remaining (relevant) ann-benchmarks million-scale datasets ([#1761](https://github.com/rapidsai/raft/pull/1761)) [@cjnolet](https://github.com/cjnolet)
- Port NN-descent algorithm to use in `cagra::build()` ([#1748](https://github.com/rapidsai/raft/pull/1748)) [@divyegala](https://github.com/divyegala)
- Adding conda build for libraft static ([#1746](https://github.com/rapidsai/raft/pull/1746)) [@cjnolet](https://github.com/cjnolet)
- [FEA] Provide device_resources_manager for easy generation of device_resources ([#1716](https://github.com/rapidsai/raft/pull/1716)) [@wphicks](https://github.com/wphicks)

## 🛠️ Improvements

- Add option to brute_force index to store non-owning reference to norms ([#1865](https://github.com/rapidsai/raft/pull/1865)) [@benfred](https://github.com/benfred)
- Pin `dask` and `distributed` for `23.10` release ([#1864](https://github.com/rapidsai/raft/pull/1864)) [@galipremsagar](https://github.com/galipremsagar)
- Update image names ([#1835](https://github.com/rapidsai/raft/pull/1835)) [@AyodeAwe](https://github.com/AyodeAwe)
- Fixes for OOM during CAGRA benchmarks ([#1832](https://github.com/rapidsai/raft/pull/1832)) [@benfred](https://github.com/benfred)
- Change CAGRA auto mode selection ([#1830](https://github.com/rapidsai/raft/pull/1830)) [@enp1s0](https://github.com/enp1s0)
- Update to clang 16.0.6. ([#1829](https://github.com/rapidsai/raft/pull/1829)) [@bdice](https://github.com/bdice)
- Add IVF-Flat C++ example ([#1828](https://github.com/rapidsai/raft/pull/1828)) [@tfeher](https://github.com/tfeher)
- matrix::select_k: extra tests and benchmarks ([#1821](https://github.com/rapidsai/raft/pull/1821)) [@achirkin](https://github.com/achirkin)
- Add index class for brute_force knn ([#1817](https://github.com/rapidsai/raft/pull/1817)) [@benfred](https://github.com/benfred)
- [FEA] Add pre-filtering to CAGRA ([#1811](https://github.com/rapidsai/raft/pull/1811)) [@enp1s0](https://github.com/enp1s0)
- More updates to ann-bench docs ([#1810](https://github.com/rapidsai/raft/pull/1810)) [@cjnolet](https://github.com/cjnolet)
- Add best deep-100M configs for IVF-PQ to ANN benchmarks ([#1807](https://github.com/rapidsai/raft/pull/1807)) [@tfeher](https://github.com/tfeher)
- A few fixes to `raft-ann-bench` recipe and docs ([#1806](https://github.com/rapidsai/raft/pull/1806)) [@cjnolet](https://github.com/cjnolet)
- Simplify wheel build scripts and allow alphas of RAPIDS dependencies ([#1804](https://github.com/rapidsai/raft/pull/1804)) [@divyegala](https://github.com/divyegala)
- Various fixes to reproducible benchmarks ([#1800](https://github.com/rapidsai/raft/pull/1800)) [@cjnolet](https://github.com/cjnolet)
- ANN-bench: more flexible cuda_stub.hpp ([#1792](https://github.com/rapidsai/raft/pull/1792)) [@achirkin](https://github.com/achirkin)
- Add RAFT devcontainers ([#1791](https://github.com/rapidsai/raft/pull/1791)) [@trxcllnt](https://github.com/trxcllnt)
- Cagra memory optimizations ([#1790](https://github.com/rapidsai/raft/pull/1790)) [@benfred](https://github.com/benfred)
- Fixing a couple security concerns in `raft-dask` nccl unique id generation ([#1785](https://github.com/rapidsai/raft/pull/1785)) [@cjnolet](https://github.com/cjnolet)
- Don&#39;t serialize dataset with CAGRA bench ([#1781](https://github.com/rapidsai/raft/pull/1781)) [@benfred](https://github.com/benfred)
- Use `copy-pr-bot` ([#1774](https://github.com/rapidsai/raft/pull/1774)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add GPU and CPU packages for ANN benchmarks ([#1773](https://github.com/rapidsai/raft/pull/1773)) [@dantegd](https://github.com/dantegd)
- Improvements to raft-ann-bench scripts, docs, and benchmarking implementations. ([#1769](https://github.com/rapidsai/raft/pull/1769)) [@cjnolet](https://github.com/cjnolet)
- [REVIEW] Introducing host API for PCG ([#1767](https://github.com/rapidsai/raft/pull/1767)) [@vinaydes](https://github.com/vinaydes)
- Unpin `dask` and `distributed` for `23.10` development ([#1760](https://github.com/rapidsai/raft/pull/1760)) [@galipremsagar](https://github.com/galipremsagar)
- Add ivf-flat notebook ([#1758](https://github.com/rapidsai/raft/pull/1758)) [@tfeher](https://github.com/tfeher)
- Update CAGRA serialization ([#1755](https://github.com/rapidsai/raft/pull/1755)) [@benfred](https://github.com/benfred)
- Remove block size template parameter from CAGRA search ([#1740](https://github.com/rapidsai/raft/pull/1740)) [@enp1s0](https://github.com/enp1s0)
- Add NVTX ranges for cagra search/serialize functions ([#1737](https://github.com/rapidsai/raft/pull/1737)) [@benfred](https://github.com/benfred)
- Improvements to ANN Benchmark Python scripts and docs ([#1734](https://github.com/rapidsai/raft/pull/1734)) [@divyegala](https://github.com/divyegala)
- Fixing forward merger for 23.08 -&gt; 23.10 ([#1731](https://github.com/rapidsai/raft/pull/1731)) [@cjnolet](https://github.com/cjnolet)
- [FEA] Use CAGRA in C++ template ([#1730](https://github.com/rapidsai/raft/pull/1730)) [@lowener](https://github.com/lowener)
- fixed box around raft image ([#1710](https://github.com/rapidsai/raft/pull/1710)) [@nwstephens](https://github.com/nwstephens)
- Enable CUTLASS-based distance kernels on CTK 12 ([#1702](https://github.com/rapidsai/raft/pull/1702)) [@ahendriksen](https://github.com/ahendriksen)
- Update bench-ann configuration ([#1696](https://github.com/rapidsai/raft/pull/1696)) [@lowener](https://github.com/lowener)
- Update to Cython 3.0.0 ([#1688](https://github.com/rapidsai/raft/pull/1688)) [@vyasr](https://github.com/vyasr)
- Update CMake version ([#1677](https://github.com/rapidsai/raft/pull/1677)) [@vyasr](https://github.com/vyasr)
- Branch 23.10 merge 23.08 ([#1672](https://github.com/rapidsai/raft/pull/1672)) [@vyasr](https://github.com/vyasr)
- ANN-benchmarks: switch to use gbench ([#1661](https://github.com/rapidsai/raft/pull/1661)) [@achirkin](https://github.com/achirkin)

# raft 23.08.00 (9 Aug 2023)

## 🚨 Breaking Changes

- Separate CAGRA index type from internal idx type ([#1664](https://github.com/rapidsai/raft/pull/1664)) [@tfeher](https://github.com/tfeher)
- Stop using setup.py in build.sh ([#1645](https://github.com/rapidsai/raft/pull/1645)) [@vyasr](https://github.com/vyasr)
- CAGRA max_queries auto configuration ([#1613](https://github.com/rapidsai/raft/pull/1613)) [@enp1s0](https://github.com/enp1s0)
- Rename the CAGRA prune function to optimize ([#1588](https://github.com/rapidsai/raft/pull/1588)) [@enp1s0](https://github.com/enp1s0)
- CAGRA pad dataset for 128bit vectorized load ([#1505](https://github.com/rapidsai/raft/pull/1505)) [@tfeher](https://github.com/tfeher)
- Sparse Pairwise Distances API Updates ([#1502](https://github.com/rapidsai/raft/pull/1502)) [@divyegala](https://github.com/divyegala)
- Cagra index construction without copying device mdarrays ([#1494](https://github.com/rapidsai/raft/pull/1494)) [@tfeher](https://github.com/tfeher)
- [FEA] Masked NN for connect_components ([#1445](https://github.com/rapidsai/raft/pull/1445)) [@tarang-jain](https://github.com/tarang-jain)
- Limiting workspace memory resource ([#1356](https://github.com/rapidsai/raft/pull/1356)) [@achirkin](https://github.com/achirkin)

## 🐛 Bug Fixes

- Remove push condition on docs-build ([#1693](https://github.com/rapidsai/raft/pull/1693)) [@raydouglass](https://github.com/raydouglass)
- IVF-PQ: Fix illegal memory access with large max_samples ([#1685](https://github.com/rapidsai/raft/pull/1685)) [@achirkin](https://github.com/achirkin)
- Fix missing parameter for select_k ([#1682](https://github.com/rapidsai/raft/pull/1682)) [@ucassjy](https://github.com/ucassjy)
- Separate CAGRA index type from internal idx type ([#1664](https://github.com/rapidsai/raft/pull/1664)) [@tfeher](https://github.com/tfeher)
- Add rmm to pylibraft run dependencies, since it is used by Cython. ([#1656](https://github.com/rapidsai/raft/pull/1656)) [@bdice](https://github.com/bdice)
- Hotfix: wrong constant in IVF-PQ fp_8bit2half ([#1654](https://github.com/rapidsai/raft/pull/1654)) [@achirkin](https://github.com/achirkin)
- Fix sparse KNN for large batches ([#1640](https://github.com/rapidsai/raft/pull/1640)) [@viclafargue](https://github.com/viclafargue)
- Fix uploading of RAFT nightly packages ([#1638](https://github.com/rapidsai/raft/pull/1638)) [@dantegd](https://github.com/dantegd)
- Fix cagra multi CTA bug ([#1628](https://github.com/rapidsai/raft/pull/1628)) [@enp1s0](https://github.com/enp1s0)
- pass correct stream to cutlass kernel launch of L2/cosine pairwise distance kernels ([#1597](https://github.com/rapidsai/raft/pull/1597)) [@mdoijade](https://github.com/mdoijade)
- Fix launchconfig y-gridsize too large in epilogue kernel ([#1586](https://github.com/rapidsai/raft/pull/1586)) [@mfoerste4](https://github.com/mfoerste4)
- Fix update version and pinnings for 23.08. ([#1556](https://github.com/rapidsai/raft/pull/1556)) [@bdice](https://github.com/bdice)
- Fix for function exposing KNN merge ([#1418](https://github.com/rapidsai/raft/pull/1418)) [@viclafargue](https://github.com/viclafargue)

## 📖 Documentation

- Critical doc fixes and updates for 23.08 ([#1705](https://github.com/rapidsai/raft/pull/1705)) [@cjnolet](https://github.com/cjnolet)
- Fix the documentation about changing the logging level ([#1596](https://github.com/rapidsai/raft/pull/1596)) [@enp1s0](https://github.com/enp1s0)
- Fix raft::bitonic_sort small usage example ([#1580](https://github.com/rapidsai/raft/pull/1580)) [@enp1s0](https://github.com/enp1s0)

## 🚀 New Features

- Use rapids-cmake new parallel testing feature ([#1623](https://github.com/rapidsai/raft/pull/1623)) [@robertmaynard](https://github.com/robertmaynard)
- Add support for row-major slice ([#1591](https://github.com/rapidsai/raft/pull/1591)) [@lowener](https://github.com/lowener)
- IVF-PQ tutorial notebook ([#1544](https://github.com/rapidsai/raft/pull/1544)) [@achirkin](https://github.com/achirkin)
- [FEA] Masked NN for connect_components ([#1445](https://github.com/rapidsai/raft/pull/1445)) [@tarang-jain](https://github.com/tarang-jain)
- raft: Build CUDA 12 packages ([#1388](https://github.com/rapidsai/raft/pull/1388)) [@vyasr](https://github.com/vyasr)
- Limiting workspace memory resource ([#1356](https://github.com/rapidsai/raft/pull/1356)) [@achirkin](https://github.com/achirkin)

## 🛠️ Improvements

- Pin `dask` and `distributed` for `23.08` release ([#1711](https://github.com/rapidsai/raft/pull/1711)) [@galipremsagar](https://github.com/galipremsagar)
- Add algo parameter for CAGRA ANN bench ([#1687](https://github.com/rapidsai/raft/pull/1687)) [@tfeher](https://github.com/tfeher)
- ANN benchmarks python wrapper for splitting billion-scale dataset groundtruth ([#1679](https://github.com/rapidsai/raft/pull/1679)) [@divyegala](https://github.com/divyegala)
- Rename CAGRA parameter num_parents to search_width ([#1676](https://github.com/rapidsai/raft/pull/1676)) [@tfeher](https://github.com/tfeher)
- Renaming namespaces to promote CAGRA from experimental ([#1666](https://github.com/rapidsai/raft/pull/1666)) [@cjnolet](https://github.com/cjnolet)
- CAGRA Python wrappers ([#1665](https://github.com/rapidsai/raft/pull/1665)) [@dantegd](https://github.com/dantegd)
- Add notebook for Vector Search - Question Retrieval ([#1662](https://github.com/rapidsai/raft/pull/1662)) [@lowener](https://github.com/lowener)
- Fix CMake CUDA support for pylibraft when raft is found. ([#1659](https://github.com/rapidsai/raft/pull/1659)) [@bdice](https://github.com/bdice)
- Cagra ANN benchmark improvements ([#1658](https://github.com/rapidsai/raft/pull/1658)) [@tfeher](https://github.com/tfeher)
- ANN-benchmarks: avoid using the dataset during search when possible ([#1657](https://github.com/rapidsai/raft/pull/1657)) [@achirkin](https://github.com/achirkin)
- Revert CUDA 12.0 CI workflows to branch-23.08. ([#1652](https://github.com/rapidsai/raft/pull/1652)) [@bdice](https://github.com/bdice)
- ANN: Optimize host-side refine ([#1651](https://github.com/rapidsai/raft/pull/1651)) [@achirkin](https://github.com/achirkin)
- Cagra template instantiations ([#1650](https://github.com/rapidsai/raft/pull/1650)) [@tfeher](https://github.com/tfeher)
- Modify comm_split to avoid ucp ([#1649](https://github.com/rapidsai/raft/pull/1649)) [@ChuckHastings](https://github.com/ChuckHastings)
- Stop using setup.py in build.sh ([#1645](https://github.com/rapidsai/raft/pull/1645)) [@vyasr](https://github.com/vyasr)
- IVF-PQ: Add a (faster) direct conversion fp8-&gt;half ([#1644](https://github.com/rapidsai/raft/pull/1644)) [@achirkin](https://github.com/achirkin)
- Simplify `bench/ann` scripts to Python based module ([#1642](https://github.com/rapidsai/raft/pull/1642)) [@divyegala](https://github.com/divyegala)
- Further removal of uses-setup-env-vars ([#1639](https://github.com/rapidsai/raft/pull/1639)) [@dantegd](https://github.com/dantegd)
- Drop blank line in `raft-dask/meta.yaml` ([#1637](https://github.com/rapidsai/raft/pull/1637)) [@jakirkham](https://github.com/jakirkham)
- Enable conservative memory allocations for RAFT IVF-Flat benchmarks. ([#1634](https://github.com/rapidsai/raft/pull/1634)) [@tfeher](https://github.com/tfeher)
- [FEA] Codepacking for IVF-flat ([#1632](https://github.com/rapidsai/raft/pull/1632)) [@tarang-jain](https://github.com/tarang-jain)
- Fixing ann bench cmake (and docs) ([#1630](https://github.com/rapidsai/raft/pull/1630)) [@cjnolet](https://github.com/cjnolet)
- [WIP] Test  CI issues ([#1626](https://github.com/rapidsai/raft/pull/1626)) [@VibhuJawa](https://github.com/VibhuJawa)
- Set pool memory resource for raft IVF ANN benchmarks ([#1625](https://github.com/rapidsai/raft/pull/1625)) [@tfeher](https://github.com/tfeher)
- Adding sort option to matrix::select_k api ([#1615](https://github.com/rapidsai/raft/pull/1615)) [@cjnolet](https://github.com/cjnolet)
- CAGRA max_queries auto configuration ([#1613](https://github.com/rapidsai/raft/pull/1613)) [@enp1s0](https://github.com/enp1s0)
- Use exceptions instead of `exit(-1)` ([#1594](https://github.com/rapidsai/raft/pull/1594)) [@benfred](https://github.com/benfred)
- [REVIEW] Add scheduler_file argument to support MNMG setup ([#1593](https://github.com/rapidsai/raft/pull/1593)) [@VibhuJawa](https://github.com/VibhuJawa)
- Rename the CAGRA prune function to optimize ([#1588](https://github.com/rapidsai/raft/pull/1588)) [@enp1s0](https://github.com/enp1s0)
- This PR adds support to __half and nb_bfloat16 to myAtomicReduce ([#1585](https://github.com/rapidsai/raft/pull/1585)) [@Kh4ster](https://github.com/Kh4ster)
- [IMP] move core CUDA RT macros to cuda_rt_essentials.hpp ([#1584](https://github.com/rapidsai/raft/pull/1584)) [@MatthiasKohl](https://github.com/MatthiasKohl)
- preprocessor syntax fix ([#1582](https://github.com/rapidsai/raft/pull/1582)) [@AyodeAwe](https://github.com/AyodeAwe)
- use rapids-upload-docs script ([#1578](https://github.com/rapidsai/raft/pull/1578)) [@AyodeAwe](https://github.com/AyodeAwe)
- Unpin `dask` and `distributed` for development and fix `merge_labels` test ([#1574](https://github.com/rapidsai/raft/pull/1574)) [@galipremsagar](https://github.com/galipremsagar)
- Remove documentation build scripts for Jenkins ([#1570](https://github.com/rapidsai/raft/pull/1570)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add support to __half and nv_bfloat16 to most math functions ([#1554](https://github.com/rapidsai/raft/pull/1554)) [@Kh4ster](https://github.com/Kh4ster)
- Add RAFT ANN benchmark for CAGRA ([#1552](https://github.com/rapidsai/raft/pull/1552)) [@enp1s0](https://github.com/enp1s0)
- Update CAGRA knn_graph_sort to use Raft::bitonic_sort ([#1550](https://github.com/rapidsai/raft/pull/1550)) [@enp1s0](https://github.com/enp1s0)
- Add identity matrix function ([#1548](https://github.com/rapidsai/raft/pull/1548)) [@lowener](https://github.com/lowener)
- Unpin scikit-build upper bound ([#1547](https://github.com/rapidsai/raft/pull/1547)) [@vyasr](https://github.com/vyasr)
- Migrate wheel workflow scripts locally ([#1546](https://github.com/rapidsai/raft/pull/1546)) [@divyegala](https://github.com/divyegala)
- Add sample filtering for ivf_flat. Filtering code refactoring and cleanup ([#1541](https://github.com/rapidsai/raft/pull/1541)) [@alexanderguzhva](https://github.com/alexanderguzhva)
- CAGRA pad dataset for 128bit vectorized load ([#1505](https://github.com/rapidsai/raft/pull/1505)) [@tfeher](https://github.com/tfeher)
- Sparse Pairwise Distances API Updates ([#1502](https://github.com/rapidsai/raft/pull/1502)) [@divyegala](https://github.com/divyegala)
- Add CAGRA gbench ([#1496](https://github.com/rapidsai/raft/pull/1496)) [@tfeher](https://github.com/tfeher)
- Cagra index construction without copying device mdarrays ([#1494](https://github.com/rapidsai/raft/pull/1494)) [@tfeher](https://github.com/tfeher)

# raft 23.06.00 (7 Jun 2023)

## 🚨 Breaking Changes

- ivf-pq::search: fix the indexing type of the query-related mdspan arguments ([#1539](https://github.com/rapidsai/raft/pull/1539)) [@achirkin](https://github.com/achirkin)
- Dropping Python 3.8 ([#1454](https://github.com/rapidsai/raft/pull/1454)) [@divyegala](https://github.com/divyegala)

## 🐛 Bug Fixes

- [HOTFIX] Fix  distance metrics L2/cosine/correlation when X &amp; Y are same buffer but with different shape and add unit test for such case. ([#1571](https://github.com/rapidsai/raft/pull/1571)) [@mdoijade](https://github.com/mdoijade)
- Using raft::resources in rsvd ([#1543](https://github.com/rapidsai/raft/pull/1543)) [@cjnolet](https://github.com/cjnolet)
- ivf-pq::search: fix the indexing type of the query-related mdspan arguments ([#1539](https://github.com/rapidsai/raft/pull/1539)) [@achirkin](https://github.com/achirkin)
- Check python brute-force knn inputs ([#1537](https://github.com/rapidsai/raft/pull/1537)) [@benfred](https://github.com/benfred)
- Fix failing TiledKNNTest unittest ([#1533](https://github.com/rapidsai/raft/pull/1533)) [@benfred](https://github.com/benfred)
- ivf-flat: fix incorrect recomputed size of the index ([#1525](https://github.com/rapidsai/raft/pull/1525)) [@achirkin](https://github.com/achirkin)
- ivf-flat: limit the workspace size of the search via batching ([#1515](https://github.com/rapidsai/raft/pull/1515)) [@achirkin](https://github.com/achirkin)
- Support uint64_t in CAGRA index data type ([#1514](https://github.com/rapidsai/raft/pull/1514)) [@enp1s0](https://github.com/enp1s0)
- Workaround for cuda 12 issue in cusparse ([#1508](https://github.com/rapidsai/raft/pull/1508)) [@cjnolet](https://github.com/cjnolet)
- Un-scale output distances ([#1499](https://github.com/rapidsai/raft/pull/1499)) [@achirkin](https://github.com/achirkin)
- Inline get_cache_idx ([#1492](https://github.com/rapidsai/raft/pull/1492)) [@ahendriksen](https://github.com/ahendriksen)
- Pin to scikit-build&lt;17.2 ([#1487](https://github.com/rapidsai/raft/pull/1487)) [@vyasr](https://github.com/vyasr)
- Remove pool_size() calls from debug printouts ([#1484](https://github.com/rapidsai/raft/pull/1484)) [@tfeher](https://github.com/tfeher)
- Add missing ext declaration for log detail::format ([#1482](https://github.com/rapidsai/raft/pull/1482)) [@tfeher](https://github.com/tfeher)
- Remove include statements from inside namespace ([#1467](https://github.com/rapidsai/raft/pull/1467)) [@robertmaynard](https://github.com/robertmaynard)
- Use pin_compatible to ensure that lower CTKs can be used ([#1462](https://github.com/rapidsai/raft/pull/1462)) [@vyasr](https://github.com/vyasr)
- fix ivf_pq n_probes ([#1456](https://github.com/rapidsai/raft/pull/1456)) [@benfred](https://github.com/benfred)
- The glog project root CMakeLists.txt is where we should build from ([#1442](https://github.com/rapidsai/raft/pull/1442)) [@robertmaynard](https://github.com/robertmaynard)
- Add missing resource factory virtual destructor ([#1433](https://github.com/rapidsai/raft/pull/1433)) [@cjnolet](https://github.com/cjnolet)
- Removing cuda stream view include from mdarray ([#1429](https://github.com/rapidsai/raft/pull/1429)) [@cjnolet](https://github.com/cjnolet)
- Fix dim param for IVF-PQ wrapper in ANN bench ([#1427](https://github.com/rapidsai/raft/pull/1427)) [@tfeher](https://github.com/tfeher)
- Remove MetricProcessor code from brute_force::knn ([#1426](https://github.com/rapidsai/raft/pull/1426)) [@benfred](https://github.com/benfred)
- Fix is_min_close ([#1419](https://github.com/rapidsai/raft/pull/1419)) [@benfred](https://github.com/benfred)
- Have consistent compile lines between BUILD_TESTS enabled or not ([#1401](https://github.com/rapidsai/raft/pull/1401)) [@robertmaynard](https://github.com/robertmaynard)
- Fix ucx-py pin in raft-dask recipe ([#1396](https://github.com/rapidsai/raft/pull/1396)) [@vyasr](https://github.com/vyasr)

## 📖 Documentation

- Various updates to the docs for 23.06 release ([#1538](https://github.com/rapidsai/raft/pull/1538)) [@cjnolet](https://github.com/cjnolet)
- Rename kernel arch finding function for dispatch ([#1536](https://github.com/rapidsai/raft/pull/1536)) [@mdoijade](https://github.com/mdoijade)
- Adding bfknn and ivf-pq python api to docs ([#1507](https://github.com/rapidsai/raft/pull/1507)) [@cjnolet](https://github.com/cjnolet)
- Add RAPIDS cuDF as a library that supports cuda_array_interface ([#1444](https://github.com/rapidsai/raft/pull/1444)) [@miguelusque](https://github.com/miguelusque)

## 🚀 New Features

- IVF-PQ: manipulating individual lists ([#1298](https://github.com/rapidsai/raft/pull/1298)) [@achirkin](https://github.com/achirkin)
- Gram matrix support for sparse input ([#1296](https://github.com/rapidsai/raft/pull/1296)) [@mfoerste4](https://github.com/mfoerste4)
- [FEA] Add randomized svd from cusolver ([#1000](https://github.com/rapidsai/raft/pull/1000)) [@lowener](https://github.com/lowener)

## 🛠️ Improvements

- Require Numba 0.57.0+ ([#1559](https://github.com/rapidsai/raft/pull/1559)) [@jakirkham](https://github.com/jakirkham)
- remove device_resources include from linalg::map ([#1540](https://github.com/rapidsai/raft/pull/1540)) [@benfred](https://github.com/benfred)
- Learn heuristic to pick fastest select_k algorithm ([#1523](https://github.com/rapidsai/raft/pull/1523)) [@benfred](https://github.com/benfred)
- [REVIEW] make raft::cache::Cache protected to allow overrides ([#1522](https://github.com/rapidsai/raft/pull/1522)) [@mfoerste4](https://github.com/mfoerste4)
- [REVIEW] Fix padding assertion in sparse Gram evaluation ([#1521](https://github.com/rapidsai/raft/pull/1521)) [@mfoerste4](https://github.com/mfoerste4)
- run docs nightly too ([#1520](https://github.com/rapidsai/raft/pull/1520)) [@AyodeAwe](https://github.com/AyodeAwe)
- Switch back to using primary shared-action-workflows branch ([#1519](https://github.com/rapidsai/raft/pull/1519)) [@vyasr](https://github.com/vyasr)
- Python API for IVF-Flat serialization ([#1516](https://github.com/rapidsai/raft/pull/1516)) [@tfeher](https://github.com/tfeher)
- Introduce sample filtering to IVFPQ index search ([#1513](https://github.com/rapidsai/raft/pull/1513)) [@alexanderguzhva](https://github.com/alexanderguzhva)
- Migrate from raft::device_resources -&gt; raft::resources ([#1510](https://github.com/rapidsai/raft/pull/1510)) [@benfred](https://github.com/benfred)
- Use rmm allocator in CAGRA prune ([#1503](https://github.com/rapidsai/raft/pull/1503)) [@enp1s0](https://github.com/enp1s0)
- Update recipes to GTest version &gt;=1.13.0 ([#1501](https://github.com/rapidsai/raft/pull/1501)) [@bdice](https://github.com/bdice)
- Remove raft/matrix/matrix.cuh includes ([#1498](https://github.com/rapidsai/raft/pull/1498)) [@benfred](https://github.com/benfred)
- Generate dataset of select_k times ([#1497](https://github.com/rapidsai/raft/pull/1497)) [@benfred](https://github.com/benfred)
- Re-use memory pool between benchmark runs ([#1495](https://github.com/rapidsai/raft/pull/1495)) [@benfred](https://github.com/benfred)
- Support CUDA 12.0 for pip wheels ([#1489](https://github.com/rapidsai/raft/pull/1489)) [@divyegala](https://github.com/divyegala)
- Update cupy dependency ([#1488](https://github.com/rapidsai/raft/pull/1488)) [@vyasr](https://github.com/vyasr)
- Enable sccache hits from local builds ([#1478](https://github.com/rapidsai/raft/pull/1478)) [@AyodeAwe](https://github.com/AyodeAwe)
- Build wheels using new single image workflow ([#1477](https://github.com/rapidsai/raft/pull/1477)) [@vyasr](https://github.com/vyasr)
- Revert shared-action-workflows pin ([#1475](https://github.com/rapidsai/raft/pull/1475)) [@divyegala](https://github.com/divyegala)
- CAGRA: Separate graph index sorting functionality from prune function ([#1471](https://github.com/rapidsai/raft/pull/1471)) [@enp1s0](https://github.com/enp1s0)
- Add generic reduction functions and separate reductions/warp_primitives ([#1470](https://github.com/rapidsai/raft/pull/1470)) [@akifcorduk](https://github.com/akifcorduk)
- [ENH] [FINAL] Header structure: combine all PRs into one ([#1469](https://github.com/rapidsai/raft/pull/1469)) [@ahendriksen](https://github.com/ahendriksen)
- use `matrix::select_k` in brute_force::knn call ([#1463](https://github.com/rapidsai/raft/pull/1463)) [@benfred](https://github.com/benfred)
- Dropping Python 3.8 ([#1454](https://github.com/rapidsai/raft/pull/1454)) [@divyegala](https://github.com/divyegala)
- Fix linalg::map to work with non-power-of-2-sized types again ([#1453](https://github.com/rapidsai/raft/pull/1453)) [@ahendriksen](https://github.com/ahendriksen)
- [ENH] Enable building with clang (limit strict error checking to GCC) ([#1452](https://github.com/rapidsai/raft/pull/1452)) [@ahendriksen](https://github.com/ahendriksen)
- Remove usage of rapids-get-rapids-version-from-git ([#1436](https://github.com/rapidsai/raft/pull/1436)) [@jjacobelli](https://github.com/jjacobelli)
- Minor Updates to Sparse Structures ([#1432](https://github.com/rapidsai/raft/pull/1432)) [@divyegala](https://github.com/divyegala)
- Use nvtx3 includes. ([#1431](https://github.com/rapidsai/raft/pull/1431)) [@bdice](https://github.com/bdice)
- Remove wheel pytest verbosity ([#1424](https://github.com/rapidsai/raft/pull/1424)) [@sevagh](https://github.com/sevagh)
- Add python bindings for matrix::select_k ([#1422](https://github.com/rapidsai/raft/pull/1422)) [@benfred](https://github.com/benfred)
- Using `raft::resources` across `raft::random` ([#1420](https://github.com/rapidsai/raft/pull/1420)) [@cjnolet](https://github.com/cjnolet)
- Generate build metrics report for test and benchmarks ([#1414](https://github.com/rapidsai/raft/pull/1414)) [@divyegala](https://github.com/divyegala)
- Update clang-format to 16.0.1. ([#1412](https://github.com/rapidsai/raft/pull/1412)) [@bdice](https://github.com/bdice)
- Use ARC V2 self-hosted runners for GPU jobs ([#1410](https://github.com/rapidsai/raft/pull/1410)) [@jjacobelli](https://github.com/jjacobelli)
- Remove uses-setup-env-vars ([#1406](https://github.com/rapidsai/raft/pull/1406)) [@vyasr](https://github.com/vyasr)
- Resolve conflicts in auto-merger of `branch-23.06` and `branch-23.04` ([#1403](https://github.com/rapidsai/raft/pull/1403)) [@galipremsagar](https://github.com/galipremsagar)
- Adding base header-only conda package without cuda math libs ([#1386](https://github.com/rapidsai/raft/pull/1386)) [@cjnolet](https://github.com/cjnolet)
- Fix IVF-PQ API to use `device_vector_view` ([#1384](https://github.com/rapidsai/raft/pull/1384)) [@lowener](https://github.com/lowener)
- Branch 23.06 merge 23.04 ([#1379](https://github.com/rapidsai/raft/pull/1379)) [@vyasr](https://github.com/vyasr)
- Forward merge branch 23.04 into 23.06 ([#1350](https://github.com/rapidsai/raft/pull/1350)) [@cjnolet](https://github.com/cjnolet)
- Fused L2 1-NN based on cutlass 3xTF32 / DMMA ([#1118](https://github.com/rapidsai/raft/pull/1118)) [@mdoijade](https://github.com/mdoijade)

# raft 23.04.00 (6 Apr 2023)

## 🚨 Breaking Changes

- Pin `dask` and `distributed` for release ([#1399](https://github.com/rapidsai/raft/pull/1399)) [@galipremsagar](https://github.com/galipremsagar)
- Remove faiss_mr.hpp ([#1351](https://github.com/rapidsai/raft/pull/1351)) [@benfred](https://github.com/benfred)
- Removing FAISS from build ([#1340](https://github.com/rapidsai/raft/pull/1340)) [@cjnolet](https://github.com/cjnolet)
- Generic linalg::map ([#1337](https://github.com/rapidsai/raft/pull/1337)) [@achirkin](https://github.com/achirkin)
- Consolidate pre-compiled specializations into single `libraft` binary ([#1333](https://github.com/rapidsai/raft/pull/1333)) [@cjnolet](https://github.com/cjnolet)
- Generic linalg::map ([#1329](https://github.com/rapidsai/raft/pull/1329)) [@achirkin](https://github.com/achirkin)
- Update and standardize IVF indexes API ([#1328](https://github.com/rapidsai/raft/pull/1328)) [@viclafargue](https://github.com/viclafargue)
- IVF-Flat index splitting ([#1271](https://github.com/rapidsai/raft/pull/1271)) [@lowener](https://github.com/lowener)
- IVF-PQ: store cluster data in individual lists and reduce templates ([#1249](https://github.com/rapidsai/raft/pull/1249)) [@achirkin](https://github.com/achirkin)
- Fix for svd API ([#1190](https://github.com/rapidsai/raft/pull/1190)) [@lowener](https://github.com/lowener)
- Remove deprecated headers ([#1145](https://github.com/rapidsai/raft/pull/1145)) [@lowener](https://github.com/lowener)

## 🐛 Bug Fixes

- Fix primitives benchmarks ([#1389](https://github.com/rapidsai/raft/pull/1389)) [@ahendriksen](https://github.com/ahendriksen)
- Fixing index-url link on pip install docs ([#1378](https://github.com/rapidsai/raft/pull/1378)) [@cjnolet](https://github.com/cjnolet)
- Adding some functions back in that seem to be a copy/paste error ([#1373](https://github.com/rapidsai/raft/pull/1373)) [@cjnolet](https://github.com/cjnolet)
- Remove usage of Dask&#39;s `get_worker` ([#1365](https://github.com/rapidsai/raft/pull/1365)) [@pentschev](https://github.com/pentschev)
- Remove MANIFEST.in use auto-generated one for sdists and package_data for wheels ([#1348](https://github.com/rapidsai/raft/pull/1348)) [@vyasr](https://github.com/vyasr)
- Revert &quot;Generic linalg::map ([#1329)&quot; (#1336](https://github.com/rapidsai/raft/pull/1329)&quot; (#1336)) [@cjnolet](https://github.com/cjnolet)
- Small follow-up to specializations cleanup ([#1332](https://github.com/rapidsai/raft/pull/1332)) [@cjnolet](https://github.com/cjnolet)
- Fixing select_k specializations ([#1330](https://github.com/rapidsai/raft/pull/1330)) [@cjnolet](https://github.com/cjnolet)
- Fixing remaining bug in ann_quantized ([#1327](https://github.com/rapidsai/raft/pull/1327)) [@cjnolet](https://github.com/cjnolet)
- Fixign a couple small kmeans bugs ([#1274](https://github.com/rapidsai/raft/pull/1274)) [@cjnolet](https://github.com/cjnolet)
- Remove no longer instantiated templates from list of extern template declarations ([#1272](https://github.com/rapidsai/raft/pull/1272)) [@vyasr](https://github.com/vyasr)
- Bump pinned deps to 23.4 ([#1266](https://github.com/rapidsai/raft/pull/1266)) [@vyasr](https://github.com/vyasr)
- Fix the destruction of interruptible token registry ([#1229](https://github.com/rapidsai/raft/pull/1229)) [@achirkin](https://github.com/achirkin)
- Expose raft::handle_t in the public header ([#1192](https://github.com/rapidsai/raft/pull/1192)) [@vyasr](https://github.com/vyasr)
- Fix for svd API ([#1190](https://github.com/rapidsai/raft/pull/1190)) [@lowener](https://github.com/lowener)

## 📖 Documentation

- Adding architecture diagram to README.md ([#1370](https://github.com/rapidsai/raft/pull/1370)) [@cjnolet](https://github.com/cjnolet)
- Adding small readme image ([#1354](https://github.com/rapidsai/raft/pull/1354)) [@cjnolet](https://github.com/cjnolet)
- Fix serialize documentation of ivf_flat ([#1347](https://github.com/rapidsai/raft/pull/1347)) [@lowener](https://github.com/lowener)
- Small updates to docs ([#1339](https://github.com/rapidsai/raft/pull/1339)) [@cjnolet](https://github.com/cjnolet)

## 🚀 New Features

- Add Options to Generate Build Metrics Report ([#1369](https://github.com/rapidsai/raft/pull/1369)) [@divyegala](https://github.com/divyegala)
- Generic linalg::map ([#1337](https://github.com/rapidsai/raft/pull/1337)) [@achirkin](https://github.com/achirkin)
- Generic linalg::map ([#1329](https://github.com/rapidsai/raft/pull/1329)) [@achirkin](https://github.com/achirkin)
- matrix::select_k specializations ([#1268](https://github.com/rapidsai/raft/pull/1268)) [@achirkin](https://github.com/achirkin)
- Use rapids-cmake new COMPONENT exporting feature ([#1154](https://github.com/rapidsai/raft/pull/1154)) [@robertmaynard](https://github.com/robertmaynard)

## 🛠️ Improvements

- Pin `dask` and `distributed` for release ([#1399](https://github.com/rapidsai/raft/pull/1399)) [@galipremsagar](https://github.com/galipremsagar)
- Pin cupy in wheel tests to supported versions ([#1383](https://github.com/rapidsai/raft/pull/1383)) [@vyasr](https://github.com/vyasr)
- CAGRA ([#1375](https://github.com/rapidsai/raft/pull/1375)) [@tfeher](https://github.com/tfeher)
- add a distance epilogue function to the bfknn call ([#1371](https://github.com/rapidsai/raft/pull/1371)) [@benfred](https://github.com/benfred)
- Relax UCX pin to allow 1.14 ([#1366](https://github.com/rapidsai/raft/pull/1366)) [@pentschev](https://github.com/pentschev)
- Generate pyproject dependencies with dfg ([#1364](https://github.com/rapidsai/raft/pull/1364)) [@vyasr](https://github.com/vyasr)
- Add nccl to dependencies.yaml ([#1361](https://github.com/rapidsai/raft/pull/1361)) [@benfred](https://github.com/benfred)
- Add extern template for ivfflat_interleaved_scan ([#1360](https://github.com/rapidsai/raft/pull/1360)) [@ahendriksen](https://github.com/ahendriksen)
- Stop setting package version attribute in wheels ([#1359](https://github.com/rapidsai/raft/pull/1359)) [@vyasr](https://github.com/vyasr)
- Fix ivf flat specialization header IdxT from uint64_t -&gt; int64_t ([#1358](https://github.com/rapidsai/raft/pull/1358)) [@ahendriksen](https://github.com/ahendriksen)
- Remove faiss_mr.hpp ([#1351](https://github.com/rapidsai/raft/pull/1351)) [@benfred](https://github.com/benfred)
- Rename optional helper function ([#1345](https://github.com/rapidsai/raft/pull/1345)) [@viclafargue](https://github.com/viclafargue)
- Pass minimum target compile options through `raft::raft` ([#1341](https://github.com/rapidsai/raft/pull/1341)) [@cjnolet](https://github.com/cjnolet)
- Removing FAISS from build ([#1340](https://github.com/rapidsai/raft/pull/1340)) [@cjnolet](https://github.com/cjnolet)
- Add dispatch based on compute architecture ([#1335](https://github.com/rapidsai/raft/pull/1335)) [@ahendriksen](https://github.com/ahendriksen)
- Consolidate pre-compiled specializations into single `libraft` binary ([#1333](https://github.com/rapidsai/raft/pull/1333)) [@cjnolet](https://github.com/cjnolet)
- Update and standardize IVF indexes API ([#1328](https://github.com/rapidsai/raft/pull/1328)) [@viclafargue](https://github.com/viclafargue)
- Using int64_t specializations for `ivf_pq` and `refine` ([#1325](https://github.com/rapidsai/raft/pull/1325)) [@cjnolet](https://github.com/cjnolet)
- Migrate as much as possible to pyproject.toml ([#1324](https://github.com/rapidsai/raft/pull/1324)) [@vyasr](https://github.com/vyasr)
- Pass `AWS_SESSION_TOKEN` and `SCCACHE_S3_USE_SSL` vars to conda build ([#1321](https://github.com/rapidsai/raft/pull/1321)) [@ajschmidt8](https://github.com/ajschmidt8)
- Numerical stability fixes for l2 pairwise distance ([#1319](https://github.com/rapidsai/raft/pull/1319)) [@benfred](https://github.com/benfred)
- Consolidate linter configuration into pyproject.toml ([#1317](https://github.com/rapidsai/raft/pull/1317)) [@vyasr](https://github.com/vyasr)
- IVF-Flat Python wrappers ([#1316](https://github.com/rapidsai/raft/pull/1316)) [@tfeher](https://github.com/tfeher)
- Add stream overloads to `ivf_pq` serialize/deserialize methods ([#1315](https://github.com/rapidsai/raft/pull/1315)) [@divyegala](https://github.com/divyegala)
- Temporary buffer to view host or device memory in device ([#1313](https://github.com/rapidsai/raft/pull/1313)) [@divyegala](https://github.com/divyegala)
- RAFT skeleton project template ([#1312](https://github.com/rapidsai/raft/pull/1312)) [@cjnolet](https://github.com/cjnolet)
- Fix docs build to be `pydata-sphinx-theme=0.13.0` compatible ([#1311](https://github.com/rapidsai/raft/pull/1311)) [@galipremsagar](https://github.com/galipremsagar)
- Update to GCC 11 ([#1309](https://github.com/rapidsai/raft/pull/1309)) [@bdice](https://github.com/bdice)
- Reduce compile times of distance specializations ([#1307](https://github.com/rapidsai/raft/pull/1307)) [@ahendriksen](https://github.com/ahendriksen)
- Fix docs upload path ([#1305](https://github.com/rapidsai/raft/pull/1305)) [@AyodeAwe](https://github.com/AyodeAwe)
- Add end-to-end CUDA ann-benchmarks to raft ([#1304](https://github.com/rapidsai/raft/pull/1304)) [@cjnolet](https://github.com/cjnolet)
- Make docs builds less verbose ([#1302](https://github.com/rapidsai/raft/pull/1302)) [@AyodeAwe](https://github.com/AyodeAwe)
- Stop using versioneer to manage versions ([#1301](https://github.com/rapidsai/raft/pull/1301)) [@vyasr](https://github.com/vyasr)
- Adding util to get the device id for a pointer address ([#1297](https://github.com/rapidsai/raft/pull/1297)) [@cjnolet](https://github.com/cjnolet)
- Enable dfg in pre-commit. ([#1293](https://github.com/rapidsai/raft/pull/1293)) [@vyasr](https://github.com/vyasr)
- Python API for brute-force KNN ([#1292](https://github.com/rapidsai/raft/pull/1292)) [@cjnolet](https://github.com/cjnolet)
- support k up to 2048 in faiss select ([#1287](https://github.com/rapidsai/raft/pull/1287)) [@benfred](https://github.com/benfred)
- CI: Remove specification of manual stage for check_style.sh script. ([#1283](https://github.com/rapidsai/raft/pull/1283)) [@csadorf](https://github.com/csadorf)
- New Sparse Matrix APIs ([#1279](https://github.com/rapidsai/raft/pull/1279)) [@cjnolet](https://github.com/cjnolet)
- fix build on cuda 11.5 ([#1277](https://github.com/rapidsai/raft/pull/1277)) [@benfred](https://github.com/benfred)
- IVF-Flat index splitting ([#1271](https://github.com/rapidsai/raft/pull/1271)) [@lowener](https://github.com/lowener)
- Remove duplicate `librmm` runtime dependency ([#1264](https://github.com/rapidsai/raft/pull/1264)) [@ajschmidt8](https://github.com/ajschmidt8)
- build.sh: Add option to log nvcc compile times ([#1262](https://github.com/rapidsai/raft/pull/1262)) [@ahendriksen](https://github.com/ahendriksen)
- Reduce error handling verbosity in CI tests scripts ([#1259](https://github.com/rapidsai/raft/pull/1259)) [@AjayThorve](https://github.com/AjayThorve)
- Update shared workflow branches ([#1256](https://github.com/rapidsai/raft/pull/1256)) [@ajschmidt8](https://github.com/ajschmidt8)
- Keeping only compute similarity specializations for uint64_t for now ([#1255](https://github.com/rapidsai/raft/pull/1255)) [@cjnolet](https://github.com/cjnolet)
- Fix compile time explosion for minkowski distance ([#1254](https://github.com/rapidsai/raft/pull/1254)) [@ahendriksen](https://github.com/ahendriksen)
- Unpin `dask` and `distributed` for development ([#1253](https://github.com/rapidsai/raft/pull/1253)) [@galipremsagar](https://github.com/galipremsagar)
- Remove gpuCI scripts. ([#1252](https://github.com/rapidsai/raft/pull/1252)) [@bdice](https://github.com/bdice)
- IVF-PQ: store cluster data in individual lists and reduce templates ([#1249](https://github.com/rapidsai/raft/pull/1249)) [@achirkin](https://github.com/achirkin)
- Fix inconsistency between the building doc and CMakeLists.txt ([#1248](https://github.com/rapidsai/raft/pull/1248)) [@yong-wang](https://github.com/yong-wang)
- Consolidating ANN benchmarks and tests ([#1243](https://github.com/rapidsai/raft/pull/1243)) [@cjnolet](https://github.com/cjnolet)
- mdspan view for IVF-PQ API ([#1236](https://github.com/rapidsai/raft/pull/1236)) [@viclafargue](https://github.com/viclafargue)
- Remove uint32 distance idx specializations ([#1235](https://github.com/rapidsai/raft/pull/1235)) [@cjnolet](https://github.com/cjnolet)
- Add innerproduct to the pairwise distance api ([#1226](https://github.com/rapidsai/raft/pull/1226)) [@benfred](https://github.com/benfred)
- Move date to build string in `conda` recipe ([#1223](https://github.com/rapidsai/raft/pull/1223)) [@ajschmidt8](https://github.com/ajschmidt8)
- Replace faiss bfKnn ([#1202](https://github.com/rapidsai/raft/pull/1202)) [@benfred](https://github.com/benfred)
- Expose KMeans `init_plus_plus` in pylibraft ([#1198](https://github.com/rapidsai/raft/pull/1198)) [@betatim](https://github.com/betatim)
- Fix `ucx-py` version ([#1184](https://github.com/rapidsai/raft/pull/1184)) [@ajschmidt8](https://github.com/ajschmidt8)
- Improve the performance of radix top-k ([#1175](https://github.com/rapidsai/raft/pull/1175)) [@yong-wang](https://github.com/yong-wang)
- Add docs build job ([#1168](https://github.com/rapidsai/raft/pull/1168)) [@AyodeAwe](https://github.com/AyodeAwe)
- Remove deprecated headers ([#1145](https://github.com/rapidsai/raft/pull/1145)) [@lowener](https://github.com/lowener)
- Simplify distance/detail to make is easier to dispatch to different kernel implementations ([#1142](https://github.com/rapidsai/raft/pull/1142)) [@ahendriksen](https://github.com/ahendriksen)
- Initial port of auto-find-k ([#1070](https://github.com/rapidsai/raft/pull/1070)) [@cjnolet](https://github.com/cjnolet)

# raft 23.02.00 (9 Feb 2023)

## 🚨 Breaking Changes

- Remove faiss ANN code from knnIndex ([#1121](https://github.com/rapidsai/raft/pull/1121)) [@benfred](https://github.com/benfred)
- Use `GenPC` (Permuted Congruential) as the default random number generator everywhere ([#1099](https://github.com/rapidsai/raft/pull/1099)) [@Nyrio](https://github.com/Nyrio)

## 🐛 Bug Fixes

- Reverting a few commits from 23.02 and speeding up end-to-end build time ([#1232](https://github.com/rapidsai/raft/pull/1232)) [@cjnolet](https://github.com/cjnolet)
- Update README.md: fix a missing word ([#1185](https://github.com/rapidsai/raft/pull/1185)) [@achirkin](https://github.com/achirkin)
- balanced-k-means: fix a too large initial memory pool size ([#1148](https://github.com/rapidsai/raft/pull/1148)) [@achirkin](https://github.com/achirkin)
- Catch signal handler change error ([#1147](https://github.com/rapidsai/raft/pull/1147)) [@tfeher](https://github.com/tfeher)
- Squared norm fix follow-up (change was lost in merge conflict) ([#1144](https://github.com/rapidsai/raft/pull/1144)) [@Nyrio](https://github.com/Nyrio)
- IVF-Flat bug fix: the *squared* norm is required for expanded distance calculations ([#1141](https://github.com/rapidsai/raft/pull/1141)) [@Nyrio](https://github.com/Nyrio)
- build.sh switch to use `RAPIDS` magic value ([#1132](https://github.com/rapidsai/raft/pull/1132)) [@robertmaynard](https://github.com/robertmaynard)
- Fix `euclidean_dist` in IVF-Flat search ([#1122](https://github.com/rapidsai/raft/pull/1122)) [@Nyrio](https://github.com/Nyrio)
- Update handle docstring ([#1103](https://github.com/rapidsai/raft/pull/1103)) [@dantegd](https://github.com/dantegd)
- Pin libcusparse and libcusolver to avoid CUDA 12 ([#1095](https://github.com/rapidsai/raft/pull/1095)) [@wphicks](https://github.com/wphicks)
- Fix race condition in `raft::random::discrete` ([#1094](https://github.com/rapidsai/raft/pull/1094)) [@Nyrio](https://github.com/Nyrio)
- Fixing libraft conda recipes ([#1084](https://github.com/rapidsai/raft/pull/1084)) [@cjnolet](https://github.com/cjnolet)
- Ensure that we get the cuda version of faiss. ([#1078](https://github.com/rapidsai/raft/pull/1078)) [@vyasr](https://github.com/vyasr)
- Fix double definition error in ANN refinement header ([#1067](https://github.com/rapidsai/raft/pull/1067)) [@tfeher](https://github.com/tfeher)
- Specify correct global targets names to raft_export ([#1054](https://github.com/rapidsai/raft/pull/1054)) [@robertmaynard](https://github.com/robertmaynard)
- Fix concurrency issues in k-means++ initialization ([#1048](https://github.com/rapidsai/raft/pull/1048)) [@Nyrio](https://github.com/Nyrio)

## 📖 Documentation

- Adding small comms tutorial to docs ([#1204](https://github.com/rapidsai/raft/pull/1204)) [@cjnolet](https://github.com/cjnolet)
- Separating more namespaces into easier-to-consume sections ([#1091](https://github.com/rapidsai/raft/pull/1091)) [@cjnolet](https://github.com/cjnolet)
- Paying down some tech debt on docs, runtime API, and cython ([#1055](https://github.com/rapidsai/raft/pull/1055)) [@cjnolet](https://github.com/cjnolet)

## 🚀 New Features

- Add function to convert mdspan to a const view ([#1188](https://github.com/rapidsai/raft/pull/1188)) [@lowener](https://github.com/lowener)
- Internal library to share headers between test and bench ([#1162](https://github.com/rapidsai/raft/pull/1162)) [@achirkin](https://github.com/achirkin)
- Add public API and tests for hierarchical balanced k-means ([#1113](https://github.com/rapidsai/raft/pull/1113)) [@Nyrio](https://github.com/Nyrio)
- Export NCCL dependency as part of raft::distributed. ([#1077](https://github.com/rapidsai/raft/pull/1077)) [@vyasr](https://github.com/vyasr)
- Serialization of IVF Flat and IVF PQ ([#919](https://github.com/rapidsai/raft/pull/919)) [@tfeher](https://github.com/tfeher)

## 🛠️ Improvements

- Pin `dask` and `distributed` for release ([#1242](https://github.com/rapidsai/raft/pull/1242)) [@galipremsagar](https://github.com/galipremsagar)
- Update shared workflow branches ([#1241](https://github.com/rapidsai/raft/pull/1241)) [@ajschmidt8](https://github.com/ajschmidt8)
- Removing interruptible from basic handle sync. ([#1224](https://github.com/rapidsai/raft/pull/1224)) [@cjnolet](https://github.com/cjnolet)
- pre-commit: Update isort version to 5.12.0 ([#1215](https://github.com/rapidsai/raft/pull/1215)) [@wence-](https://github.com/wence-)
- Pin wheel dependencies to same RAPIDS release ([#1200](https://github.com/rapidsai/raft/pull/1200)) [@sevagh](https://github.com/sevagh)
- Serializer for mdspans ([#1173](https://github.com/rapidsai/raft/pull/1173)) [@hcho3](https://github.com/hcho3)
- Use CTK 118/cp310 branch of wheel workflows ([#1169](https://github.com/rapidsai/raft/pull/1169)) [@sevagh](https://github.com/sevagh)
- Enable shallow copy of `handle_t`&#39;s resources with different workspace_resource ([#1165](https://github.com/rapidsai/raft/pull/1165)) [@cjnolet](https://github.com/cjnolet)
- Protect balanced k-means out-of-memory in some cases ([#1161](https://github.com/rapidsai/raft/pull/1161)) [@achirkin](https://github.com/achirkin)
- Use squeuclidean for metric name in ivf_pq python bindings ([#1160](https://github.com/rapidsai/raft/pull/1160)) [@benfred](https://github.com/benfred)
- ANN tests: make the min_recall check strict ([#1156](https://github.com/rapidsai/raft/pull/1156)) [@achirkin](https://github.com/achirkin)
- Make cutlass use static ctk ([#1155](https://github.com/rapidsai/raft/pull/1155)) [@sevagh](https://github.com/sevagh)
- Fix various build errors ([#1152](https://github.com/rapidsai/raft/pull/1152)) [@hcho3](https://github.com/hcho3)
- Remove faiss bfKnn call from fused_l2_knn unittest ([#1150](https://github.com/rapidsai/raft/pull/1150)) [@benfred](https://github.com/benfred)
- Fix `unary_op` docs and add `map_offset` as an improved version of `write_only_unary_op` ([#1149](https://github.com/rapidsai/raft/pull/1149)) [@Nyrio](https://github.com/Nyrio)
- Improvement of the math API wrappers ([#1146](https://github.com/rapidsai/raft/pull/1146)) [@Nyrio](https://github.com/Nyrio)
- Changing handle_t to device_resources everywhere ([#1140](https://github.com/rapidsai/raft/pull/1140)) [@cjnolet](https://github.com/cjnolet)
- Add L2SqrtExpanded support to ivf_pq ([#1138](https://github.com/rapidsai/raft/pull/1138)) [@benfred](https://github.com/benfred)
- Adding workspace resource ([#1137](https://github.com/rapidsai/raft/pull/1137)) [@cjnolet](https://github.com/cjnolet)
- Add raft::void_op functor ([#1136](https://github.com/rapidsai/raft/pull/1136)) [@ahendriksen](https://github.com/ahendriksen)
- IVF-PQ: tighten the test criteria ([#1135](https://github.com/rapidsai/raft/pull/1135)) [@achirkin](https://github.com/achirkin)
- Fix documentation author ([#1134](https://github.com/rapidsai/raft/pull/1134)) [@bdice](https://github.com/bdice)
- Add L2SqrtExpanded support to ivf_flat ANN indices ([#1133](https://github.com/rapidsai/raft/pull/1133)) [@benfred](https://github.com/benfred)
- Improvements in `matrix::gather`: test coverage, compilation errors, performance ([#1126](https://github.com/rapidsai/raft/pull/1126)) [@Nyrio](https://github.com/Nyrio)
- Adding ability to use an existing stream in the pylibraft Handle ([#1125](https://github.com/rapidsai/raft/pull/1125)) [@cjnolet](https://github.com/cjnolet)
- Remove faiss ANN code from knnIndex ([#1121](https://github.com/rapidsai/raft/pull/1121)) [@benfred](https://github.com/benfred)
- Update builds for CUDA `11.8` and Python `3.10` ([#1120](https://github.com/rapidsai/raft/pull/1120)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update workflows for nightly tests ([#1119](https://github.com/rapidsai/raft/pull/1119)) [@ajschmidt8](https://github.com/ajschmidt8)
- Enable `Recently Updated` Check ([#1117](https://github.com/rapidsai/raft/pull/1117)) [@ajschmidt8](https://github.com/ajschmidt8)
- Build wheels alongside conda CI ([#1116](https://github.com/rapidsai/raft/pull/1116)) [@sevagh](https://github.com/sevagh)
- Allow host dataset for IVF-PQ ([#1114](https://github.com/rapidsai/raft/pull/1114)) [@tfeher](https://github.com/tfeher)
- Decoupling raft handle from underlying resources ([#1111](https://github.com/rapidsai/raft/pull/1111)) [@cjnolet](https://github.com/cjnolet)
- Fixing an index error introduced in PR #1109 ([#1110](https://github.com/rapidsai/raft/pull/1110)) [@vinaydes](https://github.com/vinaydes)
- Fixing the sample-without-replacement test failures ([#1109](https://github.com/rapidsai/raft/pull/1109)) [@vinaydes](https://github.com/vinaydes)
- Remove faiss dependency from fused_l2_knn.cuh, selection_faiss.cuh, ball_cover.cuh and haversine_distance.cuh ([#1108](https://github.com/rapidsai/raft/pull/1108)) [@benfred](https://github.com/benfred)
- Remove redundant operators in sparse/distance and move others to raft/core ([#1105](https://github.com/rapidsai/raft/pull/1105)) [@Nyrio](https://github.com/Nyrio)
- Speedup `make_blobs` by up to 2x by fixing inefficient kernel launch configuration ([#1100](https://github.com/rapidsai/raft/pull/1100)) [@Nyrio](https://github.com/Nyrio)
- Use `GenPC` (Permuted Congruential) as the default random number generator everywhere ([#1099](https://github.com/rapidsai/raft/pull/1099)) [@Nyrio](https://github.com/Nyrio)
- Cleanup faiss includes ([#1098](https://github.com/rapidsai/raft/pull/1098)) [@benfred](https://github.com/benfred)
- matrix::select_k: move selection and warp-sort primitives ([#1085](https://github.com/rapidsai/raft/pull/1085)) [@achirkin](https://github.com/achirkin)
- Exclude changelog from pre-commit spellcheck ([#1083](https://github.com/rapidsai/raft/pull/1083)) [@benfred](https://github.com/benfred)
- Add GitHub Actions Workflows. ([#1076](https://github.com/rapidsai/raft/pull/1076)) [@bdice](https://github.com/bdice)
- Adding uninstall option to build.sh ([#1075](https://github.com/rapidsai/raft/pull/1075)) [@cjnolet](https://github.com/cjnolet)
- Use doctest for testing python example docstrings ([#1073](https://github.com/rapidsai/raft/pull/1073)) [@benfred](https://github.com/benfred)
- Minor cython fixes / cleanup ([#1072](https://github.com/rapidsai/raft/pull/1072)) [@benfred](https://github.com/benfred)
- IVF-PQ: tweak launch configuration ([#1069](https://github.com/rapidsai/raft/pull/1069)) [@achirkin](https://github.com/achirkin)
- Unpin `dask` and `distributed` for development ([#1068](https://github.com/rapidsai/raft/pull/1068)) [@galipremsagar](https://github.com/galipremsagar)
- Bifurcate Dependency Lists ([#1065](https://github.com/rapidsai/raft/pull/1065)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add support for 64bit svdeig ([#1060](https://github.com/rapidsai/raft/pull/1060)) [@lowener](https://github.com/lowener)
- switch mma instruction shape to 1684 from current 1688 for 3xTF32 L2/cosine kernel ([#1057](https://github.com/rapidsai/raft/pull/1057)) [@mdoijade](https://github.com/mdoijade)
- Make IVF-PQ build index in batches when necessary ([#1056](https://github.com/rapidsai/raft/pull/1056)) [@achirkin](https://github.com/achirkin)
- Remove unused setuputils modules ([#1053](https://github.com/rapidsai/raft/pull/1053)) [@vyasr](https://github.com/vyasr)
- Branch 23.02 merge 22.12 ([#1051](https://github.com/rapidsai/raft/pull/1051)) [@benfred](https://github.com/benfred)
- Shared-memory-cached kernel for `reduce_cols_by_key` to limit atomic conflicts ([#1050](https://github.com/rapidsai/raft/pull/1050)) [@Nyrio](https://github.com/Nyrio)
- Unify use of common functors ([#1049](https://github.com/rapidsai/raft/pull/1049)) [@Nyrio](https://github.com/Nyrio)
- Replace k-means++ CPU bottleneck with a `random::discrete` prim ([#1039](https://github.com/rapidsai/raft/pull/1039)) [@Nyrio](https://github.com/Nyrio)
- Add python bindings for kmeans fit ([#1016](https://github.com/rapidsai/raft/pull/1016)) [@benfred](https://github.com/benfred)
- Add MaskedL2NN ([#838](https://github.com/rapidsai/raft/pull/838)) [@ahendriksen](https://github.com/ahendriksen)
- Move contractions tiling logic outside of Contractions_NT ([#837](https://github.com/rapidsai/raft/pull/837)) [@ahendriksen](https://github.com/ahendriksen)

# raft 22.12.00 (8 Dec 2022)

## 🚨 Breaking Changes

- Make ucx linkage explicit and add a new CMake target for it ([#1032](https://github.com/rapidsai/raft/pull/1032)) [@vyasr](https://github.com/vyasr)
- IVF-Flat: make adaptive-centers behavior optional ([#1019](https://github.com/rapidsai/raft/pull/1019)) [@achirkin](https://github.com/achirkin)
- Remove make_mdspan template for memory_type enum ([#1005](https://github.com/rapidsai/raft/pull/1005)) [@wphicks](https://github.com/wphicks)
- ivf-pq performance tweaks ([#926](https://github.com/rapidsai/raft/pull/926)) [@achirkin](https://github.com/achirkin)

## 🐛 Bug Fixes

- fusedL2NN: Add input alignment checks ([#1045](https://github.com/rapidsai/raft/pull/1045)) [@achirkin](https://github.com/achirkin)
- Fix fusedL2NN bug that can happen when the same point appears in both x and y ([#1040](https://github.com/rapidsai/raft/pull/1040)) [@Nyrio](https://github.com/Nyrio)
- Fix trivial deprecated header includes ([#1034](https://github.com/rapidsai/raft/pull/1034)) [@achirkin](https://github.com/achirkin)
- Suppress ptxas stack size warning in Debug mode ([#1033](https://github.com/rapidsai/raft/pull/1033)) [@tfeher](https://github.com/tfeher)
- Don&#39;t use CMake 3.25.0 as it has a FindCUDAToolkit show stopping bug ([#1029](https://github.com/rapidsai/raft/pull/1029)) [@robertmaynard](https://github.com/robertmaynard)
- Fix for gemmi deprecation ([#1020](https://github.com/rapidsai/raft/pull/1020)) [@lowener](https://github.com/lowener)
- Remove make_mdspan template for memory_type enum ([#1005](https://github.com/rapidsai/raft/pull/1005)) [@wphicks](https://github.com/wphicks)
- Add `except +` to cython extern cdef declarations ([#1001](https://github.com/rapidsai/raft/pull/1001)) [@benfred](https://github.com/benfred)
- Changing Overloads for GCC 11/12 bug ([#995](https://github.com/rapidsai/raft/pull/995)) [@divyegala](https://github.com/divyegala)
- Changing Overloads for GCC 11/12 bugs ([#992](https://github.com/rapidsai/raft/pull/992)) [@divyegala](https://github.com/divyegala)
- Fix pylibraft docstring example code ([#980](https://github.com/rapidsai/raft/pull/980)) [@benfred](https://github.com/benfred)
- Update raft tests to compile with C++17 features enabled ([#973](https://github.com/rapidsai/raft/pull/973)) [@robertmaynard](https://github.com/robertmaynard)
- Making ivf flat gtest invoke mdspanified APIs ([#955](https://github.com/rapidsai/raft/pull/955)) [@cjnolet](https://github.com/cjnolet)
- Updates to kmeans public API to fix cuml ([#932](https://github.com/rapidsai/raft/pull/932)) [@cjnolet](https://github.com/cjnolet)
- Fix logger (vsnprintf consumes args) ([#917](https://github.com/rapidsai/raft/pull/917)) [@Nyrio](https://github.com/Nyrio)
- Adding missing include for device mdspan in `mean_squared_error.cuh` ([#906](https://github.com/rapidsai/raft/pull/906)) [@cjnolet](https://github.com/cjnolet)

## 📖 Documentation

- Add links to the docs site in the README ([#1042](https://github.com/rapidsai/raft/pull/1042)) [@benfred](https://github.com/benfred)
- Moving contributing and developer guides to main docs ([#1006](https://github.com/rapidsai/raft/pull/1006)) [@cjnolet](https://github.com/cjnolet)
- Update compiler flags in build docs ([#999](https://github.com/rapidsai/raft/pull/999)) [@cjnolet](https://github.com/cjnolet)
- Updating minimum required gcc version ([#993](https://github.com/rapidsai/raft/pull/993)) [@cjnolet](https://github.com/cjnolet)
- important doc updates for core, cluster, and neighbors ([#933](https://github.com/rapidsai/raft/pull/933)) [@cjnolet](https://github.com/cjnolet)

## 🚀 New Features

- ANN refinement Python wrapper ([#1052](https://github.com/rapidsai/raft/pull/1052)) [@tfeher](https://github.com/tfeher)
- Add ANN refinement method ([#1038](https://github.com/rapidsai/raft/pull/1038)) [@tfeher](https://github.com/tfeher)
- IVF-Flat: make adaptive-centers behavior optional ([#1019](https://github.com/rapidsai/raft/pull/1019)) [@achirkin](https://github.com/achirkin)
- Add wheel builds ([#1013](https://github.com/rapidsai/raft/pull/1013)) [@vyasr](https://github.com/vyasr)
- Update cuSparse wrappers to avoid deprecated functions ([#989](https://github.com/rapidsai/raft/pull/989)) [@wphicks](https://github.com/wphicks)
- Provide memory_type enum ([#984](https://github.com/rapidsai/raft/pull/984)) [@wphicks](https://github.com/wphicks)
- Add Tests for kmeans API ([#982](https://github.com/rapidsai/raft/pull/982)) [@lowener](https://github.com/lowener)
- mdspanifying `weighted_mean` and add `raft::stats` tests ([#910](https://github.com/rapidsai/raft/pull/910)) [@lowener](https://github.com/lowener)
- Implement `raft::stats` API with mdspan ([#802](https://github.com/rapidsai/raft/pull/802)) [@lowener](https://github.com/lowener)

## 🛠️ Improvements

- Pin `dask` and `distributed` for release ([#1062](https://github.com/rapidsai/raft/pull/1062)) [@galipremsagar](https://github.com/galipremsagar)
- IVF-PQ: use device properties helper ([#1035](https://github.com/rapidsai/raft/pull/1035)) [@achirkin](https://github.com/achirkin)
- Make ucx linkage explicit and add a new CMake target for it ([#1032](https://github.com/rapidsai/raft/pull/1032)) [@vyasr](https://github.com/vyasr)
- Fixing broken doc functions and improving coverage ([#1030](https://github.com/rapidsai/raft/pull/1030)) [@cjnolet](https://github.com/cjnolet)
- Expose cluster_cost to python ([#1028](https://github.com/rapidsai/raft/pull/1028)) [@benfred](https://github.com/benfred)
- Adding lightweight cai_wrapper to reduce boilerplate ([#1027](https://github.com/rapidsai/raft/pull/1027)) [@cjnolet](https://github.com/cjnolet)
- Change `raft` docs theme to `pydata-sphinx-theme` ([#1026](https://github.com/rapidsai/raft/pull/1026)) [@galipremsagar](https://github.com/galipremsagar)
- Revert &quot; Pin `dask` and `distributed` for release&quot; ([#1023](https://github.com/rapidsai/raft/pull/1023)) [@galipremsagar](https://github.com/galipremsagar)
- Pin `dask` and `distributed` for release ([#1022](https://github.com/rapidsai/raft/pull/1022)) [@galipremsagar](https://github.com/galipremsagar)
- Replace `dots_along_rows` with `rowNorm` and improve `coalescedReduction` performance ([#1011](https://github.com/rapidsai/raft/pull/1011)) [@Nyrio](https://github.com/Nyrio)
- Moving TestDeviceBuffer to `pylibraft.common.device_ndarray` ([#1008](https://github.com/rapidsai/raft/pull/1008)) [@cjnolet](https://github.com/cjnolet)
- Add codespell as a linter ([#1007](https://github.com/rapidsai/raft/pull/1007)) [@benfred](https://github.com/benfred)
- Fix environment channels ([#996](https://github.com/rapidsai/raft/pull/996)) [@bdice](https://github.com/bdice)
- Automatically sync handle when not passed to pylibraft functions ([#987](https://github.com/rapidsai/raft/pull/987)) [@benfred](https://github.com/benfred)
- Replace `normalize_rows` in `ann_utils.cuh` by a new `rowNormalize` prim and improve performance for thin matrices (small `n_cols`) ([#979](https://github.com/rapidsai/raft/pull/979)) [@Nyrio](https://github.com/Nyrio)
- Forward merge 22.10 into 22.12 ([#978](https://github.com/rapidsai/raft/pull/978)) [@vyasr](https://github.com/vyasr)
- Use new rapids-cmake functionality for rpath handling. ([#976](https://github.com/rapidsai/raft/pull/976)) [@vyasr](https://github.com/vyasr)
- Update cuda-python dependency to 11.7.1 ([#975](https://github.com/rapidsai/raft/pull/975)) [@galipremsagar](https://github.com/galipremsagar)
- IVF-PQ Python wrappers ([#970](https://github.com/rapidsai/raft/pull/970)) [@tfeher](https://github.com/tfeher)
- Remove unnecessary requirements for raft-dask. ([#969](https://github.com/rapidsai/raft/pull/969)) [@vyasr](https://github.com/vyasr)
- Expose `linalg::dot` in public API ([#968](https://github.com/rapidsai/raft/pull/968)) [@benfred](https://github.com/benfred)
- Fix kmeans cluster templates ([#966](https://github.com/rapidsai/raft/pull/966)) [@lowener](https://github.com/lowener)
- Run linters using pre-commit ([#965](https://github.com/rapidsai/raft/pull/965)) [@benfred](https://github.com/benfred)
- linewiseop padded span test ([#964](https://github.com/rapidsai/raft/pull/964)) [@mfoerste4](https://github.com/mfoerste4)
- Add unittest for `linalg::mean_squared_error` ([#961](https://github.com/rapidsai/raft/pull/961)) [@benfred](https://github.com/benfred)
- Exposing fused l2 knn to public APIs ([#959](https://github.com/rapidsai/raft/pull/959)) [@cjnolet](https://github.com/cjnolet)
- Remove a left over print statement from pylibraft ([#958](https://github.com/rapidsai/raft/pull/958)) [@betatim](https://github.com/betatim)
- Switch to using rapids-cmake for gbench. ([#954](https://github.com/rapidsai/raft/pull/954)) [@vyasr](https://github.com/vyasr)
- Some cleanup of k-means internals ([#953](https://github.com/rapidsai/raft/pull/953)) [@cjnolet](https://github.com/cjnolet)
- Remove stale labeler ([#951](https://github.com/rapidsai/raft/pull/951)) [@raydouglass](https://github.com/raydouglass)
- Adding optional handle to each public API function (along with example) ([#947](https://github.com/rapidsai/raft/pull/947)) [@cjnolet](https://github.com/cjnolet)
- Improving documentation across the board. Adding quick-start to breathe docs. ([#943](https://github.com/rapidsai/raft/pull/943)) [@cjnolet](https://github.com/cjnolet)
- Add unittest for `linalg::axpy` ([#942](https://github.com/rapidsai/raft/pull/942)) [@benfred](https://github.com/benfred)
- Add cutlass 3xTF32,DMMA based L2/cosine distance kernels for SM 8.0 or higher ([#939](https://github.com/rapidsai/raft/pull/939)) [@mdoijade](https://github.com/mdoijade)
- Calculate max cluster size correctly for IVF-PQ ([#938](https://github.com/rapidsai/raft/pull/938)) [@tfeher](https://github.com/tfeher)
- Add tests for `raft::matrix` ([#937](https://github.com/rapidsai/raft/pull/937)) [@lowener](https://github.com/lowener)
- Add fusedL2NN benchmark ([#936](https://github.com/rapidsai/raft/pull/936)) [@Nyrio](https://github.com/Nyrio)
- ivf-pq performance tweaks ([#926](https://github.com/rapidsai/raft/pull/926)) [@achirkin](https://github.com/achirkin)
- Adding `fused_l2_nn_argmin` wrapper to Pylibraft ([#924](https://github.com/rapidsai/raft/pull/924)) [@cjnolet](https://github.com/cjnolet)
- Moving kernel gramm primitives to `raft::distance::kernels` ([#920](https://github.com/rapidsai/raft/pull/920)) [@cjnolet](https://github.com/cjnolet)
- kmeans improvements: random initialization on GPU, NVTX markers, no batching when using fusedL2NN ([#918](https://github.com/rapidsai/raft/pull/918)) [@Nyrio](https://github.com/Nyrio)
- Moving `raft::spatial::knn` -&gt; `raft::neighbors` ([#914](https://github.com/rapidsai/raft/pull/914)) [@cjnolet](https://github.com/cjnolet)
- Create cub-based argmin primitive and replace `argmin_along_rows` in ANN kmeans ([#912](https://github.com/rapidsai/raft/pull/912)) [@Nyrio](https://github.com/Nyrio)
- Replace `map_along_rows` with `matrixVectorOp` ([#911](https://github.com/rapidsai/raft/pull/911)) [@Nyrio](https://github.com/Nyrio)
- Integrate `accumulate_into_selected` from ANN utils into `linalg::reduce_rows_by_keys` ([#909](https://github.com/rapidsai/raft/pull/909)) [@Nyrio](https://github.com/Nyrio)
- Re-enabling Fused L2 NN specializations and renaming `cub::KeyValuePair` -&gt; `raft::KeyValuePair` ([#905](https://github.com/rapidsai/raft/pull/905)) [@cjnolet](https://github.com/cjnolet)
- Unpin `dask` and `distributed` for development ([#886](https://github.com/rapidsai/raft/pull/886)) [@galipremsagar](https://github.com/galipremsagar)
- Adding padded layout &#39;layout_padded_general&#39; ([#725](https://github.com/rapidsai/raft/pull/725)) [@mfoerste4](https://github.com/mfoerste4)

# raft 22.10.00 (12 Oct 2022)

## 🚨 Breaking Changes

- Separating mdspan/mdarray infra into host_* and device_* variants ([#810](https://github.com/rapidsai/raft/pull/810)) [@cjnolet](https://github.com/cjnolet)
- Remove type punning from TxN_t ([#781](https://github.com/rapidsai/raft/pull/781)) [@wphicks](https://github.com/wphicks)
- ivf_flat::index: hide implementation details ([#747](https://github.com/rapidsai/raft/pull/747)) [@achirkin](https://github.com/achirkin)

## 🐛 Bug Fixes

- ivf-pq integration: hotfixes ([#891](https://github.com/rapidsai/raft/pull/891)) [@achirkin](https://github.com/achirkin)
- Removing cub symbol from libraft-distance instantiation. ([#887](https://github.com/rapidsai/raft/pull/887)) [@cjnolet](https://github.com/cjnolet)
- ivf-pq post integration hotfixes ([#878](https://github.com/rapidsai/raft/pull/878)) [@achirkin](https://github.com/achirkin)
- Fixing a few compile errors in new APIs ([#874](https://github.com/rapidsai/raft/pull/874)) [@cjnolet](https://github.com/cjnolet)
- Include knn.cuh in knn.cu benchmark source for finding brute_force_knn ([#855](https://github.com/rapidsai/raft/pull/855)) [@teju85](https://github.com/teju85)
- Do not use strcpy to copy 2 char ([#848](https://github.com/rapidsai/raft/pull/848)) [@mhoemmen](https://github.com/mhoemmen)
- rng_state not including necessary cstdint ([#839](https://github.com/rapidsai/raft/pull/839)) [@MatthiasKohl](https://github.com/MatthiasKohl)
- Fix integer overflow in ANN kmeans ([#835](https://github.com/rapidsai/raft/pull/835)) [@Nyrio](https://github.com/Nyrio)
- Add alignment to the TxN_t vectorized type ([#792](https://github.com/rapidsai/raft/pull/792)) [@achirkin](https://github.com/achirkin)
- Fix adj_to_csr_kernel ([#785](https://github.com/rapidsai/raft/pull/785)) [@ahendriksen](https://github.com/ahendriksen)
- Use rapids-cmake 22.10 best practice for RAPIDS.cmake location ([#784](https://github.com/rapidsai/raft/pull/784)) [@robertmaynard](https://github.com/robertmaynard)
- Remove type punning from TxN_t ([#781](https://github.com/rapidsai/raft/pull/781)) [@wphicks](https://github.com/wphicks)
- Various fixes for build.sh ([#771](https://github.com/rapidsai/raft/pull/771)) [@vyasr](https://github.com/vyasr)

## 📖 Documentation

- Fix target names in build.sh help text ([#879](https://github.com/rapidsai/raft/pull/879)) [@Nyrio](https://github.com/Nyrio)
- Document that minimum required CMake version is now 3.23.1 ([#841](https://github.com/rapidsai/raft/pull/841)) [@robertmaynard](https://github.com/robertmaynard)

## 🚀 New Features

- mdspanify raft::random functions uniformInt, normalTable, fill, bernoulli, and scaled_bernoulli ([#897](https://github.com/rapidsai/raft/pull/897)) [@mhoemmen](https://github.com/mhoemmen)
- mdspan-ify several raft::random rng functions ([#857](https://github.com/rapidsai/raft/pull/857)) [@mhoemmen](https://github.com/mhoemmen)
- Develop new mdspan-ified multi_variable_gaussian interface ([#845](https://github.com/rapidsai/raft/pull/845)) [@mhoemmen](https://github.com/mhoemmen)
- Mdspanify permute ([#834](https://github.com/rapidsai/raft/pull/834)) [@mhoemmen](https://github.com/mhoemmen)
- mdspan-ify rmat_rectangular_gen ([#833](https://github.com/rapidsai/raft/pull/833)) [@mhoemmen](https://github.com/mhoemmen)
- mdspanify sampleWithoutReplacement ([#830](https://github.com/rapidsai/raft/pull/830)) [@mhoemmen](https://github.com/mhoemmen)
- mdspan-ify make_regression ([#811](https://github.com/rapidsai/raft/pull/811)) [@mhoemmen](https://github.com/mhoemmen)
- Updating `raft::linalg` APIs to use `mdspan` ([#809](https://github.com/rapidsai/raft/pull/809)) [@divyegala](https://github.com/divyegala)
- Integrate KNN implementation: ivf-pq ([#789](https://github.com/rapidsai/raft/pull/789)) [@achirkin](https://github.com/achirkin)

## 🛠️ Improvements

- Some fixes for build.sh ([#901](https://github.com/rapidsai/raft/pull/901)) [@cjnolet](https://github.com/cjnolet)
- Revert recent fused l2 nn instantiations ([#899](https://github.com/rapidsai/raft/pull/899)) [@cjnolet](https://github.com/cjnolet)
- Update Python build instructions ([#898](https://github.com/rapidsai/raft/pull/898)) [@betatim](https://github.com/betatim)
- Adding ninja and cxx compilers to conda dev dependencies ([#893](https://github.com/rapidsai/raft/pull/893)) [@cjnolet](https://github.com/cjnolet)
- Output non-normalized distances in IVF-PQ and brute-force KNN ([#892](https://github.com/rapidsai/raft/pull/892)) [@Nyrio](https://github.com/Nyrio)
- Readme updates for 22.10 ([#884](https://github.com/rapidsai/raft/pull/884)) [@cjnolet](https://github.com/cjnolet)
- Breaking apart benchmarks into individual binaries ([#883](https://github.com/rapidsai/raft/pull/883)) [@cjnolet](https://github.com/cjnolet)
- Pin `dask` and `distributed` for release ([#858](https://github.com/rapidsai/raft/pull/858)) [@galipremsagar](https://github.com/galipremsagar)
- Mdspanifying (currently tested) `raft::matrix` ([#846](https://github.com/rapidsai/raft/pull/846)) [@cjnolet](https://github.com/cjnolet)
- Separating _RAFT_HOST and _RAFT_DEVICE macros ([#836](https://github.com/rapidsai/raft/pull/836)) [@cjnolet](https://github.com/cjnolet)
- Updating cpu job in hopes it speeds up python cpu builds ([#828](https://github.com/rapidsai/raft/pull/828)) [@cjnolet](https://github.com/cjnolet)
- Mdspan-ifying `raft::spatial` ([#827](https://github.com/rapidsai/raft/pull/827)) [@cjnolet](https://github.com/cjnolet)
- Fixing __init__.py for handle and stream ([#826](https://github.com/rapidsai/raft/pull/826)) [@cjnolet](https://github.com/cjnolet)
- Moving a few more things around ([#822](https://github.com/rapidsai/raft/pull/822)) [@cjnolet](https://github.com/cjnolet)
- Use fusedL2NN in ANN kmeans ([#821](https://github.com/rapidsai/raft/pull/821)) [@Nyrio](https://github.com/Nyrio)
- Separating test executables ([#820](https://github.com/rapidsai/raft/pull/820)) [@cjnolet](https://github.com/cjnolet)
- Separating mdspan/mdarray infra into host_* and device_* variants ([#810](https://github.com/rapidsai/raft/pull/810)) [@cjnolet](https://github.com/cjnolet)
- Fix malloc/delete mismatch ([#808](https://github.com/rapidsai/raft/pull/808)) [@mhoemmen](https://github.com/mhoemmen)
- Renaming `pyraft` -&gt; `raft-dask` ([#801](https://github.com/rapidsai/raft/pull/801)) [@cjnolet](https://github.com/cjnolet)
- Branch 22.10 merge 22.08 ([#800](https://github.com/rapidsai/raft/pull/800)) [@cjnolet](https://github.com/cjnolet)
- Statically link all CUDA toolkit libraries ([#797](https://github.com/rapidsai/raft/pull/797)) [@trxcllnt](https://github.com/trxcllnt)
- Minor follow-up fixes for ivf-flat ([#796](https://github.com/rapidsai/raft/pull/796)) [@achirkin](https://github.com/achirkin)
- KMeans benchmarks (cuML + ANN implementations) and fix for IndexT=int64_t ([#795](https://github.com/rapidsai/raft/pull/795)) [@Nyrio](https://github.com/Nyrio)
- Optimize fusedL2NN when data is skinny ([#794](https://github.com/rapidsai/raft/pull/794)) [@ahendriksen](https://github.com/ahendriksen)
- Complete the deprecation of duplicated hpp headers ([#793](https://github.com/rapidsai/raft/pull/793)) [@ahendriksen](https://github.com/ahendriksen)
- Prepare parts of the balanced kmeans for ivf-pq ([#788](https://github.com/rapidsai/raft/pull/788)) [@achirkin](https://github.com/achirkin)
- Unpin `dask` and `distributed` for development ([#783](https://github.com/rapidsai/raft/pull/783)) [@galipremsagar](https://github.com/galipremsagar)
- Exposing python wrapper for the RMAT generator logic ([#778](https://github.com/rapidsai/raft/pull/778)) [@teju85](https://github.com/teju85)
- Device, Host, Managed Accessor Types for `mdspan` ([#776](https://github.com/rapidsai/raft/pull/776)) [@divyegala](https://github.com/divyegala)
- Fix Forward-Merger Conflicts ([#768](https://github.com/rapidsai/raft/pull/768)) [@ajschmidt8](https://github.com/ajschmidt8)
- Fea 2208 kmeans use specializations ([#760](https://github.com/rapidsai/raft/pull/760)) [@cjnolet](https://github.com/cjnolet)
- ivf_flat::index: hide implementation details ([#747](https://github.com/rapidsai/raft/pull/747)) [@achirkin](https://github.com/achirkin)

# raft 22.08.00 (17 Aug 2022)

## 🚨 Breaking Changes

- Update `mdspan` to account for changes to `extents` ([#751](https://github.com/rapidsai/raft/pull/751)) [@divyegala](https://github.com/divyegala)
- Replace csr_adj_graph functions with faster equivalent ([#746](https://github.com/rapidsai/raft/pull/746)) [@ahendriksen](https://github.com/ahendriksen)
- Integrate KNN implementation: ivf-flat ([#652](https://github.com/rapidsai/raft/pull/652)) [@achirkin](https://github.com/achirkin)
- Moving kmeans from cuml to Raft ([#605](https://github.com/rapidsai/raft/pull/605)) [@lowener](https://github.com/lowener)

## 🐛 Bug Fixes

- Relax ivf-flat test recall thresholds ([#766](https://github.com/rapidsai/raft/pull/766)) [@achirkin](https://github.com/achirkin)
- Restrict the use of `]` to CXX 20 only. ([#764](https://github.com/rapidsai/raft/pull/764)) [@trivialfis](https://github.com/trivialfis)
- Update rapids-cmake version for pyraft in update-version.sh ([#749](https://github.com/rapidsai/raft/pull/749)) [@vyasr](https://github.com/vyasr)

## 📖 Documentation

- Use documented header template for doxygen ([#773](https://github.com/rapidsai/raft/pull/773)) [@galipremsagar](https://github.com/galipremsagar)
- Switch `language` from `None` to `&quot;en&quot;` in docs build ([#721](https://github.com/rapidsai/raft/pull/721)) [@galipremsagar](https://github.com/galipremsagar)

## 🚀 New Features

- Update `mdspan` to account for changes to `extents` ([#751](https://github.com/rapidsai/raft/pull/751)) [@divyegala](https://github.com/divyegala)
- Implement matrix transpose with mdspan. ([#739](https://github.com/rapidsai/raft/pull/739)) [@trivialfis](https://github.com/trivialfis)
- Implement unravel_index for row-major array. ([#723](https://github.com/rapidsai/raft/pull/723)) [@trivialfis](https://github.com/trivialfis)
- Integrate KNN implementation: ivf-flat ([#652](https://github.com/rapidsai/raft/pull/652)) [@achirkin](https://github.com/achirkin)

## 🛠️ Improvements

- Use common `js` and `css` code ([#779](https://github.com/rapidsai/raft/pull/779)) [@galipremsagar](https://github.com/galipremsagar)
- Pin `dask` &amp; `distributed` for release ([#772](https://github.com/rapidsai/raft/pull/772)) [@galipremsagar](https://github.com/galipremsagar)
- Move cmake to the build section. ([#763](https://github.com/rapidsai/raft/pull/763)) [@vyasr](https://github.com/vyasr)
- Adding old kmeans impl back in (as kmeans_deprecated) ([#761](https://github.com/rapidsai/raft/pull/761)) [@cjnolet](https://github.com/cjnolet)
- Fix for KMeans raw pointers API ([#758](https://github.com/rapidsai/raft/pull/758)) [@lowener](https://github.com/lowener)
- Fix KMeans ([#756](https://github.com/rapidsai/raft/pull/756)) [@divyegala](https://github.com/divyegala)
- Add inline to nccl_sync_stream() ([#750](https://github.com/rapidsai/raft/pull/750)) [@seunghwak](https://github.com/seunghwak)
- Replace csr_adj_graph functions with faster equivalent ([#746](https://github.com/rapidsai/raft/pull/746)) [@ahendriksen](https://github.com/ahendriksen)
- Add wrapper functions for ncclGroupStart() and ncclGroupEnd() ([#742](https://github.com/rapidsai/raft/pull/742)) [@seunghwak](https://github.com/seunghwak)
- Fix variadic template type check for mdarrays ([#741](https://github.com/rapidsai/raft/pull/741)) [@hlinsen](https://github.com/hlinsen)
- RMAT rectangular graph generator ([#738](https://github.com/rapidsai/raft/pull/738)) [@teju85](https://github.com/teju85)
- Update conda recipes to UCX 1.13.0 ([#736](https://github.com/rapidsai/raft/pull/736)) [@pentschev](https://github.com/pentschev)
- Add warp-aggregated atomic increment ([#735](https://github.com/rapidsai/raft/pull/735)) [@ahendriksen](https://github.com/ahendriksen)
- fix logic bug in include_checker.py utility ([#734](https://github.com/rapidsai/raft/pull/734)) [@grlee77](https://github.com/grlee77)
- Support 32bit and unsigned indices in bruteforce KNN ([#730](https://github.com/rapidsai/raft/pull/730)) [@achirkin](https://github.com/achirkin)
- Ability to use ccache to speedup local builds ([#729](https://github.com/rapidsai/raft/pull/729)) [@teju85](https://github.com/teju85)
- Pin max version of `cuda-python` to `11.7.0` ([#728](https://github.com/rapidsai/raft/pull/728)) [@Ethyling](https://github.com/Ethyling)
- Always add `raft::raft_nn_lib` and `raft::raft_distance_lib` aliases ([#727](https://github.com/rapidsai/raft/pull/727)) [@trxcllnt](https://github.com/trxcllnt)
- Add several type aliases and helpers for creating mdarrays ([#726](https://github.com/rapidsai/raft/pull/726)) [@achirkin](https://github.com/achirkin)
- fix nans in naive kl divergence kernel introduced by div by 0. ([#724](https://github.com/rapidsai/raft/pull/724)) [@mdoijade](https://github.com/mdoijade)
- Use rapids-cmake for cuco ([#722](https://github.com/rapidsai/raft/pull/722)) [@vyasr](https://github.com/vyasr)
- Update Python classifiers. ([#719](https://github.com/rapidsai/raft/pull/719)) [@bdice](https://github.com/bdice)
- Fix sccache ([#718](https://github.com/rapidsai/raft/pull/718)) [@Ethyling](https://github.com/Ethyling)
- Introducing raft::mdspan as an alias ([#715](https://github.com/rapidsai/raft/pull/715)) [@divyegala](https://github.com/divyegala)
- Update cuco version ([#714](https://github.com/rapidsai/raft/pull/714)) [@vyasr](https://github.com/vyasr)
- Update conda environment pinnings and update-versions.sh. ([#713](https://github.com/rapidsai/raft/pull/713)) [@bdice](https://github.com/bdice)
- Branch 22.08 merge branch 22.06 ([#712](https://github.com/rapidsai/raft/pull/712)) [@cjnolet](https://github.com/cjnolet)
- Testing conda compilers ([#705](https://github.com/rapidsai/raft/pull/705)) [@cjnolet](https://github.com/cjnolet)
- Unpin `dask` &amp; `distributed` for development ([#704](https://github.com/rapidsai/raft/pull/704)) [@galipremsagar](https://github.com/galipremsagar)
- Avoid shadowing CMAKE_ARGS variable in build.sh ([#701](https://github.com/rapidsai/raft/pull/701)) [@vyasr](https://github.com/vyasr)
- Use unique ptr in `print_device_vector` ([#695](https://github.com/rapidsai/raft/pull/695)) [@lowener](https://github.com/lowener)
- Add missing Thrust includes ([#678](https://github.com/rapidsai/raft/pull/678)) [@bdice](https://github.com/bdice)
- Consolidate C++ conda recipes and add libraft-tests package ([#641](https://github.com/rapidsai/raft/pull/641)) [@Ethyling](https://github.com/Ethyling)
- Moving kmeans from cuml to Raft ([#605](https://github.com/rapidsai/raft/pull/605)) [@lowener](https://github.com/lowener)

# raft 22.06.00 (7 Jun 2022)

## 🚨 Breaking Changes

- Rng: removed cyclic dependency creating hard-to-debug compiler errors ([#639](https://github.com/rapidsai/raft/pull/639)) [@MatthiasKohl](https://github.com/MatthiasKohl)
- Allow enabling NVTX markers by downstream projects after install ([#610](https://github.com/rapidsai/raft/pull/610)) [@achirkin](https://github.com/achirkin)
- Rng: expose host-rng-state in host-only API ([#609](https://github.com/rapidsai/raft/pull/609)) [@MatthiasKohl](https://github.com/MatthiasKohl)

## 🐛 Bug Fixes

- For fixing the cuGraph test failures with PCG ([#690](https://github.com/rapidsai/raft/pull/690)) [@vinaydes](https://github.com/vinaydes)
- Fix excessive memory used in selection test ([#689](https://github.com/rapidsai/raft/pull/689)) [@achirkin](https://github.com/achirkin)
- Revert print vector changes because of std::vector&lt;bool&gt; ([#681](https://github.com/rapidsai/raft/pull/681)) [@lowener](https://github.com/lowener)
- fix race in fusedL2knn smem read/write by adding a syncwarp ([#679](https://github.com/rapidsai/raft/pull/679)) [@mdoijade](https://github.com/mdoijade)
- gemm: fix  parameter C mistakenly set as const ([#664](https://github.com/rapidsai/raft/pull/664)) [@achirkin](https://github.com/achirkin)
- Fix SelectionTest: allow different indices when keys are equal. ([#659](https://github.com/rapidsai/raft/pull/659)) [@achirkin](https://github.com/achirkin)
- Revert recent cmake updates ([#657](https://github.com/rapidsai/raft/pull/657)) [@cjnolet](https://github.com/cjnolet)
- Don&#39;t install component dependency files in raft-header only mode ([#655](https://github.com/rapidsai/raft/pull/655)) [@robertmaynard](https://github.com/robertmaynard)
- Rng: removed cyclic dependency creating hard-to-debug compiler errors ([#639](https://github.com/rapidsai/raft/pull/639)) [@MatthiasKohl](https://github.com/MatthiasKohl)
- Fixing raft compile bug w/ RNG changes ([#634](https://github.com/rapidsai/raft/pull/634)) [@cjnolet](https://github.com/cjnolet)
- Get `libcudacxx` from `cuco` ([#632](https://github.com/rapidsai/raft/pull/632)) [@trxcllnt](https://github.com/trxcllnt)
- RNG API fixes ([#630](https://github.com/rapidsai/raft/pull/630)) [@MatthiasKohl](https://github.com/MatthiasKohl)
- Fix mdspan accessor mixin offset policy. ([#628](https://github.com/rapidsai/raft/pull/628)) [@trivialfis](https://github.com/trivialfis)
- Branch 22.06 merge 22.04 ([#625](https://github.com/rapidsai/raft/pull/625)) [@cjnolet](https://github.com/cjnolet)
- fix issue in fusedL2knn which happens when rows are multiple of 256 ([#604](https://github.com/rapidsai/raft/pull/604)) [@mdoijade](https://github.com/mdoijade)

## 🚀 New Features

- Restore changes from #653 and #655 and correct cmake component dependencies ([#686](https://github.com/rapidsai/raft/pull/686)) [@robertmaynard](https://github.com/robertmaynard)
- Adding handle and stream to pylibraft ([#683](https://github.com/rapidsai/raft/pull/683)) [@cjnolet](https://github.com/cjnolet)
- Map CMake install components to conda library packages ([#653](https://github.com/rapidsai/raft/pull/653)) [@robertmaynard](https://github.com/robertmaynard)
- Rng: expose host-rng-state in host-only API ([#609](https://github.com/rapidsai/raft/pull/609)) [@MatthiasKohl](https://github.com/MatthiasKohl)
- mdspan/mdarray template functions and utilities ([#601](https://github.com/rapidsai/raft/pull/601)) [@divyegala](https://github.com/divyegala)

## 🛠️ Improvements

- Change build.sh to find C++ library by default ([#697](https://github.com/rapidsai/raft/pull/697)) [@vyasr](https://github.com/vyasr)
- Pin `dask` and `distributed` for release ([#693](https://github.com/rapidsai/raft/pull/693)) [@galipremsagar](https://github.com/galipremsagar)
- Pin `dask` &amp; `distributed` for release ([#680](https://github.com/rapidsai/raft/pull/680)) [@galipremsagar](https://github.com/galipremsagar)
- Improve logging ([#673](https://github.com/rapidsai/raft/pull/673)) [@achirkin](https://github.com/achirkin)
- Fix minor errors in CMake configuration ([#662](https://github.com/rapidsai/raft/pull/662)) [@vyasr](https://github.com/vyasr)
- Pulling mdspan fork (from official rapids repo) into raft to remove dependency ([#649](https://github.com/rapidsai/raft/pull/649)) [@cjnolet](https://github.com/cjnolet)
- Fixing the unit test issue(s) in RAFT ([#646](https://github.com/rapidsai/raft/pull/646)) [@vinaydes](https://github.com/vinaydes)
- Build pyraft with scikit-build ([#644](https://github.com/rapidsai/raft/pull/644)) [@vyasr](https://github.com/vyasr)
- Some fixes to pairwise distances for cupy integration ([#643](https://github.com/rapidsai/raft/pull/643)) [@cjnolet](https://github.com/cjnolet)
- Require UCX 1.12.1+ ([#638](https://github.com/rapidsai/raft/pull/638)) [@jakirkham](https://github.com/jakirkham)
- Updating raft rng host public API and adding docs ([#636](https://github.com/rapidsai/raft/pull/636)) [@cjnolet](https://github.com/cjnolet)
- Build pylibraft with scikit-build ([#633](https://github.com/rapidsai/raft/pull/633)) [@vyasr](https://github.com/vyasr)
- Add `cuda_lib_dir` to `library_dirs`, allow changing `UCX`/`RMM`/`Thrust`/`spdlog` locations via envvars in `setup.py` ([#624](https://github.com/rapidsai/raft/pull/624)) [@trxcllnt](https://github.com/trxcllnt)
- Remove perf prints from MST ([#623](https://github.com/rapidsai/raft/pull/623)) [@divyegala](https://github.com/divyegala)
- Enable components installation using CMake ([#621](https://github.com/rapidsai/raft/pull/621)) [@Ethyling](https://github.com/Ethyling)
- Allow nullptr as input-indices argument of select_k ([#618](https://github.com/rapidsai/raft/pull/618)) [@achirkin](https://github.com/achirkin)
- Update CMake pinning to allow newer CMake versions ([#617](https://github.com/rapidsai/raft/pull/617)) [@vyasr](https://github.com/vyasr)
- Unpin `dask` &amp; `distributed` for development ([#616](https://github.com/rapidsai/raft/pull/616)) [@galipremsagar](https://github.com/galipremsagar)
- Improve performance of select-top-k RADIX implementation ([#615](https://github.com/rapidsai/raft/pull/615)) [@achirkin](https://github.com/achirkin)
- Moving more prims benchmarks to RAFT ([#613](https://github.com/rapidsai/raft/pull/613)) [@cjnolet](https://github.com/cjnolet)
- Allow enabling NVTX markers by downstream projects after install ([#610](https://github.com/rapidsai/raft/pull/610)) [@achirkin](https://github.com/achirkin)
- Improve performance of select-top-k  WARP_SORT implementation ([#606](https://github.com/rapidsai/raft/pull/606)) [@achirkin](https://github.com/achirkin)
- Enable building static libs ([#602](https://github.com/rapidsai/raft/pull/602)) [@trxcllnt](https://github.com/trxcllnt)
- Update `ucx-py` version ([#596](https://github.com/rapidsai/raft/pull/596)) [@ajschmidt8](https://github.com/ajschmidt8)
- Fix merge conflicts ([#587](https://github.com/rapidsai/raft/pull/587)) [@ajschmidt8](https://github.com/ajschmidt8)
- Making cuco, thrust, and mdspan optional dependencies. ([#585](https://github.com/rapidsai/raft/pull/585)) [@cjnolet](https://github.com/cjnolet)
- Some RBC3D fixes ([#530](https://github.com/rapidsai/raft/pull/530)) [@cjnolet](https://github.com/cjnolet)

# raft 22.04.00 (6 Apr 2022)

## 🚨 Breaking Changes

- Moving some of the remaining linalg prims from cuml ([#502](https://github.com/rapidsai/raft/pull/502)) [@cjnolet](https://github.com/cjnolet)
- Fix badly merged cublas wrappers ([#492](https://github.com/rapidsai/raft/pull/492)) [@achirkin](https://github.com/achirkin)
- Hiding implementation details for lap, clustering, spectral, and label ([#477](https://github.com/rapidsai/raft/pull/477)) [@cjnolet](https://github.com/cjnolet)
- Adding destructor for std comms and using nccl allreduce for barrier in mpi comms ([#473](https://github.com/rapidsai/raft/pull/473)) [@cjnolet](https://github.com/cjnolet)
- Cleaning up cusparse_wrappers ([#441](https://github.com/rapidsai/raft/pull/441)) [@cjnolet](https://github.com/cjnolet)
- Improvents to RNG ([#434](https://github.com/rapidsai/raft/pull/434)) [@vinaydes](https://github.com/vinaydes)
- Remove RAFT memory management ([#400](https://github.com/rapidsai/raft/pull/400)) [@viclafargue](https://github.com/viclafargue)
- LinAlg impl in detail ([#383](https://github.com/rapidsai/raft/pull/383)) [@divyegala](https://github.com/divyegala)

## 🐛 Bug Fixes

- Pin cmake in conda recipe to &lt;3.23 ([#600](https://github.com/rapidsai/raft/pull/600)) [@dantegd](https://github.com/dantegd)
- Fix make_device_vector_view ([#595](https://github.com/rapidsai/raft/pull/595)) [@lowener](https://github.com/lowener)
- Update cuco version. ([#592](https://github.com/rapidsai/raft/pull/592)) [@vyasr](https://github.com/vyasr)
- Fixing raft headers dir ([#574](https://github.com/rapidsai/raft/pull/574)) [@cjnolet](https://github.com/cjnolet)
- Update update-version.sh ([#560](https://github.com/rapidsai/raft/pull/560)) [@raydouglass](https://github.com/raydouglass)
- find_package(raft) can now be called multiple times safely ([#532](https://github.com/rapidsai/raft/pull/532)) [@robertmaynard](https://github.com/robertmaynard)
- Allocate sufficient memory for Hungarian if number of batches &gt; 1 ([#531](https://github.com/rapidsai/raft/pull/531)) [@ChuckHastings](https://github.com/ChuckHastings)
- Adding lap.hpp back (with deprecation) ([#529](https://github.com/rapidsai/raft/pull/529)) [@cjnolet](https://github.com/cjnolet)
- raft-config is idempotent no matter RAFT_COMPILE_LIBRARIES value ([#516](https://github.com/rapidsai/raft/pull/516)) [@robertmaynard](https://github.com/robertmaynard)
- Call initialize() in mpi_comms_t constructor. ([#506](https://github.com/rapidsai/raft/pull/506)) [@seunghwak](https://github.com/seunghwak)
- Improve row-major meanvar kernel via minimizing atomicCAS locks ([#489](https://github.com/rapidsai/raft/pull/489)) [@achirkin](https://github.com/achirkin)
- Adding destructor for std comms and using nccl allreduce for barrier in mpi comms ([#473](https://github.com/rapidsai/raft/pull/473)) [@cjnolet](https://github.com/cjnolet)

## 📖 Documentation

- Updating docs for 22.04 ([#566](https://github.com/rapidsai/raft/pull/566)) [@cjnolet](https://github.com/cjnolet)

## 🚀 New Features

- Add benchmarks ([#549](https://github.com/rapidsai/raft/pull/549)) [@achirkin](https://github.com/achirkin)
- Unify weighted mean code ([#514](https://github.com/rapidsai/raft/pull/514)) [@lowener](https://github.com/lowener)
- single-pass raft::stats::meanvar ([#472](https://github.com/rapidsai/raft/pull/472)) [@achirkin](https://github.com/achirkin)
- Move `random` package of cuML to RAFT ([#449](https://github.com/rapidsai/raft/pull/449)) [@divyegala](https://github.com/divyegala)
- mdspan integration. ([#437](https://github.com/rapidsai/raft/pull/437)) [@trivialfis](https://github.com/trivialfis)
- Interruptible execution ([#433](https://github.com/rapidsai/raft/pull/433)) [@achirkin](https://github.com/achirkin)
- make raft sources compilable with clang ([#424](https://github.com/rapidsai/raft/pull/424)) [@MatthiasKohl](https://github.com/MatthiasKohl)
- Span implementation. ([#399](https://github.com/rapidsai/raft/pull/399)) [@trivialfis](https://github.com/trivialfis)

## 🛠️ Improvements

- Adding build script for docs ([#589](https://github.com/rapidsai/raft/pull/589)) [@cjnolet](https://github.com/cjnolet)
- Temporarily disable new `ops-bot` functionality ([#586](https://github.com/rapidsai/raft/pull/586)) [@ajschmidt8](https://github.com/ajschmidt8)
- Fix commands to get conda output files ([#584](https://github.com/rapidsai/raft/pull/584)) [@Ethyling](https://github.com/Ethyling)
- Link to `cuco` and add faiss `EXCLUDE_FROM_ALL` option ([#583](https://github.com/rapidsai/raft/pull/583)) [@trxcllnt](https://github.com/trxcllnt)
- exposing faiss::faiss ([#582](https://github.com/rapidsai/raft/pull/582)) [@cjnolet](https://github.com/cjnolet)
- Pin `dask` and `distributed` version ([#581](https://github.com/rapidsai/raft/pull/581)) [@galipremsagar](https://github.com/galipremsagar)
- removing exclude_from_all from cuco ([#580](https://github.com/rapidsai/raft/pull/580)) [@cjnolet](https://github.com/cjnolet)
- Adding INSTALL_EXPORT_SET for cuco, rmm, thrust ([#579](https://github.com/rapidsai/raft/pull/579)) [@cjnolet](https://github.com/cjnolet)
- Thrust package name case ([#576](https://github.com/rapidsai/raft/pull/576)) [@trxcllnt](https://github.com/trxcllnt)
- Add missing thrust includes to transpose.cuh ([#575](https://github.com/rapidsai/raft/pull/575)) [@zbjornson](https://github.com/zbjornson)
- Use unanchored clang-format version check ([#573](https://github.com/rapidsai/raft/pull/573)) [@zbjornson](https://github.com/zbjornson)
- Fixing accidental removal of thrust target from cmakelists ([#571](https://github.com/rapidsai/raft/pull/571)) [@cjnolet](https://github.com/cjnolet)
- Don&#39;t add gtest to build export set or generate a gtest-config.cmake ([#565](https://github.com/rapidsai/raft/pull/565)) [@trxcllnt](https://github.com/trxcllnt)
- Set `main` label by default ([#559](https://github.com/rapidsai/raft/pull/559)) [@galipremsagar](https://github.com/galipremsagar)
- Add local conda channel while looking for conda outputs ([#558](https://github.com/rapidsai/raft/pull/558)) [@Ethyling](https://github.com/Ethyling)
- Updated dask and distributed to &gt;=2022.02.1 ([#557](https://github.com/rapidsai/raft/pull/557)) [@rlratzel](https://github.com/rlratzel)
- Upload packages using testing label for nightlies ([#556](https://github.com/rapidsai/raft/pull/556)) [@Ethyling](https://github.com/Ethyling)
- Add `.github/ops-bot.yaml` config file ([#554](https://github.com/rapidsai/raft/pull/554)) [@ajschmidt8](https://github.com/ajschmidt8)
- Disabling benchmarks building by default. ([#553](https://github.com/rapidsai/raft/pull/553)) [@cjnolet](https://github.com/cjnolet)
- KNN select-top-k variants ([#551](https://github.com/rapidsai/raft/pull/551)) [@achirkin](https://github.com/achirkin)
- Adding logger ([#550](https://github.com/rapidsai/raft/pull/550)) [@cjnolet](https://github.com/cjnolet)
- clang-tidy support: improved clang run scripts with latest changes (see cugraph-ops) ([#548](https://github.com/rapidsai/raft/pull/548)) [@MatthiasKohl](https://github.com/MatthiasKohl)
- Pylibraft for pairwise distances ([#540](https://github.com/rapidsai/raft/pull/540)) [@cjnolet](https://github.com/cjnolet)
- mdspan PoC for distance make_blobs ([#538](https://github.com/rapidsai/raft/pull/538)) [@cjnolet](https://github.com/cjnolet)
- Include thrust/sort.h in ball_cover.cuh ([#526](https://github.com/rapidsai/raft/pull/526)) [@akifcorduk](https://github.com/akifcorduk)
- Increase parallelism in allgatherv ([#525](https://github.com/rapidsai/raft/pull/525)) [@seunghwak](https://github.com/seunghwak)
- Moving device functions to cuh files and deprecating hpp ([#524](https://github.com/rapidsai/raft/pull/524)) [@cjnolet](https://github.com/cjnolet)
- Use `dynamic_extent` from `stdex`. ([#523](https://github.com/rapidsai/raft/pull/523)) [@trivialfis](https://github.com/trivialfis)
- Updating some of the ci check scripts ([#522](https://github.com/rapidsai/raft/pull/522)) [@cjnolet](https://github.com/cjnolet)
- Use shfl_xor in warpReduce for broadcast ([#521](https://github.com/rapidsai/raft/pull/521)) [@akifcorduk](https://github.com/akifcorduk)
- Fixing Python conda package and installation ([#520](https://github.com/rapidsai/raft/pull/520)) [@cjnolet](https://github.com/cjnolet)
- Adding instructions to install from conda and build using CPM ([#519](https://github.com/rapidsai/raft/pull/519)) [@cjnolet](https://github.com/cjnolet)
- Implement span storage optimization. ([#515](https://github.com/rapidsai/raft/pull/515)) [@trivialfis](https://github.com/trivialfis)
- RNG test fixes and improvements ([#513](https://github.com/rapidsai/raft/pull/513)) [@vinaydes](https://github.com/vinaydes)
- Moving scores and metrics over to raft::stats ([#512](https://github.com/rapidsai/raft/pull/512)) [@cjnolet](https://github.com/cjnolet)
- Random ball cover in 3d ([#510](https://github.com/rapidsai/raft/pull/510)) [@cjnolet](https://github.com/cjnolet)
- Initializing memory in RBC ([#509](https://github.com/rapidsai/raft/pull/509)) [@cjnolet](https://github.com/cjnolet)
- Adjusting conda packaging to remove duplicate dependencies ([#508](https://github.com/rapidsai/raft/pull/508)) [@cjnolet](https://github.com/cjnolet)
- Moving remaining stats prims from cuml ([#507](https://github.com/rapidsai/raft/pull/507)) [@cjnolet](https://github.com/cjnolet)
- Correcting the namespace ([#505](https://github.com/rapidsai/raft/pull/505)) [@vinaydes](https://github.com/vinaydes)
- Passing stream through commsplit ([#503](https://github.com/rapidsai/raft/pull/503)) [@cjnolet](https://github.com/cjnolet)
- Moving some of the remaining linalg prims from cuml ([#502](https://github.com/rapidsai/raft/pull/502)) [@cjnolet](https://github.com/cjnolet)
- Fixing spectral APIs ([#496](https://github.com/rapidsai/raft/pull/496)) [@cjnolet](https://github.com/cjnolet)
- Fix badly merged cublas wrappers ([#492](https://github.com/rapidsai/raft/pull/492)) [@achirkin](https://github.com/achirkin)
- Fix integer overflow in distances ([#490](https://github.com/rapidsai/raft/pull/490)) [@RAMitchell](https://github.com/RAMitchell)
- Reusing shared libs in gpu ci builds ([#487](https://github.com/rapidsai/raft/pull/487)) [@cjnolet](https://github.com/cjnolet)
- Adding fatbin to shared libs and fixing conda paths in cpu build ([#485](https://github.com/rapidsai/raft/pull/485)) [@cjnolet](https://github.com/cjnolet)
- Add CMake `install` rule for tests ([#483](https://github.com/rapidsai/raft/pull/483)) [@ajschmidt8](https://github.com/ajschmidt8)
- Adding cpu ci for conda build ([#482](https://github.com/rapidsai/raft/pull/482)) [@cjnolet](https://github.com/cjnolet)
- iUpdating codeowners to use new raft codeowners ([#480](https://github.com/rapidsai/raft/pull/480)) [@cjnolet](https://github.com/cjnolet)
- Hiding implementation details for lap, clustering, spectral, and label ([#477](https://github.com/rapidsai/raft/pull/477)) [@cjnolet](https://github.com/cjnolet)
- Define PTDS via `-D` to fix cache misses in sccache ([#476](https://github.com/rapidsai/raft/pull/476)) [@trxcllnt](https://github.com/trxcllnt)
- Unpin dask and distributed ([#474](https://github.com/rapidsai/raft/pull/474)) [@galipremsagar](https://github.com/galipremsagar)
- Replace `ccache` with `sccache` ([#471](https://github.com/rapidsai/raft/pull/471)) [@ajschmidt8](https://github.com/ajschmidt8)
- More README updates ([#467](https://github.com/rapidsai/raft/pull/467)) [@cjnolet](https://github.com/cjnolet)
- CUBLAS wrappers with switchable host/device pointer mode ([#453](https://github.com/rapidsai/raft/pull/453)) [@achirkin](https://github.com/achirkin)
- Cleaning up cusparse_wrappers ([#441](https://github.com/rapidsai/raft/pull/441)) [@cjnolet](https://github.com/cjnolet)
- Adding conda packaging for libraft and pyraft ([#439](https://github.com/rapidsai/raft/pull/439)) [@cjnolet](https://github.com/cjnolet)
- Improvents to RNG ([#434](https://github.com/rapidsai/raft/pull/434)) [@vinaydes](https://github.com/vinaydes)
- Hiding implementation details for comms ([#409](https://github.com/rapidsai/raft/pull/409)) [@cjnolet](https://github.com/cjnolet)
- Remove RAFT memory management ([#400](https://github.com/rapidsai/raft/pull/400)) [@viclafargue](https://github.com/viclafargue)
- LinAlg impl in detail ([#383](https://github.com/rapidsai/raft/pull/383)) [@divyegala](https://github.com/divyegala)

# raft 22.02.00 (2 Feb 2022)

## 🚨 Breaking Changes

- Simplify raft component CMake logic, and allow compilation without FAISS ([#428](https://github.com/rapidsai/raft/pull/428)) [@robertmaynard](https://github.com/robertmaynard)
- One cudaStream_t instance per raft::handle_t ([#291](https://github.com/rapidsai/raft/pull/291)) [@divyegala](https://github.com/divyegala)

## 🐛 Bug Fixes

- Removing extra logging from faiss mr ([#463](https://github.com/rapidsai/raft/pull/463)) [@cjnolet](https://github.com/cjnolet)
- Pin `dask` &amp; `distributed` versions ([#455](https://github.com/rapidsai/raft/pull/455)) [@galipremsagar](https://github.com/galipremsagar)
- Replace RMM CUDA Python bindings with those provided  by CUDA-Python ([#451](https://github.com/rapidsai/raft/pull/451)) [@shwina](https://github.com/shwina)
- Fix comms memory leak ([#436](https://github.com/rapidsai/raft/pull/436)) [@seunghwak](https://github.com/seunghwak)
- Fix C++ doxygen documentation ([#426](https://github.com/rapidsai/raft/pull/426)) [@achirkin](https://github.com/achirkin)
- Fix clang-format style errors ([#425](https://github.com/rapidsai/raft/pull/425)) [@achirkin](https://github.com/achirkin)
- Fix using incorrect macro RAFT_CHECK_CUDA in place of RAFT_CUDA_TRY ([#415](https://github.com/rapidsai/raft/pull/415)) [@achirkin](https://github.com/achirkin)
- Fix CUDA_CHECK_NO_THROW compatibility define ([#414](https://github.com/rapidsai/raft/pull/414)) [@zbjornson](https://github.com/zbjornson)
- Disabling fused l2 knn from bfknn ([#407](https://github.com/rapidsai/raft/pull/407)) [@cjnolet](https://github.com/cjnolet)
- Disabling expanded fused l2 knn to unblock cuml CI ([#404](https://github.com/rapidsai/raft/pull/404)) [@cjnolet](https://github.com/cjnolet)
- Reverting default knn distance to L2Unexpanded for now. ([#403](https://github.com/rapidsai/raft/pull/403)) [@cjnolet](https://github.com/cjnolet)

## 📖 Documentation

- README and build fixes before release ([#459](https://github.com/rapidsai/raft/pull/459)) [@cjnolet](https://github.com/cjnolet)
- Updates to Python and C++ Docs ([#442](https://github.com/rapidsai/raft/pull/442)) [@cjnolet](https://github.com/cjnolet)

## 🚀 New Features

- error macros: determining buffer size instead of fixed 2048 chars ([#420](https://github.com/rapidsai/raft/pull/420)) [@MatthiasKohl](https://github.com/MatthiasKohl)
- NVTX range helpers ([#416](https://github.com/rapidsai/raft/pull/416)) [@achirkin](https://github.com/achirkin)

## 🛠️ Improvements

- Splitting fused l2 knn specializations ([#461](https://github.com/rapidsai/raft/pull/461)) [@cjnolet](https://github.com/cjnolet)
- Update cuCollection git tag ([#447](https://github.com/rapidsai/raft/pull/447)) [@seunghwak](https://github.com/seunghwak)
- Remove libcudacxx patch needed for nvcc 11.4 ([#446](https://github.com/rapidsai/raft/pull/446)) [@robertmaynard](https://github.com/robertmaynard)
- Unpin `dask` and `distributed` ([#440](https://github.com/rapidsai/raft/pull/440)) [@galipremsagar](https://github.com/galipremsagar)
- Public apis for remainder of matrix and stats ([#438](https://github.com/rapidsai/raft/pull/438)) [@divyegala](https://github.com/divyegala)
- Fix bug in producer-consumer buffer exchange which occurs in UMAP test on GV100 ([#429](https://github.com/rapidsai/raft/pull/429)) [@mdoijade](https://github.com/mdoijade)
- Simplify raft component CMake logic, and allow compilation without FAISS ([#428](https://github.com/rapidsai/raft/pull/428)) [@robertmaynard](https://github.com/robertmaynard)
- Update ucx-py version on release using rvc ([#422](https://github.com/rapidsai/raft/pull/422)) [@Ethyling](https://github.com/Ethyling)
- Disabling fused l2 knn again. Not sure how this got added back. ([#421](https://github.com/rapidsai/raft/pull/421)) [@cjnolet](https://github.com/cjnolet)
- Adding no throw macro variants ([#417](https://github.com/rapidsai/raft/pull/417)) [@cjnolet](https://github.com/cjnolet)
- Remove `IncludeCategories` from `.clang-format` ([#412](https://github.com/rapidsai/raft/pull/412)) [@codereport](https://github.com/codereport)
- fix nan issues in L2 expanded sqrt KNN distances ([#411](https://github.com/rapidsai/raft/pull/411)) [@mdoijade](https://github.com/mdoijade)
- Consistent renaming of CHECK_CUDA and *_TRY macros ([#410](https://github.com/rapidsai/raft/pull/410)) [@cjnolet](https://github.com/cjnolet)
- Faster matrix-vector-ops ([#401](https://github.com/rapidsai/raft/pull/401)) [@achirkin](https://github.com/achirkin)
- Adding dev conda environment files. ([#397](https://github.com/rapidsai/raft/pull/397)) [@cjnolet](https://github.com/cjnolet)
- Update to UCX-Py 0.24 ([#392](https://github.com/rapidsai/raft/pull/392)) [@pentschev](https://github.com/pentschev)
- Branch 21.12 merge 22.02 ([#386](https://github.com/rapidsai/raft/pull/386)) [@cjnolet](https://github.com/cjnolet)
- Hiding implementation details for sparse API ([#381](https://github.com/rapidsai/raft/pull/381)) [@cjnolet](https://github.com/cjnolet)
- Adding distance specializations ([#376](https://github.com/rapidsai/raft/pull/376)) [@cjnolet](https://github.com/cjnolet)
- Use FAISS with RMM ([#363](https://github.com/rapidsai/raft/pull/363)) [@viclafargue](https://github.com/viclafargue)
- Add Fused L2 Expanded KNN kernel ([#339](https://github.com/rapidsai/raft/pull/339)) [@mdoijade](https://github.com/mdoijade)
- Update `.clang-format` to be consistent with all other RAPIDS repos ([#300](https://github.com/rapidsai/raft/pull/300)) [@codereport](https://github.com/codereport)
- One cudaStream_t instance per raft::handle_t ([#291](https://github.com/rapidsai/raft/pull/291)) [@divyegala](https://github.com/divyegala)

# raft 21.12.00 (9 Dec 2021)

## 🚨 Breaking Changes

- Use 64 bit CuSolver API for Eigen decomposition ([#349](https://github.com/rapidsai/raft/pull/349)) [@lowener](https://github.com/lowener)

## 🐛 Bug Fixes

- Fixing bad host-&gt;device copy ([#375](https://github.com/rapidsai/raft/pull/375)) [@cjnolet](https://github.com/cjnolet)
- Fix coalesced access checks in matrix_vector_op ([#372](https://github.com/rapidsai/raft/pull/372)) [@achirkin](https://github.com/achirkin)
- Port libcudacxx patch from cudf ([#370](https://github.com/rapidsai/raft/pull/370)) [@dantegd](https://github.com/dantegd)
- Fixing overflow in expanded distances ([#365](https://github.com/rapidsai/raft/pull/365)) [@cjnolet](https://github.com/cjnolet)

## 📖 Documentation

- Getting doxygen to run ([#371](https://github.com/rapidsai/raft/pull/371)) [@cjnolet](https://github.com/cjnolet)

## 🛠️ Improvements

- Upgrade `clang` to `11.1.0` ([#394](https://github.com/rapidsai/raft/pull/394)) [@galipremsagar](https://github.com/galipremsagar)
- Fix Changelog Merge Conflicts for `branch-21.12` ([#390](https://github.com/rapidsai/raft/pull/390)) [@ajschmidt8](https://github.com/ajschmidt8)
- Pin max `dask` &amp; `distributed` ([#388](https://github.com/rapidsai/raft/pull/388)) [@galipremsagar](https://github.com/galipremsagar)
- Removing conflict w/ CUDA_CHECK ([#378](https://github.com/rapidsai/raft/pull/378)) [@cjnolet](https://github.com/cjnolet)
- Update RAFT test directory ([#359](https://github.com/rapidsai/raft/pull/359)) [@viclafargue](https://github.com/viclafargue)
- Update to UCX-Py 0.23 ([#358](https://github.com/rapidsai/raft/pull/358)) [@pentschev](https://github.com/pentschev)
- Hiding implementation details for random, stats, and matrix ([#356](https://github.com/rapidsai/raft/pull/356)) [@divyegala](https://github.com/divyegala)
- README updates ([#351](https://github.com/rapidsai/raft/pull/351)) [@cjnolet](https://github.com/cjnolet)
- Use 64 bit CuSolver API for Eigen decomposition ([#349](https://github.com/rapidsai/raft/pull/349)) [@lowener](https://github.com/lowener)
- Hiding implementation details for distance primitives (dense + sparse) ([#344](https://github.com/rapidsai/raft/pull/344)) [@cjnolet](https://github.com/cjnolet)
- Unpin `dask` &amp; `distributed` in CI ([#338](https://github.com/rapidsai/raft/pull/338)) [@galipremsagar](https://github.com/galipremsagar)

# raft 21.10.00 (7 Oct 2021)

## 🚨 Breaking Changes

- Miscellaneous tech debts/cleanups ([#286](https://github.com/rapidsai/raft/pull/286)) [@viclafargue](https://github.com/viclafargue)

## 🐛 Bug Fixes

- Accounting for rmm::cuda_stream_pool not having a constructor for 0 streams ([#329](https://github.com/rapidsai/raft/pull/329)) [@divyegala](https://github.com/divyegala)
- Fix wrong lda parameter in gemv ([#327](https://github.com/rapidsai/raft/pull/327)) [@achirkin](https://github.com/achirkin)
- Fix `matrixVectorOp` to verify promoted pointer type is still aligned to vectorized load boundary ([#325](https://github.com/rapidsai/raft/pull/325)) [@viclafargue](https://github.com/viclafargue)
- Pin rmm to branch-21.10 and remove warnings from kmeans.hpp ([#322](https://github.com/rapidsai/raft/pull/322)) [@dantegd](https://github.com/dantegd)
- Temporarily pin RMM while refactor removes deprecated calls ([#315](https://github.com/rapidsai/raft/pull/315)) [@dantegd](https://github.com/dantegd)
- Fix more warnings ([#311](https://github.com/rapidsai/raft/pull/311)) [@harrism](https://github.com/harrism)

## 📖 Documentation

- Fix build doc ([#316](https://github.com/rapidsai/raft/pull/316)) [@lowener](https://github.com/lowener)

## 🚀 New Features

- Add Hamming, Jensen-Shannon, KL-Divergence, Russell rao and Correlation distance metrics support ([#306](https://github.com/rapidsai/raft/pull/306)) [@mdoijade](https://github.com/mdoijade)

## 🛠️ Improvements

- Pin max `dask` and `distributed` versions to `2021.09.1` ([#334](https://github.com/rapidsai/raft/pull/334)) [@galipremsagar](https://github.com/galipremsagar)
- Make sure we keep the rapids-cmake and raft cal version in sync ([#331](https://github.com/rapidsai/raft/pull/331)) [@robertmaynard](https://github.com/robertmaynard)
- Add broadcast with const input iterator ([#328](https://github.com/rapidsai/raft/pull/328)) [@seunghwak](https://github.com/seunghwak)
- Fused L2 (unexpanded) kNN kernel for NN &lt;= 64, without using temporary gmem to store intermediate distances ([#324](https://github.com/rapidsai/raft/pull/324)) [@mdoijade](https://github.com/mdoijade)
- Update with rapids cmake new features ([#320](https://github.com/rapidsai/raft/pull/320)) [@robertmaynard](https://github.com/robertmaynard)
- Update to UCX-Py 0.22 ([#319](https://github.com/rapidsai/raft/pull/319)) [@pentschev](https://github.com/pentschev)
- Fix Forward-Merge Conflicts ([#318](https://github.com/rapidsai/raft/pull/318)) [@ajschmidt8](https://github.com/ajschmidt8)
- Enable CUDA device code warnings as errors ([#307](https://github.com/rapidsai/raft/pull/307)) [@harrism](https://github.com/harrism)
- Remove max version pin for dask &amp; distributed on development branch ([#303](https://github.com/rapidsai/raft/pull/303)) [@galipremsagar](https://github.com/galipremsagar)
- Warnings are errors ([#299](https://github.com/rapidsai/raft/pull/299)) [@harrism](https://github.com/harrism)
- Use the new RAPIDS.cmake to fetch rapids-cmake ([#298](https://github.com/rapidsai/raft/pull/298)) [@robertmaynard](https://github.com/robertmaynard)
- ENH Replace gpuci_conda_retry with gpuci_mamba_retry ([#295](https://github.com/rapidsai/raft/pull/295)) [@dillon-cullinan](https://github.com/dillon-cullinan)
- Miscellaneous tech debts/cleanups ([#286](https://github.com/rapidsai/raft/pull/286)) [@viclafargue](https://github.com/viclafargue)
- Random Ball Cover Algorithm for 2D Haversine/Euclidean ([#213](https://github.com/rapidsai/raft/pull/213)) [@cjnolet](https://github.com/cjnolet)

# raft 21.08.00 (4 Aug 2021)

## 🚨 Breaking Changes

- expose epsilon parameter to allow precision to to be specified ([#275](https://github.com/rapidsai/raft/pull/275)) [@ChuckHastings](https://github.com/ChuckHastings)

## 🐛 Bug Fixes

- Fix support for different input and output types in linalg::reduce ([#296](https://github.com/rapidsai/raft/pull/296)) [@Nyrio](https://github.com/Nyrio)
- Const raft handle in sparse bfknn ([#280](https://github.com/rapidsai/raft/pull/280)) [@cjnolet](https://github.com/cjnolet)
- Add `cuco::cuco` to list of linked libraries ([#279](https://github.com/rapidsai/raft/pull/279)) [@trxcllnt](https://github.com/trxcllnt)
- Use nested include in destination of install headers to avoid docker permission issues ([#263](https://github.com/rapidsai/raft/pull/263)) [@dantegd](https://github.com/dantegd)
- Update UCX-Py version to 0.21 ([#255](https://github.com/rapidsai/raft/pull/255)) [@pentschev](https://github.com/pentschev)
- Fix mst knn test build failure due to RMM device_buffer change ([#253](https://github.com/rapidsai/raft/pull/253)) [@mdoijade](https://github.com/mdoijade)

## 🚀 New Features

- Add chebyshev, canberra, minkowksi and hellinger distance metrics ([#276](https://github.com/rapidsai/raft/pull/276)) [@mdoijade](https://github.com/mdoijade)
- Move FAISS ANN wrappers to RAFT ([#265](https://github.com/rapidsai/raft/pull/265)) [@cjnolet](https://github.com/cjnolet)
- Remaining sparse semiring distances ([#261](https://github.com/rapidsai/raft/pull/261)) [@cjnolet](https://github.com/cjnolet)
- removing divye from codeowners ([#257](https://github.com/rapidsai/raft/pull/257)) [@divyegala](https://github.com/divyegala)

## 🛠️ Improvements

- Pinning cuco to a specific commit hash for release ([#304](https://github.com/rapidsai/raft/pull/304)) [@rlratzel](https://github.com/rlratzel)
- Pin max `dask` &amp; `distributed` versions ([#301](https://github.com/rapidsai/raft/pull/301)) [@galipremsagar](https://github.com/galipremsagar)
- Overlap epilog compute with ldg of next grid stride in pairwise distance &amp; fusedL2NN kernels ([#292](https://github.com/rapidsai/raft/pull/292)) [@mdoijade](https://github.com/mdoijade)
- Always add faiss library alias if it&#39;s missing ([#287](https://github.com/rapidsai/raft/pull/287)) [@trxcllnt](https://github.com/trxcllnt)
- Use `NVIDIA/cuCollections` repo again ([#284](https://github.com/rapidsai/raft/pull/284)) [@trxcllnt](https://github.com/trxcllnt)
- Use the 21.08 branch of rapids-cmake as rmm requires it ([#278](https://github.com/rapidsai/raft/pull/278)) [@robertmaynard](https://github.com/robertmaynard)
- expose epsilon parameter to allow precision to to be specified ([#275](https://github.com/rapidsai/raft/pull/275)) [@ChuckHastings](https://github.com/ChuckHastings)
- Fix `21.08` forward-merge conflicts ([#274](https://github.com/rapidsai/raft/pull/274)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add lds and sts inline ptx instructions to force vector instruction generation ([#273](https://github.com/rapidsai/raft/pull/273)) [@mdoijade](https://github.com/mdoijade)
- Move ANN to RAFT (additional updates) ([#270](https://github.com/rapidsai/raft/pull/270)) [@cjnolet](https://github.com/cjnolet)
- Sparse semirings cleanup + hash table &amp; batching strategies ([#269](https://github.com/rapidsai/raft/pull/269)) [@divyegala](https://github.com/divyegala)
- Revert &quot;pin dask versions in CI ([#260)&quot; (#264](https://github.com/rapidsai/raft/pull/260)&quot; (#264)) [@ajschmidt8](https://github.com/ajschmidt8)
- Pass stream to device_scalar::value() calls. ([#259](https://github.com/rapidsai/raft/pull/259)) [@harrism](https://github.com/harrism)
- Update get_rmm.cmake to better support CalVer ([#258](https://github.com/rapidsai/raft/pull/258)) [@harrism](https://github.com/harrism)
- Add Grid stride pairwise dist and fused L2 NN kernels ([#250](https://github.com/rapidsai/raft/pull/250)) [@mdoijade](https://github.com/mdoijade)
- Fix merge conflicts ([#236](https://github.com/rapidsai/raft/pull/236)) [@ajschmidt8](https://github.com/ajschmidt8)

# raft 21.06.00 (9 Jun 2021)

## 🐛 Bug Fixes

- Update UCX-Py version to 0.20 ([#254](https://github.com/rapidsai/raft/pull/254)) [@pentschev](https://github.com/pentschev)
- cuco git tag update (again) ([#248](https://github.com/rapidsai/raft/pull/248)) [@seunghwak](https://github.com/seunghwak)
- Revert PR #232 for 21.06 release ([#246](https://github.com/rapidsai/raft/pull/246)) [@dantegd](https://github.com/dantegd)
- Python comms to hold onto server endpoints ([#241](https://github.com/rapidsai/raft/pull/241)) [@cjnolet](https://github.com/cjnolet)
- Fix Thrust 1.12 compile errors ([#231](https://github.com/rapidsai/raft/pull/231)) [@trxcllnt](https://github.com/trxcllnt)
- Make sure we use CalVer when checking out rapids-cmake ([#230](https://github.com/rapidsai/raft/pull/230)) [@robertmaynard](https://github.com/robertmaynard)
- Loss of Precision in MST weight alteration ([#223](https://github.com/rapidsai/raft/pull/223)) [@divyegala](https://github.com/divyegala)

## 🛠️ Improvements

- cuco git tag update ([#243](https://github.com/rapidsai/raft/pull/243)) [@seunghwak](https://github.com/seunghwak)
- Update `CHANGELOG.md` links for calver ([#233](https://github.com/rapidsai/raft/pull/233)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add Grid stride pairwise dist and fused L2 NN kernels ([#232](https://github.com/rapidsai/raft/pull/232)) [@mdoijade](https://github.com/mdoijade)
- Updates to enable HDBSCAN ([#208](https://github.com/rapidsai/raft/pull/208)) [@cjnolet](https://github.com/cjnolet)

# raft 0.19.0 (21 Apr 2021)

## 🐛 Bug Fixes

- Exposing spectral random seed property ([#193](https://github.com//rapidsai/raft/pull/193)) [@cjnolet](https://github.com/cjnolet)
- Fix pointer arithmetic in spmv smem kernel ([#183](https://github.com//rapidsai/raft/pull/183)) [@lowener](https://github.com/lowener)
- Modify default value for rowMajorIndex and rowMajorQuery in bf-knn ([#173](https://github.com//rapidsai/raft/pull/173)) [@viclafargue](https://github.com/viclafargue)
- Remove setCudaMallocWarning() call for libfaiss[@v1.7.0 ([#167](https://github.com//rapidsai/raft/pull/167)) @trxcllnt](https://github.com/v1.7.0 ([#167](https://github.com//rapidsai/raft/pull/167)) @trxcllnt)
- Add const to KNN handle ([#157](https://github.com//rapidsai/raft/pull/157)) [@hlinsen](https://github.com/hlinsen)

## 🚀 New Features

- Moving optimized L2 1-nearest neighbors implementation from cuml ([#158](https://github.com//rapidsai/raft/pull/158)) [@cjnolet](https://github.com/cjnolet)

## 🛠️ Improvements

- Fixing codeowners ([#194](https://github.com//rapidsai/raft/pull/194)) [@cjnolet](https://github.com/cjnolet)
- Adjust Hellinger pairwise distance to vaoid NaNs ([#189](https://github.com//rapidsai/raft/pull/189)) [@lowener](https://github.com/lowener)
- Add column major input support in contractions_nt kernels with new kernel policy for it ([#188](https://github.com//rapidsai/raft/pull/188)) [@mdoijade](https://github.com/mdoijade)
- Dice formula correction ([#186](https://github.com//rapidsai/raft/pull/186)) [@lowener](https://github.com/lowener)
- Scaling knn graph fix connectivities algorithm ([#181](https://github.com//rapidsai/raft/pull/181)) [@cjnolet](https://github.com/cjnolet)
- Fixing RAFT CI &amp; a few small updates for SLHC Python wrapper ([#178](https://github.com//rapidsai/raft/pull/178)) [@cjnolet](https://github.com/cjnolet)
- Add Precomputed to the DistanceType enum (for cuML DBSCAN) ([#177](https://github.com//rapidsai/raft/pull/177)) [@Nyrio](https://github.com/Nyrio)
- Enable matrix::copyRows for row major input ([#176](https://github.com//rapidsai/raft/pull/176)) [@tfeher](https://github.com/tfeher)
- Add Dice distance to distancetype enum ([#174](https://github.com//rapidsai/raft/pull/174)) [@lowener](https://github.com/lowener)
- Porting over recent updates to distance prim from cuml ([#172](https://github.com//rapidsai/raft/pull/172)) [@cjnolet](https://github.com/cjnolet)
- Update KNN ([#171](https://github.com//rapidsai/raft/pull/171)) [@viclafargue](https://github.com/viclafargue)
- Adding translations parameter to brute_force_knn ([#170](https://github.com//rapidsai/raft/pull/170)) [@viclafargue](https://github.com/viclafargue)
- Update Changelog Link ([#169](https://github.com//rapidsai/raft/pull/169)) [@ajschmidt8](https://github.com/ajschmidt8)
- Map operation ([#168](https://github.com//rapidsai/raft/pull/168)) [@viclafargue](https://github.com/viclafargue)
- Updating sparse prims based on recent changes ([#166](https://github.com//rapidsai/raft/pull/166)) [@cjnolet](https://github.com/cjnolet)
- Prepare Changelog for Automation ([#164](https://github.com//rapidsai/raft/pull/164)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update 0.18 changelog entry ([#163](https://github.com//rapidsai/raft/pull/163)) [@ajschmidt8](https://github.com/ajschmidt8)
- MST symmetric/non-symmetric output for SLHC ([#162](https://github.com//rapidsai/raft/pull/162)) [@divyegala](https://github.com/divyegala)
- Pass pre-computed colors to MST ([#154](https://github.com//rapidsai/raft/pull/154)) [@divyegala](https://github.com/divyegala)
- Streams upgrade in RAFT handle (RMM backend + create handle from parent&#39;s pool) ([#148](https://github.com//rapidsai/raft/pull/148)) [@afender](https://github.com/afender)
- Merge branch-0.18 into 0.19 ([#146](https://github.com//rapidsai/raft/pull/146)) [@dantegd](https://github.com/dantegd)
- Add device_send, device_recv, device_sendrecv, device_multicast_sendrecv ([#144](https://github.com//rapidsai/raft/pull/144)) [@seunghwak](https://github.com/seunghwak)
- Adding SLHC prims. ([#140](https://github.com//rapidsai/raft/pull/140)) [@cjnolet](https://github.com/cjnolet)
- Moving cuml sparse prims to raft ([#139](https://github.com//rapidsai/raft/pull/139)) [@cjnolet](https://github.com/cjnolet)

# raft 0.18.0 (24 Feb 2021)

## Breaking Changes 🚨

- Make NCCL root initialization configurable. (#120) @drobison00

## Bug Fixes 🐛

- Add idx_t template parameter to matrix helper routines (#131) @tfeher
- Eliminate CUDA 10.2 as valid for large svd solving (#129) @wphicks
- Update check to allow svd solver on CUDA&gt;=10.2 (#125) @wphicks
- Updating gpu build.sh and debugging threads CI issue (#123) @dantegd

## New Features 🚀

- Adding additional distances (#116) @cjnolet

## Improvements 🛠️

- Update stale GHA with exemptions &amp; new labels (#152) @mike-wendt
- Add GHA to mark issues/prs as stale/rotten (#150) @Ethyling
- Prepare Changelog for Automation (#135) @ajschmidt8
- Adding Jensen-Shannon and BrayCurtis to DistanceType for Nearest Neighbors (#132) @lowener
- Add brute force KNN (#126) @hlinsen
- Make NCCL root initialization configurable. (#120) @drobison00
- Auto-label PRs based on their content (#117) @jolorunyomi
- Add gather &amp; gatherv to raft::comms::comms_t (#114) @seunghwak
- Adding canberra and chebyshev to distance types (#99) @cjnolet
- Gpuciscripts clean and update (#92) @msadang

# RAFT 0.17.0 (10 Dec 2020)

## New Features
- PR #65: Adding cuml prims that break circular dependency between cuml and cumlprims projects
- PR #101: MST core solver
- PR #93: Incorporate Date/Nagi implementation of Hungarian Algorithm
- PR #94: Allow generic reductions for the map then reduce op
- PR #95: Cholesky rank one update prim

## Improvements
- PR #108: Remove unused old-gpubuild.sh
- PR #73: Move DistanceType enum from cuML to RAFT
- pr #92: Cleanup gpuCI scripts
- PR #98: Adding InnerProduct to DistanceType
- PR #103: Epsilon parameter for Cholesky rank one update
- PR #100: Add divyegala as codeowner
- PR #111: Cleanup gpuCI scripts
- PR #120: Update NCCL init process to support root node placement.

## Bug Fixes
- PR #106: Specify dependency branches to avoid pip resolver failure
- PR #77: Fixing CUB include for CUDA < 11
- PR #86: Missing headers for newly moved prims
- PR #102: Check alignment before binaryOp dispatch
- PR #104: Fix update-version.sh
- PR #109: Fixing Incorrect Deallocation Size and Count Bugs

# RAFT 0.16.0 (Date TBD)

## New Features

- PR #63: Adding MPI comms implementation
- PR #70: Adding CUB to RAFT cmake

## Improvements
- PR #59: Adding csrgemm2 to cusparse_wrappers.h
- PR #61: Add cusparsecsr2dense to cusparse_wrappers.h
- PR #62: Adding `get_device_allocator` to `handle.pxd`
- PR #67: Remove dependence on run-time type info

## Bug Fixes
- PR #56: Fix compiler warnings.
- PR #64: Remove `cublas_try` from `cusolver_wrappers.h`
- PR #66: Fixing typo `get_stream` to `getStream` in `handle.pyx`
- PR #68: Change the type of recvcounts & displs in allgatherv from size_t[] to size_t* and int[] to size_t*, respectively.
- PR #69: Updates for RMM being header only
- PR #74: Fix std_comms::comm_split bug
- PR #79: remove debug print statements
- PR #81: temporarily expose internal NCCL communicator

# RAFT 0.15.0 (Date TBD)

## New Features
- PR #12: Spectral clustering.
- PR #7: Migrating cuml comms -> raft comms_t
- PR #18: Adding commsplit to cuml communicator
- PR #15: add exception based error handling macros
- PR #29: Add ceildiv functionality
- PR #44: Add get_subcomm and set_subcomm to handle_t

## Improvements
- PR #13: Add RMM_INCLUDE and RMM_LIBRARY options to allow linking to non-conda RMM
- PR #22: Preserve order in comms workers for rank initialization
- PR #38: Remove #include <cudar_utils.h> from `raft/mr/`
- PR #39: Adding a virtual destructor to `raft::handle_t` and `raft::comms::comms_t`
- PR #37: Clean-up CUDA related utilities
- PR #41: Upgrade to `cusparseSpMV()`, alg selection, and rectangular matrices.
- PR #45: Add Ampere target to cuda11 cmake
- PR #47: Use gtest conda package in CMake/build.sh by default

## Bug Fixes
- PR #17: Make destructor inline to avoid redeclaration error
- PR #25: Fix bug in handle_t::get_internal_streams
- PR #26: Fix bug in RAFT_EXPECTS (add parentheses surrounding cond)
- PR #34: Fix issue with incorrect docker image being used in local build script
- PR #35: Remove #include <nccl.h> from `raft/error.hpp`
- PR #40: Preemptively fixed future CUDA 11 related errors.
- PR #43: Fixed CUDA version selection mechanism for SpMV.
- PR #46: Fix for cpp file extension issue (nvcc-enforced).
- PR #48: Fix gtest target names in cmake build gtest option.
- PR #49: Skip raft comms test if raft module doesn't exist

# RAFT 0.14.0 (Date TBD)

## New Features
- Initial RAFT version
- PR #3: defining raft::handle_t, device_buffer, host_buffer, allocator classes

## Bug Fixes
- PR #5: Small build.sh fixes
