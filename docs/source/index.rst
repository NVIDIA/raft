NVIDIA RAFT Documentation
=========================

.. image:: ../../img/raft-tech-stack.svg
  :width: 800
  :alt: RAFT Tech Stack


Useful Resources
################

.. _raft_reference: https://docs.rapids.ai/api/raft/stable/

- `RAPIDS Community <https://rapids.ai/community.html>`_: Get help, contribute, and collaborate.
- `GitHub repository <https://github.com/NVIDIA/raft>`_: Download the RAFT source code.
- `Issue tracker <https://github.com/NVIDIA/raft/issues>`_: Report issues or request features.


What is NVIDIA RAFT?
####################

NVIDIA RAFT contains fundamental widely-used algorithms and primitives for machine learning and information retrieval. The algorithms are CUDA-accelerated and form building blocks for more easily writing high performance applications.

By taking a primitives-based approach to algorithm development, RAFT

- accelerates algorithm construction time
- reduces the maintenance burden by maximizing reuse across projects, and
- centralizes core reusable computations, allowing future optimizations to benefit all algorithms that use them.

While not exhaustive, the following general categories help summarize the accelerated building blocks that RAFT contains:

.. list-table::
   :widths: 25 50
   :header-rows: 1

   * - Category
     - Examples
   * - Data Formats
     - sparse & dense, conversions, data generation
   * - Dense Operations
     - linear algebra, matrix and vector operations, slicing, norms, factorization, least squares, svd & eigenvalue problems
   * - Sparse Operations
     - linear algebra, eigenvalue problems, slicing, norms, reductions, factorization, symmetrization, components & labeling
   * - Solvers
     - combinatorial optimization, iterative solvers
   * - Statistics
     - sampling, moments and summary statistics, metrics
   * - Tools & Utilities
     - common utilities for developing CUDA applications, multi-node multi-gpu infrastructure

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   quick_start.md
   build.md
   cpp_api.rst
   pylibraft_api.rst
   raft_dask_api.rst
   using_raft_comms.rst
   developer_guide.md
   contributing.md


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
