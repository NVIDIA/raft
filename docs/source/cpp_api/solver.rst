Solvers
=======

This page provides C++ class references for the publicly-exposed elements of the iterative and combinatorial solvers package.

.. role:: py(code)
   :language: c++
   :class: highlight


Linear Assignment Problem
#########################

``#include <raft/solver/linear_assignment.cuh>``

.. doxygenclass:: raft::solver::LinearAssignmentProblem
    :project: RAFT
    :members:

Batched Non-negative Least Squares
##################################

``#include <raft/solver/nnls.cuh>``

.. doxygenstruct:: raft::solver::nnls_params
    :project: RAFT
    :members:

.. doxygenfunction:: raft::solver::nnls
    :project: RAFT

Minimum Spanning Tree
#####################

``#include <raft/sparse/solver/mst.cuh>``

.. doxygenfunction:: raft::sparse::solver::mst
    :project: RAFT
