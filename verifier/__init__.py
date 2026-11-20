"""Independent verifier.

This package must not import ``sciagent.mfem_runner`` or ``sciagent.controller``
and never reads a solver's ``passed``-style flag as evidence of correctness.
It reads only: the frozen specification files of a run, and the raw artifacts
written by the solver (nodal values, element connectivity, boundary geometry,
and the solver's self-report, which it treats as a claim to be checked).
"""
