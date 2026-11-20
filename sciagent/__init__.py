"""sciagent: deterministic orchestration layer for MFEM-based verified runs.

The package deliberately contains no LLM dependency. Every decision taken by
the controller is a deterministic function of the frozen specification, the
capability matrix, the environment probe and the independent verifier's
typed verdict.
"""

__version__ = "0.1.0"
