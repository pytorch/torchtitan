"""Agentic compute-efficiency optimizer for TorchTitan pretraining.

Three actors interact through fixed contracts: the Human authors the binding
`Constitution.md` and advisory `Ideas.md`; the Harness (this package) enforces
the constitution, runs/measures/judges candidates, and defines the Agent API; a
pluggable Agent proposes candidate recipes and sees the system only through the
Harness. Objective: climb throughput, floor quality. See `ARCHITECTURE.md`.
"""
