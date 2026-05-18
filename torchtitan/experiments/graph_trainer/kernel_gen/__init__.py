"""Offline kernel generation and benchmarking tooling.

Workflow:
  1. Train with --compile.fused_kernel_dir to extract problems
  2. python -m torchtitan.experiments.graph_trainer.kernel_gen.run_all --dir <path> to generate kernels
  3. python -m torchtitan.experiments.graph_trainer.kernel_gen.benchmark_all --dir <path> to benchmark
  4. Train again — auto-picks up best backend per op
"""

from .bridge import generate_kernel, optimize_kernel
