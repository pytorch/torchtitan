"""KernelAgent integration for FX graph kernel optimization.

Workflow:
  1. dump_graph_for_analysis(gm) — get graph summary for LLM to identify fusion candidates
  2. extract_subgraph_as_problem(gm, nodes) — convert FX subgraph to KernelAgent problem
  3. generate_kernel(problem) — call KernelAgent to produce a Triton kernel
  4. register_triton_op(name, kernel_fn, ...) — register as torch custom op
  5. Write a graph pass that replaces matched patterns with the custom op
"""

from .extract_subgraph import (
    dump_graph_for_analysis,
    extract_subgraph_as_problem,
    find_pattern_instances,
)
from .bridge import generate_kernel, optimize_kernel
from .integrate import load_kernel, register_triton_op
