#!/usr/bin/env python3
"""End-to-end example: fuse SwiGLU (silu + mul) via KernelAgent.

Demonstrates the full workflow:
  1. Analyze an FX graph to identify fusion candidates
  2. Extract the SwiGLU subgraph as a KernelAgent problem description
  3. Generate a Triton kernel via KernelAgent
  4. Register the kernel as a custom op
  5. Create a graph pass that replaces silu+mul with the fused kernel
  6. Verify correctness

Usage:
  # Step 1-2: Extract problem (works without GPU)
  python -m autoresearch.kernel_agent.example_swiglu --extract-only

  # Step 3: Generate kernel (requires KernelAgent + LLM API key)
  python -m autoresearch.kernel_agent.example_swiglu --generate

  # Step 4-6: Integrate and test (requires GPU)
  python -m autoresearch.kernel_agent.example_swiglu --integrate --kernel-path generated/swiglu/kernel.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

aten = torch.ops.aten


# ---------------------------------------------------------------------------
# Step 1-2: Build a toy FX graph and extract the SwiGLU subgraph
# ---------------------------------------------------------------------------

def _make_toy_graph() -> torch.fx.GraphModule:
    """Create a minimal FX graph containing a SwiGLU pattern.

    Uses make_fx to produce aten-level ops with FakeTensor metadata,
    matching what graph_trainer's aot_fx_trace produces for the MLP gate path.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx

    class ToyMLP(torch.nn.Module):
        def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
            return F.silu(x) * gate

    with FakeTensorMode() as fake_mode:
        x = torch.randn(2, 4096, 14336, dtype=torch.bfloat16, device="cpu")
        gate = torch.randn(2, 4096, 14336, dtype=torch.bfloat16, device="cpu")
        gm = make_fx(ToyMLP(), tracing_mode="fake")(
            fake_mode.from_tensor(x), fake_mode.from_tensor(gate)
        )

    return gm


def demo_extract():
    """Demonstrate graph analysis and subgraph extraction."""
    from autoresearch.kernel_agent import (
        dump_graph_for_analysis,
        extract_subgraph_as_problem,
        find_pattern_instances,
    )

    gm = _make_toy_graph()

    # Step 1: Analyze the graph
    print("=" * 80)
    print("Step 1: Graph Analysis")
    print("=" * 80)
    summary = dump_graph_for_analysis(gm)
    print(summary)

    # Step 2: Find SwiGLU pattern instances
    print("\n" + "=" * 80)
    print("Step 2: Find SwiGLU patterns")
    print("=" * 80)

    def match_swiglu(silu_node: torch.fx.Node):
        for user in silu_node.users:
            if user.op == "call_function" and user.target == aten.mul.Tensor:
                return [silu_node, user]
        return None

    instances = find_pattern_instances(gm, aten.silu.default, match_swiglu)
    print(f"Found {len(instances)} SwiGLU instance(s)")

    if not instances:
        # Try with the symbolic_trace target names
        for node in gm.graph.nodes:
            if node.op == "call_function":
                print(f"  Node: {node.name}, target: {node.target}")
        return None

    # Step 3: Extract as KernelAgent problem
    print("\n" + "=" * 80)
    print("Step 3: Extract SwiGLU as KernelAgent problem")
    print("=" * 80)
    problem = extract_subgraph_as_problem(
        gm,
        instances[0],
        description=(
            "Fused SwiGLU activation: silu(x) * gate.\n"
            "This pattern appears 64 times per Llama3 8B training step "
            "(32 layers x fwd + recompute).\n"
            "Both inputs are bfloat16 tensors of shape (batch*seq, intermediate_size)."
        ),
    )
    print(problem)

    # Save for KernelAgent
    out_dir = Path("autoresearch/kernel_agent/generated/swiglu")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "problem.py").write_text(problem)
    print(f"\nSaved problem to {out_dir / 'problem.py'}")

    return problem


# ---------------------------------------------------------------------------
# Step 3: Generate kernel via KernelAgent
# ---------------------------------------------------------------------------

def demo_generate(problem: str | None = None):
    """Generate a Triton kernel for SwiGLU using KernelAgent."""
    from autoresearch.kernel_agent import generate_kernel

    if problem is None:
        problem_path = Path("autoresearch/kernel_agent/generated/swiglu/problem.py")
        if not problem_path.exists():
            print("No problem file found. Run --extract-only first.")
            return None
        problem = problem_path.read_text()

    print("=" * 80)
    print("Step 3: Generate Triton kernel via KernelAgent")
    print("=" * 80)

    result = generate_kernel(
        problem,
        num_workers=4,
        max_rounds=8,
        output_dir="autoresearch/kernel_agent/generated/swiglu/logs",
    )

    if result["success"]:
        kernel_code = result["kernel_code"]
        out_path = Path("autoresearch/kernel_agent/generated/swiglu/kernel.py")
        out_path.write_text(kernel_code)
        print(f"Kernel generated successfully! Saved to {out_path}")
        print(f"  Worker {result['worker_id']}, {result['rounds']} rounds")
        print(f"\nKernel code:\n{kernel_code[:500]}...")
        return kernel_code
    else:
        print(f"Generation failed: {result['message']}")
        return None


# ---------------------------------------------------------------------------
# Step 4-5: Integrate kernel as a graph pass
# ---------------------------------------------------------------------------

def make_swiglu_replacement_pass(kernel_path: str | Path):
    """Create a graph pass that replaces silu+mul with the KernelAgent kernel.

    Returns a pass function with the standard signature:
        def pass_fn(gm: GraphModule, example_inputs) -> GraphModule
    """
    from autoresearch.kernel_agent import load_kernel, register_triton_op

    kernel_fn = load_kernel(kernel_path)

    fused_swiglu_op = register_triton_op(
        "fused_swiglu",
        kernel_fn,
        schema="(Tensor x, Tensor gate) -> Tensor",
        fake_fn=lambda x, gate: torch.empty_like(x),
    )

    def swiglu_kernel_pass(
        gm: torch.fx.GraphModule,
        example_inputs: tuple | None = None,
    ) -> torch.fx.GraphModule:
        """Replace silu+mul patterns with the fused KernelAgent Triton kernel."""
        from autoresearch.kernel_agent.integrate import replace_pattern

        def match_swiglu(silu_node: torch.fx.Node):
            for user in silu_node.users:
                if (
                    user.op == "call_function"
                    and user.target == aten.mul.Tensor
                ):
                    return [silu_node, user]
            return None

        def replace_swiglu(
            nodes: list[torch.fx.Node], graph: torch.fx.Graph
        ) -> torch.fx.Node:
            silu_node, mul_node = nodes
            x = silu_node.args[0]
            gate = mul_node.args[1] if mul_node.args[0] is silu_node else mul_node.args[0]
            with graph.inserting_after(mul_node):
                new_node = graph.call_function(fused_swiglu_op, args=(x, gate))
                new_node.meta = mul_node.meta.copy()
            return new_node

        count = replace_pattern(
            gm,
            match_swiglu,
            replace_swiglu,
            anchor_target=aten.silu.default,
        )
        if count > 0:
            print(f"Replaced {count} SwiGLU patterns with fused kernel")
        return gm

    return swiglu_kernel_pass


# ---------------------------------------------------------------------------
# Step 6: Verify correctness
# ---------------------------------------------------------------------------

def demo_verify(kernel_path: str | Path):
    """Verify the fused kernel produces identical results."""
    from autoresearch.kernel_agent import load_kernel

    kernel_fn = load_kernel(kernel_path)

    print("=" * 80)
    print("Step 6: Verify correctness")
    print("=" * 80)

    # Test shapes matching the FX graph
    x = torch.randn(2, 4096, 14336, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn(2, 4096, 14336, dtype=torch.bfloat16, device="cuda")

    ref = F.silu(x) * gate
    out = kernel_fn(x, gate)

    if torch.allclose(ref, out, rtol=1e-3, atol=1e-3):
        max_diff = (ref - out).abs().max().item()
        print(f"PASS (max diff: {max_diff:.2e})")
    else:
        max_diff = (ref - out).abs().max().item()
        print(f"FAIL (max diff: {max_diff:.2e})")

    # Benchmark
    import time
    torch.cuda.synchronize()

    for label, fn in [("aten silu+mul", lambda: F.silu(x) * gate), ("fused kernel", lambda: kernel_fn(x, gate))]:
        # warmup
        for _ in range(20):
            fn()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            fn()
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / 100
        print(f"  {label}: {elapsed:.3f} ms/call")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SwiGLU fusion via KernelAgent")
    parser.add_argument("--extract-only", action="store_true", help="Only extract the problem description")
    parser.add_argument("--generate", action="store_true", help="Generate kernel via KernelAgent")
    parser.add_argument("--integrate", action="store_true", help="Integrate and test kernel")
    parser.add_argument("--kernel-path", type=str, default="autoresearch/kernel_agent/generated/swiglu/kernel.py")
    args = parser.parse_args()

    if args.extract_only or (not args.generate and not args.integrate):
        demo_extract()

    if args.generate:
        demo_generate()

    if args.integrate:
        demo_verify(args.kernel_path)
        print("\nTo use as a graph pass in graph_trainer:")
        print(f"  pass_fn = make_swiglu_replacement_pass('{args.kernel_path}')")
        print("  # Add pass_fn to compile_time_passes in passes.py")


if __name__ == "__main__":
    main()
