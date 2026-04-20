# Manifesto

As accelerators get faster, CPU-side kernel launch overhead dominates —
you can't launch kernels fast enough to keep the GPU fed. CUDAGraph is
already a must on GB200, and this is the direction all hardware is heading.

GraphTrainer exists because distributed training at scale will require a
compiler. The question is when, not if. It captures the full training step
— forward, loss, backward — as a single FX graph, then transforms and
optimizes that graph before execution. Built as an experiment on top of
torchtitan, it serves as both a proving ground for compiler-driven
training and a demonstration of how to use PyTorch's compiler as a
toolkit.

## Eager Challenges

**Composability is fragile.** Getting FSDP2, activation checkpointing,
torch.compile, and CUDAGraph to all work together is a minefield.
Each feature is its own system with its own interception points, and they
interfere with each other in non-obvious ways: compile graph-breaks on
FSDP2, AC recomputes FSDP2 all-gathers in backward, AC with compile
graph-breaks invalidates AC and causes OOM, and so on. Each combination
needs its own workaround, and workarounds for one pair can break another.

**CUDAGraph is hard.** Making CUDAGraph work in eager requires deep
understanding of PyTorch autograd engine internals and careful memory
management. Wrapping a full training step is tractable; regional CUDAGraph
is much harder.

**Scheduling is coarse.** `autograd.Function` and hooks are the only
mechanisms for fine-grain scheduling in eager, and they're not granular
enough — code gets ugly fast. Microbatch overlap for MoE is a case in
point. With a graph representation, scheduling is just node reordering.

## What GraphTrainer Bets On

**Single unified graph.** Capture the entire training step — forward, loss,
backward — as one flat FX graph. No separate graphs stitched together, no
opaque boundaries. Full visibility into every operation — in particular, all backward
computations are explicit.

**Every optimization is a graph pass.** Activation checkpointing, CUDAGraph,
CPU offload, communication overlap, kernel fusion — all expressed as
transformations on the same graph. Passes compose naturally because they share a common
representation. Adding a new optimization means writing a new pass, not
threading hooks through the entire stack.

**SimpleFSDP.** A compiler-friendly replacement for FSDP2 that expresses
all-gather and reduce-scatter as traceable DTensor operations. The
collectives show up as nodes in the graph, so they can be reordered,
fused, and overlapped by passes — not hidden behind opaque module hooks.

**CUDAGraph becomes manageable.** With a graph, all computation is explicit
— no hidden autograd state, no opaque memory management. Piecewise
CUDAGraph wrapping is straightforward because you can see exactly what
needs to be captured.

**Debuggability.** The graph is inspectable. You can dump it, diff it
before and after a pass, and see exactly what changed. When something
goes wrong, you're reading a concrete program — not stepping through
callback chains.
