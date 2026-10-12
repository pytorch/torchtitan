# Harness and checks (copy-paste)

## Where regions live

- Declare: `@local_compile("name", batch_invariant=..., options={...})` on a function (`torchtitan/distributed/local_compile.py`). `grep -rn "@local_compile(" torchtitan/` lists existing names.
- Enable: the model config's `local_compile_regions` list (e.g. `["loss", "swiglu"]`), set in the model's config builder or appended in a recipe (`config.local_compile_regions.append("router")`).
- Bind: `apply_local_compile(regions)` (the trainer calls it); in a harness call it yourself after building the model.
- The API differs by torchtitan version. Newer main keeps regions in the model config (`local_compile_regions`). Older trees set them at recipe level, `Trainer.Config.compile = LocalCompileConfig(regions=...)` with a shared default list, bound with `.apply_local_compile()`. Check which applies: `grep -rn "LocalCompileConfig\|local_compile_regions" torchtitan/`.

## Bandwidth (once per GPU)

```python
x = torch.empty(2**30, dtype=torch.uint8, device="cuda"); y = torch.empty_like(x)
for _ in range(3): y.copy_(x)
s, e = torch.cuda.Event(True), torch.cuda.Event(True)
s.record(); [y.copy_(x) for _ in range(20)]; e.record(); torch.cuda.synchronize()
tb_s = 2 * x.numel() * 20 / (s.elapsed_time(e) / 1e3) / 1e12   # read + write
```

## 2-layer step at native dims

```python
torch.manual_seed(0)
torch.backends.cuda.matmul.fp32_precision = ...       # whatever this tree's trainer sets (grep fp32_precision torchtitan/)
cfg = ModelConfig(...)                                 # the model's real flavor config
blocks = nn.ModuleList([Block(cfg, layer_id=i) for i in (a, b)]).cuda().to(torch.bfloat16)
# pick layer ids that cover each block type (e.g. one dense + one MoE layer)
apply_local_compile(regions)                           # [] for eager, the upstream list, or your variant
x = torch.randn(tokens, cfg.dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
# read Model.forward for what each block takes (positions, rope cache, masks); build those once here,
# with requires_grad where training has it (e.g. a residual stream passed between blocks)
extra = dict(...)

def model(x, **extra):
    for block in blocks:
        x = block(x, **extra)
    return x

def step():
    for p in blocks.parameters(): p.grad = None
    x.grad = None
    out = model(x, **extra)
    out.float().square().mean().backward()

for _ in range(5): step()                   # warmup (compiles)
torch.cuda.synchronize()
s, e = torch.cuda.Event(True), torch.cuda.Event(True)
s.record()
for _ in range(20): step()
e.record(); torch.cuda.synchronize()
wall_ms = s.elapsed_time(e) / 20
```

- When production calls a region with a dynamic small dim (e.g. a stack width that varies per layer), warm the harness at several widths first; otherwise it times a static graph production never runs.
- MoE layers with an aux loss need `AuxLoss.set_step_denominator(...)` before the first forward, as the trainer does.
- Run each variant in a **fresh process** for cited numbers; one process with interleaved variants for screening.
- Record `torch.cuda.max_memory_allocated()` (reset before the timed loop) and `nvidia-smi --query-compute-apps=pid` before and after.
- **EP work on one GPU:** a harness with expert parallelism off silently skips the dispatcher's EP-only ops (e.g. `_permute` / `_unpermute`; on DSv3 the pair cost 9.36 ms per MoE layer at 16k at EP=2 shapes, 4.08 after a rewrite). Benchmark EP-path ops on one GPU at a real rank's shapes, with a mocked EP mesh or by calling the ops directly.
- **MoE on one GPU:**
  - local (no-EP) dispatcher, with local experts set to one EP rank's share (e.g. 256 experts / EP8 = 32) so grouped GEMM shapes match a real rank;
  - the router then scores fewer experts, so also time the router at full width;
  - libraries that need a process group: a 1-rank group for timing; a fake process group (`torch.testing._internal.distributed.fake_pg`) only for FakeTensor tracing (fake peers alias storage).
- Benchmark the code you will ship, not a harness copy of it: the shipped version may compute extra things (an in-graph inverse permutation cost ~18 µs the copy didn't).
- Before a multi-GPU cluster job, build the model at the target world size on meta tensors or a fake PG first; a world-1 smoke run won't catch config checks that fire only at EP > 1.

## Profiler kernel time

```python
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(5): step()
torch.cuda.synchronize()
rows = [(e.key, e.device_time_total / 5, e.count / 5) for e in prof.key_averages() if e.device_time_total > 0]
rows.sort(key=lambda r: -r[1])
kernel_ms = sum(r[1] for r in rows) / 1e3
for name, us, n in rows[:40]: print(f"{us:9.1f} us  x{n:4.1f}  {name[:110]}")
```

`key_averages` groups by name: identical kernels merge, so look at the count column.

## Mapping Triton kernels to source

- Inductor names kernels after the fused ops: `triton_poi_fused_add_mul_silu_1` (pointwise), `triton_red_*` (looped reduction), `triton_per_*` (persistent reduction).
- `TORCH_LOGS=output_code` prints the generated code with the source line of each fused node; `TORCH_COMPILE_DEBUG=1` writes it to disk.
- `TORCH_LOGS=graph_breaks,recompiles` shows why a region breaks or recompiles (guard failures name the shape or global state).

## Counting graphs

```python
torch._dynamo.reset()
from torch._dynamo.utils import counters
# run every shape the model uses: each training token count, each call site, then a no_grad pass
print(counters["stats"]["unique_graphs"])
```

Or for one function: `torch._dynamo.eval_frame._debug_get_cache_entry_list(fn.__code__)` (one entry per graph). The limit is `torch._dynamo.config.recompile_limit` (8); with `fullgraph=True`, hitting it raises.

## Eager cost and GraphTrainer timing

- Eager cost of a rewrite: force the new branch on in eager (patch the gate) and time the 2-layer step with inputs requiring grad.
- GraphTrainer (only if the model trains with it): trace the 2-layer step with make_fx and compile it with `full_inductor_compilation_pass` (`torchtitan/experiments/graph_trainer/inductor_passes.py`), then time old vs new branch in the same process from profiler kernel time.

## CUDA-graph and SAC checks

```python
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): step()               # warm up on a side stream
torch.cuda.current_stream().wait_stream(s)
with torch.cuda.graph(g): step()
g.replay(); torch.cuda.synchronize()        # compare outputs/grads to an uncaptured step
```

Steps that zero grads inside the capture run extra kernels; compare captured vs captured. For SAC, apply the model's AC the way the trainer does and time again. On main (RegionAC / `torch_remat`), wrapping blocks with `_wrap_block` alone fails ("Duplicate torch_remat region name"); call `SelectiveAC.apply` on a holder module like the trainer. On older trees eager SAC replays region forwards in the backward.

## Environment gotchas

- `python -c` run from the repo root imports the main checkout, not your worktree on `PYTHONPATH`: run harnesses from another directory.
- `git stash` and branch names are shared across worktrees; use WIP commits instead of stashes.
- Subagents' file-writing tools may refuse report files; writing them via the shell works.
