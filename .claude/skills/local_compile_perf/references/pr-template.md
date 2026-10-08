# Writing the stack: PRs, findings doc, code

Reader: a tired junior engineer. They should get each PR in one read, without opening the code.

## Writing rules

- **TLDR first**, with the headline number.
- **Problem and Solution as numbered baby steps**, 4-6 each:
  - one fact per line; each line follows from the one before; if a line needs "and", split it;
  - start from what the code does today, not from the fix;
  - end the Problem with the cost (a number); end the Solution with what stays unchanged.
- **Name real things:** the function, the tensor and its shape, the config field.
- **Diagram** when data moves between stages: eager vs region, fwd vs bwd, before vs after.
- **Numbers in code blocks:** `before -> after`, one line per workload, conditions on one line above.
- **Plain words:** explain a term on first use or say what it does ("autograd copies the two grads back into one tensor", not `UnbindBackward0`).
- **Direction and both numbers:** "prevents a 2.7x slower backward (1089 vs 2987 us)", not "makes backward ~2x slower".
- **Say whether real runs hit it:** which recipes enable the path. If none today, call it headroom.
- **No paragraphs, no markdown tables.** One idea per line.
- **No `#N` or GitHub URLs** to other people's PRs or issues.
- **Last check:** can it be shorter?

## PR template

````markdown
**TLDR:** <what changes, why it is faster, headline number>

**Problem**
1. <what the code does today>
2. <the next fact, following from 1>
3. <where it goes wrong>
4. <the cost, with a number>

**Solution**
1. <the idea in one line>
2. <what this PR adds: function, region, file>
3. <what stays unchanged>

<optional before/after diagram>

**Results**
```text
<model, layers, tokens/rank, fwd+bwd, wall or kernel, hardware, median of N fresh processes>
<workload>   <before> -> <after> ms (<delta>)
correctness: outputs + all grads vs eager, bf16 and fp32: <...>; vs fp64: <...>
graphs: <region> N for <shapes> + no_grad (limit L)
```

You may ask: <rejected alternative: why, one line each>
NOTE: <not tested: loss curve, multi-GPU, ...>

**Test plan**
<test names or the exact command>

<details><summary>Stack summary (same in every PR)</summary>

<the summary block from FINDINGS.md>
</details>
````

## When the template doesn't fit

- **Cost step:** take the glue's own kernel time from the profile. Else "85.6 ms of the 240 ms step goes away (Results)".
- **Numbers from elsewhere:** label the line: `(measured one base earlier, same code)`, `(1 process, indicative)`.
- **Alternatives with numbers:** an `alternatives` section in the Results block; one line each.
- **Big PR:** correctness gets its own code block.
- **Region options** (`prefer_nd_tiling`, `recompile_limit`): a short list after Solution, one line each with why.
- **Skill / docs PR:** TLDR, "What it contains" (numbered), "How it was validated", Test plan.
- **Fix PR:** Problem / Solution as usual; Results is the correctness check, no speed number needed.
- **Same commit in several stacks:** end with `Stack: 3 of 7. Same commit as <other stack>/<branch>.`
- **Supersedes, stack-level correctness:** FINDINGS only, not PR bodies.
- **Blocked by compile:** only real blockers (a crash, a silent dtype or value change, a missing fusion worth >1%). One line in "You may ask" with the number: `Why not fold the residual add into addmm? Inductor copies the residual first: 64 extra copies per step.` None hit: say nothing.
- **Stack summary:** collapsed in every PR with `<- this PR` on its line; expanded in PR 1 with Rejected, Compile limits and Not tested.

## Filled example

````markdown
**TLDR:** GPT-OSS's per-expert bias backward becomes one fp32 GEMM instead of a bf16 scatter-add: 16k step -9.7%, deterministic.

**Problem**
1. **Bias per expert.** Each GPT-OSS expert projection adds its expert's bias to every routed row.
2. **Expanded in eager.** `repeat_interleave` copies the bias out to all R rows, then adds it.
3. **Scatter in backward.** Autograd sums the R row grads back into the bias with a bf16 scatter-add.
4. **Cost.** The scatter alone is 2.65 ms per 2 layers at 16k tokens, and its result changes run to run.

**Solution**
1. **Gather, don't expand.** Each row reads its expert's bias by index.
2. **Bias grad as a GEMM.** `add_grouped_bias` computes it as `one_hot[E+1, R] @ grad[R, O]`, accumulated in fp32.
3. **Region.** `grouped_expert_bias` fuses gather + add; width marked static (w13 and w2 differ), recompile_limit 16.
4. **Unchanged.** Expert weights, routing and the dispatcher.

```text
before:  bias[E,O] --repeat_interleave--> [R,O] --add-->   bwd: scatter_add bf16 (atomics)
after:   bias[E,O] --gather(expert_id)--> add   (region)   bwd: one_hot[E+1,R] @ grad  (fp32)
```

**Results**
```text
one projection, 16 experts, R = 4T, fwd+bwd kernel time, GB300
T=16k O=5760   1593 -> 615 us   (ideal ~319)
T=4k  O=5760    313 -> 161 us

GPT-OSS 20b layers 0-1, 16k tokens, fwd+bwd wall, GB300, 1 process (indicative)
step           28.46 -> 25.70 ms  (-9.7%)
correctness: fp64 CPU test vs the repeat_interleave reference, incl. an empty expert and padding rows
graphs: grouped_expert_bias 6 (2 widths x static/dynamic rows x grad mode), limit 16
```

You may ask: why not let block compile handle it? Inductor's scatter there uses atomics: nondeterministic.
NOTE: not tested: EP>1 with padded rows from TorchAO, loss curve.

**Test plan**
`pytest tests/unit_tests/cpu/test_gpt_oss_compile.py`
````

## Findings doc skeleton

Its top block becomes the Stack summary of every PR.

````markdown
# <model>: local compile results

```text
<model, layers, tokens/rank, AC mode, hardware, process count>
upstream regions -> stack   16k  240.1 -> 75.1 ms (-69%)
                             4k   80.2 -> 32.2 ms (-60%)
baselines: eager 243.9, per-layer compile 213.1

stack (biggest first)
1 skill
2 mhc                    -35.7%
3 complex_rope code       0      (enables 4)
4 attention RoPE glue    -22.2%
...
```

Rejected: <idea: number, why>, one per line
Compile limits hit: <limit: cost, workaround, repro>, one per line; omit if none
Not tested: <...>
Skill feedback: <missing patterns, wrong advice>
````

## Code rules

- **Minimal diff.** No defensive fallbacks, shims, single-use helpers or `f -> _f -> _bound_f` chains.
- **Names say what it holds;** no single letters or cryptic abbreviations. Shapes in a comment when the type is ambiguous: `routed_bias  # [R, O]`.
- **Docstring first line:** what the caller gets. Non-trivial public function: `Args:` plus an `Example:` with shapes; the example replaces prose.
- **Comments: the local why only.** An invariant that looks unnecessary, in terms the reader has:
  ```python
  # Detached inputs: requires_grad is part of the guards, so no_grad and training share one graph.
  ```
  No measured ratios, no narration of each line. A mechanism of 3+ steps gets numbered one-line steps.
- **Config field over hardcoded value** when two users need different values (e.g. a region's `recompile_limit`).
- **Gated paths side by side:** say what runs eager vs compiled in one place.
- **Leave clear upstream text alone;** match the repo style guide (torchtitan's `.claude/CLAUDE.md` and `.claude/rules/`).
