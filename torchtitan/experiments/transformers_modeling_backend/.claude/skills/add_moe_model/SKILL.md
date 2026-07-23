---
name: add_moe_model
description: Add a new HF MoE model to the transformers_modeling_backend experiment. Use when the user provides a HuggingFace model ID and wants to integrate it with Titan's MoE, or invokes /add_moe_model.
---

# Add HF MoE Model to Transformers Backend

Add support for a new HuggingFace MoE model in the
`torchtitan/experiments/transformers_modeling_backend/` experiment.

## Scope of changes

**ONLY modify code under `torchtitan/experiments/transformers_modeling_backend/`.**
If supporting the new model appears to require a change outside this folder
(e.g. in `torchtitan/models/`, `torchtitan/distributed/`, `torchtitan/components/`,
or any other core path), **STOP and escalate to the user** — describe the change
you believe is needed and why, and wait for their decision instead of editing
core yourself.

The one exception is the **temporary, must-be-reverted diagnostic patches** in
Phase 3c (numerical-gap analysis): those edit core only to measure where titan
diverges from HF, are reverted before the change is done, and are never
committed. Any *permanent* change outside the experiment folder still requires
escalation.

**DO modify experiment code** to support the new model. The experiment folder
(`torchtitan/experiments/transformers_modeling_backend/`) is designed to be
extended for new models. Expected changes include:

- `state_dict_adapter.py` — add regex patterns for new key naming conventions,
  handle transposed weight layouts, fix round-trip mappings
- `moe_replacement.py` — add detection logic for new scoring functions, router
  types, expert layouts, or shared expert patterns in `_probe_hf_moe_block`
  and its helper functions
- `hf_sharding.py` — give **every parameter/buffer-bearing module** the model
  introduces a sharding config. This is not limited to attention projections:
  `set_hf_sharding_configs` covers the root-level `tok_embeddings`, `norm`,
  `lm_head`, and `rotary_emb` (and their buffers), and `_set_layer_sharding_configs`
  covers per-layer norms, the attention block, and the dense MLP. A module with
  params/buffers but no config will mix a plain tensor with a DTensor and crash
  under TP (the `_assert_all_states_sharded` backstop now makes that a loud
  setup-time error for any param/buffer-bearing dense-path module — root
  modules and every decoder layer, excluding the Titan-swapped MoE). For a genuinely new module, shard it
  (`colwise_config`/`rowwise_config`) if it has a clear TP layout, otherwise
  default to `_replicate_config(module)` (replicate weights, no TP). If a new
  module type doesn't fit any existing helper, that's a signal to stop and think
  (or escalate) rather than leave it unconfigured.
- `model.py` — handle alternative submodule names: HF models name the same
  component differently (feed-forward `mlp` vs `feed_forward`, final norm `norm`
  vs `final_layernorm`). Add a new model's variant to the relevant lookup
  (`_get_moe_attr_name`, the `HFTransformerModel` accessor properties).
- `scripts/numerical_equivalence.py` — add the model's synthetic test config

When blockers are found in Phase 1, fix them in the experiment code before
proceeding to Phase 2. Do NOT stop and ask — implement the fixes, then
continue the workflow.

## Input

The user provides a HuggingFace model ID (e.g. `Qwen/Qwen3-30B-A3B`).
Optionally, a local path to downloaded weights for pretrained testing.

Before starting, ask the user (if not already specified):
1. **Which branch to work on** — should changes go on the current branch,
   or a new branch? If running in a worktree, clarify whether the commit
   should target the main feature branch or stay on the worktree branch.
2. **Whether to commit** — after all tests pass, should the changes be
   committed automatically, or left as uncommitted changes for the user
   to review first?

## Phase 1: Compatibility Probe

Load the model config and a single MoE layer on meta device to verify
titan compatibility. No file changes in this phase — but record all
issues found. After the probe, fix any blockers in the experiment code
before moving to Phase 2.

### 1a. Load and inspect config

```python
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
```

Record: `model_type`, `hidden_size`, `intermediate_size`,
`moe_intermediate_size`, `num_attention_heads`, `num_key_value_heads`,
`num_hidden_layers`.

### 1b. Instantiate on meta device

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
```

### 1c. Find MoE layers

Scan `model.model.layers` for layers where `hasattr(mlp, "gate") or
hasattr(mlp, "router")` AND `hasattr(mlp, "experts")`.

Record:
- Which layer indices are MoE vs dense
- The mechanism: `first_k_dense_replace`, `decoder_sparse_step`, or
  `mlp_layer_types`
- Minimum `num_hidden_layers` needed to have at least one MoE layer

### 1d. Probe MoE block

Call `_probe_hf_moe_block(moe_block, config)` from
`torchtitan/experiments/transformers_modeling_backend/moe_replacement.py`.

**Check each field succeeds without error:**

| Field | How it's resolved | Blocker if missing |
|-------|-------------------|--------------------|
| `num_experts` | `experts.num_experts`, `gate.weight.shape[0]`, or config attrs | Yes |
| `dim` | `config.hidden_size` | Yes |
| `moe_intermediate_size` | `experts.gate_up_proj.shape[1] // 2` or config | Yes |
| `top_k` | `moe_block.top_k`, `config.num_experts_per_tok` | Yes |
| `score_func` | `sigmoid` if `e_score_correction_bias` exists, else `softmax` | Blocker if neither |
| `route_norm` | `config.norm_topk_prob` or default `True` | No |
| `route_scale` | `config.routed_scaling_factor` or `1.0` | No |
| `num_expert_groups` | `config.n_group` or `None` | No |
| `shared_expert_info` | Probes `shared_expert`/`shared_experts`/`shared_mlp` | No |

**Issues to detect and fix in experiment code:**
- Expert weight layout transposed (e.g. `(E, H, 2*I)` instead of
  `(E, 2*I, H)`) — fix `state_dict_adapter.py` to transpose during
  conversion, and `moe_replacement.py` to detect the layout
- Experts stored as `nn.ModuleList` of individual MLPs instead of fused
  3D tensors — this is a true blocker (needs HF upstream change), report
  to user and stop
- Scoring function not `softmax`/`sigmoid` but detectable from the model
  code (e.g. `torch.sigmoid` call without `e_score_correction_bias`) —
  fix `_resolve_score_func()` in `moe_replacement.py`
- Router uses `nn.Linear` directly (key is `router.weight` not
  `gate.weight`) — add pattern to `state_dict_adapter.py`
- MoE layer attribute is not `mlp` (e.g. `feed_forward`) — fix
  `model.py` and `moe_replacement.py` to handle the alternative name
- Shared expert round-trip fails (singular/plural mismatch) — fix
  reverse patterns in `state_dict_adapter.py`
- Model doesn't support `grouped_mm` expert implementation — adjust
  the numerical test to skip the `grouped_mm` alignment for this model
- Layer-level MoE: router and experts are siblings of the dense MLP
  at the decoder layer level, not nested inside `layer.mlp` (e.g.
  Gemma4 has `layer.mlp` + `layer.router` + `layer.experts`). In this
  case the dense MLP is functionally a **shared expert** — it runs on
  every token and its output is summed with the routed expert output.
  The correct mapping is to treat the dense MLP as `shared_experts`
  inside Titan's `MoE` module and replace `layer.mlp` with the
  full Titan MoE (router + experts + shared_experts). This way the
  entire block has a single `ShardingConfig` and TP/EP work correctly.
  Fix in `moe_replacement.py`: when probing, detect that the dense MLP
  is a shared expert, include it in `shared_expert_info`, and when
  swapping, replace `layer.mlp` with the Titan MoE that includes the
  dense MLP as its shared expert. Wire the router and expert weights
  from the layer-level attributes into the Titan MoE.

### 1e. Probe attention for ShardingConfig

Check which sharding configs `_set_layer_sharding_configs` (in
`hf_sharding.py`) needs:

Q and KV are detected **independently**, so any mix of full-rank/low-rank
works (e.g. DeepSeek-V2-Lite has full-rank `q_proj` + low-rank KV):

| Pattern | Projections | Already supported? |
|---------|-------------|--------------------|
| Full-rank Q | `q_proj` | Yes (`colwise_config`) |
| Low-rank Q (MLA) | `q_a_proj`, `q_b_proj` | Yes (`_replicate_config`/`colwise_config`) |
| Standard KV (GQA) | `k_proj`, `v_proj` | Yes (`colwise_config`) |
| Low-rank KV (MLA) | `kv_a_proj_with_mqa`, `kv_b_proj` | Yes (`_replicate_config`/`colwise_config`) |
| Output | `o_proj` or `dense` | Yes (`rowwise_config`) |
| Q/K norms | `q_norm`, `k_norm` | Yes (`_replicate_config`, each independent) |
| V-norm | `v_norm` | Yes (`_replicate_config`) |
| DSA indexer | `indexer` | FSDP/EP only — **NOT under TP** (fails loud; the indexer's scatter_/index ops need the `spmd_types` backend, see `hf_sharding.py`) |

The `_assert_all_states_sharded` backstop will fail loud at setup if any
param/buffer-bearing attention (or other dense-path) module is left without a
config, so an unhandled projection surfaces immediately. If found, report:
"attention has unsupported projection `<name>`, needs
`_set_layer_sharding_configs` update in `hf_sharding.py`."

Also check: does the layer have `post_attention_layernorm`? (Some models
omit it — already handled.) Does the embedding have extra buffers (e.g.
Gemma4's `embed_scale`)? (Already handled — buffers are enumerated
dynamically.)

### 1f. State dict adapter round-trip

```python
from torchtitan.experiments.transformers_modeling_backend.state_dict_adapter import (
    hf_to_titan_moe_state_dict, titan_to_hf_moe_state_dict,
)

hf_sd = model.state_dict()
titan_sd = hf_to_titan_moe_state_dict(hf_sd)
roundtrip_sd = titan_to_hf_moe_state_dict(titan_sd)
```

Verify: all original keys present, no extra keys, values match.
If keys are missing, the adapter needs new patterns for this model's
key naming convention.

### 1g. Report

Print a compatibility summary table:

```
Compatibility report for <model_id>
  model_type:          <type>
  hidden_size:         <H>
  moe_intermediate:    <I>
  num_experts:         <E>
  top_k:               <K>
  score_func:          softmax|sigmoid
  route_norm:          True|False
  shared_experts:      none|additive|sigmoid-gated
  attention:           gqa|mla|mla+dsa
  MoE layers:          <indices>
  min_layers_needed:   <N>
  state_dict_roundtrip: PASS|FAIL
  issues:              none | <list>
```

### 1h. Fix issues

If issues were found, fix them in the experiment code now. Consult the
"Issue Resolution Reference" at the bottom of this document for where
each fix goes. The only true blocker is `nn.ModuleList` experts — for
everything else, implement the fix and re-run the probe to verify.

After all issues are resolved and the probe passes cleanly, proceed to
Phase 2.

## Phase 2: Add Synthetic Test Config

Add an entry to `_MODEL_CONFIGS` in
`torchtitan/experiments/transformers_modeling_backend/.claude/skills/add_moe_model/scripts/numerical_equivalence.py`.

### Config rules

Use production-scale `hidden_size` and `moe_intermediate_size` from the
real config. Override everything else for a small single-GPU test:

```python
"<model_type>": dict(
    model_type="<model_type>",
    hidden_size=<real>,           # from pretrained config
    intermediate_size=<real>,     # from pretrained config
    moe_intermediate_size=<real>, # from pretrained config
    num_local_experts=8,          # reduced from real
    num_experts_per_tok=2,        # reduced from real
    num_hidden_layers=<min>,      # minimum for MoE layer to exist
    num_attention_heads=<fits>,   # must divide hidden_size evenly
    num_key_value_heads=<fits>,   # must divide num_attention_heads evenly
    vocab_size=256,
    max_position_embeddings=64,
    attn_implementation="sdpa",
    use_cache=False,
    # Include model-specific fields:
    # first_k_dense_replace, decoder_sparse_step, mlp_layer_types,
    # n_group, topk_group, n_shared_experts, norm_topk_prob,
    # routed_scaling_factor, q_lora_rank, kv_lora_rank, etc.
)
```

**Attention dimension rules:**
- `num_attention_heads` must divide `hidden_size` evenly
- `num_key_value_heads` must divide `num_attention_heads` evenly
- For MLA models: use small `q_lora_rank` (16), `kv_lora_rank` (16),
  `qk_rope_head_dim` (8), `qk_nope_head_dim` (8), `v_head_dim` (16)
- For DSA models: use small `index_n_heads` (2), `index_head_dim` (8),
  `index_topk` (8)

**MoE layer selection rules:**
- If `first_k_dense_replace > 0`: set it to 0 (force first layer to be MoE)
- If model uses `decoder_sparse_step`: set to 1
- If model uses `mlp_layer_types`: set to `["sparse"]`

**Expert group routing:**
- If `n_group` is set, `num_local_experts` must be divisible by `n_group`
  and `num_local_experts / n_group >= 2`

## Phase 3: Run Numerical Equivalence

**HF is the gold standard.** The numerical equivalence test compares
Titan's MoE output against HF's output. The test must run HF's
forward pass unmodified — do NOT patch, monkey-patch, or alter HF's
computation to make the test pass. The only permitted modifications to
close accuracy gaps are **temporary changes to titan core code**, which
must be reverted after verification.

Specifically:
- Do NOT patch the HF model's activation function to match titan's
- Do NOT replace HF's routing with titan's routing
- Do NOT modify the test's comparison logic or thresholds
- Do NOT skip or disable parts of the comparison
- Do NOT cast HF tensors to different dtypes to hide precision diffs
  (the existing float32 gate cast is an exception — it aligns titan's
  autocast behavior, not the other way around)
- If titan produces different results from HF, the gap analysis
  (Phase 3c) must identify WHY by temporarily changing titan to match
  HF, not by changing HF to match titan

**Do NOT modify the pass/warn/fail thresholds.** The thresholds
(KL < 1e-6 PASS, KL < 1e-3 WARN, KL >= 1e-3 FAIL) are calibrated for
the known dispatcher precision gap. If a new model hits WARN, verify
via Phase 3c that the diff is from known causes. If it hits FAIL, fix
the integration, do not loosen the thresholds.

### 3a. Synthetic test

```bash
CUDA_VISIBLE_DEVICES=<gpu> python \
    torchtitan/experiments/transformers_modeling_backend/.claude/skills/add_moe_model/scripts/numerical_equivalence.py \
    --models <model_type>
```

**Expected:** PASS with KL < 1e-6 and round-trip max_diff = 0.00.

If it fails, diagnose:
- `ValueError` from `_probe_hf_moe_block` — missing config attribute,
  fix the synthetic config
- Shape mismatch in `load_state_dict` — state dict adapter needs update
- `RuntimeError` from `grouped_mm` — check expert weight shapes
- Numerical FAIL — investigate routing or weight transfer issue

### 3b. Pretrained test (if weights available)

```bash
CUDA_VISIBLE_DEVICES=<gpu> python \
    torchtitan/experiments/transformers_modeling_backend/.claude/skills/add_moe_model/scripts/numerical_equivalence_pretrained.py \
    --model_dir <path>
```

**Expected:** PASS with KL < 1e-3 (known dispatcher precision gap).

### 3c. Verify numerical gaps via temporary core alignment (MANDATORY)

**This step is NOT optional.** Execute it fully and automatically whenever
any test shows WARN or higher. Do NOT ask the user before proceeding —
apply patches, run tests, record results, report, revert. The entire
sequence runs without user interaction.

The goal: bring max_diff to 0.00 by temporarily making titan's
implementation match HF's exactly. Every remaining diff must be
experimentally explained — never claim a root cause without running
code that proves it.

**Rules:**

- **Never give up on a patch because it crashes.** If a patch causes an
  error (e.g. dtype mismatch in scatter), fix the error by extending the
  patch. The patch is temporary — make it work, don't declare it
  "not applicable."
- **Never claim a root cause without verification.** If you think "the
  diff is from bmm vs grouped_mm", prove it: make both sides use the
  same kernel and show the diff drops. If you can't change one side
  (e.g. HF rejects grouped_mm), change the other side (e.g. make titan
  use bmm temporarily, or manually compute the expert forward with the
  same kernel as HF using titan's weights).
- **Adapt patches to the model's code path.** The known patches target
  specific code paths (e.g. f32 accumulation in
  `LocalTokenDispatcher.combine()`). If the model uses a different path,
  write an equivalent patch for that path. The principle is the same --
  align titan's operations with HF's.
- **Keep going until max_diff=0.00.** If the known patches don't get
  there, trace the forward step by step to find additional differences.

**Procedure:**

1. **Record baseline** — note current max_diff and KL before any patches.

2. **Apply patches incrementally** — apply each applicable patch one at
   a time. After each, re-run the test and record the new max_diff.
   If a patch crashes, fix the crash (extend the patch as needed) and
   re-run. If a patch has no effect (e.g. top_k=1 makes reshape+sum
   trivial), record "no effect (top_k=1)" and move on.

3. **Handle kernel mismatches** — if the model doesn't support
   `grouped_mm` on the HF side, the test can't force the same kernel.
   In that case, verify experimentally:
   - Extract titan's expert weights after weight transfer
   - Run the expert forward manually using HF's kernel (e.g. `torch.bmm`)
     with titan's weights and the same input
   - Compare that output against HF's expert output
   - If they match (0.00), the remaining diff is confirmed as kernel
     difference, not a bug
   - Record this in the report as a verified kernel difference

4. **Investigate remaining diffs** — if max_diff is still non-zero after
   all patches and kernel alignment, trace the forward path step by step:
   router logits → routing weights → per-expert gate/up/down → score
   application → combine. Compare intermediate values between HF and
   titan to find where they first diverge. Make additional minimal
   patches to test hypotheses — if a patch doesn't help, revert it
   and keep digging.

5. **Report** — present a table to the user showing each patch and its
   measured impact:

```
Numerical gap analysis for <model_name>
| Patch | max_diff | KL | Description |
|-------|----------|----|-------------|
| baseline | X.XXe-XX | X.XXe-XX | no patches |
| +patch 1 | X.XXe-XX | X.XXe-XX | f32 accumulation in combine |
| +patch 2 | X.XXe-XX | X.XXe-XX | reshape+sum (no effect, top_k=1) |
| +kernel align | X.XXe-XX | X.XXe-XX | titan experts via bmm |
| final | 0.00e+00 | X.XXe-XX | all diffs explained |
```

6. **Revert ALL patches** — restore all core files:
   ```bash
   git checkout torchtitan/models/common/token_dispatcher.py torchtitan/models/common/moe.py
   ```
   Verify with `git diff` that the revert is clean. The temporary patches
   are NOT part of the commit.

**Known sources of numerical difference:**

Do NOT copy-paste patches from this document — the titan core code may
have changed since these were written. Instead, read the current code in
`torchtitan/models/common/token_dispatcher.py` and
`torchtitan/models/common/moe.py`, understand the principle behind each
fix, and write a patch that applies cleanly to the current code.

The three known differences and how to fix them:

**1. bf16 accumulation before scatter_add** —
`LocalTokenDispatcher.combine()` multiplies expert output by routing
scores in f32 but casts back to bf16 before `deterministic_scatter_add`,
so the accumulation happens in bf16. **Fix principle:** remove the cast
to bf16, keep everything in f32 through the scatter_add, allocate an
f32 output buffer, cast to the input dtype only at the very end.

Example (illustrative, may not match current code):
```python
# principle: remove .to(routed_output.dtype), use f32 buffer
routed_output = routed_output.to(torch.float32) * scores
out = scatter_add(f32_buffer, ..., routed_output)
return out.to(x.dtype)
```

**2. scatter_add vs reshape+sum accumulation order** —
`scatter_add` accumulates top_k contributions in expert-sorted order.
HF uses `reshape(N, K, D).sum(dim=1)` which accumulates in token order
(consecutive top_k entries per token). Different f32 summation order
produces different results. **Fix principle:** store the argsort
permutation from dispatch, in combine unsort back to original token
order, then reshape and sum over the top_k dimension.
For `top_k=1` this has no effect (only one element to sum).

**3. topk sorted flag** — titan uses `torch.topk(sorted=False)`, HF
defaults to `sorted=True`. With `sorted=True` topk values come back in
descending order; the normalization sum is computed over differently
ordered values, producing a small f32 rounding difference. **Fix
principle:** change `sorted=False` to `sorted=True` in the router's
topk call.
For `top_k=1` this has no effect (only one element).

With all three fixed, **all existing models produce max_diff=0.00.**
If the new model still shows non-zero diff after all applicable fixes,
the remaining difference is model-specific. Investigate as described
in step 4 above.


## Phase 4: Parallelism Integration Tests

Run the new model through the same parallelism configurations used for
existing models. The integration test infrastructure is in
`torchtitan/experiments/transformers_modeling_backend/tests/integration_tests.py`.

### 4a. Determine available GPUs

```bash
nvidia-smi -L | wc -l
```

### 4b. Run parallelism configs

Test the new model with at least these configurations (adapt GPU counts
to what is available). Use `--training.steps 2` to keep runs short — the
goal is to verify the model initializes, parallelizes, and runs
forward/backward without errors, not to converge.

```bash
# FSDP only (minimum viable test)
torchrun --nproc-per-node <NGPU> --module transformers_modeling_backend \
    --config transformers_modeling_backend_debugmodel_moe \
    --hf_model <model_id> \
    --training.steps 2

# FSDP + EP
torchrun --nproc-per-node <NGPU> --module transformers_modeling_backend \
    --config transformers_modeling_backend_debugmodel_moe \
    --hf_model <model_id> \
    --parallelism.data_parallel_shard_degree -1 \
    --parallelism.expert_parallel_degree <EP> \
    --training.steps 2

# FSDP + TP + EP (if NGPU >= 4)
torchrun --nproc-per-node <NGPU> --module transformers_modeling_backend \
    --config transformers_modeling_backend_debugmodel_moe \
    --hf_model <model_id> \
    --parallelism.data_parallel_shard_degree -1 \
    --parallelism.tensor_parallel_degree 2 \
    --parallelism.expert_parallel_degree <EP> \
    --training.steps 2
```

**Every config must be run.** There are only three — run all of them.
Do NOT skip, do NOT mark as "NOT TESTED", do NOT mark as "SKIPPED".
Every config in the final report must show either PASS or FAIL with a
diagnosed error message and root cause.

If a config fails:
1. Read the error traceback
2. Diagnose the root cause
3. If fixable in experiment code, fix it and re-run
4. If it's a fundamental limitation (e.g. DTensor mixing), report FAIL
   with the specific error and why it can't be fixed in experiment code

**Redundant compute is a blocker.** If the only way to make TP work for
a module is to run the full computation redundantly on every TP rank
(e.g. all-gather input, run full forward, slice output), that is NOT
an acceptable fix — it defeats the purpose of TP. If a model component
(e.g. a custom attention mechanism) cannot be properly sharded, report
it as a FAIL with the reason "no proper TP sharding available, would
require redundant compute". Do NOT implement redundant-compute
workarounds with pre/post hooks that all-gather and re-shard.

**Use ShardingConfig, not hooks.** The HF backend sets
``_sharding_config`` on every sub-module so ``model.parallelize()``
handles all DTensor distribution and forward wrapping. Available
helpers from ``torchtitan/models/common/decoder_sharding.py`` and
``torchtitan/experiments/transformers_modeling_backend/hf_sharding.py``:

- ``colwise_config()`` — for linear projections (weight Shard(0),
  output Shard(-1))
- ``rowwise_config(output_sp=True)`` — for output projections (weight
  Shard(1), output Shard(1) for SP)
- ``_hf_norm_config(enable_sp=True)`` — for norms (weight Replicate,
  activations Shard(1) when SP enabled)
- ``_replicate_config(module)`` — for modules that should have
  Replicate params/buffers without TP sharding (dynamically enumerates
  params and buffers to avoid ``_shard_states`` errors)
- ``ShardingConfig(in_src_shardings=..., in_dst_shardings=...)`` —
  for input redistribution at module boundaries (e.g. attention,
  MLP gather from Shard(1) to Replicate)

Do NOT use ``register_forward_pre_hook`` / ``register_forward_hook``
to manually convert DTensor ↔ local tensors. Do NOT use
``parallelize_module`` from ``torch.distributed.tensor.parallel``.
Use ``_sharding_config`` exclusively.

The `--hf_model <model_id>` flag overrides the default model in the
base config. If the base config has incompatible defaults for the new
model (e.g. wrong tokenizer path), override those too with additional
`--` flags. Do NOT give up because "the config is incompatible" — fix
the incompatibility with overrides.

Common failure modes and fixes:
- TP failures from unsupported attention projections — fix in
  `hf_sharding.py` by adding ShardingConfig in `_set_layer_sharding_configs`
- EP failures from expert weight sharding — check FSDP `shard_placement_fn`
- Shape mismatches from model-specific layer structure — fix in experiment code
- Config incompatibility — add more `--override.flags` to the command
- Tokenizer mismatch — override `--hf_assets_path` to point to the model
- Custom attention with no standard Q/K/V/O projections — FAIL, not fixable
  without redundant compute
- Mixed DTensor / plain Tensor — a module has buffers not declared in its
  ShardingConfig's ``state_shardings``. Fix by using ``_replicate_config(module)``
  which dynamically enumerates all params and buffers

**Note:** If no GPUs are available, skip this phase and note it in the
report. But if GPUs are available, all configs must be run and reported.

## Phase 5: Update Model Compatibility Doc

Update `torchtitan/experiments/transformers_modeling_backend/MODEL_COMPATIBILITY.md`
with the new model:

1. Add a row to the **Numerical Equivalence Summary** table with the
   model's status, KL, max_diff, cos_sim, and round-trip result.

2. Add a section under **Architectural Differences per Model** describing:
   - Router type and features (softmax/sigmoid, normalization, bias, scaling)
   - Expert format (standard/transposed, activation function)
   - Shared experts (none/additive/sigmoid-gated/layer-level)
   - Attention type (GQA/MLA/DSA/custom)
   - Any unsupported features and their impact
   - Summary of differences from Titan's MoE

3. Add a row to the **Parallelism Support** table (FSDP and FSDP+EP).

4. If the model has unsupported features requiring core changes, add
   or update entries under **Core Changes Needed for Full Support**.

## Phase 6: Lint and Commit

1. Run pre-commit on all changed files:
   ```bash
   SKIP=pyrefly-check pre-commit run --files <changed_files>
   ```
   Fix any failures and re-run until clean.

2. Verify no core files were modified:
   ```bash
   git diff --stat  # only files in experiments/transformers_modeling_backend/
   ```

3. Create commit:
   ```
   Add <ModelName> to transformers MoE backend

   Adds synthetic test config with production-scale dimensions
   (H=<hidden_size>, I=<moe_intermediate_size>, E=8). Verified via
   numerical equivalence test (KL=<value>, max_diff=<value>).
   ```

## Issue Resolution Reference

Most issues are fixable in the experiment folder. Only `nn.ModuleList`
experts is a true blocker requiring upstream changes.

| Symptom | Where to fix | What to do |
|---------|-------------|------------|
| Expert weights transposed `(E, H, 2*I)` | `state_dict_adapter.py` | Add transpose in conversion; fix shape detection in `moe_replacement.py` |
| `experts` is `nn.ModuleList` | **Cannot fix** — needs HF upstream | Report to user, stop |
| Sigmoid scoring not auto-detected | `moe_replacement.py` | Add detection path in `_resolve_score_func()` (e.g. check router source for `sigmoid`) |
| Router key `router.weight` not converted | `state_dict_adapter.py` | Add regex pattern in `_build_hf_to_titan_patterns()` |
| Shared expert round-trip broken | `state_dict_adapter.py` | Fix reverse patterns in `_build_titan_to_hf_patterns()` |
| MoE layer attr is `feed_forward` not `mlp` | `model.py`, `moe_replacement.py` | Handle alternative attribute name |
| Attention projection not in ShardingConfig | `hf_sharding.py` | Add case to `_set_layer_sharding_configs` using `colwise_config()`/`_replicate_config()` |
| Model rejects `grouped_mm` | `scripts/numerical_equivalence.py` | Skip `_experts_implementation` for this model |
| `trust_remote_code` import error | Config loading | Try without `trust_remote_code` first |
| `num_hidden_layers=1` has no MoE layer | Test config | Set `first_k_dense_replace=0` or increase layers |
| State dict keys missing after round-trip | `state_dict_adapter.py` | Add patterns in `_build_hf_to_titan_patterns()` / `_build_titan_to_hf_patterns()` |
| Layer-level MoE (router/experts are siblings of dense MLP) | `moe_replacement.py` | Treat dense MLP as shared expert inside the Titan MoE; replace `layer.mlp` with full MoE block containing router + experts + shared_experts(=dense MLP) |
| TP fails with mixed DTensor/Tensor | `hf_sharding.py` | Module has undeclared buffers — use `_replicate_config(module)` which dynamically enumerates all params and buffers |
| Embedding has extra buffers (e.g. `embed_scale`) | `hf_sharding.py` | Already handled — `set_hf_sharding_configs` enumerates embedding buffers dynamically |

## Files touched (experiment folder only)

| File | Change |
|------|--------|
| `scripts/numerical_equivalence.py` | Add entry to `_MODEL_CONFIGS` |
| `state_dict_adapter.py` | Add key patterns (only if round-trip fails) |
| `hf_sharding.py` | Add ShardingConfig entries (only if new projection/norm names) |
| `moe_replacement.py` | Add probing logic (only if new MoE pattern) |
