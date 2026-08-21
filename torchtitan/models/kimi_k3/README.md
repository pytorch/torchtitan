# Kimi K3 (KDA + MLA + MoE + Block Attention Residuals)

Torchtitan implementation of the **Kimi K3 architecture family**: the
[Kimi-Linear](https://arxiv.org/pdf/2510.26692) backbone (Kimi Delta
Attention + MLA + sigmoid-gated MoE) with **Block Attention Residuals**
([arXiv:2603.15031](https://arxiv.org/abs/2603.15031)) woven in.
[Kimi K3](https://www.kimi.com/blog/kimi-k3) (2026-07-16) confirmed
AttnRes + KDA as production architecture components; open weights and the
tech report are due 2026-07-27, and this experiment's configs will be
aligned to the official release (structure details currently pending hold
placeholder interfaces).

> **Status (2026-07-18).** RFC
> [pytorch/torchtitan#3029](https://github.com/pytorch/torchtitan/issues/3029)
> was gated by reviewers on the Kimi K3 release -- that gate is now met. A
> follow-up RFC proposing this experiment is in preparation.

## What's in this folder

| File | Role |
| --- | --- |
| [`model.py`](./model.py) | K3 backbone: `KimiDeltaAttention` (KDA via `fla-core`), `KimiMLAAttention`, `KimiMoE`, `KimiDecoderLayer`, `KimiK3Model` |
| [`attn_res_model.py`](./attn_res_model.py) | `KimiK3AttnResModel`: AttnRes weave over the backbone (per-block-start RMSNorm + zero-init pseudo-queries) |
| [`attn_res.py`](./attn_res.py) | `block_attn_res()` primitive, `AttnResConfig`, `AttnResProjection`, `stack_blocks` / `unstack_blocks` |
| [`multimodal_model.py`](./multimodal_model.py) | `KimiK3LlavaMultimodalModel` + `KimiVisionProjector` (SigLIP-splice scaffold for the vision-native path) |
| [`parallelize.py`](./parallelize.py) | `parallelize_kimi_k3`: FSDP2/HSDP + TP + EP (CP blocked on fla-core `chunk_kda`) |
| [`pipeline_adapter.py`](./pipeline_adapter.py) | Cross-stage caching adapter + `pipelining_fn` (Interleaved1F1B), private to this experiment. Opt-in via `TORCHTITAN_ATTNRES_CACHE=1`. |
| [`layout.py`](./layout.py) | Static block-delta layout tables consumed by the PP adapter |
| [`model_configs.py`](./model_configs.py) | Architecture-side builders: AttnRes tech-report Table 2 scaling-law table (194m..528m), the SGLang-aligned 447m carrier, the 48B-A3B layout, `build_kimi_linear_config` |
| [`config_registry.py`](./config_registry.py) | Trainer configs for every `kimi_linear_<size>_<variant>` flavor (variants: baseline / block_attn_res / full_attn_res; + fp8 rowwise) |
| [`__init__.py`](./__init__.py) | `model_registry` -> `ModelSpec` (fla-core guarded) |
| [`tests/`](./tests/) | CPU unit tests: AttnRes primitive, KDA/MLA/MoE layers, AttnRes model, multimodal splice, pipeline-adapter wiring, all-flavor registry sweep |

## Running

```bash
# Unit tests (CPU; KDA falls back to fla-core's CPU path)
pytest torchtitan/models/kimi_k3/tests/ -v

# Single-node FSDP, 447M carrier
bash run_train.sh --module kimi_k3 --config kimi_linear_447m_aligned_block_attn_res_n4     --training.steps 100

# PP with the cross-stage cache adapter
TORCHTITAN_ATTNRES_CACHE=1 torchrun --nproc_per_node=4 ...     --module kimi_k3 --config kimi_linear_436m_block_attn_res     --parallelism.pipeline_parallel_degree 4     --parallelism.pipeline_parallel_schedule Interleaved1F1B
```

Dependencies: `pip install fla-core` (KDA kernels; CPU fallback exists for
tests, training needs the triton path).

## Design notes

- **Zero-init pseudo-queries.** AttnRes projections are zero-initialized so
  softmax weights are uniform at step 0 and the model is numerically
  equivalent to standard residuals on the first forward -- also the anchor
  for grafting AttnRes onto the released Kimi-Linear-48B checkpoint.
- **PP cross-stage cache adapter.** Producer stages publish each committed
  block once; consumers on the same rank read it back through a
  detached-leaf cache + gradient bridge, so backward through cached
  tensors does not double-accumulate into the producer. Delta mode sends
  only newly committed blocks.
- **Context parallelism is per layer kind, and both kinds run together.**
  The KDA layers use KCP (report sec 5.1.2): the sequence stays sharded end
  to end via a prefix scan over state fragments, plus a fixed-size halo for
  the short convolutions. fla-core >= 0.5.1 provides both
  (`chunk_kda(cp_context=...)`, `causal_conv1d_cp`). The MLA layers use
  Ulysses head sharding, which is unrelated -- KCP decomposes the delta-rule
  recurrence and says nothing about softmax attention -- so a CP run is KCP
  on the KDA layers *and* Ulysses on the MLA layers simultaneously.
  `kda_cp_mode` selects the KDA side and defaults to `"kcp"`; `"ulysses"`
  there is kept as an A/B, and is not what K3 does: it gives every rank the
  whole sequence for its head subset, so activation memory does not fall
  with `cp` and the context lengths K3 targets are out of reach.
  KCP's varlen path takes no batch axis (fla asserts `[1, T, D]`), so the
  batch is looped -- flattening it into one packed sequence would not match,
  because fla cuts the *global* packed sequence into contiguous rank-ordered
  pieces while a rank holds piece `r` of every sequence.
- **TP and CP interact through the head count, and the KDA layers pay for
  it.** KDA is `NoParallel` under TP (replicated), so its attention compute
  is duplicated across the TP axis: at the report's 3:1 KDA:MLA ratio, three
  quarters of the attention layers compute redundantly at `tp > 1`. TP is
  there for MLA and the MoE; KDA scales on CP. On the MLA side both axes cut
  the same heads, so `num_attention_heads % (tp * cp) == 0` is enforced, and
  the quotient is also a performance floor -- 96 heads at `tp=8, cp=4` leaves
  3 heads a rank, where the all-to-all payload and SDPA's own head
  parallelism both get thin. Prefer spending the budget on `cp` over `tp`
  once that quotient drops into the single digits.
- **Expert parallelism is the standard all-to-all, not MoonEP.** Report sec
  5.2.1 describes a balanced EP implementation that is not reproduced here;
  what this folder has is torchtitan's `ExpertParallel` on the routed-expert
  container, i.e. dispatch and combine all-to-alls with no load-balancing
  transport of its own. See
  [MoonshotAI/MoonEP](https://github.com/MoonshotAI/MoonEP). Note that at
  896 experts with top-16 the report itself (sec 2.3) puts the sparsity
  beyond where fixed-step auxiliary-loss-free bias updates are known to
  behave, so this is not only a throughput gap. `quantile_balance.py`
  addresses the *router* half of that (sec 2.3.3: it solves for the bias
  instead of nudging it, removing the step size); it does not make the
  transport balanced.
- **`V=1` is a supported PP mode, not a degradation.** With one virtual
  stage per rank the adapter runs the naive chain relay, and that is the
  bandwidth lower bound rather than a fallback: with no second virtual stage
  on the rank there is no cached prefix to diff against, so every hop must
  carry the blocks the next stage reads. Delta mode needs `V >= 2` to have
  anything to omit.

## Evidence

Development history, pretraining runs, and PP pressure tests live in the
companion logbook repo
[QIU023/torchtitan_attention_residual](https://github.com/QIU023/torchtitan_attention_residual):

- **PP adapter numerics**: naive-vs-adapter |dLoss| <= 0.011 across PPxVP
  shapes up to PP=8 x VP=4 (32 virtual stages), incl. a 48B-layout
  carrier -- [pressure-test report](https://github.com/QIU023/torchtitan_attention_residual/blob/main/phase3_attnres_pp_integration/PRESSURE_TEST_REPORT_2026-05-12.md).
- **12.5K-step pretraining** on the 436M/447M shapes --
  [phase-4 log](https://github.com/QIU023/torchtitan_attention_residual/blob/main/phase4_kimi_attnres_lm_pretrain/README.md).
- **Dense A/B + adapter test grid**: the Llama3-shape/DSv3-shape AttnRes
  test carrier (paper Table 1 reproduction; 1460-line PP adapter test
  grid) was developed here and now lives at
  [phase3 `dense_carrier/`](https://github.com/QIU023/torchtitan_attention_residual/tree/main/phase3_attnres_pp_integration/dense_carrier)
  (runnable against fork history <= `666cf7ad6`).
- **HF reference blueprint** (`modeling_kimi.py`, for correctness diffs):
  [phase4 `hf_reference/`](https://github.com/QIU023/torchtitan_attention_residual/tree/main/phase4_kimi_attnres_lm_pretrain/hf_reference).

## Ownership

- Owner: [@QIU023](https://github.com/QIU023) -- open issues on the fork
  repo for technical questions.
