# DCPv1 vs DCPv2 256-GPU checkpoint producers

This experiment package compares native DCP and DCPv2 checkpoint writes for a
matched DeepSeek-V3 671B training configuration. The `dcpv2_512` package name
is retained from the earlier checkpoint panel, but the current experiment runs
on 256 H100 GPUs.

Use module `torchtitan_recipes.dcpv2_512` with one of these configurations:

| Configuration | Purpose |
|---|---|
| `dsv3_671b_save_dcp` | Native DCP producer |
| `dsv3_671b_save_dcpv2` | Matched DCPv2 producer |
| `dsv3_debug_save_dcp` | One-host native DCP save smoke test |
| `dsv3_debug_save_dcpv2` | One-host DCPv2 save smoke test |

## Producer configuration

- DeepSeek-V3 671B on 256 ranks: PP1, DP replicate 1, DP shard 256, CP1,
  TP1, and EP64.
- EP64 overlays the DP/TP mesh; it does not multiply world size. The expert
  FSDP degree is `DP shard 256 / EP64 = 4`.
- BF16 training, local batch size 4, sequence length 4096, and full activation
  checkpointing.
- Deterministic seed 42 with forced-balanced MoE routing.
- Fresh initialization: no initial checkpoint load.
- Pinned-memory asynchronous checkpointing for both backends. Native DCP
  selects `async_with_pinned_mem` explicitly; the DCPv2 manager does not expose
  an `async_mode` config field and uses its pinned-memory async implementation.

The run lasts 151 steps and writes full training state at steps 50, 100, and
150. Step 151 is an explicit BF16 model-only Hugging Face export. Downstream
full-state load or resharding experiments must name `checkpoint/step-150`
explicitly rather than selecting the latest checkpoint. Hugging Face load
experiments should use the final `checkpoint/step-151` export.

`keep_latest_k=2` preserves step 150 in both backends, although their purge
timing can leave different older checkpoint sets. The first periodic save is a
warm-up sample; steps 100 and 150 are the steady-state comparison samples.

The earlier 512-GPU load and resharding configurations are intentionally not
carried forward. They assumed a DP-shard128/TP4/EP32 source checkpoint and are
not valid consumers of this DP-shard256/TP1/EP64 producer without redesigning
each destination topology.

The one-host smoke configurations use DP shard 8 and EP2, preserving expert
FSDP degree 4. They run three steps, write a full-state checkpoint at step 2,
and finish with a BF16 model-only Hugging Face export at step 3.
