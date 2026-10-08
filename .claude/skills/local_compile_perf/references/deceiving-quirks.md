# Results that change when the setup changes

Each row fooled us or nearly did. Before claiming a win, check the rows that apply.

| Dimension | What flipped | Check |
|---|---|---|
| TP=1 harness vs a TP>1 recipe | a one-GPU harness at TP=1 is only a proxy when the recipe runs TP>1 (GPT-OSS RL: TP=2, EP=2) | say so in the numbers; EP-only ops: harness.md (EP work on one GPU) |
| TP=1 vs TP>1 | folding the residual add into the MoE combine looked free at TP=1 but is wrong under TP (the output is partial before its all-reduce) | the sharding of every tensor a rewrite touches |
| TP=1 vs TP>1 | a per-row score scale before w2 is exact only because routed w2 isn't sharded across the row | same |
| World 1 vs EP>1 | zero-token ranks exist only at EP>1: HybridEP's combine backward skipped probs there, so router grads went missing silently | a real multi-GPU run for cross-rank paths |
| World 1 vs EP>1 | dense top-k input and a fused permute were neutral or slower at world 1; their possible wins appear only at scale | measure at the target scale before rejecting for good |
| H100 vs GB300 | masked-load reductions ~2x slower on GB300; FMA changed bits on GB300 (an upstream CosSinRoPE test failed only there); bandwidth ~7.1 vs ~3.35 TB/s; attention backend rankings (cuDNN vs FA4) differ | label hardware; don't transfer rankings |
| Config default vs recipe flags | blocking dispatch (the HybridEP default) had a wrong fake and the zero-row bug while every test recipe used non-blocking; a win reached only test recipes because no production recipe ran HybridEP | which configs actually reach the change |
| Activation checkpointing | no-AC wins shrink or flip under SAC (regions replayed, ~20 µs CPU per eager op) | measure with the recipe's AC mode (gating-and-options.md) |
| Eager vs compiled vs GraphTrainer | a rewrite's sign can differ per mode and per model | negative-results.md; measure each mode |
| Missing optional deps | DSv4's venv lacked FA4 and had attn-gym 0.0.13 (0.0.16 required): `gather_attn` silently fell back to Triton and attention took ~115 of 180 ms | check attention kernel names in the profile; pin optional deps. Also: varlen attention on Blackwell needs FA4 |
| Attention backend | a head-major q / o-grad layout was -3.8% with Triton attention and neutral with FA4; deterministic mode switches backends | measure layout changes with the production backend and mode |
| Small shapes | host-bound: kernel time and wall time disagree | report both |
| Harness | detached inputs, accumulated grads, unseeded buffers, kernels merged by name, order effects, shared GPU, timing a harness copy | measurement.md (traps) |
| Cold compile and graph budget | a ~2% layer win cost ~3 min of cold compile per start; the limit was often blown only by the first `no_grad` pass or an untested shape | report the trade-off; count all shapes |
