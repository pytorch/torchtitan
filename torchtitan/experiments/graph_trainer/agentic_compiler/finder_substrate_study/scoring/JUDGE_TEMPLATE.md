# Task: adjudicate kernel-fusion opportunity cards for one training step

Several independent analyses each listed kernel-fusion opportunities in the same {MODEL_NAME} training
step. Their findings are pooled as cards in `{BOX}/cards.json`, in random order with random IDs. Your job is
to turn the cards into one adjudicated list of opportunities, decide which are real, and estimate what each
is worth. Don't try to work out which analysis a card came from; judge each card on its content.

## What you have

- `{BOX}/cards.json`: the cards. Each has a title, a description, phases, a footprint (the op instances it
  covers: phase, module path with layer indices as `*`, layers, op names), shapes, instance counts,
  measured and ideal times, the claimed gap, a bitwise tier, the dataflow argument and blockers. Text was
  normalized: `<value>` stands for a tensor name, `<program>` / `<profile>` / `<label>` / `<callback>` for
  tool-specific words.
- `{BOX}/overlaps.json`: card pairs whose footprints share (phase, module, op) keys; a hint for grouping,
  neither complete nor authoritative.
- `{BOX}/hw_profile.json`: the roofline numbers every card was meant to use. Ideal time =
  `max(bytes / hbm_copy_gbps, flops / GEMM peak of the matching precision)`.
- **The program:** {PROGRAM} No activation recompute, deterministic algorithms on, one process with a fake
  process group (collectives are emulated by local copies and are not fusion targets themselves). The
  forward and backward are in scope; the optimizer step is not.
- The repo snapshot `{REPO}` (read-only), to check what the model code computes when a card's dataflow
  claim needs verifying.

## Rules

- Use only `{BOX}` and `{REPO}`. Don't read anything else on this machine, don't use the network, don't use
  skills, don't spawn subagents, don't use a GPU.
- Write your outputs under `{BOX}/out/`.
- Run `date` first. Have everything written within 90 minutes.

## Steps

1. **Group cards into opportunities.** An opportunity is a set of ops that one fused kernel (or one fused
   forward kernel plus its backward kernel) would replace. Cards that describe the same ops belong to the
   same opportunity, even if their boundaries, wording or numbers differ.
   - Make opportunities fine enough that their savings add up without double counting. If card A covers
     the ops of cards B and C together, make one opportunity for B's ops and one for C's, and let A cover
     both. If fusing across B and C saves more than the two parts (for example an intermediate tensor no
     longer goes through memory), record the extra as its own opportunity, covered only by cards that fuse
     across.
   - Every card goes into at least one opportunity.
2. **Decide whether each opportunity is real.** Real means:
   - the ops exist in this program as described and the dataflow allows the fusion (no dependency through
     an op that can't be part of the fused kernel, unless the opportunity is a GEMM or attention
     prologue/epilogue, tier T3);
   - fusing would save time: the measured time of its ops is above the ideal of the fused kernel;
   - it is in scope: forward or backward, not the optimizer step, not the collectives themselves.
   Library ownership or integration difficulty doesn't make an opportunity unreal; note it.
3. **Estimate the achievable saving** of each real opportunity in ms per step: measured time of the ops it
   replaces minus the ideal time of the fused kernel(s), from `hw_profile.json` and the cards' bytes, shapes
   and instance counts. Recompute the ideal yourself; don't take a card's claimed gap on trust. When cards
   report different measured times for the same ops, use the smaller and say so.
4. **Record coverage.** For every card in an opportunity, the fraction (0 to 1) of that opportunity's
   saving the card's proposal would capture if implemented as described: 1 if it covers all the ops, less
   if it covers part of them or proposes a fusion that keeps some of the traffic.
5. **Classify** each opportunity: bitwise tier (T0-T3, as defined on the cards), and one category from
   `norm`, `rope`, `attention-glue`, `router`, `moe-dispatch-combine`, `activation`, `loss`, `residual`,
   `quantization`, `communication-glue`, `weight-prep`, `gemm-adjacent`, `horizontal`, `other`.

## Output

`{BOX}/out/judgement.json`:

```json
{
  "opportunities": [{
    "id": "O-01",
    "title": "",
    "ops_summary": "phase / module / ops it replaces",
    "real": true,
    "not_real_reason": "",
    "tier": "T0",
    "category": "norm",
    "saving_ms": 0.0,
    "saving_basis": "how you computed it (bytes, instances, measured ms used)",
    "cards": [{"card": "C-0000", "coverage": 1.0, "note": ""}],
    "notes": ""
  }],
  "card_notes": [{"card": "C-0000", "note": "anything wrong with this card's numbers or claims"}]
}
```

`{BOX}/out/judgement.md`: a short summary: number of opportunities, real vs not real, the total real
saving, how you grouped tricky cases, and any card claims you corrected.

Reply with the path to `judgement.json`, the number of opportunities (real / not real), and the total real
saving in ms.
