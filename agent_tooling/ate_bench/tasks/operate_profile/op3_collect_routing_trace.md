# OP3: Collect Routing Trace

Instrument training to dump the **per-token MoE routing trace** for the first
**8M training tokens**. For **each MoE layer**, capture the **top-k expert IDs**
and their **gating weights** for **every token in the global batch**.

Resume training from the released HuggingFace checkpoint so the router is in its
trained, load-balanced regime — routing decisions are model-intrinsic and valid
from step 1, so no warmup is required. (The MoE routing lives in
`torchtitan/models/common/moe.py`.)

**Output schema:** one `step-<step_id:08d>.npz` file per step under
```
{{ROUTING_TRACES_DIR}}/
```
Each `.npz` must contain, for each MoE layer index `L` (0-based), two arrays of
shape `[num_tokens, top_k]` covering every token in the global batch
(`num_tokens = global_batch_size * seq_len`):
- `layer{L:02d}_expert_ids`  — int array of selected expert IDs
- `layer{L:02d}_gating_weights` — float array of the corresponding gating weights

plus a scalar array `num_experts` (the MoE expert count). The grader checks that:
- the expected number of per-step files are present,
- arrays have the correct shape for every MoE layer over every token,
- expert-ID values are within `[0, num_experts)`,
- gating weights are non-negative and sum to 1 over the top-k selected experts for
  every token.
