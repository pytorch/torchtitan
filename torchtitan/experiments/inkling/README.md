# Inkling 2 Mini: native single-Spark training

This is the **text target** of `thinkingmachines/Inkling-2-Mini`. TorchTitan
owns the full-sequence forward and backward. It does not call attention-gym
during training. The middle 12 of the 24 layers run three times with shared
parameters, for 48 layer visits. Each later pass receives gated messages from
the previous one. Attention includes sliding/full masks, relative logits,
sinks, and short convolutions. CUDA uses FlexAttention with grouped-query heads.
It applies the learned sink using the softmax log-sum-exp; this preserves its
gradient but changes BF16 rounding relative to eager attention. A relative-bias
tensor is still materialized, even though the attention probabilities are not.

The single-device recipe adapts **`layers.*.attn.wq_du`**, the internal query
projections, not the language head. The optimizer updates FP32 adapter masters;
the training forward uses BF16 projections with reduced-precision BF16 GEMM
reductions disabled. The backbone stays in its published
formats. On CUDA, tokens are grouped by expert; each projection decodes packed
weight tiles inside the grouped matrix multiply. The backward pass computes
input gradients through those frozen weights using a straight-through
estimator for activation quantization. The unchanged vision tower and
speculative drafter are not training components.

## Train

Run from the TorchTitan checkout with its dependencies installed and the
published checkpoint downloaded (including `hf_quant_config.json`, tokenizer
assets, and the weights). To download the tested snapshot:

```bash
.venv/bin/hf download thinkingmachines/Inkling-2-Mini --revision c9d028302cd0f035fdf8e7e91f089a4ce94d2195
```

Create an importable recipe, for example `inkling_run.py` in the checkout:

```python
from pathlib import Path

from torchtitan_recipes.models.inkling import inkling_2_mini_lora


def spark():
    checkpoint = (
        Path.home() / ".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini"
        / "snapshots/c9d028302cd0f035fdf8e7e91f089a4ce94d2195"
    )
    return inkling_2_mini_lora(
        checkpoint=checkpoint, text=Path("train.txt"), seq_len=8, steps=10
    )
```

```bash
run="$HOME/checkpoints/inkling-query-lora"
.venv/bin/torchrun --standalone --nproc-per-node=1 \
  --module torchtitan.train --module inkling_run --config spark --output-dir "$run"
```

Supply a UTF-8 corpus with one document per line. The example uses short
contexts to exercise training on a single GB10, not to establish convergence
or long-context capacity. The recipe uses the standard Trainer, Grain data
loader, next-token cross entropy, optimizer, and distributed checkpointing.
Native checkpoints live in `$run/checkpoint`, separate from the source snapshot.
They save the adapter and training state, not a second copy of the frozen model. Keep the source checkpoint for resume and allow about
19 GiB of disk space per merged HF export. The trainer records the original
snapshot and corpus contents and the sequence length in `run-manifest.json`.
It verifies them before loading a native checkpoint and again before export.
To continue training, use the same output directory, snapshot, corpus, and
sequence length, and increase `steps` in the recipe. A step-qualified export
leaves the previous result intact.

## Use the trained checkpoint in WarpDrive

The export folds the internal query updates into their BF16 tensors. Packed
expert payloads, the language head, tokenizer, and drafter retain the source
checkpoint format. Use the **merged directory**, not the original Hub ID:

```bash
cd ../attention-gym
.venv/bin/python -m attn_gym.warpdrive.examples.inkling_megakernel.chat \
  --model "$run/merged-step-10" --backend megakernel-spec --verify-tokens 4 \
  --prompt 'What is the capital of France?' --reasoning-effort low --max-new-tokens 64
```

Training does not require an attention-gym source checkout. For the WarpDrive
example use its own environment and a revision with GB10 Inkling support
(e.g. `5c2e8d594dafaff6d65541f5a02a923df5b532e1`). Internal LoRA can change
the drafter's acceptance rate. Full-parameter training, visual inputs, and
distributed model parallelism are not supported by this recipe.
