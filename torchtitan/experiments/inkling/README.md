# Inkling 2 Mini: native single-Spark training

This is the **text target** of `thinkingmachines/Inkling-2-Mini`. TorchTitan
owns the full-sequence forward and backward. It does not call attention-gym
during training. The middle 12 of the 24 layers run three times with shared
parameters, for 48 layer visits. Each later pass receives gated messages from
the previous one. Attention includes sliding/full masks, relative logits,
sinks, and short convolutions.

The single-device recipe adapts **`layers.*.attn.wq_du`**, the internal query
projections, not the language head. The backbone stays in its published
formats. The MoE path decodes only the experts used by each input and computes
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

```bash
checkpoint="$HOME/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/c9d028302cd0f035fdf8e7e91f089a4ce94d2195"
run="$HOME/checkpoints/inkling-query-lora"
.venv/bin/torchrun --standalone --nproc-per-node=1 \
  --module torchtitan_recipes.models.inkling \
  --checkpoint "$checkpoint" --text train.txt --output-dir "$run" \
  --seq-len 8 --steps 10
```

Supply a UTF-8 corpus with one document per line. The example uses short
contexts to exercise training on a single GB10, not to establish convergence
or long-context capacity. The recipe uses the standard Trainer, Grain data
loader, next-token cross entropy, optimizer, and distributed checkpointing.
The native step checkpoint saves the adapter and training state, not a second
copy of the frozen model. Keep the source checkpoint for resume and allow about
19 GiB of disk space per merged HF export. The launcher verifies the contents of the original snapshot when
resuming and before export. To continue training, use the same output directory,
snapshot and corpus, and increase the total number of steps. A step-qualified
export leaves the previous result intact.

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
