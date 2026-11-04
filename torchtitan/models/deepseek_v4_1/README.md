# DeepSeek V4.1 Flash

The text model implements teacher-forced training through the complete causal
encoder and decoder. It reuses TorchTitan's shared MoE, RoPE, normalization,
expert dispatch, activation rematerialization, and FSDP lifecycle.

## Architecture

The released model has 40 blocks and an encoder boundary after block 20.
All block indices below are zero based.

- Blocks 0 and 1 use sliding-window attention only.
- Blocks 2 through 39 combine a 128-token local window with indexed global keys.
- Blocks 2, 8, 14, and 20 project a new global KV pool. Encoder pools compress
  two tokens per entry; the decoder pool retains one entry per token.
- Reindex blocks update selections; Reuse blocks keep the previous selections.
- Each block computes two sets of single-pass mHC coefficients over four
  residual streams. The incoming read-in coefficients feed the current sublayer.
- Engram uses causal 2-, 3-, and 4-gram hashes with gated phrase embeddings.

```text
tokens -> embeddings -> four residual streams -> 40 blocks -> norm -> logits
                            |                       ^
                            +-> Engram hashes        |
Full producer -> global KV/index keys -> Reindex/Reuse consumers
```

The debug flavor retains all attention modes, the encoder boundary, three KV
producers, a decoder Reindex block, two Engram layers, and MoE routing.

## Configuration

```python
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.deepseek_v4_1 import build_model_config

config = build_model_config(
    "debugmodel", parallelism=ParallelismConfig(), seq_len=128,
)
```

This increment supports FSDP with TP=EP=PP=1 and CP=1. The standard CUDA recipe
uses the public attention-gym fused kernels. The explicit `reference` backend
uses eager PyTorch attention for independent CPU comparisons. Training runs the
full graph; optimized serving prefill is not implemented here.

The adapter maps released HF text checkpoint keys and reads block-scaled FP8
and packed FP4 tensors through slice-aware DCP loading. The manifest check does
not download the released parameter values.

## Images

Set `vision_enabled=True` to include DeepSeek ViT, the padded 3x3 unshuffle
aligner, and image marker embeddings. The released flavor includes them by
default; the debug flavor uses a smaller encoder.

```text
flattened patches -> ViT with two-axis RoPE -> 3x3 unshuffle -> aligner
                                                                |
packed tokens -> token embeddings + image markers + visual tokens -> backbone
```

Preprocessing accepts `pixel_values`, `grid_thw` with time=1, and `token_types`.
Types -1, 0, 1, 2, and 3 denote text, image start, aligned visual token,
newline, and image end. Visual positions receive no Engram residual update and
use their own expert-balancing counters. Text-only ranks execute a dummy visual
dependency so FSDP can reduce the visual parameters when another rank has images.
The image-balancing hook reduces all layers' counts together and is registered
only for vision-enabled models. Distributed HF conversion keeps both stacked
ViT projections on each rank and gathers the three marker rows before splitting
them into the released checkpoint names.

## Validation

```bash
pytest tests/unit_tests/cpu/models/deepseek_v4_1
python -m scripts.checkpoint_conversion.numerical_tests_deepseek_v4_1 \
  --reference-dir=/path/to/pinned/inference \
  --output-json=/tmp/deepseek-v4-1-reference.json
```

See [VALIDATION.md](VALIDATION.md) for the reference revision and current evidence.
