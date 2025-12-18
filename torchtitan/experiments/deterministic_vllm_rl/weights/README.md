# Weight Converter: vLLM ↔ TorchTitan

Minimal weight conversion between vLLM/HuggingFace and TorchTitan formats for Qwen3-1.7B.

## Files

- **`weight_converter.py`**: Core conversion functions
- **`test_converter.py`**: Download & test script (weight comparison)
- **`test_forward_passes.py`**: Forward pass test (logits comparison)

## Quick Start

### 1. Install dependencies

```bash
pip install torch safetensors huggingface_hub transformers
```

### 2. Run weight conversion test (downloads Qwen3-1.7B automatically)

```bash
python test_converter.py
```

This will:
1. Download Qwen3-1.7B from HuggingFace (~3.5GB)
2. Convert to TorchTitan format
3. Convert back to vLLM format (round-trip test)
4. Verify all weights match

### 3. Run forward pass test (validates conversion accuracy)

```bash
python test_forward_passes.py
```

This will:
1. Download Qwen3-1.7B (if not already cached)
2. Convert weights to TorchTitan format
3. Run forward pass on both vLLM (via transformers) and TorchTitan
4. Compare logits to verify conversion accuracy
5. Report differences and top token predictions

### 4. Use custom directories

```bash
python test_converter.py ./custom_cache ./custom_output
python test_forward_passes.py ./custom_cache ./custom_output
```

## Manual Usage

### Convert vLLM to TorchTitan

```python
from weight_converter import vllm_to_torchtitan
from safetensors.torch import save_file

# Convert
titan_weights = vllm_to_torchtitan("path/to/vllm/model")

# Save
save_file(titan_weights, "qwen3_torchtitan.safetensors")
```

### Convert TorchTitan to vLLM

```python
from weight_converter import torchtitan_to_vllm
from safetensors.torch import load_file, save_file

# Load TorchTitan weights
titan_weights = load_file("qwen3_torchtitan.safetensors")

# Convert
vllm_weights = torchtitan_to_vllm(titan_weights)

# Save
save_file(vllm_weights, "qwen3_vllm.safetensors")
```

## Command-line Interface

```bash
# vLLM → TorchTitan
python weight_converter.py vllm_to_titan <vllm_path> <output.safetensors>

# TorchTitan → vLLM
python weight_converter.py titan_to_vllm <titan_checkpoint.safetensors> <output.safetensors>
```

## Key Differences

### Weight Name Mappings

| vLLM/HuggingFace | TorchTitan |
|------------------|------------|
| `model.embed_tokens.weight` | `tok_embeddings.weight` |
| `model.layers.{N}.self_attn.q_proj.weight` | `layers.{N}.attention.wq.weight` |
| `model.layers.{N}.self_attn.k_proj.weight` | `layers.{N}.attention.wk.weight` |
| `model.layers.{N}.self_attn.v_proj.weight` | `layers.{N}.attention.wv.weight` |
| `model.layers.{N}.self_attn.o_proj.weight` | `layers.{N}.attention.wo.weight` |
| `model.layers.{N}.mlp.gate_proj.weight` | `layers.{N}.feed_forward.w1.weight` |
| `model.layers.{N}.mlp.up_proj.weight` | `layers.{N}.feed_forward.w3.weight` |
| `model.layers.{N}.mlp.down_proj.weight` | `layers.{N}.feed_forward.w2.weight` |
| `model.norm.weight` | `norm.weight` |
| `lm_head.weight` | `output.weight` |

### Notes

- Rotary embedding frequencies (`rotary_emb.inv_freq`) are computed on-the-fly in TorchTitan, so they're skipped during conversion
- Both formats support `.safetensors` and `.bin` (PyTorch) files
- Qwen3 uses q_norm/k_norm for attention normalization, which are preserved in both formats

## Model Support

Currently tested with:
- **Qwen3-1.7B** ✅

Should work with other Qwen3 models with same architecture.
