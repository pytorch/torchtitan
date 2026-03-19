"""Compare forward pass between torchtitan's Qwen3 and HF's Qwen3.

Loads the same weights, feeds the same input, compares logits.

Usage:
    CUDA_VISIBLE_DEVICES=6 python scripts/compare_forward.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Use titan-rl env's HF model, but load torchtitan's model too
# We'll compare on CPU to avoid kernel differences

MODEL_PATH = "./assets/hf/Qwen3-0.6B"

print("Loading HF model...")
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
hf_model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Tokenize a sample
messages = [
    {"role": "user", "content": "What is 2 + 3?"},
    {"role": "assistant", "content": "5"},
]
text = tokenizer.apply_chat_template(messages, tokenize=False)
input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]

print(f"Input shape: {input_ids.shape}")
print(f"Input tokens: {input_ids[0].tolist()[:20]}...")

# HF forward
with torch.no_grad():
    hf_out = hf_model(input_ids)
    hf_logits = hf_out.logits

print(f"HF logits shape: {hf_logits.shape}")
print(f"HF logits[0,0,:5]: {hf_logits[0, 0, :5].tolist()}")
print(f"HF logits[0,-1,:5]: {hf_logits[0, -1, :5].tolist()}")

# Now load torchtitan's Qwen3 with the same weights
print("\nLoading torchtitan model...")
import sys
sys.path.insert(0, ".")
from torchtitan.models.qwen3 import Qwen3Model
from torchtitan.models.qwen3 import model_registry as qwen3_registry

model_spec = qwen3_registry("0.6B")
model_config = model_spec.model

# Build torchtitan model
tt_model = Qwen3Model(model_config)

# Load HF weights into torchtitan model via state dict adapter
from safetensors.torch import load_file
import glob

# Load all safetensor files
state_dict = {}
for f in sorted(glob.glob(f"{MODEL_PATH}/model*.safetensors")):
    state_dict.update(load_file(f))

# Use the state dict adapter to convert HF keys to torchtitan keys
adapter = model_spec.state_dict_adapter(model_config, MODEL_PATH)
converted = adapter.from_hf(state_dict)
tt_model.load_state_dict(converted, strict=True)
tt_model.eval()
tt_model = tt_model.bfloat16()

# torchtitan forward
# torchtitan expects input WITHOUT the last token (pre-shifted)
# But for comparing logits, let's pass the same input
with torch.no_grad():
    tt_logits = tt_model(input_ids)  # torchtitan expects [batch, seq]

print(f"\nTT logits shape: {tt_logits.shape}")
print(f"TT logits[0,:5]: {tt_logits[0, :5].tolist()}")
print(f"TT logits[-1,:5]: {tt_logits[-1, :5].tolist()}")

# Compare
# HF logits are [batch, seq, vocab], TT may be [seq, vocab] or [batch, seq, vocab]
if tt_logits.dim() == 2:
    tt_compare = tt_logits
    hf_compare = hf_logits[0]
else:
    tt_compare = tt_logits[0]
    hf_compare = hf_logits[0]

print(f"\n=== Comparison ===")
print(f"Max abs diff: {(tt_compare - hf_compare).abs().max().item():.10f}")
print(f"Mean abs diff: {(tt_compare - hf_compare).abs().mean().item():.10f}")
print(f"Allclose (atol=1e-5): {torch.allclose(tt_compare, hf_compare, atol=1e-5)}")
print(f"Allclose (atol=1e-4): {torch.allclose(tt_compare, hf_compare, atol=1e-4)}")
print(f"Allclose (atol=1e-3): {torch.allclose(tt_compare, hf_compare, atol=1e-3)}")

# Check per-position max diff
per_pos_diff = (tt_compare - hf_compare).abs().max(dim=-1).values
print(f"\nPer-position max diff (first 10): {per_pos_diff[:10].tolist()}")
print(f"Per-position max diff (last 5): {per_pos_diff[-5:].tolist()}")
