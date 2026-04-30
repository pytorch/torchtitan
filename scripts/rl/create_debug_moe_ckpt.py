# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Generate a debug HF-compatible checkpoint for the debugmodel_moe Qwen3 config.

Uses the model's own ``init_weights()`` for properly-scaled initialization
(scaled output projections, etc.), which prevents activation explosion that
naive ``randn * 0.02`` produces across deep models in bf16.

Output: /tmp/debug_moe_ckpt/ with config.json, model.safetensors, tokenizer files.
"""
import json
import os
import shutil

import torch
from safetensors.torch import save_file

from torchtitan.models.qwen3 import model_registry

OUT = "/tmp/debug_moe_ckpt"
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(42)

ms = model_registry("debugmodel_moe", attn_backend="varlen")
with torch.device("meta"):
    model = ms.model.build()
model.to_empty(device="cpu")
with torch.no_grad():
    model.init_weights(buffer_device=None)

sd_adapter = ms.state_dict_adapter(ms.model, OUT)
hf_state_dict = sd_adapter.to_hf(model.state_dict())
hf_state_dict = {k: v.to(torch.bfloat16) for k, v in hf_state_dict.items()}

save_file(hf_state_dict, os.path.join(OUT, "model.safetensors"))

mc = ms.model
attn = mc.layers[0].attention
moe = mc.layers[0].moe
n_heads = attn.n_heads
n_kv_heads = attn.n_kv_heads or n_heads
head_dim = attn.head_dim if attn.head_dim is not None else mc.dim // n_heads

config = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 0,
    "decoder_sparse_step": 1,
    "eos_token_id": 1,
    "head_dim": head_dim,
    "hidden_act": "silu",
    "hidden_size": mc.dim,
    "initializer_range": 0.02,
    "intermediate_size": moe.experts.hidden_dim,
    "max_position_embeddings": 4096,
    "max_window_layers": len(mc.layers),
    "mlp_only_layers": [],
    "model_type": "qwen3_moe",
    "moe_intermediate_size": moe.experts.hidden_dim,
    "norm_topk_prob": True,
    "num_attention_heads": n_heads,
    "num_experts": moe.experts.num_experts,
    "num_experts_per_tok": moe.router.top_k,
    "num_hidden_layers": len(mc.layers),
    "num_key_value_heads": n_kv_heads,
    "output_router_logits": False,
    "rms_norm_eps": 1e-6,
    "rope_scaling": None,
    "rope_theta": 1000000.0,
    "router_aux_loss_coef": 0.001,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.45.0",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": mc.vocab_size,
}
with open(os.path.join(OUT, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

with open(os.path.join(OUT, "generation_config.json"), "w") as f:
    json.dump(
        {
            "bos_token_id": 0,
            "eos_token_id": 1,
            "transformers_version": "4.45.0",
        },
        f,
        indent=2,
    )

src = "/data/users/jianiw/model/Qwen3-30B-A3B"
for f in (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
):
    sp = os.path.join(src, f)
    if os.path.exists(sp):
        shutil.copy(sp, OUT)

total = sum(t.numel() for t in hf_state_dict.values())
print(f"Created debug MoE checkpoint at {OUT}")
print(f"  Total params: {total / 1e6:.1f}M")
print(f"  Files: {sorted(os.listdir(OUT))}")
