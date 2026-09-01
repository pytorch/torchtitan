# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
StateDictAdapter for Muse Glimmer (text decoder + vision tower).

Converts between torchtitan's native Muse Glimmer state dict and the released
HuggingFace ``MuseGlimmerForConditionalGeneration`` safetensors layout
(``meta-models/Muse-Glimmer-30B``), so checkpoints can be saved/loaded in HF
format (``checkpoint.last_save_in_hf`` / ``checkpoint.initial_load_in_hf``) and
so a HF<->titan parity check is possible. Both the text decoder and the vision
stack (encoder/adapter/projection) are mapped, matching the VLM convention used
by ``qwen3_5`` / ``kimi_k2_7``.

This mapping was validated against the *actual* released checkpoint's
``model.safetensors.index.json``, its ``config.json``, and the
``transformers`` ``modeling_muse_glimmer.py`` source (transformers >= 5.15).
Key facts that shaped the mapping (all verified, not assumed):

Text decoder (HF prefix ``model.language_model.``):
* RoPE: HF ``muse_glimmer`` applies rotary embeddings with the **split-half**
  ``rotate_half`` convention (``x1=x[..., :d/2]``, ``x2=x[..., d/2:]``), whereas
  torchtitan's ``ComplexRoPE`` pairs **adjacent** head-dim components
  (``view_as_complex(reshape(..., -1, 2))``). The two are related by a
  permutation of the q/k projection rows within each head, so we apply the same
  Q/K (reverse-)permute as ``Llama3StateDictAdapter`` (llama3 also runs
  ComplexRoPE against HF split-half checkpoints). Verified by numerical parity
  against the released checkpoint (with the permute, logits match HF; without
  it, cosine drops to ~0.82).
* Attention output gate: torchtitan ``attention.o_gate`` == HF
  ``self_attn.gate_proj`` (``attn_output * sigmoid(gate_proj(x))``).
* Norm conventions match on both sides (gain-centered ``1+weight`` for the
  per-layer norms, plain ``weight`` for the final norm), so no +/-1 offset.
* Scaleless norms (qk_norm, token-embedding norm) have no learnable params.
* ``tie_word_embeddings=False``; ``lm_head`` maps directly.

Vision stack (HF prefix ``model.vision_tower.`` / ``model.vision_adapter.`` /
``model.vision_projection``):
* HF uses **split** ``q_proj/k_proj/v_proj`` with bias; torchtitan also stores
  split ``wq/wk/wv`` with bias -> straight 1:1 map (no fuse/split, unlike
  qwen3_5 whose HF vision is fused).
* Vision RoPE: HF applies 2D ``rotate_half`` (split-half) while torchtitan's
  vision attention uses the complex backend (adjacent-pair). As on the text
  side, q/k projection rows are (reverse-)permuted to reconcile the two.
* Patch embedding is a ``Linear`` on both sides (HF
  ``patch_embedder.patch_embedding``, shape [dim, in_ch*T*P*P]); no Conv3d
  reshape needed. The learned position table maps to torchtitan's raw
  ``positional_embedding_vlm`` parameter.
* Everything else (proj, mlp fc1/fc2, norm1/norm2, ln_pre/ln_post, adapter
  fc1/fc2, vision_projection) is a straight rename incl. biases.

torchtitan mid-layer name note: torchtitan exposes the pre-FFN norm as
``ffn_norm`` and the post-FFN norm as ``post_ffn_norm``; these map to HF
``pre_feedforward_layernorm`` and ``post_feedforward_layernorm`` respectively.
"""

import functools
import re
from typing import Any

from torch.distributed.tensor import DTensor, Replicate

from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .model import MuseGlimmerModel

# HF prefixes in the released MuseGlimmerForConditionalGeneration.
_HF_TEXT_PREFIX = "model.language_model."
_HF_VISION_PREFIX = "model.vision_tower."


def _dtensor_safe(fn):
    """Run a row-reshaping permute that is invalid on a tensor sharded along the
    permuted (row) dim.

    In the live save/load path ``to_hf`` / ``from_hf`` receive DTensors that
    FSDP shards along dim 0 (the q/k output rows). The head-splitting
    ``view(n_heads, ...)`` cannot unflatten an unevenly-sharded dim and raises
    ``Cannot unflatten unevenly sharded tensor``. Redistribute to Replicate,
    permute the full local tensor, then restore the original placements. Plain
    (non-DTensor) tensors take the fast path unchanged.
    """

    @functools.wraps(fn)
    def wrapper(self, w, *args, **kwargs):
        if isinstance(w, DTensor):
            placements = w.placements
            mesh = w.device_mesh
            replicated = w.redistribute(
                device_mesh=mesh, placements=[Replicate()] * mesh.ndim
            )
            local = fn(self, replicated.to_local(), *args, **kwargs)
            out = DTensor.from_local(
                local, mesh, [Replicate()] * mesh.ndim, run_check=False
            )
            return out.redistribute(device_mesh=mesh, placements=placements)
        return fn(self, w, *args, **kwargs)

    return wrapper


class MuseGlimmerStateDictAdapter(StateDictAdapter):
    def __init__(
        self,
        model_config: "MuseGlimmerModel.Config",
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)
        self.model_config = model_config
        self.hf_assets_path = hf_assets_path

        p = _HF_TEXT_PREFIX
        v = _HF_VISION_PREFIX
        # torchtitan native FQN (values)  <->  released HF FQN (keys)
        self.from_hf_map = {
            # ===== Text decoder =====
            f"{p}embed_tokens.weight": "tok_embeddings.embedding.weight",
            # Attention (q/k get a RoPE permute -- see class docstring)
            f"{p}layers.{{}}.self_attn.q_proj.weight": "layers.{}.attention.qkv_linear.wq.weight",
            f"{p}layers.{{}}.self_attn.k_proj.weight": "layers.{}.attention.qkv_linear.wk.weight",
            f"{p}layers.{{}}.self_attn.v_proj.weight": "layers.{}.attention.qkv_linear.wv.weight",
            f"{p}layers.{{}}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            # Attention output gate: HF gate_proj == titan o_gate
            f"{p}layers.{{}}.self_attn.gate_proj.weight": "layers.{}.attention.o_gate.weight",
            # MLP (SwiGLU): w1=gate, w3=up, w2=down
            f"{p}layers.{{}}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            f"{p}layers.{{}}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            f"{p}layers.{{}}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            # Norms (gain-centered / plain conventions match -> no offset)
            f"{p}layers.{{}}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            f"{p}layers.{{}}.post_attention_layernorm.weight": "layers.{}.post_attention_norm.weight",
            f"{p}layers.{{}}.pre_feedforward_layernorm.weight": "layers.{}.ffn_norm.weight",
            f"{p}layers.{{}}.post_feedforward_layernorm.weight": "layers.{}.post_ffn_norm.weight",
            # Final norm + head
            f"{p}norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
            # ===== Vision tower =====
            # Patch embedding (Linear on both sides) + learned position table
            f"{v}patch_embedder.patch_embedding.weight": "vision_encoder.conv1_linear.weight",
            f"{v}patch_embedder.position_embedding_table.weight": "vision_encoder.positional_embedding_vlm",
            # Pre/post layernorms
            f"{v}ln_pre.weight": "vision_encoder.ln_pre.weight",
            f"{v}ln_pre.bias": "vision_encoder.ln_pre.bias",
            f"{v}ln_post.weight": "vision_encoder.ln_post.weight",
            f"{v}ln_post.bias": "vision_encoder.ln_post.bias",
            # Vision transformer blocks (HF: vision_tower.layers, split q/k/v; q/k permuted)
            f"{v}layers.{{}}.attn.q_proj.weight": "vision_encoder.layers.{}.attn.wq.weight",
            f"{v}layers.{{}}.attn.q_proj.bias": "vision_encoder.layers.{}.attn.wq.bias",
            f"{v}layers.{{}}.attn.k_proj.weight": "vision_encoder.layers.{}.attn.wk.weight",
            f"{v}layers.{{}}.attn.k_proj.bias": "vision_encoder.layers.{}.attn.wk.bias",
            f"{v}layers.{{}}.attn.v_proj.weight": "vision_encoder.layers.{}.attn.wv.weight",
            f"{v}layers.{{}}.attn.v_proj.bias": "vision_encoder.layers.{}.attn.wv.bias",
            f"{v}layers.{{}}.attn.proj.weight": "vision_encoder.layers.{}.attn.proj.weight",
            f"{v}layers.{{}}.attn.proj.bias": "vision_encoder.layers.{}.attn.proj.bias",
            f"{v}layers.{{}}.mlp.fc1.weight": "vision_encoder.layers.{}.mlp.linear_fc1.weight",
            f"{v}layers.{{}}.mlp.fc1.bias": "vision_encoder.layers.{}.mlp.linear_fc1.bias",
            f"{v}layers.{{}}.mlp.fc2.weight": "vision_encoder.layers.{}.mlp.linear_fc2.weight",
            f"{v}layers.{{}}.mlp.fc2.bias": "vision_encoder.layers.{}.mlp.linear_fc2.bias",
            f"{v}layers.{{}}.norm1.weight": "vision_encoder.layers.{}.norm1.weight",
            f"{v}layers.{{}}.norm1.bias": "vision_encoder.layers.{}.norm1.bias",
            f"{v}layers.{{}}.norm2.weight": "vision_encoder.layers.{}.norm2.weight",
            f"{v}layers.{{}}.norm2.bias": "vision_encoder.layers.{}.norm2.bias",
            # Vision adapter + projection to LLM dim
            "model.vision_adapter.fc1.weight": "vision_adapter.c_fc.weight",
            "model.vision_adapter.fc2.weight": "vision_adapter.c_proj.weight",
            "model.vision_projection.weight": "vision_projection.weight",
        }

    # ---- HF split-half <-> ComplexRoPE interleaved permute (as in Llama3) ----
    # These reshape the row (dim-0) axis, which is invalid on a tensor sharded
    # along that axis, so they are wrapped to redistribute DTensors to Replicate
    # first (see _dtensor_safe).
    @_dtensor_safe
    def _permute(self, w, n_heads_arg, dim1=None, dim2=None):
        """native (interleaved) -> HF (split-half) row order for q/k."""
        if dim1 is None:
            dim1 = w.shape[0]
        if dim2 is None:
            dim2 = w.shape[1]
        return (
            w.view(n_heads_arg, dim1 // n_heads_arg // 2, 2, dim2)
            .transpose(1, 2)
            .reshape(dim1, dim2)
            .clone()
        )

    @_dtensor_safe
    def _reverse_permute(self, w, n_heads_arg, dim1=None, dim2=None):
        """HF (split-half) -> native (interleaved) row order for q/k."""
        if dim1 is None:
            dim1 = w.shape[0]
        if dim2 is None:
            dim2 = w.shape[1]
        return (
            w.view(n_heads_arg, 2, dim1 // n_heads_arg // 2, dim2)
            .transpose(1, 2)
            .reshape(dim1, dim2)
            .clone()
        )

    @_dtensor_safe
    def _permute_1d(self, w, n_heads_arg):
        """Permute a 1D bias/vector (split-half -> interleaved, and inverse via
        the same transpose structure). For q/k biases in the vision tower."""
        dim = w.shape[0]
        return (
            w.view(n_heads_arg, dim // n_heads_arg // 2, 2)
            .transpose(1, 2)
            .reshape(dim)
            .clone()
        )

    @_dtensor_safe
    def _reverse_permute_1d(self, w, n_heads_arg):
        dim = w.shape[0]
        return (
            w.view(n_heads_arg, 2, dim // n_heads_arg // 2)
            .transpose(1, 2)
            .reshape(dim)
            .clone()
        )

    def _attn_geometry(self):
        # pyrefly: ignore [missing-attribute]
        attn = self.model_config.layers[0].attention
        n_heads = attn.n_heads
        n_kv_heads = attn.n_kv_heads if attn.n_kv_heads is not None else n_heads
        # pyrefly: ignore [missing-attribute]
        dim = self.model_config.dim
        head_dim = attn.head_dim or dim // n_heads
        return n_heads, n_kv_heads, dim, head_dim

    def _vision_num_heads(self):
        # pyrefly: ignore [missing-attribute]
        ve = self.model_config.vision_encoder
        return ve.num_heads if ve is not None else None

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        n_heads, n_kv_heads, dim, head_dim = self._attn_geometry()
        v_heads = self._vision_num_heads()
        to_hf_map = {val: k for k, val in self.from_hf_map.items() if val is not None}
        hf_state_dict: dict[str, Any] = {}

        for key, value in state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)
                new_key = to_hf_map.get(abstract_key)
                if new_key is None:
                    continue
                # Text: interleaved (titan) -> split-half (HF) for q/k
                if abstract_key == "layers.{}.attention.qkv_linear.wq.weight":
                    value = self._permute(value, n_heads)
                elif abstract_key == "layers.{}.attention.qkv_linear.wk.weight":
                    value = self._permute(value, n_kv_heads, head_dim * n_kv_heads, dim)
                # Vision: same permute for q/k weight+bias
                elif abstract_key == "vision_encoder.layers.{}.attn.wq.weight":
                    value = self._permute(value, v_heads)
                elif abstract_key == "vision_encoder.layers.{}.attn.wk.weight":
                    value = self._permute(value, v_heads)
                elif abstract_key == "vision_encoder.layers.{}.attn.wq.bias":
                    value = self._permute_1d(value, v_heads)
                elif abstract_key == "vision_encoder.layers.{}.attn.wk.bias":
                    value = self._permute_1d(value, v_heads)
                new_key = new_key.format(layer_num)
            else:
                new_key = to_hf_map.get(key)
                if new_key is None:
                    continue
            hf_state_dict[new_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        # All Muse Glimmer text layers carry a ComplexRoPE config (NoPE layers
        # still build one; RoPE is guarded in forward), so this validates cleanly.
        self._validate_hf_rope_config(ComplexRoPE.Config)

        n_heads, n_kv_heads, dim, head_dim = self._attn_geometry()
        v_heads = self._vision_num_heads()
        state_dict: dict[str, Any] = {}
        for key, value in hf_state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)
                new_key = self.from_hf_map.get(abstract_key)
                if new_key is None:
                    continue
                # Text: split-half (HF) -> interleaved (titan) for q/k
                if (
                    abstract_key
                    == f"{_HF_TEXT_PREFIX}layers.{{}}.self_attn.q_proj.weight"
                ):
                    value = self._reverse_permute(value, n_heads)
                elif (
                    abstract_key
                    == f"{_HF_TEXT_PREFIX}layers.{{}}.self_attn.k_proj.weight"
                ):
                    value = self._reverse_permute(
                        value, n_kv_heads, head_dim * n_kv_heads, dim
                    )
                # Vision: reverse permute for q/k weight+bias
                elif (
                    abstract_key == f"{_HF_VISION_PREFIX}layers.{{}}.attn.q_proj.weight"
                ):
                    value = self._reverse_permute(value, v_heads)
                elif (
                    abstract_key == f"{_HF_VISION_PREFIX}layers.{{}}.attn.k_proj.weight"
                ):
                    value = self._reverse_permute(value, v_heads)
                elif abstract_key == f"{_HF_VISION_PREFIX}layers.{{}}.attn.q_proj.bias":
                    value = self._reverse_permute_1d(value, v_heads)
                elif abstract_key == f"{_HF_VISION_PREFIX}layers.{{}}.attn.k_proj.bias":
                    value = self._reverse_permute_1d(value, v_heads)
                new_key = new_key.format(layer_num)
            else:
                new_key = self.from_hf_map.get(key)
                if new_key is None:
                    continue
            state_dict[new_key] = value

        return state_dict
