# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
StateDictAdapter for Muse Glimmer.

Converts between torchtitan's native Muse Glimmer state dict and the released
HuggingFace ``MuseGlimmerForConditionalGeneration`` safetensors layout
(``meta-models/Muse-Glimmer-30B``), so checkpoints can be saved/loaded in HF
format (``checkpoint.last_save_in_hf`` / ``checkpoint.initial_load_in_hf``) and
so a HF<->titan parity check is possible.

This mapping was validated against the *actual* released checkpoint's
``model.safetensors.index.json``, its ``config.json``, and the
``transformers`` ``modeling_muse_glimmer.py`` source (transformers >= 5.15).
Key facts that shaped the mapping (all verified, not assumed):

* HF prefix is ``model.language_model.`` (the released model is the multimodal
  ``MuseGlimmerForConditionalGeneration`` with a vision tower/adapter). The
  text-decoder weights this adapter handles live under that prefix; ``lm_head``
  is top-level.
* RoPE: this is the subtle one. HF ``muse_glimmer`` applies rotary embeddings
  with the **split-half** ``rotate_half`` convention (``x1=x[..., :d/2]``,
  ``x2=x[..., d/2:]``), whereas torchtitan's ``ComplexRoPE`` pairs **adjacent**
  head-dim components (``view_as_complex(reshape(..., -1, 2))`` -> pairs
  ``(0,1),(2,3),...``). The two conventions are related by a permutation of the
  q/k projection rows within each head. We therefore apply the **same Q/K
  (reverse-)permute as ``Llama3StateDictAdapter``** (llama3 also uses
  ComplexRoPE against HF split-half checkpoints). ``_permute`` and
  ``_reverse_permute`` are exact inverses. This was VERIFIED by numerical
  parity against the released checkpoint: with the permute, torchtitan logits
  match HF (see the parity test); without it, cosine sim drops to ~0.82.
* Attention output gate: torchtitan ``attention.o_gate`` == HF
  ``self_attn.gate_proj`` (HF applies ``attn_output * sigmoid(gate_proj(x))``,
  shape [n_heads*head_dim, dim]). Real HF key, no extension needed.
* Norm conventions (verified against both codebases -- NO +/-1 offset needed):
    - Per-layer input/post_attention/pre_feedforward/post_feedforward norms use
      HF ``MuseGlimmerTextCenteredRMSNorm`` (effective scale ``1 + weight``,
      weight stored centered on 0). torchtitan uses ``RMSGainCenterNorm`` with
      ``gain_center=1.0`` (also stored centered on 0). Same storage -> pass
      through.
    - Final ``model.norm`` uses HF plain ``MuseGlimmerRMSNorm`` (scale =
      ``weight``). torchtitan final norm is ``RMSGainCenterNorm`` with
      ``gain_center=0.0`` (scale = ``weight``). Same storage -> pass through.
* Scaleless norms (qk_norm, token-embedding norm) have no learnable params on
  either side, so there is nothing to map.
* ``tie_word_embeddings=False`` and ``lm_head.weight`` is a distinct tensor in
  the released index, so lm_head maps directly (no embed<->head aliasing).

torchtitan mid-layer name note: torchtitan exposes the pre-FFN norm as
``ffn_norm`` and the post-FFN norm as ``post_ffn_norm``; these map to HF
``pre_feedforward_layernorm`` and ``post_feedforward_layernorm`` respectively.
"""

import re
from typing import Any

from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.protocols.state_dict_adapter import StateDictAdapter
from .model import MuseGlimmerModel

# HF text-decoder prefix in the released MuseGlimmerForConditionalGeneration.
_HF_TEXT_PREFIX = "model.language_model."


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
        # torchtitan native FQN (values)  <->  released HF FQN (keys)
        self.from_hf_map = {
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
        }

    # ---- HF split-half <-> ComplexRoPE interleaved permute (as in Llama3) ----
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

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        n_heads, n_kv_heads, dim, head_dim = self._attn_geometry()
        to_hf_map = {v: k for k, v in self.from_hf_map.items() if v is not None}
        hf_state_dict: dict[str, Any] = {}

        for key, value in state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)
                new_key = to_hf_map.get(abstract_key)
                if new_key is None:
                    continue
                # interleaved (titan) -> split-half (HF) for q/k
                if abstract_key == "layers.{}.attention.qkv_linear.wq.weight":
                    value = self._permute(value, n_heads)
                elif abstract_key == "layers.{}.attention.qkv_linear.wk.weight":
                    value = self._permute(value, n_kv_heads, head_dim * n_kv_heads, dim)
                new_key = new_key.format(layer_num)
            else:
                new_key = to_hf_map.get(key)
                if new_key is None:
                    continue
            hf_state_dict[new_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        # All Muse Glimmer layers carry a ComplexRoPE config (NoPE layers still
        # build one; RoPE is guarded in forward), so this validates cleanly.
        self._validate_hf_rope_config(ComplexRoPE.Config)

        n_heads, n_kv_heads, dim, head_dim = self._attn_geometry()
        state_dict: dict[str, Any] = {}
        for key, value in hf_state_dict.items():
            # Skip multimodal (vision tower/adapter/projection) weights -- this
            # adapter handles the text decoder; the vision path is loaded
            # separately by the MM model spec when enabled.
            if (
                key.startswith("model.vision_")
                or key.startswith("model.perception")
                or ".vision_" in key
            ):
                continue

            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)
                new_key = self.from_hf_map.get(abstract_key)
                if new_key is None:
                    continue
                # split-half (HF) -> interleaved (titan) for q/k
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
                new_key = new_key.format(layer_num)
            else:
                new_key = self.from_hf_map.get(key)
                if new_key is None:
                    continue
            state_dict[new_key] = value

        return state_dict
