# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Weight reconstruction and name conversion for native vLLM model updates.

Reconstructs full (unsharded) tensors from per-rank TP shards produced by
the torchtitan trainer, then converts parameter names from torchtitan
format to HuggingFace format suitable for vLLM's ``load_weights()``.
"""

import torch

# TorchTitan parameter name suffixes and their TP shard reconstruction strategy.
#
# ColwiseParallel: each rank holds a slice along dim 0 -> cat on dim 0
# RowwiseParallel: each rank holds a slice along dim 1 -> cat on dim 1
# Replicated (norms): all ranks hold identical copies -> take rank 0

_COLWISE_SUFFIXES = (
    "attention.wq.weight",
    "attention.wk.weight",
    "attention.wv.weight",
    "feed_forward.w1.weight",
    "feed_forward.w3.weight",
    "output.weight",
)

_ROWWISE_SUFFIXES = (
    "tok_embeddings.weight",
    "attention.wo.weight",
    "feed_forward.w2.weight",
)

# Name mapping: torchtitan -> HuggingFace.
# Derived by inverting VLLM_TO_TITAN_MAP from
# torchtitan/experiments/rl/vllm_compat/weights/converter.py.
# Layer-indexed entries use ``{}`` as a placeholder for the layer number.
_TITAN_TO_HF_MAP = {
    "tok_embeddings.weight": "model.embed_tokens.weight",
    "layers.{}.attention.wq.weight": "model.layers.{}.self_attn.q_proj.weight",
    "layers.{}.attention.wk.weight": "model.layers.{}.self_attn.k_proj.weight",
    "layers.{}.attention.wv.weight": "model.layers.{}.self_attn.v_proj.weight",
    "layers.{}.attention.wo.weight": "model.layers.{}.self_attn.o_proj.weight",
    "layers.{}.attention.q_norm.weight": "model.layers.{}.self_attn.q_norm.weight",
    "layers.{}.attention.k_norm.weight": "model.layers.{}.self_attn.k_norm.weight",
    "layers.{}.feed_forward.w1.weight": "model.layers.{}.mlp.gate_proj.weight",
    "layers.{}.feed_forward.w3.weight": "model.layers.{}.mlp.up_proj.weight",
    "layers.{}.feed_forward.w2.weight": "model.layers.{}.mlp.down_proj.weight",
    "layers.{}.attention_norm.weight": "model.layers.{}.input_layernorm.weight",
    "layers.{}.ffn_norm.weight": "model.layers.{}.post_attention_layernorm.weight",
    "norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}


def _get_cat_dim(param_name: str) -> int | None:
    """Return the concatenation dimension for a sharded parameter.

    Returns 0 for ColwiseParallel, 1 for RowwiseParallel, or ``None`` for
    replicated parameters (norms).
    """
    for suffix in _COLWISE_SUFFIXES:
        if param_name.endswith(suffix):
            return 0
    for suffix in _ROWWISE_SUFFIXES:
        if param_name.endswith(suffix):
            return 1
    return None


def _titan_name_to_hf(titan_name: str) -> str:
    """Convert a torchtitan parameter name to HuggingFace format."""
    if "layers." in titan_name:
        parts = titan_name.split(".")
        layer_idx = parts[1]
        abstract_key = titan_name.replace(f".{layer_idx}.", ".{}.")
        if abstract_key in _TITAN_TO_HF_MAP:
            return _TITAN_TO_HF_MAP[abstract_key].format(layer_idx)
    else:
        if titan_name in _TITAN_TO_HF_MAP:
            return _TITAN_TO_HF_MAP[titan_name]
    raise ValueError(f"No HF name mapping for torchtitan param: {titan_name}")


def _detect_weight_tying(
    rank_0_sd: dict[str, torch.Tensor],
) -> bool:
    """Detect whether tok_embeddings and output share the same weight (tying).

    When weight tying is active, ``parallelize_module`` converts the shared
    parameter to ``Shard(0)`` (from ``output``'s ColwiseParallel, which runs
    after ``tok_embeddings``'s RowwiseParallel).  Both state-dict entries
    then have the **same** local shape ``[V/tp, H]``.

    Without tying the shapes differ: ``tok_embeddings`` is ``[V, H/tp]``
    (RowwiseParallel) while ``output`` is ``[V/tp, H]`` (ColwiseParallel).
    Since ``V != H`` for all practical language models, a shape comparison
    reliably distinguishes the two cases.
    """
    tok = rank_0_sd.get("tok_embeddings.weight")
    out = rank_0_sd.get("output.weight")
    if tok is None or out is None:
        return False
    # Fast path: same storage after Monarch transfer (not always preserved)
    if tok.data_ptr() == out.data_ptr():
        return True
    # Fallback: identical shapes imply shared Shard(0) placement
    return tok.shape == out.shape


def convert_to_hf(
    state_dict: dict[str, torch.Tensor],
) -> list[tuple[str, torch.Tensor]]:
    """Convert torchtitan parameter names to HF format.

    Unlike ``reconstruct_and_convert_to_hf``, this assumes the tensors are
    already full (unsharded) — as produced by ``get_weights()`` which calls
    ``.full_tensor()`` on DTensors.

    Args:
        state_dict: Flat mapping of torchtitan param names to full tensors.

    Returns:
        List of ``(hf_name, full_tensor)`` pairs suitable for passing to
        vLLM's ``model.load_weights()``.
    """
    return [(_titan_name_to_hf(name), tensor) for name, tensor in state_dict.items()]


def reconstruct_and_convert_to_hf(
    per_rank_state_dicts: dict[int, dict[str, torch.Tensor]],
    tp_degree: int,
) -> list[tuple[str, torch.Tensor]]:
    """Reconstruct full tensors from TP shards and convert names to HF format.

    Args:
        per_rank_state_dicts: Mapping from rank index to that rank's state dict
            (local tensors exported by the trainer via ``to_local()``).
        tp_degree: Tensor parallel degree (number of TP ranks).

    Returns:
        List of ``(hf_name, full_tensor)`` pairs suitable for passing to
        vLLM's ``model.load_weights()``.
    """
    rank_0_sd = per_rank_state_dicts[0]
    results: list[tuple[str, torch.Tensor]] = []

    # When weight tying is active, tok_embeddings.weight shares storage with
    # output.weight and ends up with Shard(0) placement (ColwiseParallel from
    # ``output``).  We must cat on dim 0 instead of the usual dim 1.
    tied_embeddings = _detect_weight_tying(rank_0_sd)

    for param_name in rank_0_sd:
        cat_dim = _get_cat_dim(param_name)

        if tied_embeddings and param_name == "tok_embeddings.weight":
            cat_dim = 0

        if cat_dim is not None:
            shards = [per_rank_state_dicts[r][param_name] for r in range(tp_degree)]
            full_tensor = torch.cat(shards, dim=cat_dim)
        else:
            # Replicated: all ranks hold identical copies
            full_tensor = rank_0_sd[param_name]

        hf_name = _titan_name_to_hf(param_name)
        results.append((hf_name, full_tensor))

    return results
