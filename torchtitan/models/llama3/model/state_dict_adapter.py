# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Any

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .args import TransformerModelArgs
from torchtitan.tools.logging import logger
import torch.nn.functional as F
from torch.distributed.tensor.placement_types import Replicate
import hashlib


def calculate_hash(tensor_data: np.ndarray) -> str:
    """Calculate SHA-256 hash of a tensor."""
    tensor_bytes = tensor_data.tobytes()
    return hashlib.sha256(tensor_bytes).hexdigest()

class Llama3StateDictAdapter(StateDictAdapter):
    def __init__(self, model_args: TransformerModelArgs):
        self.model_args = model_args
        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }

    # HuggingFace permutation function (exact copy from their conversion script)
    def _permute(self, w, n_heads_arg, dim1=None, dim2=None):
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
        if dim1 is None:
            dim1 = w.shape[0]
        if dim2 is None:
            dim2 = w.shape[1]
        return (
            w.view(n_heads_arg, 2, dim1 // n_heads_arg // 2, dim2)
            .transpose(1, 2)
            .reshape(dim1, dim2)
        )

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        to_hf_map = {v: k for k, v in self.from_hf_map.items()}

        n_heads = self.model_args.n_heads
        n_kv_heads = (
            self.model_args.n_kv_heads
            if self.model_args.n_kv_heads is not None
            else n_heads
        )
        dim = self.model_args.dim
        head_dim = dim // n_heads
        hf_state_dict = {}

        for key, value in state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_key = to_hf_map[abstract_key]
                # We need to permute the weights in wq and wk layer in order to account for the difference between
                # the native Llama and huggingface RoPE implementation.
                if abstract_key == "layers.{}.attention.wq.weight":
                    # input_full_tensor = value.clone().full_tensor()
                    if layer_num == "0":
                        full_value = value.redistribute(placements=[Replicate(), Replicate()]).to_local()
                        logger.info(f"[To hf] Before permute {key}, the dtensor full value is {full_value}, hash {calculate_hash(full_value.detach().cpu().numpy())}")
                        logger.info(f"[To hf] Before permute {key}, the dtensor shape is {value.shape}, placement is {value.placements}, device_mesh is {value.device_mesh}")
                    value = self._permute(value, n_heads)
                    # output_full_tensor = value.clone().full_tensor()
                    if layer_num == "0":
                        full_value = value.redistribute(placements=[Replicate(), Replicate()]).to_local()
                        logger.info(f"[To hf] After permute {key}, the dtensor full value is {full_value}, , hash {calculate_hash(full_value.detach().cpu().numpy()))}")
                        logger.info(f"[To hf] After permute {key}, the dtensor shape is {value.shape}, placement is {value.placements}, device_mesh is {value.device_mesh}")
                    # # logger.info(f"[To hf] KL divergence between input and output tensors for {key}: {loss_fn(self._permute(input_full_tensor, n_heads), output_full_tensor).item()}")
                    # are_tensors_equal = torch.allclose(
                    #     self._permute(input_full_tensor, n_heads), 
                    #     output_full_tensor,
                    #     rtol=1e-5, 
                    #     atol=1e-5
                    # )
                    # logger.info(f"[To hf] Are input and output tensors equivalent for {key}: {are_tensors_equal}")

                if abstract_key == "layers.{}.attention.wk.weight":
                    # input_full_tensor = value.clone().full_tensor()
                    key_value_dim = head_dim * n_kv_heads
                    # logger.info(f"[To hf] Before permute {key}, the dtensor shape is {value.shape}, placement is {value.placements}, device_mesh is {value.device_mesh}")
                    value = self._permute(value, n_kv_heads, key_value_dim, dim)
                    # output_full_tensor = value.clone().full_tensor()
                    # logger.info(f"[To hf] After permute {key}, the dtensor shape is {value.shape}, placement is {value.placements}, device_mesh is {value.device_mesh}")
                    # logger.info(f"[To hf] KL divergence between input and output tensors for {key}: {loss_fn(self._permute(input_full_tensor, n_heads), output_full_tensor).item()}")

                
                # if abstract_key == "layers.{}.attention.wq.weight":
                #     full_value = value.redistribute(placements=[Replicate(), Replicate()])
                #     value = self._permute(full_value, n_heads)

                # if abstract_key == "layers.{}.attention.wk.weight":
                #     key_value_dim = head_dim * n_kv_heads
                #     full_value = value.redistribute(placements=[Replicate(), Replicate()])
                #     logger.info(f"[to hf] Before _reverse_permute {key}, the dtensor shape is {full_value.shape}, placement is {full_value.placements}, device_mesh is {full_value.device_mesh}")
                #     value = self._permute(full_value, n_kv_heads, key_value_dim, dim)
                #     logger.info(f"[to hf] After _reverse_permute {key}, the dtensor shape is {full_value.shape}, placement is {full_value.placements}, device_mesh is {full_value.device_mesh}")

                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
            else:
                new_key = to_hf_map[key]

            hf_state_dict[new_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        n_heads = self.model_args.n_heads
        n_kv_heads = (
            self.model_args.n_kv_heads
            if self.model_args.n_kv_heads is not None
            else n_heads
        )
        dim = self.model_args.dim
        head_dim = dim // n_heads
        state_dict = {}

        for key, value in hf_state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_key = self.from_hf_map[abstract_key]

                # We need to permute the weights in wq and wk layer in order to account for the difference between
                # # the native Llama and huggingface RoPE implementation.
                if abstract_key == "model.layers.{}.self_attn.q_proj.weight":
                    if layer_num == "0": 
                        full_value = value.redistribute(placements=[Replicate(), Replicate()]).to_local()
                        logger.info(f"[From hf] Before _reverse_permute {key}, the dtensor full value is {full_value}, hash {calculate_hash(full_value.detach().cpu().numpy())}")
                        logger.info(f"[From hf] Before _reverse_permute {abstract_key}, the dtensor shape is {value.shape}, placement is {value.placements}, device_mesh is {value.device_mesh}")
                    value = self._reverse_permute(value, n_heads)
                    if layer_num == "0":
                        full_value = value.redistribute(placements=[Replicate(), Replicate()]).to_local()
                        logger.info(f"[From hf] After _reverse_permute {key}, the dtensor full value is {full_value}, hash {calculate_hash(full_value.detach().cpu().numpy())}")
                        logger.info(f"[From hf] After _reverse_permute {abstract_key}, the dtensor shape is {value.shape}, placement is {value.placements}, device_mesh is {value.device_mesh}")

                if abstract_key == "model.layers.{}.self_attn.k_proj.weight":
                    key_value_dim = head_dim * n_kv_heads
                    value = self._reverse_permute(value, n_kv_heads, key_value_dim, dim)
                
                # if abstract_key == "model.layers.{}.self_attn.q_proj.weight":
                #     full_value = value.redistribute(placements=[Replicate(), Replicate()])
                #     value = self._reverse_permute(full_value, n_heads)
                
                # if abstract_key == "model.layers.{}.self_attn.k_proj.weight":
                #     key_value_dim = head_dim * n_kv_heads
                #     full_value = value.redistribute(placements=[Replicate(), Replicate()])
                #     logger.info(f"[From hf] Before _reverse_permute {key}, the dtensor shape is {full_value.shape}, placement is {full_value.placements}, device_mesh is {full_value.device_mesh}")
                #     value = self._reverse_permute(full_value, n_kv_heads, key_value_dim, dim)
                #     logger.info(f"[From hf] After _reverse_permute {key}, the dtensor shape is {full_value.shape}, placement is {full_value.placements}, device_mesh is {full_value.device_mesh}")

                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
            else:
                new_key = self.from_hf_map[key]

            state_dict[new_key] = value
        return state_dict
