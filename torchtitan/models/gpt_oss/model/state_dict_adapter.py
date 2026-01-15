# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import math
import os
import re
from typing import Any

import torch
from safetensors import safe_open
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torchtitan.models.utils import MoEStateDictAdapter
from torchtitan.tools.logging import logger

from .args import GptOssModelArgs

# MXFP4 lookup table for dequantization
FP4_VALUES = [
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


class GptOssStateDictAdapter(MoEStateDictAdapter):
    def __init__(self, model_args: GptOssModelArgs, hf_assets_path: str | None):
        super().__init__(model_args, hf_assets_path)
        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            # Attention module
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "model.layers.{}.self_attn.q_proj.bias": "layers.{}.attention.wq.bias",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
            "model.layers.{}.self_attn.k_proj.bias": "layers.{}.attention.wk.bias",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
            "model.layers.{}.self_attn.v_proj.bias": "layers.{}.attention.wv.bias",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.o_proj.bias": "layers.{}.attention.wo.bias",
            "model.layers.{}.self_attn.sinks": "layers.{}.attention.sinks",
            # Transformer layer
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            # MoE - quantized format (with _blocks suffix)
            "model.layers.{}.mlp.experts.gate_up_proj_blocks": "layers.{}.moe.experts.mlp1_weight",
            "model.layers.{}.mlp.experts.gate_up_proj_bias": "layers.{}.moe.experts.mlp1_bias",
            "model.layers.{}.mlp.experts.down_proj_blocks": "layers.{}.moe.experts.mlp2_weight",
            "model.layers.{}.mlp.experts.down_proj_bias": "layers.{}.moe.experts.mlp2_bias",
            # MoE - unquantized format (without _blocks suffix)
            "model.layers.{}.mlp.experts.gate_up_proj": "layers.{}.moe.experts.mlp1_weight",
            "model.layers.{}.mlp.experts.down_proj": "layers.{}.moe.experts.mlp2_weight",
            # Router
            "model.layers.{}.mlp.router.weight": "layers.{}.moe.router.gate.weight",
            "model.layers.{}.mlp.router.bias": "layers.{}.moe.router.gate.bias",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }

    def get_hf_storage_reader(
        self, path: str, from_quantized: bool = False
    ) -> HuggingFaceStorageReader:
        """
        Override default get_hf_storage_reader function to return QuantizedHFStorageReader.
        """
        if from_quantized:
            from torch.distributed.checkpoint.quantized_hf_storage import (
                QuantizedHuggingFaceStorageReader,
            )

            # NOTE: Now we use Quantized HF storage reader to read GPT-OSS model where
            # expert weights are saved in MXFP4 format.
            # If loading checkpoints without quantization, use HuggingFaceStorageReader instead
            return QuantizedHuggingFaceStorageReader(
                path=path,
                thread_count=4,
            )
        else:
            return HuggingFaceStorageReader(path)

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Convert from a tt model state dict to a hf format state dict.

        For unquantized exports, transposes expert weights from TorchTitan format
        (experts, out_dim, in_dim) to HuggingFace format (experts, in_dim, out_dim)
        for compatibility with transformers/vLLM.

        For loading from quantized checkpoints, the QuantizedHuggingFaceStorageReader
            will handle dequantization during load.

        Warning: Conversion does not support saving to mxfp4 quantization format.
                 One can save into unquantized hf checkpoints with last_save_in_hf = true.
        """
        to_hf_map = {v: k for k, v in self.from_hf_map.items()}
        hf_state_dict = {}

        for key, value in state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                if abstract_key not in to_hf_map:
                    continue
                # pyrefly: ignore
                layer_num = re.search(r"\d+", key).group(0)
                hf_key = to_hf_map[abstract_key]
                hf_key = hf_key.format(layer_num)

                # Transpose expert weights for HF format (OSS package compatibility)
                # TorchTitan: (experts, out_dim, in_dim) -> HF: (experts, in_dim, out_dim)
                if 'moe.experts' in key and ('mlp1_weight' in key or 'mlp2_weight' in key):
                    if value.ndim == 3:  # Weight tensor (not bias)
                        value = value.transpose(1, 2).contiguous()
                        logger.info(f"Transposed for HF export: {key} -> {hf_key} {tuple(value.shape)}")

                hf_state_dict[hf_key] = value
            else:
                if key not in to_hf_map:
                    continue
                hf_key = to_hf_map[key]
                hf_state_dict[hf_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Convert from hf format state dict to tt model state dict.

        Skips quantization-specific keys (e.g., *_scales) that don't exist
        in unquantized TorchTitan models.

        Transposes expert weights from HF format (experts, in_dim, out_dim)
        to TorchTitan format (experts, out_dim, in_dim).

        Warning:
            This function may mutate the input `hf_state_dict` by calling
            `hf_state_dict.clear()` when expert weight transposes are required.
            This is an intentional memory optimization to release ~12GB of
            original expert weights before broadcast, since the transposed
            weights are stored in a new dict. Callers should not rely on
            `hf_state_dict` contents after this call if loading HF checkpoints
            with MoE expert weights that require transposition.
        """

        logger.info(f"from_hf() called with {len(hf_state_dict)} keys")
        state_dict = {}
        transpose_count = 0

        # First pass: Detect if HF format by checking any gate_up_proj shape
        # HF format: dim[1] < dim[2] (in_dim < out_dim, i.e., 2880 < 5760)
        # TorchTitan format: dim[1] > dim[2] (out_dim > in_dim, i.e., 5760 > 2880)
        needs_transpose = None
        for key, value in hf_state_dict.items():
            if 'gate_up_proj' in key and '_blocks' not in key and value.ndim == 3:
                if value.shape[1] < value.shape[2]:
                    needs_transpose = True
                    logger.info(f"Detected HF format (needs transpose): {key} {tuple(value.shape)}")
                else:
                    needs_transpose = False
                    logger.info(f"Detected TorchTitan format (no transpose needed): {key} {tuple(value.shape)}")
                break

        # Default to transpose if we couldn't detect (safe assumption for HF checkpoints)
        if needs_transpose is None:
            needs_transpose = True
            logger.info("Could not detect format from gate_up_proj, defaulting to transpose")

        # Second pass: Convert all keys
        for key, value in hf_state_dict.items():
            # Skip quantization metadata (scales, zero_points, etc.)
            if 'scales' in key or 'zero_point' in key:
                continue

            if "layers" in key:
                # pyrefly: ignore
                layer_num = re.search(r"\d+", key).group(0)
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)

                # Skip if not in mapping (e.g., quantization artifacts)
                if abstract_key not in self.from_hf_map:
                    continue

                tt_key = self.from_hf_map[abstract_key]
                tt_key = tt_key.format(layer_num)

                # Transpose expert weights from HF format to TorchTitan format
                # Expected shapes after transpose (for 20B model):
                #   gate_up_proj: (32, 5760, 2880) = (experts, hidden_dim*2, dim)
                #   down_proj: (32, 2880, 2880) = (experts, dim, hidden_dim)
                if 'mlp.experts' in key and ('gate_up_proj' in key or 'down_proj' in key):
                    # Skip MXFP4 blocks - they're already correct after dequantization
                    if '_blocks' in key:
                        pass  # MXFP4 dequant produces correct orientation
                    # Check if this is unquantized (3D) weight tensor
                    elif value.ndim == 3 and not key.endswith('_bias') and needs_transpose:
                        logger.info(f"Transposing HF->TorchTitan: {key} {tuple(value.shape)}")
                        value = value.transpose(-2, -1).contiguous()
                        transpose_count += 1
                        logger.info(f"  Result: {tuple(value.shape)}")

                state_dict[tt_key] = value
            else:
                # Skip if not in mapping
                if key not in self.from_hf_map:
                    continue

                tt_key = self.from_hf_map[key]
                state_dict[tt_key] = value

        # If we transposed expert weights, free memory before broadcast
        if transpose_count > 0:
            import gc
            # Clear input dict to release ~12GB of original expert weights
            hf_state_dict.clear()
            gc.collect()
            logger.info(f"Freed memory after {transpose_count} transposes (before broadcast)")

        return state_dict

    def _dequantize_mxfp4(
        self,
        blocks: torch.Tensor,
        scales: torch.Tensor,
        target_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """
        Dequantize MXFP4 blocks and scales to full precision tensor.

        Adapted from OpenAI's gpt_oss reference implementation.

        Args:
            blocks: Quantized weight blocks, shape [..., G, B] where G=groups, B=16 bytes
            scales: Scale exponents, shape [..., G] in uint8 (subtract 127 for actual exp)
            target_dtype: Output dtype (default bfloat16)

        Returns:
            Dequantized tensor of shape [..., G * B * 2] (2 values per byte)
        """
        scales = scales.to(torch.int32) - 127

        assert blocks.shape[:-1] == scales.shape, (
            f"Block shape {blocks.shape} doesn't match scale shape {scales.shape}"
        )

        lut = torch.tensor(FP4_VALUES, dtype=target_dtype, device=blocks.device)

        *prefix_shape, G, B = blocks.shape
        rows_total = math.prod(prefix_shape) * G

        blocks_flat = blocks.reshape(rows_total, B)
        scales_flat = scales.reshape(rows_total, 1)

        out = torch.empty(rows_total, B * 2, dtype=target_dtype, device=blocks.device)

        # Extract low and high nibbles, map through lookup table
        idx_lo = (blocks_flat & 0x0F).to(torch.long)
        idx_hi = (blocks_flat >> 4).to(torch.long)

        # Interleave: [lo0, hi0, lo1, hi1, ...] for swiglu compatibility
        out[:, 0::2] = lut[idx_lo]
        out[:, 1::2] = lut[idx_hi]

        # Apply scale (2^exp multiplication)
        torch.ldexp(out, scales_flat, out=out)

        return out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)

    def load_hf_safetensors_direct(
        self,
        checkpoint_path: str,
        target_dtype: torch.dtype = torch.float32,
    ) -> dict[str, torch.Tensor]:
        """
        Load HuggingFace safetensors checkpoint with manual MXFP4 dequantization.

        This bypasses the buggy QuantizedHuggingFaceStorageReader.dcp.load() which
        doesn't correctly handle the metadata shape update for MXFP4 tensors.

        Args:
            checkpoint_path: Path to HuggingFace checkpoint directory
            target_dtype: Target dtype for dequantization (default float32 for best precision)
                         Model dtype will be applied during set_model_state_dict()

        Returns:
            State dict with TorchTitan key names and dequantized weights
        """
        # Load index to find tensor locations
        index_path = os.path.join(checkpoint_path, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})

        # Group tensors by file
        file_to_tensors: dict[str, list[str]] = {}
        for tensor_name, file_name in weight_map.items():
            if file_name not in file_to_tensors:
                file_to_tensors[file_name] = []
            file_to_tensors[file_name].append(tensor_name)

        # Load all tensors with MXFP4 dequantization
        hf_state_dict: dict[str, torch.Tensor] = {}
        loaded_count = 0
        mxfp4_count = 0

        for file_name, tensor_names in file_to_tensors.items():
            file_path = os.path.join(checkpoint_path, file_name)
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for name in tensor_names:
                    # Skip scale tensors - they're loaded with their blocks
                    if name.endswith("_scales"):
                        continue

                    if name.endswith("_blocks"):
                        # MXFP4 quantized tensor - need to dequantize
                        blocks = f.get_tensor(name)
                        scales_name = name.replace("_blocks", "_scales")
                        scales = f.get_tensor(scales_name)

                        # Dequantize to target dtype (fp32 for best precision)
                        dequantized = self._dequantize_mxfp4(blocks, scales, target_dtype)

                        # Explicitly verify dtype and pin memory (match unquantized path)
                        if dequantized.dtype != target_dtype and dequantized.is_floating_point():
                            dequantized = dequantized.to(target_dtype)
                        # Pin memory for fast broadcast transfers
                        dequantized = dequantized.pin_memory()

                        hf_state_dict[name] = dequantized
                        mxfp4_count += 1
                    else:
                        # Regular tensor - load directly
                        tensor = f.get_tensor(name)
                        # Explicit dtype cast (same as unquantized path)
                        if tensor.dtype != target_dtype and tensor.is_floating_point():
                            tensor = tensor.to(target_dtype)
                        # Pin memory for fast broadcast transfers
                        tensor = tensor.pin_memory()
                        hf_state_dict[name] = tensor

                    loaded_count += 1

        logger.info(
            f"Loaded {loaded_count} tensors from HF checkpoint "
            f"({mxfp4_count} MXFP4 dequantized)"
        )

        # Free memory before calling from_hf() and broadcast
        # Dequantization creates large temp buffers that are no longer needed
        if mxfp4_count > 0:
            import gc
            gc.collect()
            logger.info(f"Freed memory after MXFP4 dequantization (before from_hf)")

        # Convert HF keys to TorchTitan keys
        return self.from_hf(hf_state_dict)
