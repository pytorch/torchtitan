# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import json
import math
import os
import pprint
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torchtitan.components.checkpoint import MODEL
from torchtitan.config_manager import ConfigManager, JobConfig
from torchtitan.tools.logging import init_logger, logger
from torchtitan.train import Trainer


def extract_layer_number_expert_number(s):
    import re

    match_layer = re.search(r"layers\.(\d+)", s)
    match_expert = re.search(r"experts\.(\d+)", s)
    layer_id = int(match_layer.group(1)) if match_layer else None
    expert_id = int(match_expert.group(1)) if match_expert else None
    return layer_id, expert_id


def is_quantization_tensor(fqn: str) -> bool:
    """
    Check if a tensor is related to quantization. Eg, model.layers.0.mlp.down_proj.weight_scale_inv
    """
    return any(
        suffix in fqn
        for suffix in [
            "weight_scale_inv",
            "weight_scale",
            "_scale",
        ]
    )


# Dictionary to map HF expert projection types to TorchTitan types
expert_mapping = {
    "gate_proj.weight": "w1",
    "up_proj.weight": "w3",
    "down_proj.weight": "w2",
}


def convert_to_titan_fqns(fqn: str) -> list[str]:
    """Converts a fqn from the stored checkpoint to the fqn in the TorchTitan model."""

    # Skip quantization-related tensors
    if is_quantization_tensor(fqn):
        return []

    layer, expert = extract_layer_number_expert_number(fqn)

    if layer is None:
        if "embed_tokens.weight" in fqn:
            return ["tok_embeddings.weight"]
        elif "norm.weight" in fqn:
            return ["norm.weight"]
        elif "lm_head.weight" in fqn:
            return ["output.weight"]
        else:
            raise ValueError(f"Unknown fqn {fqn}")

    # MoE layer
    # 1) Experts's weights -> Need to fuse to GroupedExperts
    if expert is not None and "mlp.experts" in fqn:
        # Check if this is one of the projection weights we need to handle
        for proj_type, titan_proj_type in expert_mapping.items():
            if f"mlp.experts.{expert}.{proj_type}" in fqn:
                # Store the mapping for later concatenation
                titan_fqn = f"layers.{layer}.moe.experts.{titan_proj_type}"
                return [titan_fqn]

    # 2) Router's weights
    elif "mlp.gate.weight" in fqn:
        return [f"layers.{layer}.moe.router.gate.weight"]

    # 3) Shared expert's weights
    elif "mlp.shared_experts.down_proj.weight" in fqn:
        return [f"layers.{layer}.moe.shared_expert.w2"]
    elif "mlp.shared_experts.gate_proj.weight" in fqn:
        return [f"layers.{layer}.moe.shared_expert.w1"]
    elif "mlp.shared_experts.up_proj.weight" in fqn:
        return [f"layers.{layer}.moe.shared_expert.w3"]

    # Dense Layer
    # down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    # self.w2(F.silu(self.w1(x)) * self.w3(x))
    elif "mlp.gate_proj.weight" in fqn:
        return [f"layers.{layer}.feed_forward.w1.weight"]
    elif "mlp.down_proj.weight" in fqn:
        return [f"layers.{layer}.feed_forward.w2.weight"]
    elif "mlp.up_proj.weight" in fqn:
        return [f"layers.{layer}.feed_forward.w3.weight"]

    # Transformer layer
    elif "input_layernorm.weight" in fqn:
        return [f"layers.{layer}.attention_norm.weight"]
    elif "post_attention_layernorm.weight" in fqn:
        return [f"layers.{layer}.ffn_norm.weight"]
    # Attention layer
    elif "self_attn.q_a_proj" in fqn:
        return [f"layers.{layer}.attention.wq_a.weight"]
    elif "self_attn.q_a_layernorm" in fqn:
        return [f"layers.{layer}.attention.q_norm.weight"]
    elif "self_attn.q_b_proj" in fqn:
        return [f"layers.{layer}.attention.wq_b.weight"]
    elif "self_attn.kv_a_proj_with_mqa" in fqn:
        return [f"layers.{layer}.attention.wkv_a.weight"]
    elif "self_attn.kv_a_layernorm" in fqn:
        return [f"layers.{layer}.attention.kv_norm.weight"]
    elif "self_attn.kv_b_proj" in fqn:
        return [f"layers.{layer}.attention.wkv_b.weight"]
    elif "self_attn.o_proj" in fqn:
        return [f"layers.{layer}.attention.wo.weight"]

    else:
        raise ValueError(f"Unknown fqn {fqn}")


def convert_to_hf_shape(
    fqn: str, titan_fqns: list[str], dtensor: DTensor
) -> torch.Size:
    if "shared_expert" in fqn:
        s = dtensor.shape  # torchtitan shape: [1, s[1], s[2]]
        # TODO: this is not right but I have to do this to load the checkpoint.
        return torch.Size((s[2], s[1]))
    elif "mlp.experts" in fqn:
        # For MoE expert weights, the HF checkpoint has 2D tensors for each expert
        # while TorchTitan has a single 3D tensor for all experts
        s = dtensor.shape
        if len(s) == 3:  # This is a 3D tensor [num_experts, dim1, dim2]
            # Return the shape of a single expert. And we are using nn.Parameter,
            # while HF is using nn.Linear. So we need to transpose the weight.
            return torch.Size([s[2], s[1]])
    return dtensor.shape


def convert_to_titan_tensors(fqn: str, full_tensor: torch.Tensor) -> list[torch.Tensor]:
    if "shared_experts" in fqn:
        # HF shape: (s[2], s[1]) -> torchtitan shape: (1, s[1], s[2])
        full_tensor = full_tensor.transpose(1, 0)
        full_tensors = [full_tensor.unsqueeze(0)]
    else:
        full_tensors = [full_tensor]
    return full_tensors


@dataclass
class _Assignment:
    loader_id: int
    filename: str
    fqns: list[str]
    shapes: list[torch.Size]
    dtypes: list[torch.dtype]


@dataclass
class _AssignmentRound:
    loader_assignments: dict[int, _Assignment]  # List of assignments for each loader


@dataclass
class TensorMetadata:
    fqn: str
    shape: torch.Size
    dtype: torch.dtype


class CheckpointConverter:
    def __init__(
        self,
        process_group: dist.ProcessGroup,
        path: str,
        token: Optional[str] = None,
        loader_every_n_ranks: int = 8,
    ) -> None:
        self.path = path
        self.token = token
        self.pg = process_group
        self.my_rank = dist.get_rank(self.pg)

        self.loader_every_n_ranks = loader_every_n_ranks
        self.loader_id = self.my_rank // loader_every_n_ranks
        self.should_load = self.my_rank % loader_every_n_ranks == 0
        self.total_loader = dist.get_world_size(self.pg) // loader_every_n_ranks

        self.titan_fqn_to_stored_fqn: dict[str, str] = {}
        self.stored_fqn_to_titan_fqn: dict[str, list[str]] = {}
        self.total_send_bytes = 0
        self.total_recv_bytes = 0

        # Dictionary to track expert weights for concatenation
        self.expert_weights_by_layer = {}  # {layer: {type: {expert_id: tensor}}}

    def convert(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        begin = time.time()
        # Load HF fqns
        self._load_metadata()
        # Create the mapping from the stored checkpoint keys to TorchTitan keys.
        self._create_fqn_mappings(state_dict)
        rounds = self._get_load_assignments(state_dict)

        logger.info(f"Got {len(rounds)} rounds of assignments.")
        for idx, assignments in enumerate(rounds):
            loader_assignments = assignments.loader_assignments
            loaded_state_dict = None
            # Let each loader to load its own data and move to its GPU.
            logger.info(f"Loading round {idx}")
            for i in range(self.total_loader):
                # This loader doesn't have any loading assignment for this round.
                if i not in loader_assignments:
                    continue
                # This rank is not the loader
                if i != self.loader_id or not self.should_load:
                    continue

                # Loaded state_dict should be HF state_dict
                loaded_state_dict = self._load_round(loader_assignments[i])

            torch.cuda.synchronize()
            logger.info(f"Loading round {idx} finished")
            for i in range(self.total_loader):
                if i not in loader_assignments:
                    continue

                logger.info(f"Resharding round {idx} loader {i} data. ")
                if i == self.loader_id and self.should_load:
                    # This rank is the loader. It needs to send the loaded data to
                    # the other ranks.
                    assert loaded_state_dict is not None
                    results = self._reshard_send(
                        loader_assignments[i], loaded_state_dict
                    )
                else:
                    results = self._reshard_receive(loader_assignments[i], state_dict)
                torch.cuda.synchronize()

                logger.info(f"Communication round {idx} loader {i} is done.")
                self._reshard(results, state_dict)
                logger.info(f"Resharding round {idx} loader {i} is done.")
                self._reshard(results, state_dict)
                torch.cuda.synchronize()

        dist.barrier()
        torch.cuda.synchronize()
        logger.info(f"Checkpoint conversion took {time.time() - begin:.2f} seconds.")
        logger.info(f"Total send bytes: {self.total_send_bytes / 1e9:.2f} GB")
        logger.info(f"Total recv bytes: {self.total_recv_bytes / 1e9:.2f} GB")
        return state_dict

    def _load_metadata(self) -> None:
        metadata_path = os.path.join(self.path, "model.safetensors.index.json")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)["weight_map"]

        # # TODO: Limit the layers for conversion. Now only keep the first 5 layers: 3 dense layers and 2 MoE layers
        # keys_to_remove = []
        # for key in self.metadata:
        #
        #     for layer in range(5, 62):
        #         if f"model.layers.{layer}" in key:
        #             keys_to_remove.append(key)
        #             break
        # for key in keys_to_remove:
        #     self.metadata.pop(key)

    def _create_fqn_mappings(self, state_dict: dict[str, torch.Tensor]) -> None:
        if not self.metadata:
            return

        # Create the mapping from the stored checkpoint keys to TorchTitan keys.
        for fqn in list(self.metadata.keys()):
            # Skip quantization-specific tensors
            if is_quantization_tensor(fqn):
                # logger.info(f"Skipping quantization tensor: {fqn}")
                self.metadata.pop(fqn)
                continue

            # We don't know how to process _extra_state
            # And we don't have e_score_correction_bias in torchtitan implementation
            if (
                "_extra_state" in fqn
                or "mlp.gate.e_score_correction_bias" in fqn
                or "tokens_per_expert" in fqn
                or "expert_bias" in fqn
            ):
                self.metadata.pop(fqn)
                continue

            titan_fqns = convert_to_titan_fqns(fqn)

            # Skip if no mapping was found (e.g., for quantization tensors)
            if not titan_fqns:
                self.metadata.pop(fqn)
                continue

            if titan_fqns[0] not in state_dict:
                for titan_fqn in titan_fqns:
                    assert titan_fqn not in state_dict
                self.metadata.pop(fqn)
                continue

            self.stored_fqn_to_titan_fqn[fqn] = titan_fqns
            for titan_fqn in titan_fqns:
                self.titan_fqn_to_stored_fqn[titan_fqn] = fqn

        # print("self.titan_fqn_to_stored_fqn.keys(): ", self.titan_fqn_to_stored_fqn.keys())

        torchtitan_extra = sorted(
            list(set(state_dict.keys()) - set(self.titan_fqn_to_stored_fqn.keys()))
        )
        converted_extra = sorted(
            list(set(self.titan_fqn_to_stored_fqn.keys()) - set(state_dict.keys()))
        )
        state_dict_keys = [
            x
            for x in state_dict.keys()
            if "expert_bias" not in x and "tokens_per_expert" not in x
        ]

        assert set(state_dict_keys) == set(self.titan_fqn_to_stored_fqn.keys()), (
            f"{pprint.pformat(torchtitan_extra)}",
            f"{pprint.pformat(converted_extra)}",
        )

    def _get_load_assignments(
        self, state_dict: dict[str, Any]
    ) -> list[_AssignmentRound]:
        if self.my_rank == 0:
            filename_to_metas = defaultdict(list)
            for hf_fqn, filename in self.metadata.items():
                titan_fqns = self.stored_fqn_to_titan_fqn[hf_fqn]
                # The shape is wrong for the following keys. We need to convert it.
                shape = convert_to_hf_shape(
                    hf_fqn, titan_fqns, state_dict[titan_fqns[0]]
                )
                # Here we are expecting the converted weight tensor to be bfloat16.
                # This is to save GPU memory and time for communication
                meta = TensorMetadata(
                    fqn=hf_fqn,
                    shape=shape,
                    dtype=torch.bfloat16,  # TODO: don't hardcode this
                )
                filename_to_metas[filename].append(meta)

            loader_filename_to_metas = [{} for _ in range(self.total_loader)]
            for idx, (filename, metas) in enumerate(filename_to_metas.items()):
                loader_id = idx % self.total_loader
                loader_filename_to_metas[loader_id][filename] = metas

            rounds = []
            while any(len(remain) > 0 for remain in loader_filename_to_metas):
                round_assignment = _AssignmentRound(loader_assignments={})
                for loader_id in range(self.total_loader):
                    if not loader_filename_to_metas[loader_id]:
                        continue

                    filename, metas = loader_filename_to_metas[loader_id].popitem()
                    round_assignment.loader_assignments[loader_id] = _Assignment(
                        filename=filename,
                        fqns=[meta.fqn for meta in metas],
                        shapes=[meta.shape for meta in metas],
                        dtypes=[meta.dtype for meta in metas],
                        loader_id=loader_id,
                    )

                rounds.append(round_assignment)

            object_list: list[Any] = [
                rounds,
                self.titan_fqn_to_stored_fqn,
                self.stored_fqn_to_titan_fqn,
            ]
        else:
            object_list = [None, None, None]

        dist.broadcast_object_list(object_list, src=0, group=self.pg)
        rounds = object_list[0]
        self.titan_fqn_to_stored_fqn = object_list[1]
        self.stored_fqn_to_titan_fqn = object_list[2]
        return rounds

    def _dequantize_weight(
        self, weight: torch.Tensor, scale_inv: torch.Tensor, dtype=torch.bfloat16
    ) -> torch.Tensor:
        """
        Dequantize an FP8 weight tensor using its weight_scale_inv tensor.

        The FP8 weight file includes a weight_scale_inv field, which stores the
        dequantization scale for each weight block.

        Dequantization Formula:
        - If the weight block is not aligned to 128, it is zero-padded to 128 before
          calculating the scale. After quantization, the padded portion is removed.
        - The dequantization process is performed as: (128x128 weight block) * weight_scale_inv.
        - This enables online quantization at a granularity of per-token-per-128-channel.
        """
        # Convert to float32 for computation
        float_weight = weight.to(torch.float32)

        # Get original dimensions
        orig_shape = weight.shape

        # Fixed block size of 128x128 as specified in the algorithm
        BLOCK_SIZE = 128

        # Calculate number of blocks needed
        block_rows = (orig_shape[0] + BLOCK_SIZE - 1) // BLOCK_SIZE
        block_cols = (orig_shape[1] + BLOCK_SIZE - 1) // BLOCK_SIZE

        # Verify scale_inv shape matches expected block dimensions
        expected_scale_shape = (block_rows, block_cols)
        if scale_inv.shape != expected_scale_shape:
            logger.warning(
                f"scale_inv shape {scale_inv.shape} doesn't match expected shape {expected_scale_shape}"
            )

        dequantized = torch.zeros(orig_shape, dtype=dtype, device="cuda")

        # Apply scaling factors to each block
        for i in range(block_rows):
            row_start = i * BLOCK_SIZE
            row_end = min(row_start + BLOCK_SIZE, orig_shape[0])

            for j in range(block_cols):
                col_start = j * BLOCK_SIZE
                col_end = min(col_start + BLOCK_SIZE, orig_shape[1])

                # Get the block
                block = float_weight[row_start:row_end, col_start:col_end]

                scale = scale_inv[i, j]
                block = block * scale

                # Explicitly convert block to dtype
                block_converted = block.to(dtype=dtype)
                # Store the dequantized block
                dequantized[row_start:row_end, col_start:col_end] = block_converted

        return dequantized

    def _load_round(self, assignment: _Assignment) -> dict[str, Any]:
        from safetensors.torch import load_file as hf_load_file

        path = os.path.join(self.path, assignment.filename)
        state_dict = hf_load_file(path)

        # Group quantized weights with their scales
        weight_groups = defaultdict(dict)
        missing_scale_inv = []

        # First pass: collect weights and available scale_inv values
        for k, v in state_dict.items():
            base_name = k.split(".weight")[0]
            if (
                f"{base_name}.weight" in assignment.fqns
            ):  # HF fqn, but not inlucde weight_scale_inv
                # Extract base name without quantization suffix
                if ".weight_" in k:
                    base_name = k.split(".weight_")[0]
                    weight_groups[base_name][k] = v.to(device="cuda")
                elif ".weight" in k:
                    base_name = k.split(".weight")[0]
                    weight_groups[base_name][k] = v.to(device="cuda")
                    # Check if scale_inv is missing
                    if f"{base_name}.weight_scale_inv" not in state_dict:
                        if base_name in ["model.embed_tokens", "model.norm", "lm_head"]:
                            # These are the only two cases where scale_inv is not needed
                            continue
                        if (
                            "input_layernorm" in base_name
                            or "post_attention_layernorm" in base_name
                        ):
                            # These are the only two cases where scale_inv is not needed
                            continue
                        missing_scale_inv.append(base_name)
                        logger.info(
                            f"Need to look for {base_name}.weight_scale_inv in other files"
                        )
                else:
                    # Regular tensor
                    weight_groups[k][k] = v.to(device="cuda")

        # If we have missing scale_inv values, try to find them in other files
        # Simplified approach: Just check the next file in the metadata list
        if missing_scale_inv:
            logger.info(
                f"Looking for {len(missing_scale_inv)} missing scale_inv values"
            )

            # Get all available files from metadata
            all_files = sorted(list(set(self.metadata.values())))
            current_file = assignment.filename

            # Find the next file after the current one
            try:
                current_idx = all_files.index(current_file)
                next_file = all_files[(current_idx + 1) % len(all_files)]

                try:
                    other_path = os.path.join(self.path, next_file)
                    other_state_dict = hf_load_file(other_path)

                    # Check for each missing scale_inv
                    for base_name in missing_scale_inv:
                        scale_inv_key = f"{base_name}.weight_scale_inv"
                        if scale_inv_key in other_state_dict:
                            logger.info(f"Found {scale_inv_key} in {next_file}")
                            weight_groups[base_name][scale_inv_key] = other_state_dict[
                                scale_inv_key
                            ].to(device="cuda")
                        else:
                            logger.warning(
                                f"Could not find {scale_inv_key} in next file"
                            )

                except Exception as e:
                    logger.warning(f"Error loading file {next_file}: {e}")
            except ValueError:
                logger.warning(
                    f"Could not find current file {current_file} in metadata files"
                )

        # Process and dequantize weights
        result_dict = {}
        for base_name, tensors in weight_groups.items():
            # Check if this is a quantized weight that needs dequantization
            weight_key = (
                f"{base_name}.weight" if f"{base_name}.weight" in tensors else base_name
            )
            scale_inv_key = f"{base_name}.weight_scale_inv"

            if weight_key in tensors and scale_inv_key in tensors:
                # This is a quantized weight that needs dequantization
                weight = tensors[weight_key]
                scale_inv = tensors[scale_inv_key]

                dequantized_weight = self._dequantize_weight(
                    weight, scale_inv, dtype=torch.bfloat16
                )

                result_dict[weight_key] = dequantized_weight
            else:
                # Regular tensors or already in full precision
                for k, v in tensors.items():
                    if k in assignment.fqns and not is_quantization_tensor(k):
                        result_dict[k] = v

        logger.info(f"Processed {len(result_dict)} tensors in this round")
        return result_dict

    def _reshard_send(
        self,
        assignment: _Assignment,
        loaded_state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        assert self.loader_id == assignment.loader_id
        rank = self.loader_id * self.loader_every_n_ranks
        assert rank == self.my_rank

        logger.info(f"Sending {assignment.filename} from {rank} {self.loader_id}")

        # Send each tensor individually
        for fqn in assignment.fqns:
            # Skip quantization tensors
            if is_quantization_tensor(fqn):
                continue

            tensor = loaded_state_dict[fqn]
            logger.info(
                f"Sending tensor {fqn} with shape {tensor.shape} and dtype {tensor.dtype}"
            )

            # Send tensor shape first (needed for receiving side to allocate correctly)
            tensor_shape = torch.tensor(tensor.shape, dtype=torch.long, device="cuda")
            dist.broadcast(tensor_shape, src=rank, group=self.pg)

            # Flatten and send the actual tensor
            flat_tensor = tensor.flatten()
            dist.broadcast(flat_tensor, src=rank, group=self.pg)
            self.total_send_bytes += flat_tensor.numel() * flat_tensor.element_size()

        return loaded_state_dict

    def _reshard_receive(
        self, assignment: _Assignment, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        rank = assignment.loader_id * self.loader_every_n_ranks
        logger.info(f"Receiving {assignment.filename} from {rank}")

        ret: dict[str, torch.Tensor] = {}

        # Receive each tensor individually
        for i, (fqn, expected_shape, dtype) in enumerate(
            zip(assignment.fqns, assignment.shapes, assignment.dtypes)
        ):
            # Skip quantization tensors
            if is_quantization_tensor(fqn):
                continue

            # Log the tensor we're about to receive
            size = math.prod(expected_shape)
            logger.info(
                f"Receiving tensor {i + 1}/{len(assignment.fqns)}: {fqn}, Shape: {expected_shape}, Size: {size}, \
                    GB: {size * 2 / 1e9}, expected dtype: {dtype}"
            )

            # Receive tensor shape first
            tensor_shape = torch.empty(
                len(expected_shape), dtype=torch.long, device="cuda"
            )
            dist.broadcast(tensor_shape, src=rank, group=self.pg)
            actual_shape = tuple(tensor_shape.tolist())

            # Verify shape matches expected shape
            if actual_shape != expected_shape:
                logger.warning(
                    f"Shape mismatch for {fqn}: expected {expected_shape}, got {actual_shape}"
                )

            # Allocate memory for this tensor only
            n_ele = math.prod(actual_shape)
            flat_tensor = torch.empty(n_ele, dtype=dtype, device="cuda")

            # Receive the flattened tensor
            dist.broadcast(flat_tensor, src=rank, group=self.pg)
            self.total_recv_bytes += flat_tensor.numel() * flat_tensor.element_size()

            # Reshape and store
            tensor = flat_tensor.view(actual_shape)
            ret[fqn] = tensor

        return ret

    def _reshard(
        self,
        result: dict[str, torch.Tensor],
        state_dict: dict[str, torch.Tensor],
    ) -> None:
        def _inplace_copy(fqn: str, full_tensors: list[torch.Tensor]):
            titan_fqns = self.stored_fqn_to_titan_fqn[fqn]
            assert len(titan_fqns) == len(full_tensors)
            for titan_fqn, full_tensor in zip(titan_fqns, full_tensors):
                dtensor = state_dict[titan_fqn]
                assert isinstance(dtensor, DTensor)

                # Special handling for MoE expert weights
                if "moe.experts" in titan_fqn:
                    # Extract layer and projection type
                    parts = titan_fqn.split(".")
                    layer = int(parts[1])
                    proj_type = parts[-1]  # w1, w2, or w3

                    # Extract expert ID from the original fqn
                    expert_match = re.search(r"experts\.(\d+)", fqn)
                    if expert_match:
                        expert_id = int(expert_match.group(1))

                        # Store this expert tensor for later concatenation
                        if layer not in self.expert_weights_by_layer:
                            self.expert_weights_by_layer[layer] = {}
                        if proj_type not in self.expert_weights_by_layer[layer]:
                            self.expert_weights_by_layer[layer][proj_type] = {}

                        self.expert_weights_by_layer[layer][proj_type][
                            expert_id
                        ] = full_tensor

                        # We'll handle the actual copying after collecting all experts
                        continue

                # Regular tensor handling (non-MoE experts)
                assert dtensor.shape == full_tensor.shape, (
                    (fqn, titan_fqn),
                    dtensor.shape,
                    full_tensor.shape,
                )
                shape, offset = compute_local_shape_and_global_offset(
                    full_tensor.shape, dtensor.device_mesh, dtensor.placements
                )
                slices = [
                    slice(cur_offset, cur_offset + cur_shape)
                    for cur_shape, cur_offset in zip(shape, offset)
                ]
                logger.debug(
                    f"Copying {titan_fqn} with {slices=} {dtensor._local_tensor.shape=} "
                    f"{shape=} {offset=} {self.my_rank=} {dtensor.shape=} {full_tensor.shape=} "
                    f"{dtensor.placements=} {dtensor.device_mesh=} "
                )
                dtensor.to_local().copy_(full_tensor[slices].to(dtensor.dtype))

        for fqn, full_tensor in result.items():
            full_tensors = convert_to_titan_tensors(fqn, full_tensor)
            _inplace_copy(fqn, full_tensors)

        # After processing all tensors in this round, check if we can concatenate any expert weights
        self._concatenate_expert_weights(state_dict)

    def _concatenate_expert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """
        Concatenate collected expert weights into 3D tensors for TorchTitan.
        This should be called after all tensors for a layer have been processed.
        """
        for layer, proj_types in list(self.expert_weights_by_layer.items()):
            for proj_type, experts in list(proj_types.items()):
                # Check if we have collected all experts for this layer and projection type
                titan_fqn = f"layers.{layer}.moe.experts.{proj_type}"

                # Get the expected number of experts from the state_dict tensor shape
                if titan_fqn in state_dict:
                    dtensor = state_dict[titan_fqn]
                    expected_num_experts = dtensor.shape[0]

                    # If we have all the experts, concatenate them
                    if len(experts) == expected_num_experts:
                        logger.info(
                            f"Concatenating {len(experts)} experts for {titan_fqn}"
                        )

                        sorted_expert_ids = sorted(experts.keys())
                        sorted_experts = [experts[i] for i in sorted_expert_ids]

                        # print info
                        for i in range(len(sorted_experts)):
                            logger.info(
                                f"Expert {sorted_expert_ids[i]} - Shape: {sorted_experts[i].shape}, \
                                    Dtype: {sorted_experts[i].dtype}, Device: {sorted_experts[i].device}"
                            )

                        stacked_tensor = torch.stack(sorted_experts, dim=0).transpose(
                            1, 2
                        )

                        # Copy to the state_dict
                        shape, offset = compute_local_shape_and_global_offset(
                            stacked_tensor.shape,
                            dtensor.device_mesh,
                            dtensor.placements,
                        )
                        slices = [
                            slice(cur_offset, cur_offset + cur_shape)
                            for cur_shape, cur_offset in zip(shape, offset)
                        ]

                        # Target shape is (num_experts, hidden_size, hidden_size)
                        # stack_tensor: ([256, 2048, 7168])
                        logger.info(
                            f"Copying concatenated experts to {titan_fqn} with stacked_tensor shape {stacked_tensor.shape}, \
                                titan dtensor_shape: {dtensor.shape}"
                        )
                        dtensor.to_local().copy_(
                            stacked_tensor[slices].to(dtensor.dtype)
                        )

                        # Remove these experts from the tracking dict to free memory
                        del self.expert_weights_by_layer[layer][proj_type]
                        if not self.expert_weights_by_layer[layer]:
                            del self.expert_weights_by_layer[layer]


if __name__ == "__main__":
    init_logger()

    @dataclass
    class Checkpoint:
        convert_path: str = ""
        """Specify the path of the target checkpoint to convert."""

        convert_hf_token: str = ""
        """Specify the HuggingFace token to use when downloading checkpoints."""

        convert_load_every_n_ranks: int = 1
        """
        Specify the interval at which ranks are assigned to load checkpoints.

        For example, if this number is 4, then ranks 0, 4, 8, ... will load the
        checkpoint. Each loader is responsible for loading one file. If there
        are more loaders than files, only the first few loaders will be assigned
        to load the checkpoint. The default value is 8.
        """

    @dataclass
    class MyJobConfig:
        checkpoint: Checkpoint = field(default_factory=Checkpoint)

    MergedJobConfig = ConfigManager._merge_configs(JobConfig, MyJobConfig)
    config_manager = ConfigManager(config_cls=MergedJobConfig)
    config = config_manager.parse_args()

    assert config.checkpoint.convert_path != ""

    trainer: Optional[Trainer] = None

    try:
        trainer = Trainer(config)
        if os.path.exists(trainer.checkpointer.folder):
            raise RuntimeError(
                "The checkpoint folder already exists. Abort to avoid overwriting "
                f"the checkpoint. {trainer.checkpointer.folder=}"
            )

        state_dict = trainer.checkpointer.states[MODEL].state_dict()

        size = 0
        for v in state_dict.values():
            size += v.numel() * v.element_size()
        logger.info(f"Total size of the model: {size / 1e9:.2f} GB")

        state_dict = CheckpointConverter(
            process_group=trainer.parallel_dims.world_mesh.get_group(),
            path=config.checkpoint.convert_path,
            token=config.checkpoint.convert_hf_token,
            loader_every_n_ranks=config.checkpoint.convert_load_every_n_ranks,
        ).convert(state_dict)

        class DummyModel:
            def __init__(self, state_dict: dict[str, torch.Tensor]) -> None:
                self._state_dict = state_dict

            def state_dict(self) -> dict[str, torch.Tensor]:
                return self._state_dict

        trainer.checkpointer.states[MODEL] = DummyModel(state_dict)
        trainer.checkpointer.last_save_model_only = True
        trainer.checkpointer.export_dtype = next(iter(state_dict.values())).dtype
        logger.info("Export checkpoint dtype: ", trainer.checkpointer.export_dtype)
        trainer.checkpointer.save(curr_step=0, last_step=True)
        time.sleep(2)
    finally:
        pass
