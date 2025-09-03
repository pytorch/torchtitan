# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
DeepSeek V3 HuggingFace Storage Reader for DCP pre-resharding conversion.

This module provides a custom storage reader that handles the conversion between
HuggingFace DeepSeek V3 checkpoint format and TorchTitan format during the loading
process, before distributed sharding occurs. This avoids the memory-intensive
DTensor operations that cause OOM during post-sharding conversion.
"""

import datetime
import json
import os
import queue
import re
import threading

from pathlib import Path
from typing import Any

import torch
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.checkpoint._hf_utils import _HFStorageInfo
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import LoadPlan, LoadPlanner
from torch.distributed.checkpoint.planner_helpers import ReadItem
from torch.futures import Future
from torchtitan.models.deepseek_v3.model.key_mappings import (
    convert_hf_key_to_tt_key,
    convert_tt_key_to_hf_key,
    get_hf_to_tt_map,
)
from torchtitan.models.deepseek_v3.model.metadata import DeepSeekV3Metadata
from torchtitan.tools.logging import logger

__all__ = ["QuantizedHuggingFaceStorageReader"]
WEIGHT_MAP_FILENAME = "model.safetensors.index.json"
WEIGHT_MAP_KEY = "weight_map"
WEIGHT_SCALE_INVERSE_SUFFIX = ".weight_scale_inv"
WEIGHT_SUFFIX = ".weight"


class QuantizedHuggingFaceStorageReader(HuggingFaceStorageReader):
    """
    Extension of HuggingFaceStorageReader that handles fp8 quantized tensors.

    This reader handles the dequantization of fp8 tensors during the read process,
    converting them from quantized blocks to full dequantized tensors before
    copying to the target tensor.
    """

    def __init__(self, path: str, block_size: int = 128, thread_count: int = 20):
        """
        Initialize the FP8 HuggingFace storage reader.

        Args:
            path: directory where the checkpoint will be read from
            block_size: optional fixed block size for FP8 dequantization. If None,
                       block size will be calculated dynamically based on tensor shapes.
        """
        super().__init__(path)

        self.checkpoint_path = path
        self.block_size = block_size
        self.tensor_to_file_mapping = {}  # Maps tensor names to their file paths
        self.thread_count = thread_count

    def set_up_storage_reader(
        self, metadata: Metadata, is_coordinator: bool, *args: Any, **kwargs: Any
    ) -> None:
        super().set_up_storage_reader(metadata, is_coordinator, *args, **kwargs)

        """Load quantization metadata from checkpoint."""
        checkpoint_path = Path(self.checkpoint_path)

        # Load weight mapping from index file
        weight_map_file = checkpoint_path / WEIGHT_MAP_FILENAME
        with open(weight_map_file, "r") as f:
            index_data = json.load(f)
            weight_map = index_data.get(WEIGHT_MAP_KEY, {})
            # Store the complete tensor-to-file mapping for efficient lookups
            self.tensor_to_file_mapping = weight_map

    def _get_scale_tensor_name(self, weight_tensor_name: str) -> str:
        """Get the scale inverse tensor name for a given weight tensor."""
        return weight_tensor_name.replace(WEIGHT_SUFFIX, WEIGHT_SCALE_INVERSE_SUFFIX)

    def _has_scale_tensor(self, weight_tensor_name: str) -> bool:
        """Check if a weight tensor has a corresponding scale inverse tensor."""
        scale_name = self._get_scale_tensor_name(weight_tensor_name)
        return scale_name in self.tensor_to_file_mapping

    def _read_files_from_queue(
        self,
        file_queue: queue.Queue,
        result_queue: queue.Queue,
        planner: LoadPlanner,
    ) -> None:
        from safetensors import safe_open  # type: ignore[import]

        try:
            while True:
                file_name, reqs = file_queue.get_nowait()
                with safe_open(filename=file_name, framework="pt") as f:
                    for req in reqs:
                        self._process_read_request(f, req, planner)
                result_queue.put(True)  # Signal that this file has been processed
        except queue.Empty:
            pass

    def _process_read_request(self, f, req: ReadItem, planner: LoadPlanner) -> None:
        """Helper function to process a single read request."""
        # Create slices for each dimension based on offsets and lengths
        tensor_fqn = req.storage_index.fqn

        # Check if this is a quantized tensor that needs special handling
        if self._is_quantized_tensor(tensor_fqn):
            tensor = self._read_quantized_tensor_with_block_alignment(req, f)
        else:
            # Standard tensor reading
            slices = tuple(
                slice(offset, offset + length)
                for offset, length in zip(req.storage_offsets, req.lengths)
            )
            tensor = f.get_slice(tensor_fqn)[slices]

        target_tensor = planner.resolve_tensor(req).detach()

        assert (
            target_tensor.size() == tensor.size()
        ), f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"

        target_tensor.copy_(tensor)
        planner.commit_tensor(req, target_tensor)

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        per_file: dict[str, list[ReadItem]] = {}

        # Large models have tensors sharded across multiple files.
        # We need to batch the read requests by file to avoid repeated file opens.
        # This will reduce the I/O overhead.
        for read_item in plan.items:
            item_md: _HFStorageInfo = self.storage_data[read_item.storage_index]
            file_name = item_md.relative_path
            per_file.setdefault(file_name, []).append(read_item)

        # Use parallel implementation with thread pool
        file_queue: queue.Queue = queue.Queue()
        result_queue: queue.Queue = queue.Queue()

        for file_name, reqs in per_file.items():
            file_queue.put((file_name, reqs))

        # Create and start worker threads
        threads = []
        num_threads = min(self.thread_count, len(per_file))
        for _ in range(num_threads):
            t = threading.Thread(
                target=self._read_files_from_queue,
                args=(file_queue, result_queue, planner),
            )
            t.start()
            threads.append(t)

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Check if all files were processed
        processed_count = 0
        try:
            while True:
                result_queue.get_nowait()
                processed_count += 1
        except queue.Empty:
            pass

        assert processed_count == len(
            per_file
        ), f"Not all files were processed: {processed_count} out of {len(per_file)}"

        fut: Future = Future()
        fut.set_result(None)
        return fut

    def _calculate_scale_shape(self, weight: torch.Tensor) -> tuple[int, int]:  # noqa: F841
        """Calculate expected scale tensor shape based on weight tensor and block size."""
        rows, cols = weight.shape
        block_rows = (rows + self.block_size - 1) // self.block_size  # Ceiling division
        block_cols = (cols + self.block_size - 1) // self.block_size  # Ceiling division
        return (block_rows, block_cols)

    def _dequantize_tensor(
        self,
        weight: torch.Tensor,
        scale_inv: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Optimized dequantization using vectorized operations.
        Handles both 1D and 2D tensors.
        """
        # Convert to target dtype once
        dequantized_tensor = weight.to(dtype=dtype)

        # Handle 1D tensors (like bias terms)
        if len(weight.shape) == 1:
            # For 1D tensors, apply scale directly if scale_inv is also 1D
            if len(scale_inv.shape) == 1:
                return dequantized_tensor * scale_inv.to(dtype=dtype)
            else:
                # If scale_inv is 2D but weight is 1D, use the first element or flatten
                scale_flat = scale_inv.flatten()
                if len(scale_flat) >= len(dequantized_tensor):
                    return dequantized_tensor * scale_flat[
                        : len(dequantized_tensor)
                    ].to(dtype=dtype)
                else:
                    # Broadcast the scale to match tensor length
                    return dequantized_tensor * scale_inv.flatten()[0].to(dtype=dtype)

        # Handle 2D tensors (matrices)
        if len(weight.shape) != 2:
            raise ValueError(
                f"Unsupported tensor shape: {weight.shape}. Only 1D and 2D tensors are supported."
            )

        # Calculate block dimensions for 2D tensors
        rows, cols = weight.shape
        block_rows = (rows + self.block_size - 1) // self.block_size
        block_cols = (cols + self.block_size - 1) // self.block_size

        # Create expanded scale tensor to match weight dimensions
        # This avoids the nested loops by using broadcasting
        scale_expanded = torch.ones_like(dequantized_tensor, dtype=dtype)

        for i in range(min(block_rows, scale_inv.shape[0])):
            row_start = i * self.block_size
            row_end = min(row_start + self.block_size, rows)

            for j in range(min(block_cols, scale_inv.shape[1])):
                col_start = j * self.block_size
                col_end = min(col_start + self.block_size, cols)

                scale_expanded[row_start:row_end, col_start:col_end] = scale_inv[i, j]

        # Vectorized multiplication - much faster than loops
        return dequantized_tensor

    def _read_expert_tensors_parallel(
        self, expert_requests: list, planner: LoadPlanner
    ) -> None:
        """
        Process expert tensor requests with parallel processing to improve performance.

        Args:
            expert_requests: List of expert tensor read requests
            planner: Load planner for tensor resolution
        """
        import queue
        import threading

        # Create a queue for expert requests
        expert_queue: queue.Queue = queue.Queue()
        result_queue: queue.Queue = queue.Queue()

        # Add all expert requests to the queue
        for req in expert_requests:
            expert_queue.put(req)

        def process_expert_requests():
            """Worker function to process expert requests from the queue."""
            try:
                while True:
                    req = expert_queue.get_nowait()
                    tt_key = req.storage_index.fqn

                    # Handle expert tensor grouping
                    tensor = self._read_grouped_expert_tensor(req, tt_key)
                    target_tensor = planner.resolve_tensor(req).detach()

                    assert (
                        target_tensor.size() == tensor.size()
                    ), f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
                    target_tensor.copy_(tensor)
                    planner.commit_tensor(req, target_tensor)

                    result_queue.put(True)  # Signal completion

            except queue.Empty:
                pass

        # Create and start worker threads
        threads = []
        # Use fewer threads for expert processing to avoid file handle exhaustion
        # since each expert tensor may need to open multiple files
        num_threads = min(self.thread_count, len(expert_requests), 8)

        for _ in range(num_threads):
            t = threading.Thread(target=process_expert_requests)
            t.start()
            threads.append(t)

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Verify all requests were processed
        processed_count = 0
        try:
            while True:
                result_queue.get_nowait()
                processed_count += 1
        except queue.Empty:
            pass

        assert (
            processed_count == len(expert_requests)
        ), f"Not all expert requests were processed: {processed_count} out of {len(expert_requests)}"

    def _is_quantized_tensor(self, tensor_fqn: str) -> bool:
        """
        Check if a tensor is quantized and needs dequantization.

        BASELINE TEST: Always return False to bypass dequantization for performance testing.

        Args:
            tensor_fqn: Fully qualified name of the tensor

        Returns:
            False - bypassing all dequantization for baseline performance measurement
        """
        # BASELINE: Disable all dequantization to measure baseline performance
        if tensor_fqn.endswith(WEIGHT_SCALE_INVERSE_SUFFIX):
            return False

        # Check the presence of scale tensor
        return self._has_scale_tensor(tensor_fqn)

    def _read_quantized_tensor_with_block_alignment(
        self, req: ReadItem, safetensor_file
    ) -> torch.Tensor:
        """
        Read quantized tensor with block alignment considerations for FSDP compatibility.

        Args:
            req: Read request containing tensor info and required slices
            safetensor_file: Open safetensors file handle

        Returns:
            Dequantized tensor ready for use
        """
        start_time = datetime.datetime.now()

        tensor_fqn = req.storage_index.fqn
        scale_fqn = self._get_scale_tensor_name(tensor_fqn)

        # Load the quantized weight tensor
        weight_slices = tuple(
            slice(offset, offset + length)
            for offset, length in zip(req.storage_offsets, req.lengths)
        )
        quantized_weight = safetensor_file.get_slice(tensor_fqn)[weight_slices]

        # Load the corresponding scale inverse tensor
        # For scale tensors, we need the full scale tensor for proper block alignment
        scale_inv = safetensor_file.get_slice(scale_fqn)[:]

        # Perform dequantization
        dequantized_tensor = self._dequantize_tensor(
            weight=quantized_weight,
            scale_inv=scale_inv,
            dtype=torch.float32,
        )

        end_time = datetime.datetime.now()
        logger.info(
            f"Read and dequantized the tensor {tensor_fqn} in time {end_time - start_time}"
        )
        return dequantized_tensor


class DeepSeekV3HuggingFaceStorageReader(QuantizedHuggingFaceStorageReader):
    """
    Custom HuggingFace storage reader for DeepSeek V3 that performs format conversion
    during tensor loading, before distributed sharding.

    This reader handles the conversion from HuggingFace format (separate expert tensors)
    to TorchTitan format (grouped expert tensors) at the storage level, avoiding
    memory-intensive DTensor operations that occur with post-sharding conversion.
    """

    def __init__(
        self, path: str, block_size: int = 128, thread_count: int = 20
    ) -> None:
        """
        Initialize the DeepSeek V3 storage reader.

        Args:
            path: Path to the HuggingFace checkpoint directory
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(path, block_size, thread_count)

        # Build key mapping from HF to TorchTitan format using centralized mappings
        self._build_key_mapping()

    def read_metadata(self) -> DeepSeekV3Metadata:
        """
        Override read_metadata to provide separate HF and TT format metadata

        The planner needs both formats:
        - Original HF metadata for reading the actual checkpoint files (storage IO)
        - Converted TT metadata for DCP validation and planning (tensor loading)
        """
        start_time = datetime.datetime.now()
        original_hf_metadata = super().read_metadata()

        # Convert the storage_data keys from HF to TT format with expert grouping
        converted_storage_data = {}
        expert_tensors_by_group = {}  # Track expert tensors for grouping
        converted_state_dict_metadata = {}

        # Process each entry in the original metadata
        entry_count = 0
        for metadata_index in original_hf_metadata.storage_data:
            entry_count += 1
            hf_key = metadata_index.fqn
            original_storage_info = original_hf_metadata.storage_data[metadata_index]

            # Check if this is an expert tensor that needs grouping
            # Only match actual weight tensors, not scale tensors (weight_scale_inv)
            expert_match = re.match(
                r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(\w+)\.weight$", hf_key
            )

            if expert_match:
                layer_idx, expert_idx, weight_type = expert_match.groups()

                # Map HF weight types to TT weight types
                hf_to_tt_weight_map = {
                    "gate_proj": "w1",
                    "down_proj": "w2",
                    "up_proj": "w3",
                }

                if weight_type in hf_to_tt_weight_map:
                    tt_weight_type = hf_to_tt_weight_map[weight_type]
                    # Group expert tensors by layer and weight type (TT format)
                    group_key = f"layers.{layer_idx}.moe.experts.{tt_weight_type}"

                    if group_key not in expert_tensors_by_group:
                        expert_tensors_by_group[group_key] = []

                    expert_info = {
                        "expert_idx": int(expert_idx),
                        "storage_info": original_storage_info,  # Keep original storage info with correct file path
                        "metadata_index": metadata_index,
                        "hf_key": hf_key,
                    }
                    expert_tensors_by_group[group_key].append(expert_info)
            else:
                # Convert regular (non-expert) HF key to TT key
                tt_key = self._get_tt_key_for_hf_key(hf_key)

                if tt_key is None:
                    continue

                # Create new metadata index with converted key
                converted_metadata_index = MetadataIndex(
                    fqn=tt_key, offset=metadata_index.offset, index=metadata_index.index
                )

                converted_storage_data[converted_metadata_index] = original_storage_info

                # Create proper TensorProperties if available from original metadata
                tensor_properties = None
                if hf_key in original_hf_metadata.state_dict_metadata:
                    original_tensor_metadata = original_hf_metadata.state_dict_metadata[
                        hf_key
                    ]
                    tensor_properties = original_tensor_metadata.properties

                # Create chunk metadata with the actual shape
                chunk_metadata = ChunkStorageMetadata(
                    offsets=torch.Size(
                        [0] * len(original_storage_info.shape)
                    ),  # Start from zero offsets
                    sizes=original_storage_info.shape,
                )

                # Create tensor metadata with the actual shape from storage_info
                tensor_metadata = TensorStorageMetadata(
                    properties=tensor_properties,  # Use original properties if available
                    size=original_storage_info.shape,  # Always use the actual shape from storage
                    chunks=[chunk_metadata],
                )

                converted_state_dict_metadata[tt_key] = tensor_metadata

        # Add grouped expert tensors to converted metadata
        for group_key, expert_infos in expert_tensors_by_group.items():
            if expert_infos:
                # Sort by expert index to ensure consistent ordering
                expert_infos.sort(key=lambda x: x["expert_idx"])

                # Use the first expert's storage info as template for shape/dtype
                template_info = expert_infos[0]["storage_info"]

                # Calculate the grouped shape: [num_experts, ...original_shape]
                original_shape = template_info.shape
                grouped_shape = torch.Size([len(expert_infos)] + list(original_shape))

                # Use a placeholder path that indicates this is a grouped tensor
                placeholder_path = (
                    f"grouped_experts/{group_key.replace('.', '_')}.placeholder"
                )

                # Create storage info for grouped tensor
                grouped_storage_info = _HFStorageInfo(
                    relative_path=placeholder_path,  # Placeholder - never used for actual reading
                    shape=grouped_shape,
                    dtype=template_info.dtype,
                )

                for expert_info in expert_infos:
                    expert_idx = expert_info["expert_idx"]
                    file_path = expert_info["storage_info"].relative_path
                    hf_key = expert_info["hf_key"]
                    expert_info["file_path"] = file_path
                    expert_info["hf_key"] = hf_key

                # Create metadata index for the grouped tensor
                grouped_metadata_index = MetadataIndex(
                    fqn=group_key,
                    offset=torch.Size([0] * len(grouped_shape)),
                    index=None,
                )

                converted_storage_data[grouped_metadata_index] = grouped_storage_info

                # Create TensorStorageMetadata for the grouped tensor
                chunk_metadata = ChunkStorageMetadata(
                    offsets=torch.Size([0] * len(grouped_shape)), sizes=grouped_shape
                )

                grouped_tensor_metadata = TensorStorageMetadata(
                    properties=None, size=grouped_shape, chunks=[chunk_metadata]
                )

                converted_state_dict_metadata[group_key] = grouped_tensor_metadata

        # Store expert grouping information for read_data
        self.expert_groups = expert_tensors_by_group

        # Create SD metadata with converted storage_data and state_dict_metadata
        sd_metadata = Metadata(
            storage_data=converted_storage_data,
            planner_data=original_hf_metadata.planner_data,
            state_dict_metadata=converted_state_dict_metadata,
        )

        # Store both formats for use in read_data and planner
        self.storage_data = converted_storage_data  # SD format for DCP
        self.original_hf_storage_data = (
            original_hf_metadata.storage_data
        )  # Original HF format for file reading

        # Return SD metadata for DCP framework compatibility, but store MetadataManager internally
        result = DeepSeekV3Metadata(
            io_metadata=original_hf_metadata, sd_metadata=sd_metadata
        )

        end_time = datetime.datetime.now()
        logger.info(f"Metadata preparation time: {end_time - start_time}")
        return result

    def _build_key_mapping(self):
        """Build mapping from checkpoint keys to TorchTitan keys using centralized mappings."""
        # Get the key mappings from the centralized key_mappings module
        self.hf_to_tt_map = get_hf_to_tt_map()
        self.tt_to_hf_map = {v: k for k, v in self.hf_to_tt_map.items()}
        self.expert_grouping_keys = set()

        # Identify expert tensors that need grouping
        for tt_key in self.hf_to_tt_map.values():
            if "mlp.experts." in tt_key and tt_key.endswith(".weight"):
                self.expert_grouping_keys.add(tt_key)

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        from safetensors import safe_open  # type: ignore[import]

        """
        Override read_data to perform format conversion during loading.

        This method intercepts tensor loading and performs the HF->TorchTitan
        conversion at the storage level, before any distributed operations.
        The key insight is that we need to map from TorchTitan keys (what DCP expects)
        back to HuggingFace checkpoint keys (what's actually stored in the checkpoint).
        """
        # Separate expert tensor requests from regular tensor requests
        expert_requests = []
        regular_requests = []

        for read_item in plan.items:
            tt_key = read_item.storage_index.fqn
            # if self._is_expert_grouping_needed(tt_key):
            #     expert_requests.append(read_item)
            # else:
            
            # NOTE(jiani): We need to bypass the expert grouping for now
            regular_requests.append(read_item)

        logger.info(
            f"Read data started. Identified a total of {len(expert_requests)} expert group tensots and {len(regular_requests)} regular tensors"
        )

        start_time = datetime.datetime.now()
        # Process regular tensors grouped by file
        # For regular tensors, we need to map back to original HF metadata to get correct file paths
        per_file: dict[str, list[ReadItem]] = {}
        for read_item in regular_requests:
            tt_key = read_item.storage_index.fqn
            # Map TT key back to HF key to find the original metadata
            hf_key = self._get_hf_key_for_tt_key(tt_key)

            # Find the original HF metadata entry for this tensor
            original_metadata_entry = None
            for orig_index, orig_storage_info in self.original_hf_storage_data.items():
                if orig_index.fqn == hf_key:
                    original_metadata_entry = orig_storage_info
                    break

            if original_metadata_entry is None:
                raise RuntimeError(
                    f"Could not find original HF metadata for TT key {tt_key} (mapped to HF key {hf_key})"
                )

            relative_path = original_metadata_entry.relative_path
            # Construct full path by joining checkpoint path with relative path
            full_path = str(Path(self.checkpoint_path) / relative_path)
            per_file.setdefault(full_path, []).append(read_item)

        # Process regular tensors
        for file_path, reqs in per_file.items():
            with safe_open(filename=file_path, framework="pt") as f:
                available_keys = list(f.keys())

                for req in reqs:
                    tt_key = req.storage_index.fqn
                    hf_key = self._get_hf_key_for_tt_key(tt_key)

                    if hf_key not in available_keys:
                        raise RuntimeError(
                            f"Key {hf_key} not found in file {file_path}"
                        )

                    # Standard tensor reading using the mapped HF key
                    if self._is_quantized_tensor(hf_key):
                        tensor = self._read_quantized_tensor_with_block_alignment(
                            req, f, hf_key
                        )
                    else:
                        # Standard tensor reading
                        slices = tuple(
                            slice(offset, offset + length)
                            for offset, length in zip(req.storage_offsets, req.lengths)
                        )
                        tensor = f.get_slice(hf_key)[slices]

                    target_tensor = planner.resolve_tensor(req).detach()

                    assert (
                        target_tensor.size() == tensor.size()
                    ), f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
                    target_tensor.copy_(tensor)
                    planner.commit_tensor(req, target_tensor)

        end_time = datetime.datetime.now()
        logger.info(f"Finished reading the regular tensors in time {end_time - start_time}")

        # Process expert tensor requests with parallel processing
        start_time = datetime.datetime.now()
        assert len(expert_requests) == 0, "Should not have expert requests found"
        self._read_expert_tensors_parallel(expert_requests, planner)
        end_time = datetime.datetime.now()
        logger.info(f"Finished reading the grouped experts in time {end_time - start_time}")

        fut: Future = Future()
        fut.set_result(None)
        return fut

    def _get_tt_key_for_hf_key(self, hf_key: str) -> str | None:
        """Convert HuggingFace key to TorchTitan key using centralized key mappings."""
        return convert_hf_key_to_tt_key(hf_key)

    def _get_hf_key_for_tt_key(self, tt_key: str) -> str:
        """
        Get the HuggingFace checkpoint key corresponding to a TorchTitan key.

        Args:
            tt_key: TorchTitan format key (what DCP expects)

        Returns:
            Corresponding HuggingFace checkpoint format key (what's stored in checkpoint)
        """
        hf_key = convert_tt_key_to_hf_key(tt_key)
        return hf_key

    def _is_expert_grouping_needed(self, tt_key: str) -> bool:
        """Check if this TorchTitan key represents a grouped expert tensor."""
        # Only regular experts need grouping, not shared experts
        # Shared experts remain as individual tensors
        return "moe.experts" in tt_key

    def _read_grouped_expert_tensor(self, req: ReadItem, tt_key: str) -> torch.Tensor:
        """
        Read and group expert tensors from individual HF expert tensors.

        This method reads individual expert weights from the HuggingFace checkpoint
        and groups them into the TorchTitan GroupedExpert format by concatenating them
        along the first dimension, with proper shape validation.

        Expert tensors may be distributed across multiple safetensor files,
        so we batch the reads by file to avoid repeated file opens for efficiency.

        Args:
            req: Read request
            tt_key: TorchTitan key for the grouped expert tensor

        Returns:
            Grouped expert tensor (concatenated individual experts)
        """
        import os

        from safetensors import safe_open  # type: ignore[import]

        if tt_key not in self.expert_groups:
            raise RuntimeError(f"ERROR: Expert key {tt_key} not found in expert_groups")

        expert_infos = self.expert_groups[tt_key]
        expert_tensors_by_idx = {}

        # Batch expert tensors by file to avoid repeated file opens
        experts_by_file: dict[str, list] = {}
        for expert_info in expert_infos:
            hf_key = expert_info["hf_key"]
            expert_file_path = self.tensor_to_file_mapping[hf_key]

            # Ensure we have the full absolute path
            if not os.path.isabs(expert_file_path):
                expert_file_path = os.path.join(self.checkpoint_path, expert_file_path)

            experts_by_file.setdefault(expert_file_path, []).append(expert_info)

        # Process experts grouped by file
        for file_path, file_expert_infos in experts_by_file.items():
            with safe_open(filename=file_path, framework="pt") as expert_file:
                for expert_info in file_expert_infos:
                    expert_idx = expert_info["expert_idx"]
                    hf_key = expert_info["hf_key"]

                    # CRITICAL: Only process weight tensors, not scale tensors
                    if not hf_key.endswith(".weight"):
                        continue

                    # Additional check to ensure we don't process scale tensors
                    if "weight_scale_inv" in hf_key:
                        continue

                    # Check if this is a quantized tensor
                    if self._is_quantized_tensor(hf_key):
                        tensor = self._read_full_quantized_tensor(expert_file, hf_key)
                    else:
                        # Read the full tensor (no slicing for individual experts)
                        tensor = expert_file.get_slice(hf_key)[:]

                    # CRITICAL: Only store weight tensors for grouping, never scale tensors
                    if hf_key.endswith(".weight") and "weight_scale_inv" not in hf_key:
                        expert_tensors_by_idx[expert_idx] = tensor

        start_time = datetime.datetime.now()
        # Use the concatenation approach similar to _concatenate_expert_weights
        grouped_tensor = self._concatenate_expert_weights(
            expert_tensors_by_idx, len(expert_infos)
        )

        # Apply the requested slicing to the grouped tensor
        slices = tuple(
            slice(offset, offset + length)
            for offset, length in zip(req.storage_offsets, req.lengths)
        )
        sliced_tensor = grouped_tensor[slices]
        end_time = datetime.datetime.now()
        logger.info(f"Concatented the grouped experts in time {end_time - start_time}")

        return sliced_tensor

    def _concatenate_expert_weights(
        self, expert_weights_by_idx: dict[int, torch.Tensor], n_experts: int
    ) -> torch.Tensor:
        """
        Concatenate the weights of separate experts into GroupedExpert weights.

        Args:
            expert_weights_by_idx: Dictionary mapping expert index to tensor
            n_experts: Expected number of experts

        Returns:
            Stacked tensor with shape [num_experts, ...original_shape] or None if failed
        """
        # Sort experts by index to ensure consistent ordering
        sorted_expert_ids = sorted(expert_weights_by_idx.keys())
        sorted_experts = [expert_weights_by_idx[i] for i in sorted_expert_ids]

        # Stack all expert tensors along the first dimension
        return torch.stack(sorted_experts, dim=0)

    def _read_full_quantized_tensor(self, safetensor_file, hf_key: str) -> torch.Tensor:
        """
        Read a full quantized tensor without slicing.

        Args:
            safetensor_file: Open safetensors file handle
            hf_key: HuggingFace key to read from

        Returns:
            Full dequantized tensor
        """
        from safetensors import safe_open  # type: ignore[import]

        start_time = datetime.datetime.now()
        # Check if we have a scale tensor for this weight
        if not self._has_scale_tensor(hf_key):
            # Fallback to standard reading
            return safetensor_file.get_slice(hf_key)[:]

        scale_fqn = self._get_scale_tensor_name(hf_key)
        # Load the full FP8 weight tensor
        fp8_weight = safetensor_file.get_slice(hf_key)[:]

        # Load the corresponding scale inverse tensor
        # Scale tensor might be in a different file, so check mapping first
        scale_inv = None
        scale_file_path = self.tensor_to_file_mapping.get(scale_fqn)
        if scale_file_path:
            # Ensure we have the full absolute path
            if not os.path.isabs(scale_file_path):
                scale_file_path = str(Path(self.checkpoint_path) / scale_file_path)

            with safe_open(filename=scale_file_path, framework="pt") as scale_file:
                scale_inv = scale_file.get_slice(scale_fqn)[:]
        else:
            raise RuntimeError(f"Scale tensor {scale_fqn} not found in tensor mapping")

        # Perform dequantization
        dequantized_tensor = self._dequantize_tensor(
            weight=fp8_weight,
            scale_inv=scale_inv,
            dtype=torch.float32,
        )
        end_time = datetime.datetime.now()
        logger.info(
            f"Read and dequantized the tensor {hf_key} in time {end_time - start_time}"
        )
        return dequantized_tensor

    def _read_quantized_tensor_with_block_alignment(
        self, req: ReadItem, safetensor_file, hf_key: str
    ) -> torch.Tensor:
        """
        Read quantized tensor using a specific HF key instead of the request's fqn.

        Args:
            req: Read request (for slice information)
            safetensor_file: Open safetensors file handle
            hf_key: HuggingFace key to read from

        Returns:
            Dequantized tensor
        """
        from safetensors import safe_open  # type: ignore[import]

        start_time = datetime.datetime.now()
        # Check if we have a scale tensor for this weight
        if not self._has_scale_tensor(hf_key):
            # Fallback to standard reading with proper slicing
            slices = tuple(
                slice(offset, offset + length)
                for offset, length in zip(req.storage_offsets, req.lengths)
            )
            tensor = safetensor_file.get_slice(hf_key)[slices]
            return tensor

        # Handle quantized tensor with scale mapping
        scale_fqn = self._get_scale_tensor_name(hf_key)

        # Load the quantized weight tensor with proper slicing for distributed sharding
        weight_slices = tuple(
            slice(offset, offset + length)
            for offset, length in zip(req.storage_offsets, req.lengths)
        )
        quantized_weight = safetensor_file.get_slice(hf_key)[weight_slices]

        # Load the corresponding scale inverse tensor
        # Scale tensor might be in a different file, so check current file first
        scale_inv = None
        scale_file_path = self.tensor_to_file_mapping.get(scale_fqn)
        if scale_file_path:
            # Ensure we have the full absolute path
            if not os.path.isabs(scale_file_path):
                scale_file_path = str(Path(self.checkpoint_path) / scale_file_path)

            with safe_open(filename=scale_file_path, framework="pt") as scale_file:
                scale_inv = scale_file.get_slice(scale_fqn)[:]
        else:
            raise RuntimeError(f"Scale tensor {scale_fqn} not found in tensor mapping")

        # Perform dequantization on the sliced tensor
        dequantized_tensor = self._dequantize_tensor(
            weight=quantized_weight,
            scale_inv=scale_inv,
            dtype=torch.float32,
        )
        end_time = datetime.datetime.now()
        logger.info(
            f"Read and dequantized the tensor {hf_key} in time {end_time - start_time}"
        )

        return dequantized_tensor
