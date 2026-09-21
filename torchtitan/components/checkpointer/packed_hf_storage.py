# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hugging Face storage reader for paired packed weights and block scales."""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import LoadPlanner, ReadItem

__all__ = ["PackedPairHuggingFaceStorageReader", "PackedPairSpec"]


@dataclass(frozen=True)
class PackedPairSpec:
    """Describe how a packed tensor and its scales expose a virtual weight."""

    packed_suffix: str
    scale_suffix: str
    virtual_suffix: str
    block_size: int
    packed_values_per_byte: int
    target_dtype: torch.dtype
    target_fqns: frozenset[str]
    decode: Callable[[torch.Tensor, torch.Tensor, int, torch.dtype], torch.Tensor]

    def __post_init__(self) -> None:
        if not self.packed_suffix or not self.scale_suffix or not self.virtual_suffix:
            raise ValueError("packed, scale, and virtual suffixes must be non-empty")
        if len({self.packed_suffix, self.scale_suffix, self.virtual_suffix}) != 3:
            raise ValueError("packed, scale, and virtual suffixes must be distinct")
        if (
            self.block_size <= 0
            or self.packed_values_per_byte <= 0
            or self.block_size % self.packed_values_per_byte != 0
        ):
            raise ValueError(
                "block_size must be positive and divisible by packed_values_per_byte"
            )


@dataclass(frozen=True)
class _PackedPair:
    packed_fqn: str
    scale_fqn: str
    packed_path: str
    scale_path: str


class PackedPairHuggingFaceStorageReader(HuggingFaceStorageReader):
    """Present selected uint8 packed/scale pairs as unquantized DCP tensors."""

    def __init__(
        self,
        path: str,
        spec: PackedPairSpec,
        thread_count: int = 1,
    ) -> None:
        super().__init__(path=path, thread_count=thread_count)
        self.spec = spec
        self._pairs: dict[str, _PackedPair] = {}

    def _replace_suffix(self, fqn: str, source: str, destination: str) -> str:
        if not fqn.endswith(source):
            raise ValueError(f"{fqn!r} does not end with {source!r}")
        return fqn[: -len(source)] + destination

    def _validate_pair(
        self,
        virtual_fqn: str,
        packed_metadata: TensorStorageMetadata,
        scale_metadata: TensorStorageMetadata,
    ) -> None:
        if virtual_fqn not in self.spec.target_fqns:
            raise ValueError(
                f"Packed tensor {virtual_fqn!r} is outside the packed-weight policy."
            )
        if (
            packed_metadata.properties.dtype != torch.uint8
            or scale_metadata.properties.dtype != torch.uint8
        ):
            raise ValueError(
                f"Packed tensor pair for {virtual_fqn!r} must use torch.uint8."
            )
        packed_shape = packed_metadata.size
        scale_shape = scale_metadata.size
        if len(packed_shape) != 2 or len(scale_shape) != 2:
            raise ValueError(
                f"Packed tensor pair for {virtual_fqn!r} must be two-dimensional."
            )
        if (
            packed_shape[0] != scale_shape[0]
            or packed_shape[1] * self.spec.packed_values_per_byte
            != scale_shape[1] * self.spec.block_size
        ):
            raise ValueError(
                f"Packed tensor pair for {virtual_fqn!r} has incompatible packed "
                f"and scale shapes: {tuple(packed_shape)} and {tuple(scale_shape)}."
            )

    def _virtual_chunk(self, chunk: ChunkStorageMetadata) -> ChunkStorageMetadata:
        offsets = list(chunk.offsets)
        sizes = list(chunk.sizes)
        offsets[-1] *= self.spec.packed_values_per_byte
        sizes[-1] *= self.spec.packed_values_per_byte
        return dataclasses.replace(
            chunk,
            offsets=torch.Size(offsets),
            sizes=torch.Size(sizes),
        )

    # pyrefly: ignore [bad-override]
    def read_metadata(self) -> Any:
        metadata = super().read_metadata()
        state_dict_metadata = metadata.state_dict_metadata
        storage_paths: dict[str, str] = {}
        for index, storage_info in metadata.storage_data.items():
            previous = storage_paths.setdefault(index.fqn, storage_info.relative_path)
            if previous != storage_info.relative_path:
                raise ValueError(
                    f"Tensor {index.fqn!r} is split across safetensors files; "
                    "packed-pair loading requires each source tensor in one file."
                )
        packed_fqns = {
            fqn for fqn in state_dict_metadata if fqn.endswith(self.spec.packed_suffix)
        }
        scale_fqns = {
            fqn for fqn in state_dict_metadata if fqn.endswith(self.spec.scale_suffix)
        }

        pairs: dict[str, _PackedPair] = {}
        for packed_fqn in sorted(packed_fqns):
            virtual_fqn = self._replace_suffix(
                packed_fqn, self.spec.packed_suffix, self.spec.virtual_suffix
            )
            scale_fqn = self._replace_suffix(
                packed_fqn, self.spec.packed_suffix, self.spec.scale_suffix
            )
            if scale_fqn not in scale_fqns:
                raise ValueError(
                    f"Packed tensor {packed_fqn!r} is missing scale tensor "
                    f"{scale_fqn!r}."
                )
            if virtual_fqn in state_dict_metadata:
                raise ValueError(
                    f"Virtual tensor {virtual_fqn!r} conflicts with a stored tensor."
                )
            packed_metadata = state_dict_metadata[packed_fqn]
            scale_metadata = state_dict_metadata[scale_fqn]
            if not isinstance(packed_metadata, TensorStorageMetadata) or not isinstance(
                scale_metadata, TensorStorageMetadata
            ):
                raise TypeError(
                    f"Packed tensor pair for {virtual_fqn!r} must contain tensors."
                )
            self._validate_pair(virtual_fqn, packed_metadata, scale_metadata)
            pairs[virtual_fqn] = _PackedPair(
                packed_fqn,
                scale_fqn,
                storage_paths[packed_fqn],
                storage_paths[scale_fqn],
            )

        paired_scales = {pair.scale_fqn for pair in pairs.values()}
        orphan_scales = sorted(scale_fqns - paired_scales)
        if orphan_scales:
            raise ValueError(f"Found orphan scale tensor {orphan_scales[0]!r}.")

        missing = self.spec.target_fqns - pairs.keys()
        if missing:
            raise ValueError(
                f"Packed-weight policy requires missing pairs: {sorted(missing)[:10]}."
            )

        virtual_state_dict_metadata = {
            fqn: tensor_metadata
            for fqn, tensor_metadata in state_dict_metadata.items()
            if fqn not in packed_fqns and fqn not in scale_fqns
        }
        for virtual_fqn, pair in pairs.items():
            packed_metadata = state_dict_metadata[pair.packed_fqn]
            virtual_state_dict_metadata[virtual_fqn] = dataclasses.replace(
                packed_metadata,
                properties=dataclasses.replace(
                    packed_metadata.properties, dtype=self.spec.target_dtype
                ),
                size=torch.Size(
                    (
                        packed_metadata.size[0],
                        packed_metadata.size[1] * self.spec.packed_values_per_byte,
                    )
                ),
                chunks=[self._virtual_chunk(chunk) for chunk in packed_metadata.chunks],
            )

        virtual_storage_data: dict[MetadataIndex, Any] = {}
        for index, storage_info in metadata.storage_data.items():
            if index.fqn in scale_fqns:
                continue
            if index.fqn not in packed_fqns:
                virtual_storage_data[index] = storage_info
                continue
            virtual_fqn = self._replace_suffix(
                index.fqn, self.spec.packed_suffix, self.spec.virtual_suffix
            )
            virtual_offset = list(index.offset or ())
            if virtual_offset:
                virtual_offset[-1] *= self.spec.packed_values_per_byte
            virtual_index = dataclasses.replace(
                index,
                fqn=virtual_fqn,
                offset=torch.Size(virtual_offset),
            )
            virtual_storage_data[virtual_index] = dataclasses.replace(
                storage_info,
                shape=torch.Size(
                    (
                        storage_info.shape[0],
                        storage_info.shape[1] * self.spec.packed_values_per_byte,
                    )
                ),
                dtype=self.spec.target_dtype,
            )

        self._pairs = pairs
        return dataclasses.replace(
            metadata,
            state_dict_metadata=virtual_state_dict_metadata,
            storage_data=virtual_storage_data,
        )

    def _process_read_request(
        self,
        f: Any,
        req: ReadItem,
        planner: LoadPlanner,
    ) -> None:
        virtual_fqn = req.storage_index.fqn
        pair = self._pairs.get(virtual_fqn)
        if pair is None:
            super()._process_read_request(f, req, planner)
            return
        if len(req.storage_offsets) != 2 or len(req.lengths) != 2:
            raise ValueError(
                f"Packed tensor {virtual_fqn!r} requires a two-dimensional read; "
                f"got offsets {tuple(req.storage_offsets)} and lengths "
                f"{tuple(req.lengths)}."
            )

        row_start, column_start = req.storage_offsets
        row_count, column_count = req.lengths
        row_stop = row_start + row_count
        column_stop = column_start + column_count
        group_start = column_start // self.spec.block_size
        group_stop = (column_stop + self.spec.block_size - 1) // self.spec.block_size
        packed_values_per_group = (
            self.spec.block_size // self.spec.packed_values_per_byte
        )
        packed_start = group_start * packed_values_per_group
        packed_stop = group_stop * packed_values_per_group

        packed = f.get_slice(pair.packed_fqn)[
            slice(row_start, row_stop), slice(packed_start, packed_stop)
        ]
        scale_slices = (slice(row_start, row_stop), slice(group_start, group_stop))
        if pair.scale_path == pair.packed_path:
            scales = f.get_slice(pair.scale_fqn)[scale_slices]
        else:
            from safetensors import safe_open  # type: ignore[import]

            with safe_open(pair.scale_path, framework="pt") as scale_file:
                scales = scale_file.get_slice(pair.scale_fqn)[scale_slices]

        dequantized = self.spec.decode(
            packed,
            scales,
            self.spec.block_size,
            self.spec.target_dtype,
        )
        crop_start = column_start - group_start * self.spec.block_size
        tensor = dequantized[:, crop_start : crop_start + column_count]
        target_tensor = planner.resolve_tensor(req).detach()
        if target_tensor.size() != tensor.size():
            raise AssertionError(
                f"req {req.storage_index} mismatch sizes "
                f"{target_tensor.size()} vs {tensor.size()}"
            )
        target_tensor.copy_(tensor)
        planner.commit_tensor(req, target_tensor)
