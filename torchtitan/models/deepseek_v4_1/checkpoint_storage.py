# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Slice-aware HF loading for block-scaled FP8 and packed E2M1 weights."""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from safetensors import safe_open
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata


@dataclass(frozen=True)
class BlockScaledLayout:
    scale_key: str
    block_rows: int
    block_cols: int
    packed_fp4: bool = False

    def __post_init__(self):
        if self.block_rows <= 0 or self.block_cols <= 0:
            raise ValueError("Quantization block sizes must be positive.")
        if self.packed_fp4 and self.block_cols % 2:
            raise ValueError("Packed FP4 blocks require an even column count.")


class BlockScaledHuggingFaceStorageReader(HuggingFaceStorageReader):
    """Dequantize only the requested rows and the intersecting column blocks.

    Layouts are supplied by the model adapter. Packed FP4 metadata exposes
    logical columns to DCP, while storage reads use packed byte offsets.
    """

    def __init__(self, path, *, layouts: dict[str, BlockScaledLayout], thread_count=1):
        super().__init__(path, thread_count=thread_count)
        self.layouts = layouts
        self._logical_shapes = {}
        with open(Path(path) / "model.safetensors.index.json") as file:
            self._weight_files = json.load(file)["weight_map"]

    def read_metadata(self):
        metadata = super().read_metadata()
        for key, layout in self.layouts.items():
            entry = metadata.state_dict_metadata[key]
            assert isinstance(entry, TensorStorageMetadata)
            if layout.packed_fp4:
                if len(entry.size) != 2 or len(entry.chunks) != 1:
                    raise ValueError(
                        "Packed FP4 HF tensors must be complete two-dimensional matrices."
                    )
                entry.size = torch.Size((entry.size[0], entry.size[1] * 2))
                entry.chunks[0].sizes = entry.size
            entry.properties.dtype = torch.float32
            self._logical_shapes[key] = entry.size
        return metadata

    @staticmethod
    def _fp4_values(packed):
        packed = packed.view(torch.uint8)
        codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2)
        table = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ]
        )
        return table[codes.long()]

    def read_data(self, plan, planner):
        quantized_by_file = {}
        unquantized = []
        for request in plan.items:
            key = request.storage_index.fqn
            if key in self.layouts:
                filename = self._weight_files[key]
                quantized_by_file.setdefault(filename, []).append(request)
            else:
                unquantized.append(request)
        for filename, requests in quantized_by_file.items():
            with safe_open(str(Path(self.path) / filename), framework="pt") as file:
                for request in requests:
                    self._read_quantized_request(file, request, planner)
        return super().read_data(replace(plan, items=unquantized), planner)

    def _read_quantized_request(self, f, req, planner):
        key = req.storage_index.fqn
        layout = self.layouts[key]
        row, col = req.storage_offsets
        num_rows, num_cols = req.lengths
        first_col = col // layout.block_cols * layout.block_cols
        last_col = min(
            (col + num_cols + layout.block_cols - 1)
            // layout.block_cols
            * layout.block_cols,
            self._logical_shapes[key][1],
        )
        divisor = 2 if layout.packed_fp4 else 1
        weight = f.get_slice(key)[
            row : row + num_rows, first_col // divisor : last_col // divisor
        ]
        scale_rows = slice(
            row // layout.block_rows,
            (row + num_rows + layout.block_rows - 1) // layout.block_rows,
        )
        scale_cols = slice(
            first_col // layout.block_cols,
            (last_col + layout.block_cols - 1) // layout.block_cols,
        )
        if self._weight_files[key] == self._weight_files[layout.scale_key]:
            scales = f.get_slice(layout.scale_key)[scale_rows, scale_cols].float()
        else:
            with safe_open(
                str(Path(self.path) / self._weight_files[layout.scale_key]),
                framework="pt",
                device="cpu",
            ) as scale_file:
                scales = scale_file.get_slice(layout.scale_key)[
                    scale_rows, scale_cols
                ].float()
        values = self._fp4_values(weight) if layout.packed_fp4 else weight.float()
        row_blocks = (
            torch.arange(row, row + num_rows) // layout.block_rows - scale_rows.start
        )
        col_blocks = (
            torch.arange(first_col, last_col) // layout.block_cols - scale_cols.start
        )
        values = values * scales[row_blocks[:, None], col_blocks[None, :]]
        values = values[:, col - first_col : col - first_col + num_cols]
        target = planner.resolve_tensor(req).detach()
        target.copy_(values)
        planner.commit_tensor(req, target)
