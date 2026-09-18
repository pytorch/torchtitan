# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.checkpoint.planner import LoadItemType, ReadItem

from torchtitan.components.checkpointer.packed_hf_storage import (
    PackedPairHuggingFaceStorageReader,
    PackedPairSpec,
)

_PACKED_KEY = "model.layers.0.experts.0.w1.weight_packed"
_SCALE_KEY = "model.layers.0.experts.0.w1.weight_scale"
_VIRTUAL_KEY = "model.layers.0.experts.0.w1.weight"


def _is_expert_weight(key: str) -> bool:
    return ".experts." in key


def _decode_mxfp4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    block_size: int,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    from torchao.prototype.mx_formats.mx_tensor import MXTensor

    return MXTensor(
        packed,
        scales.view(torch.float8_e8m0fnu),
        torch.float4_e2m1fn_x2,
        block_size,
        target_dtype,
        None,
        None,
        False,
    ).dequantize(target_dtype)


def _spec() -> PackedPairSpec:
    return PackedPairSpec(
        packed_suffix=".weight_packed",
        scale_suffix=".weight_scale",
        virtual_suffix=".weight",
        block_size=32,
        packed_values_per_byte=2,
        target_dtype=torch.bfloat16,
        is_target=_is_expert_weight,
        decode=_decode_mxfp4,
    )


def _packed_fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nibbles = torch.arange(2 * 64, dtype=torch.uint8).reshape(2, 64) % 16
    packed = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
    scales = torch.tensor([[127, 128], [126, 255]], dtype=torch.uint8)
    lookup = torch.tensor(
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
        ],
        dtype=torch.float32,
    )
    decoded = torch.stack(
        (lookup[(packed & 0x0F).long()], lookup[(packed >> 4).long()]), dim=-1
    ).flatten(-2)
    expanded_scales = scales.repeat_interleave(32, dim=-1)
    expected = torch.ldexp(decoded, expanded_scales.to(torch.int32) - 127)
    expected = torch.where(expanded_scales == 255, torch.nan, expected).to(
        torch.bfloat16
    )
    return packed, scales, expected


class _Planner:
    def __init__(self, destination: torch.Tensor) -> None:
        self.destination = destination
        self.committed = False

    def resolve_tensor(self, _read_item: ReadItem) -> torch.Tensor:
        return self.destination

    def commit_tensor(self, _read_item: ReadItem, tensor: torch.Tensor) -> None:
        self.committed = tensor.data_ptr() == self.destination.data_ptr()


class _RecordingSlice:
    def __init__(self, tensor_slice: object, calls: list[tuple[str, object]], key: str):
        self.tensor_slice = tensor_slice
        self.calls = calls
        self.key = key

    def __getitem__(self, slices: object) -> torch.Tensor:
        self.calls.append((self.key, slices))
        return self.tensor_slice[slices]  # type: ignore[index]


class _RecordingFile:
    def __init__(self, handle: object) -> None:
        self.handle = handle
        self.calls: list[tuple[str, object]] = []

    def get_slice(self, key: str) -> _RecordingSlice:
        return _RecordingSlice(self.handle.get_slice(key), self.calls, key)  # type: ignore[attr-defined]


class PackedPairHuggingFaceStorageReaderMetadataTest(unittest.TestCase):
    def _write_checkpoint(self, tensors: dict[str, torch.Tensor]) -> str:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        save_file(tensors, Path(directory.name) / "model.safetensors")
        return directory.name

    def test_read_metadata_exposes_virtual_weight(self) -> None:
        path = self._write_checkpoint(
            {
                _PACKED_KEY: torch.zeros((2, 32), dtype=torch.uint8),
                _SCALE_KEY: torch.full((2, 2), 127, dtype=torch.uint8),
                "dense.weight": torch.ones((3, 4), dtype=torch.bfloat16),
            }
        )

        metadata = PackedPairHuggingFaceStorageReader(path, _spec()).read_metadata()

        self.assertEqual(
            set(metadata.state_dict_metadata), {_VIRTUAL_KEY, "dense.weight"}
        )
        virtual = metadata.state_dict_metadata[_VIRTUAL_KEY]
        self.assertEqual(virtual.size, torch.Size((2, 64)))
        self.assertEqual(virtual.properties.dtype, torch.bfloat16)
        self.assertEqual(
            {index.fqn for index in metadata.storage_data},
            {_VIRTUAL_KEY, "dense.weight"},
        )

    def test_read_metadata_rejects_missing_scale(self) -> None:
        path = self._write_checkpoint(
            {_PACKED_KEY: torch.zeros((2, 32), dtype=torch.uint8)}
        )

        with self.assertRaisesRegex(ValueError, "missing scale tensor"):
            PackedPairHuggingFaceStorageReader(path, _spec()).read_metadata()

    def test_read_metadata_rejects_orphan_scale(self) -> None:
        path = self._write_checkpoint(
            {_SCALE_KEY: torch.full((2, 2), 127, dtype=torch.uint8)}
        )

        with self.assertRaisesRegex(ValueError, "orphan scale tensor"):
            PackedPairHuggingFaceStorageReader(path, _spec()).read_metadata()

    def test_read_metadata_rejects_non_uint8_payload(self) -> None:
        path = self._write_checkpoint(
            {
                _PACKED_KEY: torch.zeros((2, 32), dtype=torch.int16),
                _SCALE_KEY: torch.full((2, 2), 127, dtype=torch.uint8),
            }
        )

        with self.assertRaisesRegex(ValueError, "must use torch.uint8"):
            PackedPairHuggingFaceStorageReader(path, _spec()).read_metadata()

    def test_read_metadata_rejects_incompatible_shapes(self) -> None:
        path = self._write_checkpoint(
            {
                _PACKED_KEY: torch.zeros((2, 31), dtype=torch.uint8),
                _SCALE_KEY: torch.full((2, 2), 127, dtype=torch.uint8),
            }
        )

        with self.assertRaisesRegex(ValueError, "incompatible packed and scale shapes"):
            PackedPairHuggingFaceStorageReader(path, _spec()).read_metadata()

    def test_read_metadata_rejects_pair_outside_target_policy(self) -> None:
        path = self._write_checkpoint(
            {
                "model.layers.0.dense.weight_packed": torch.zeros(
                    (2, 32), dtype=torch.uint8
                ),
                "model.layers.0.dense.weight_scale": torch.full(
                    (2, 2), 127, dtype=torch.uint8
                ),
            }
        )

        with self.assertRaisesRegex(ValueError, "outside the packed-weight policy"):
            PackedPairHuggingFaceStorageReader(path, _spec()).read_metadata()


class PackedPairHuggingFaceStorageReaderReadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        packed, scales, self.expected = _packed_fixture()
        save_file(
            {_PACKED_KEY: packed, _SCALE_KEY: scales},
            Path(self.directory.name) / "model.safetensors",
        )

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_dcp_load_dequantizes_full_tensor(self) -> None:
        destination = torch.empty((2, 64), dtype=torch.bfloat16)

        dcp.load(
            {_VIRTUAL_KEY: destination},
            storage_reader=PackedPairHuggingFaceStorageReader(
                self.directory.name, _spec()
            ),
        )

        torch.testing.assert_close(
            destination, self.expected, rtol=0, atol=0, equal_nan=True
        )

    def test_unaligned_read_uses_only_intersecting_groups(self) -> None:
        reader = PackedPairHuggingFaceStorageReader(self.directory.name, _spec())
        reader.read_metadata()
        destination = torch.empty((2, 38), dtype=torch.bfloat16)
        planner = _Planner(destination)
        request = ReadItem(
            type=LoadItemType.TENSOR,
            dest_index=MetadataIndex(_VIRTUAL_KEY, [0, 0]),
            dest_offsets=torch.Size((0, 0)),
            storage_index=MetadataIndex(_VIRTUAL_KEY, [0, 0]),
            storage_offsets=torch.Size((0, 7)),
            lengths=torch.Size((2, 38)),
        )

        with safe_open(
            Path(self.directory.name) / "model.safetensors", framework="pt"
        ) as handle:
            recording_file = _RecordingFile(handle)
            reader._process_read_request(recording_file, request, planner)

        self.assertTrue(planner.committed)
        torch.testing.assert_close(
            destination, self.expected[:, 7:45], rtol=0, atol=0, equal_nan=True
        )
        self.assertEqual(
            recording_file.calls,
            [
                (_PACKED_KEY, (slice(0, 2), slice(0, 32))),
                (_SCALE_KEY, (slice(0, 2), slice(0, 2))),
            ],
        )

    def test_aligned_read_dequantizes_second_group(self) -> None:
        reader = PackedPairHuggingFaceStorageReader(self.directory.name, _spec())
        reader.read_metadata()
        destination = torch.empty((1, 32), dtype=torch.bfloat16)
        planner = _Planner(destination)
        request = ReadItem(
            type=LoadItemType.TENSOR,
            dest_index=MetadataIndex(_VIRTUAL_KEY, [0, 0]),
            dest_offsets=torch.Size((0, 0)),
            storage_index=MetadataIndex(_VIRTUAL_KEY, [0, 0]),
            storage_offsets=torch.Size((1, 32)),
            lengths=torch.Size((1, 32)),
        )

        with safe_open(
            Path(self.directory.name) / "model.safetensors", framework="pt"
        ) as handle:
            reader._process_read_request(handle, request, planner)

        torch.testing.assert_close(
            destination, self.expected[1:2, 32:64], rtol=0, atol=0, equal_nan=True
        )

    def test_e8m0_extreme_bytes_follow_mx_semantics(self) -> None:
        packed = torch.full((1, 48), 0x11, dtype=torch.uint8)
        scales = torch.tensor([[0, 254, 255]], dtype=torch.uint8)
        with tempfile.TemporaryDirectory() as directory:
            save_file(
                {_PACKED_KEY: packed, _SCALE_KEY: scales},
                Path(directory) / "model.safetensors",
            )
            destination = torch.empty((1, 96), dtype=torch.bfloat16)
            dcp.load(
                {_VIRTUAL_KEY: destination},
                storage_reader=PackedPairHuggingFaceStorageReader(directory, _spec()),
            )

        expected = torch.cat(
            (
                torch.full(
                    (32,),
                    torch.ldexp(torch.tensor(0.5), torch.tensor(-127)).item(),
                ),
                torch.full(
                    (32,),
                    torch.ldexp(torch.tensor(0.5), torch.tensor(127)).item(),
                ),
                torch.full((32,), torch.nan),
            )
        ).reshape(1, 96)
        torch.testing.assert_close(
            destination,
            expected.to(torch.bfloat16),
            rtol=0,
            atol=0,
            equal_nan=True,
        )

if __name__ == "__main__":
    unittest.main()
