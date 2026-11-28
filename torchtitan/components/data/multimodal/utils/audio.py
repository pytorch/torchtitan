# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Audio processing utilities for multimodal datasets."""

import wave
from pathlib import Path

import numpy as np
import torch


def load_pcm_waveform(
    path: str | Path,
    *,
    expected_num_channels: int,
) -> tuple[torch.Tensor, int]:
    """Load an uncompressed signed PCM16 WAVE file."""
    try:
        with wave.open(str(path), "rb") as input_file:
            compression_type = input_file.getcomptype()
            if compression_type != "NONE":
                raise ValueError(
                    f"Expected uncompressed PCM audio, got {compression_type}"
                )

            sample_width = input_file.getsampwidth()
            if sample_width != 2:
                raise ValueError(
                    f"Expected signed 16-bit PCM audio, got {sample_width * 8}-bit"
                )

            num_channels = input_file.getnchannels()
            if num_channels != expected_num_channels:
                raise ValueError(
                    f"Expected {expected_num_channels} audio channels, "
                    f"got {num_channels}"
                )

            sample_rate = input_file.getframerate()
            if sample_rate <= 0:
                raise ValueError(
                    f"WAVE sample rate must be positive, got {sample_rate}"
                )

            num_samples = input_file.getnframes()
            if num_samples <= 0:
                raise ValueError("WAVE audio is empty")

            frame_bytes = input_file.readframes(num_samples + 1)
    except (EOFError, wave.Error) as error:
        raise ValueError(f"Invalid WAVE file {path}: {error}") from error

    expected_num_bytes = num_samples * num_channels * sample_width
    if len(frame_bytes) != expected_num_bytes:
        raise ValueError(
            f"WAVE audio has incomplete frame data: expected {expected_num_bytes} "
            f"bytes, got {len(frame_bytes)}"
        )

    pcm = np.frombuffer(frame_bytes, dtype="<i2").reshape(num_samples, num_channels)
    normalized = np.ascontiguousarray(pcm.astype(np.float32) / np.float32(32768.0))
    return torch.from_numpy(normalized), sample_rate


def resample_waveform(
    waveform: torch.Tensor,
    *,
    source_sample_rate: int,
    target_sample_rate: int,
) -> torch.Tensor:
    """Resample a channels-last float32 waveform with linear interpolation."""
    if source_sample_rate == target_sample_rate:
        return waveform.contiguous()

    num_input_samples = waveform.shape[0]
    num_output_samples = max(
        1,
        num_input_samples * target_sample_rate // source_sample_rate,
    )
    output_indices = torch.arange(
        num_output_samples,
        dtype=torch.float32,
        device=waveform.device,
    )
    source_positions = output_indices * source_sample_rate / target_sample_rate
    lower_indices = source_positions.floor().to(torch.int64)
    upper_indices = (lower_indices + 1).clamp(max=num_input_samples - 1)
    upper_weights = (source_positions - lower_indices).unsqueeze(1)

    lower_values = waveform[lower_indices]
    upper_values = waveform[upper_indices]
    return (lower_values + (upper_values - lower_values) * upper_weights).contiguous()
