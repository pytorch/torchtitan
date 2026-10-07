# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from torchtitan.hf_datasets.multimodal.utils.audio import (
    load_pcm_waveform,
    resample_waveform,
)


def _write_wave(
    path: Path,
    samples: list[int],
    *,
    sample_rate: int,
    num_channels: int = 1,
    sample_width: int = 2,
) -> None:
    dtype = "<i2" if sample_width == 2 else "u1"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(num_channels)
        output.setsampwidth(sample_width)
        output.setframerate(sample_rate)
        output.writeframes(np.asarray(samples, dtype=dtype).tobytes())


def test_load_pcm_waveform_decodes_channels_last_float32(tmp_path):
    path = tmp_path / "stereo.wav"
    _write_wave(
        path,
        [-32768, 32767, 0, -16384],
        sample_rate=8_000,
        num_channels=2,
    )

    waveform, sample_rate = load_pcm_waveform(path, expected_num_channels=2)

    assert sample_rate == 8_000
    assert waveform.dtype == torch.float32
    assert waveform.is_contiguous()
    assert torch.equal(
        waveform,
        torch.tensor([[-1.0, 32767 / 32768], [0.0, -0.5]], dtype=torch.float32),
    )


def test_load_pcm_waveform_rejects_non_pcm16_input(tmp_path):
    path = tmp_path / "pcm8.wav"
    _write_wave(path, [0, 255], sample_rate=8_000, sample_width=1)

    with pytest.raises(ValueError, match="16-bit"):
        load_pcm_waveform(path, expected_num_channels=1)


def test_resample_waveform_interpolates_channels_last_input():
    waveform = torch.tensor(
        [[0.0, 10.0], [3.0, 13.0], [6.0, 16.0], [9.0, 19.0]],
        dtype=torch.float32,
    )

    result = resample_waveform(
        waveform,
        source_sample_rate=4,
        target_sample_rate=6,
    )

    assert torch.equal(
        result,
        torch.tensor(
            [
                [0.0, 10.0],
                [2.0, 12.0],
                [4.0, 14.0],
                [6.0, 16.0],
                [8.0, 18.0],
                [9.0, 19.0],
            ],
            dtype=torch.float32,
        ),
    )
