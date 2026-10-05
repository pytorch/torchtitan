#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generate deterministic paired audio-text and video-text test fixtures."""

import argparse
import io
import json
import sys
import wave
from pathlib import Path

import numpy as np


NUM_ROWS = 32
NUM_MEDIA_FILES = 4
AUDIO_SPECS = (
    (8_000, 257),
    (12_000, 383),
    (16_000, 509),
    (22_050, 641),
)
VIDEO_SPECS = (
    (2, 28, 36),
    (3, 32, 24),
    (4, 40, 30),
    (5, 26, 42),
)


def _canonical_jsonl(rows: list[dict[str, object]]) -> bytes:
    return (
        "".join(
            json.dumps(row, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
            + "\n"
            for row in rows
        )
    ).encode("utf-8")


def _audio_samples(index: int, num_samples: int) -> np.ndarray:
    sample_indices = np.arange(num_samples, dtype=np.int64)
    samples = (sample_indices * (index + 3) * 97 + index * 7_919) % 65_536 - 32_768
    return samples.astype("<i2")


def _wave_bytes(index: int, sample_rate: int, num_samples: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(_audio_samples(index, num_samples).tobytes())
    return output.getvalue()


def _video_frames(index: int, shape: tuple[int, int, int]) -> np.ndarray:
    num_frames, height, width = shape
    time, row, column, channel = np.indices(
        (num_frames, height, width, 3), dtype=np.int64
    )
    frames = (
        time * (17 + index) + row * 11 + column * 7 + channel * 53 + index * 29
    ) % 256

    square_size = min(height, width) // 4
    for frame_index in range(num_frames):
        top = (frame_index * 2 + index) % (height - square_size + 1)
        left = (frame_index * 3 + index * 2) % (width - square_size + 1)
        frames[
            frame_index,
            top : top + square_size,
            left : left + square_size,
        ] = (255, 32 + index * 40, frame_index * 31)
    return np.ascontiguousarray(frames, dtype=np.uint8)


def _audio_rows() -> list[dict[str, object]]:
    rows = []
    for row_index in range(NUM_ROWS):
        media_index = row_index % NUM_MEDIA_FILES
        rows.append(
            {
                "audio_path": f"audio_{media_index:02d}.wav",
                "text": (
                    f"Synthetic waveform {media_index:02d}, "
                    f"example {row_index:03d}."
                ),
            }
        )
    return rows


def _video_rows() -> list[dict[str, object]]:
    return [
        {
            "video_path": f"video_{row_index % NUM_MEDIA_FILES:02d}.npy",
            "text": (
                f"A synthetic square moves in video {row_index % NUM_MEDIA_FILES:02d}, "
                f"example {row_index:03d}."
            ),
        }
        for row_index in range(NUM_ROWS)
    ]


def _expected_file_names(kind: str) -> set[str]:
    suffix = "wav" if kind == "audio" else "npy"
    return {
        "data.jsonl",
        *(f"{kind}_{index:02d}.{suffix}" for index in range(NUM_MEDIA_FILES)),
    }


def _write_fixtures(output_dir: Path) -> None:
    audio_dir = output_dir / "audio_text"
    video_dir = output_dir / "video_text"
    audio_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    (audio_dir / "data.jsonl").write_bytes(_canonical_jsonl(_audio_rows()))
    for index, (sample_rate, num_samples) in enumerate(AUDIO_SPECS):
        (audio_dir / f"audio_{index:02d}.wav").write_bytes(
            _wave_bytes(index, sample_rate, num_samples)
        )

    (video_dir / "data.jsonl").write_bytes(_canonical_jsonl(_video_rows()))
    for index, shape in enumerate(VIDEO_SPECS):
        with (video_dir / f"video_{index:02d}.npy").open("wb") as output:
            np.save(output, _video_frames(index, shape), allow_pickle=False)


def _check_directory_entries(
    fixture_dir: Path, expected_names: set[str], errors: list[str]
) -> bool:
    if not fixture_dir.is_dir():
        errors.append(f"missing directory: {fixture_dir}")
        return False

    actual_names = {path.name for path in fixture_dir.iterdir()}
    missing_names = sorted(expected_names - actual_names)
    unexpected_names = sorted(actual_names - expected_names)
    for name in missing_names:
        errors.append(f"missing fixture: {fixture_dir / name}")
    for name in unexpected_names:
        errors.append(f"unexpected fixture: {fixture_dir / name}")
    return not missing_names


def _check_exact_bytes(
    path: Path, expected: bytes, description: str, errors: list[str]
) -> None:
    if not path.is_file():
        return
    if path.read_bytes() != expected:
        errors.append(f"stale {description}: {path}")


def _check_fixtures(output_dir: Path) -> list[str]:
    errors: list[str] = []
    audio_dir = output_dir / "audio_text"
    video_dir = output_dir / "video_text"
    _check_directory_entries(audio_dir, _expected_file_names("audio"), errors)
    _check_directory_entries(video_dir, _expected_file_names("video"), errors)

    _check_exact_bytes(
        audio_dir / "data.jsonl",
        _canonical_jsonl(_audio_rows()),
        "audio manifest",
        errors,
    )
    for index, (sample_rate, num_samples) in enumerate(AUDIO_SPECS):
        _check_exact_bytes(
            audio_dir / f"audio_{index:02d}.wav",
            _wave_bytes(index, sample_rate, num_samples),
            "WAV data",
            errors,
        )

    _check_exact_bytes(
        video_dir / "data.jsonl",
        _canonical_jsonl(_video_rows()),
        "video manifest",
        errors,
    )
    for index, shape in enumerate(VIDEO_SPECS):
        path = video_dir / f"video_{index:02d}.npy"
        if not path.is_file():
            continue
        try:
            actual = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as error:
            errors.append(f"invalid NPY data: {path}: {error}")
            continue
        expected = _video_frames(index, shape)
        if (
            actual.dtype != np.uint8
            or not actual.flags.c_contiguous
            or not np.array_equal(actual, expected)
        ):
            errors.append(f"stale NPY data: {path}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Fixture root to generate or check",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify fixtures without modifying them",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()

    if not args.check:
        _write_fixtures(output_dir)
        return 0

    errors = _check_fixtures(output_dir)
    if errors:
        print("Synthetic fixture check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
