#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import gzip
import json
import random
import shutil
import subprocess
import sys
from array import array
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torchtitan.components.tokenizer import HuggingFaceTokenizer


SUPPORTED_JSONL_PATTERNS = ("*.jsonl", "*.jsonl.gz", "*.jsonl.zst")
FORMAT_VERSION = 2


def _input_files(input_dir: Path, pattern: str | None) -> list[Path]:
    if pattern is not None:
        return sorted(input_dir.glob(pattern))
    return sorted(
        {
            path
            for supported_pattern in SUPPORTED_JSONL_PATTERNS
            for path in input_dir.glob(supported_pattern)
        }
    )


def _iter_jsonl_zst(path: Path) -> Iterator[dict]:
    with subprocess.Popen(
        ["zstdcat", str(path)],
        stdout=subprocess.PIPE,
        text=True,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            yield json.loads(line)
        return_code = proc.wait()
        if return_code != 0:
            raise RuntimeError(f"zstdcat failed for {path} with code {return_code}")


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def _iter_jsonl_gz(path: Path) -> Iterator[dict]:
    with gzip.open(path, mode="rt") as f:
        for line in f:
            yield json.loads(line)


def _iter_records(path: Path) -> Iterator[dict]:
    if path.suffix == ".zst":
        yield from _iter_jsonl_zst(path)
    elif path.suffix == ".gz":
        yield from _iter_jsonl_gz(path)
    else:
        yield from _iter_jsonl(path)


def _shuffled_records(
    path: Path,
    *,
    shuffle_seed: int,
    source_file_idx: int,
) -> list[dict]:
    """Load and deterministically shuffle the documents in one input shard."""
    records = list(_iter_records(path))
    random.Random(shuffle_seed + source_file_idx).shuffle(records)
    return records


def _append_tokens(output_path: Path, tokens: list[int]) -> None:
    token_array = array("I", tokens)
    if token_array.itemsize != 4:
        raise RuntimeError("array('I') is not 4 bytes on this platform")
    with output_path.open("ab") as f:
        token_array.tofile(f)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    tmp_path.replace(path)


class TokenBinWriter:
    def __init__(
        self,
        output_dir: Path,
        *,
        output_prefix: str,
        tokens_per_bin: int,
        chunk_size: int,
        progress: dict[str, Any] | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.output_prefix = output_prefix
        self.tokens_per_bin = tokens_per_bin
        self.chunk_size = chunk_size
        self.buffer: list[int] = []

        progress = progress or {}
        self.data_files: list[dict[str, int | str]] = list(
            progress.get("data_files", [])
        )
        self.part_idx = int(progress.get("part_idx", len(self.data_files)))
        self.part_token_count = int(progress.get("part_token_count", 0))
        self.current_tmp_rollback_source_file_idx = progress.get(
            "current_tmp_rollback_source_file_idx"
        )
        self.current_tmp_rollback_record_idx = progress.get(
            "current_tmp_rollback_record_idx", 0
        )
        self.current_tmp_rollback_token_offset = progress.get(
            "current_tmp_rollback_token_offset", 0
        )
        self.current_tmp_rollback_num_documents = progress.get(
            "current_tmp_rollback_num_documents"
        )
        self.current_tmp_rollback_num_tokens = progress.get(
            "current_tmp_rollback_num_tokens"
        )
        if (
            self.part_token_count > 0
            and not self.data_files
            and self.current_tmp_rollback_source_file_idx is None
        ):
            self.current_tmp_rollback_source_file_idx = 0
            self.current_tmp_rollback_record_idx = 0
            self.current_tmp_rollback_token_offset = 0
            self.current_tmp_rollback_num_documents = 0
            self.current_tmp_rollback_num_tokens = 0

        current_tmp_file = progress.get("current_tmp_file")
        self.current_path = (
            self.output_dir / current_tmp_file if current_tmp_file is not None else None
        )
        if self.current_path is not None:
            self.current_path.parent.mkdir(parents=True, exist_ok=True)
            expected_size = self.part_token_count * 4
            if not self.current_path.exists():
                if expected_size > 0:
                    raise ValueError(
                        f"progress.json points to missing tmp bin "
                        f"{self.current_path}. Pass --overwrite to restart."
                    )
                self.current_path.touch()
            current_size = self.current_path.stat().st_size
            if current_size != expected_size:
                raise ValueError(
                    f"Tmp bin {self.current_path} size does not match "
                    f"progress.json ({current_size} bytes vs {expected_size} "
                    "bytes)."
                )

    def _bin_name(self) -> str:
        return f"{self.output_prefix}-{self.part_idx:06d}.bin"

    def _tmp_name(self) -> str:
        return self._bin_name() + ".tmp"

    def _start_part(
        self,
        *,
        rollback_source_file_idx: int,
        rollback_record_idx: int,
        rollback_token_offset: int,
        rollback_num_documents: int,
        rollback_num_tokens: int,
    ) -> None:
        self.current_path = self.output_dir / self._tmp_name()
        self.current_path.touch()
        self.current_tmp_rollback_source_file_idx = rollback_source_file_idx
        self.current_tmp_rollback_record_idx = rollback_record_idx
        self.current_tmp_rollback_token_offset = rollback_token_offset
        self.current_tmp_rollback_num_documents = rollback_num_documents
        self.current_tmp_rollback_num_tokens = rollback_num_tokens

    def _flush(self) -> None:
        if not self.buffer:
            return
        assert self.current_path is not None
        _append_tokens(self.current_path, self.buffer)
        self.buffer.clear()

    def _finish_part_if_full(
        self,
        *,
        end_source_file_idx: int,
        end_record_idx: int,
        end_token_offset: int,
        end_num_documents: int,
        end_num_tokens: int,
    ) -> None:
        if self.part_token_count < self.tokens_per_bin:
            return
        self._flush()
        assert self.current_path is not None
        final_name = self._bin_name()
        self.current_path.replace(self.output_dir / final_name)
        self.data_files.append(
            {
                "data_file": final_name,
                "num_tokens": self.part_token_count,
                "end_source_file_idx": end_source_file_idx,
                "end_record_idx": end_record_idx,
                "end_token_offset": end_token_offset,
                "end_num_documents": end_num_documents,
                "end_num_tokens": end_num_tokens,
            }
        )
        self.part_idx += 1
        self.part_token_count = 0
        self.current_path = None
        self.current_tmp_rollback_source_file_idx = None
        self.current_tmp_rollback_record_idx = 0
        self.current_tmp_rollback_token_offset = 0
        self.current_tmp_rollback_num_documents = None
        self.current_tmp_rollback_num_tokens = None

    def append(
        self,
        token_ids: list[int],
        *,
        source_file_idx: int,
        record_idx: int,
        token_offset_start: int,
        num_documents_before: int,
        num_tokens_before: int,
    ) -> None:
        offset = token_offset_start
        while offset < len(token_ids):
            if self.current_path is None:
                self._start_part(
                    rollback_source_file_idx=source_file_idx,
                    rollback_record_idx=record_idx,
                    rollback_token_offset=offset,
                    rollback_num_documents=num_documents_before,
                    rollback_num_tokens=(
                        num_tokens_before + offset - token_offset_start
                    ),
                )

            capacity = self.tokens_per_bin - self.part_token_count
            take = min(capacity, len(token_ids) - offset)
            self.buffer.extend(token_ids[offset : offset + take])
            self.part_token_count += take
            offset += take

            if len(self.buffer) >= self.chunk_size:
                self._flush()
            self._finish_part_if_full(
                end_source_file_idx=source_file_idx,
                end_record_idx=record_idx,
                end_token_offset=offset,
                end_num_documents=(
                    num_documents_before + 1
                    if offset == len(token_ids)
                    else num_documents_before
                ),
                end_num_tokens=num_tokens_before + offset - token_offset_start,
            )

    def checkpoint_state(self) -> dict[str, Any]:
        self._flush()
        return {
            "data_files": self.data_files,
            "part_idx": self.part_idx,
            "part_token_count": self.part_token_count,
            "current_tmp_file": (
                self.current_path.name if self.current_path is not None else None
            ),
            "current_tmp_rollback_source_file_idx": (
                self.current_tmp_rollback_source_file_idx
            ),
            "current_tmp_rollback_record_idx": self.current_tmp_rollback_record_idx,
            "current_tmp_rollback_token_offset": (
                self.current_tmp_rollback_token_offset
            ),
            "current_tmp_rollback_num_documents": (
                self.current_tmp_rollback_num_documents
            ),
            "current_tmp_rollback_num_tokens": self.current_tmp_rollback_num_tokens,
        }

    def close(self) -> list[dict[str, int | str]]:
        self._flush()
        if self.current_path is not None and self.part_token_count > 0:
            final_name = self._bin_name()
            self.current_path.replace(self.output_dir / final_name)
            self.data_files.append(
                {"data_file": final_name, "num_tokens": self.part_token_count}
            )
            self.part_idx += 1
            self.part_token_count = 0
            self.current_path = None
            self.current_tmp_rollback_source_file_idx = None
            self.current_tmp_rollback_num_documents = None
            self.current_tmp_rollback_num_tokens = None
        return self.data_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-tokenize Dolma JSONL files into raw uint32 token bins."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "/home/ruisizhang123/ruisizhang123_data/tree/"
            "dolma3_mix-6T-1025-7B/data/common_crawl-religion-0016"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for part bins and metadata.json. Defaults to "
            "<dolma-root>/pre-tokenize-data/<input-dir-name>."
        ),
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path("./assets/hf/Olmo-3-1025-7B"),
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help=(
            "Glob for input files. Defaults to all supported JSONL formats: "
            "*.jsonl, *.jsonl.gz, and *.jsonl.zst."
        ),
    )
    parser.add_argument("--text-field", default="text")
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=34521,
        help=(
            "Seed for deterministic document shuffling within each input file. "
            "Each input file is materialized in memory before tokenization."
        ),
    )
    parser.add_argument("--output-prefix", default="part")
    parser.add_argument("--tokens-per-bin", type=int, default=8_000_000_000)
    parser.add_argument("--add-bos", action="store_true")
    parser.add_argument("--no-add-eos", dest="add_eos", action="store_false")
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(add_eos=True)
    return parser.parse_args()


def _load_progress(progress_path: Path) -> dict[str, Any] | None:
    if not progress_path.exists():
        return None
    with progress_path.open() as f:
        return json.load(f)


def _validate_completed_metadata(
    metadata_path: Path,
    *,
    shuffle_seed: int,
) -> None:
    with metadata_path.open() as file:
        metadata = json.load(file)
    if (
        metadata.get("format_version") != FORMAT_VERSION
        or metadata.get("shuffle_seed") != shuffle_seed
    ):
        raise ValueError(
            f"Completed output at {metadata_path.parent} is incompatible with "
            "the configured document shuffle. Pass --overwrite to regenerate it."
        )


def _cursor_from_last_data_file(
    progress: dict[str, Any],
) -> tuple[int, int, int, int, int] | None:
    data_files = progress.get("data_files", [])
    if not data_files:
        return None

    last_data_file = data_files[-1]
    required_keys = (
        "end_source_file_idx",
        "end_record_idx",
        "end_token_offset",
        "end_num_documents",
        "end_num_tokens",
    )
    if not all(key in last_data_file for key in required_keys):
        return None
    return (
        int(last_data_file["end_source_file_idx"]),
        int(last_data_file["end_record_idx"]),
        int(last_data_file["end_token_offset"]),
        int(last_data_file["end_num_documents"]),
        int(last_data_file["end_num_tokens"]),
    )


def _current_tmp_rollback_state(
    progress: dict[str, Any],
) -> tuple[int, int, int, int, int] | None:
    rollback_source_file_idx = progress.get("current_tmp_rollback_source_file_idx")
    rollback_num_documents = progress.get("current_tmp_rollback_num_documents")
    rollback_num_tokens = progress.get("current_tmp_rollback_num_tokens")
    if (
        rollback_source_file_idx is not None
        and rollback_num_documents is not None
        and rollback_num_tokens is not None
    ):
        return (
            int(rollback_source_file_idx),
            int(progress.get("current_tmp_rollback_record_idx", 0)),
            int(progress.get("current_tmp_rollback_token_offset", 0)),
            int(rollback_num_documents),
            int(rollback_num_tokens),
        )
    if progress.get("current_tmp_file") is not None and not progress.get("data_files"):
        return (0, 0, 0, 0, 0)
    return None


def _repair_current_tmp_for_resume(
    output_dir: Path, progress_path: Path, progress: dict[str, Any]
) -> dict[str, Any]:
    current_tmp_file = progress.get("current_tmp_file")
    if current_tmp_file is None:
        return progress

    current_path = output_dir / current_tmp_file
    expected_size = int(progress.get("part_token_count", 0)) * 4
    current_size = current_path.stat().st_size if current_path.exists() else None
    if current_size == expected_size:
        return progress

    rollback_state = _cursor_from_last_data_file(progress)
    if rollback_state is None:
        rollback_state = _current_tmp_rollback_state(progress)
    if rollback_state is None:
        actual_size = "missing" if current_size is None else f"{current_size} bytes"
        raise ValueError(
            f"Tmp bin {current_path} is {actual_size}, but progress.json "
            f"expects {expected_size} bytes and has no finalized-bin cursor "
            "or tmp rollback cursor. Pass --overwrite to restart."
        )

    (
        rollback_source_file_idx,
        rollback_record_idx,
        rollback_token_offset,
        rollback_num_documents,
        rollback_num_tokens,
    ) = rollback_state
    actual_size = "missing" if current_size is None else f"{current_size} bytes"
    print(
        f"Discarding tmp bin {current_path} ({actual_size}; expected "
        f"{expected_size} bytes) and rewinding to source file "
        f"{rollback_source_file_idx}, record {rollback_record_idx}, "
        f"token offset {rollback_token_offset}",
        flush=True,
    )
    current_path.unlink(missing_ok=True)

    repaired_progress = dict(progress)
    repaired_progress["next_source_file_idx"] = rollback_source_file_idx
    repaired_progress["next_record_idx"] = rollback_record_idx
    repaired_progress["next_token_offset"] = rollback_token_offset
    repaired_progress["num_documents"] = rollback_num_documents
    repaired_progress["num_tokens"] = rollback_num_tokens
    repaired_progress["part_idx"] = len(repaired_progress.get("data_files", []))
    repaired_progress["part_token_count"] = 0
    repaired_progress["current_tmp_file"] = None
    repaired_progress["current_tmp_rollback_source_file_idx"] = None
    repaired_progress["current_tmp_rollback_record_idx"] = 0
    repaired_progress["current_tmp_rollback_token_offset"] = 0
    repaired_progress["current_tmp_rollback_num_documents"] = None
    repaired_progress["current_tmp_rollback_num_tokens"] = None
    repaired_progress["current_tmp_source_file_start_idx"] = None
    repaired_progress["current_tmp_source_file_end_idx"] = None
    _write_json_atomic(progress_path, repaired_progress)
    return repaired_progress


def _cleanup_for_resume(
    output_dir: Path, *, progress: dict[str, Any] | None, overwrite: bool
) -> None:
    metadata_path = output_dir / "metadata.json"
    progress_path = output_dir / "progress.json"

    if overwrite:
        for pattern in ("*.bin", "*.bin.tmp"):
            for path in output_dir.glob(pattern) if output_dir.exists() else []:
                path.unlink()
        metadata_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
        return

    if progress is None:
        existing_bins = []
        existing_tmp_bins = []
        if output_dir.exists():
            existing_bins.extend(output_dir.glob("*.bin"))
            existing_tmp_bins.extend(output_dir.glob("*.bin.tmp"))
        if existing_bins:
            raise ValueError(
                f"Output directory {output_dir} contains bin files but no "
                "progress.json. Pass --overwrite to replace them."
            )
        for path in existing_tmp_bins:
            print(
                f"Removing stale tmp bin with no progress.json: {path}",
                flush=True,
            )
            path.unlink()
        return

    kept_bins = {entry["data_file"] for entry in progress.get("data_files", [])}
    current_tmp_file = progress.get("current_tmp_file")
    for path in output_dir.glob("*.bin"):
        if path.name not in kept_bins:
            print(
                f"Removing stale bin not tracked by progress.json: {path}",
                flush=True,
            )
            path.unlink()
    for path in output_dir.glob("*.bin.tmp"):
        if path.name != current_tmp_file:
            print(
                f"Removing stale tmp bin not tracked by progress.json: {path}",
                flush=True,
            )
            path.unlink()


def _make_progress_payload(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    source_files: list[Path],
    next_source_file_idx: int,
    num_documents: int,
    num_tokens: int,
    writer: TokenBinWriter,
) -> dict[str, Any]:
    payload = {
        "format": "pretokenized_uint32_bins_progress",
        "format_version": FORMAT_VERSION,
        "dtype": "uint32",
        "num_tokens": num_tokens,
        "num_documents": num_documents,
        "tokens_per_bin": args.tokens_per_bin,
        "next_source_file_idx": next_source_file_idx,
        "next_record_idx": 0,
        "next_token_offset": 0,
        "num_source_files": len(source_files),
        "source_name": args.input_dir.name,
        "source_dir": str(args.input_dir),
        "source_files": [str(path) for path in source_files],
        "tokenizer_path": str(args.tokenizer_path),
        "text_field": args.text_field,
        "add_bos": args.add_bos,
        "add_eos": args.add_eos,
        "shuffle_seed": args.shuffle_seed,
    }
    payload.update(writer.checkpoint_state())
    if (
        payload.get("current_tmp_file") is not None
        and payload.get("current_tmp_rollback_source_file_idx") is not None
    ):
        payload["current_tmp_source_file_start_idx"] = payload[
            "current_tmp_rollback_source_file_idx"
        ]
        payload["current_tmp_source_file_end_idx"] = next_source_file_idx
    else:
        payload["current_tmp_source_file_start_idx"] = None
        payload["current_tmp_source_file_end_idx"] = None
    return payload


def main() -> None:
    args = parse_args()

    if not args.input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {args.input_dir}")
    if args.tokens_per_bin <= 0:
        raise ValueError("--tokens-per-bin must be positive")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    output_dir = args.output_dir
    if output_dir is None:
        if args.input_dir.parent.name != "data":
            raise ValueError(
                "Cannot infer output directory because input-dir is not under "
                "a data/ directory. Pass --output-dir explicitly."
            )
        output_dir = (
            args.input_dir.parent.parent / "pre-tokenize-data" / args.input_dir.name
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    progress_path = output_dir / "progress.json"
    if metadata_path.exists() and not args.overwrite:
        _validate_completed_metadata(metadata_path, shuffle_seed=args.shuffle_seed)
        print(
            f"Skipping {args.input_dir.name}: found complete metadata at "
            f"{metadata_path}. Pass --overwrite to regenerate.",
            flush=True,
        )
        return
    progress = None if args.overwrite else _load_progress(progress_path)
    _cleanup_for_resume(output_dir, progress=progress, overwrite=args.overwrite)

    source_files = _input_files(args.input_dir, args.pattern)
    if not source_files:
        patterns = (
            [args.pattern] if args.pattern is not None else SUPPORTED_JSONL_PATTERNS
        )
        raise ValueError(
            f"No files matched {patterns!r} under {args.input_dir}"
        )

    if (
        any(path.suffix == ".zst" for path in source_files)
        and shutil.which("zstdcat") is None
    ):
        raise RuntimeError("zstdcat is required to read .zst files")

    if progress is not None:
        if progress.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                "progress.json was written by an incompatible pretokenization "
                "format. Pass --overwrite to restart."
            )
        if progress.get("source_files") != [str(path) for path in source_files]:
            raise ValueError(
                "progress.json source file list does not match current input. "
                "Pass --overwrite to restart."
            )
        if progress.get("tokens_per_bin") != args.tokens_per_bin:
            raise ValueError(
                "progress.json tokens_per_bin does not match current args. "
                "Pass --overwrite to restart."
            )
        if progress.get("shuffle_seed") != args.shuffle_seed:
            raise ValueError(
                "progress.json shuffle_seed does not match current args. "
                "Pass --overwrite to restart."
            )
        progress = _repair_current_tmp_for_resume(
            output_dir,
            progress_path,
            progress,
        )

    tokenizer = HuggingFaceTokenizer(tokenizer_path=str(args.tokenizer_path))

    start_source_file_idx = (
        int(progress.get("next_source_file_idx", 0)) if progress else 0
    )
    start_record_idx = int(progress.get("next_record_idx", 0)) if progress else 0
    start_token_offset = int(progress.get("next_token_offset", 0)) if progress else 0
    num_documents = int(progress.get("num_documents", 0)) if progress else 0
    num_tokens = int(progress.get("num_tokens", 0)) if progress else 0

    writer = TokenBinWriter(
        output_dir,
        output_prefix=args.output_prefix,
        tokens_per_bin=args.tokens_per_bin,
        chunk_size=args.chunk_size,
        progress=progress,
    )

    if progress is not None:
        current_tmp_file = progress.get("current_tmp_file")
        print(
            f"Resuming {args.input_dir.name} from source file "
            f"{start_source_file_idx}/{len(source_files)}, record "
            f"{start_record_idx}, token offset {start_token_offset} with "
            f"{num_documents} documents and {num_tokens} tokens",
            flush=True,
        )
        if current_tmp_file is not None:
            print(
                f"Reusing tmp bin {output_dir / current_tmp_file} at "
                f"{writer.part_token_count} tokens",
                flush=True,
            )

    stop_after_current_file = False
    for source_file_idx in range(start_source_file_idx, len(source_files)):
        source_file = source_files[source_file_idx]
        resume_record_idx = (
            start_record_idx if source_file_idx == start_source_file_idx else 0
        )
        records = _shuffled_records(
            source_file,
            shuffle_seed=args.shuffle_seed,
            source_file_idx=source_file_idx,
        )
        for record_idx, record in enumerate(records):
            if record_idx < resume_record_idx:
                continue

            text = record.get(args.text_field)
            if not isinstance(text, str):
                continue

            token_ids = tokenizer.encode(
                text,
                add_bos=args.add_bos,
                add_eos=args.add_eos,
            )
            token_offset_start = (
                start_token_offset
                if (
                    source_file_idx == start_source_file_idx
                    and record_idx == start_record_idx
                )
                else 0
            )
            if token_offset_start > len(token_ids):
                raise ValueError(
                    f"Resume token offset {token_offset_start} exceeds "
                    f"record {record_idx} token count {len(token_ids)} in "
                    f"{source_file}. Pass --overwrite to restart."
                )
            if token_offset_start == len(token_ids):
                continue

            writer.append(
                token_ids,
                source_file_idx=source_file_idx,
                record_idx=record_idx,
                token_offset_start=token_offset_start,
                num_documents_before=num_documents,
                num_tokens_before=num_tokens,
            )
            num_documents += 1
            num_tokens += len(token_ids) - token_offset_start

            if args.max_documents is not None and num_documents >= args.max_documents:
                stop_after_current_file = True
                break

        if stop_after_current_file:
            break

        progress_payload = _make_progress_payload(
            args=args,
            output_dir=output_dir,
            source_files=source_files,
            next_source_file_idx=source_file_idx + 1,
            num_documents=num_documents,
            num_tokens=num_tokens,
            writer=writer,
        )
        _write_json_atomic(progress_path, progress_payload)

    data_files = writer.close()

    metadata = {
        "format": "pretokenized_uint32_bins",
        "format_version": FORMAT_VERSION,
        "dtype": "uint32",
        "num_tokens": num_tokens,
        "num_documents": num_documents,
        "tokens_per_bin": args.tokens_per_bin,
        "data_files": data_files,
        "source_name": args.input_dir.name,
        "source_dir": str(args.input_dir),
        "source_files": [str(path) for path in source_files],
        "tokenizer_path": str(args.tokenizer_path),
        "text_field": args.text_field,
        "add_bos": args.add_bos,
        "add_eos": args.add_eos,
        "shuffle_seed": args.shuffle_seed,
    }
    _write_json_atomic(metadata_path, metadata)
    progress_path.unlink(missing_ok=True)

    print(
        f"Wrote {num_tokens} tokens from {num_documents} documents "
        f"to {len(data_files)} bin file(s) under {output_dir}"
    )
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
