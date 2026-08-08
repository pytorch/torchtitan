#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
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


def _iter_records(path: Path) -> Iterator[dict]:
    if path.suffix == ".zst":
        yield from _iter_jsonl_zst(path)
    else:
        yield from _iter_jsonl(path)


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

        current_tmp_file = progress.get("current_tmp_file")
        self.current_path = (
            self.output_dir / current_tmp_file if current_tmp_file is not None else None
        )
        if self.current_path is not None:
            self.current_path.parent.mkdir(parents=True, exist_ok=True)
            self.current_path.touch(exist_ok=True)
            with self.current_path.open("r+b") as f:
                f.truncate(self.part_token_count * 4)

    def _bin_name(self) -> str:
        return f"{self.output_prefix}-{self.part_idx:06d}.bin"

    def _tmp_name(self) -> str:
        return self._bin_name() + ".tmp"

    def _start_part(self) -> None:
        self.current_path = self.output_dir / self._tmp_name()
        self.current_path.touch()

    def _flush(self) -> None:
        if not self.buffer:
            return
        assert self.current_path is not None
        _append_tokens(self.current_path, self.buffer)
        self.buffer.clear()

    def _finish_part_if_full(self) -> None:
        if self.part_token_count < self.tokens_per_bin:
            return
        self._flush()
        assert self.current_path is not None
        final_name = self._bin_name()
        self.current_path.replace(self.output_dir / final_name)
        self.data_files.append(
            {"data_file": final_name, "num_tokens": self.part_token_count}
        )
        self.part_idx += 1
        self.part_token_count = 0
        self.current_path = None

    def append(self, token_ids: list[int]) -> None:
        offset = 0
        while offset < len(token_ids):
            if self.current_path is None:
                self._start_part()

            capacity = self.tokens_per_bin - self.part_token_count
            take = min(capacity, len(token_ids) - offset)
            self.buffer.extend(token_ids[offset : offset + take])
            self.part_token_count += take
            offset += take

            if len(self.buffer) >= self.chunk_size:
                self._flush()
            self._finish_part_if_full()

    def checkpoint_state(self) -> dict[str, Any]:
        self._flush()
        return {
            "data_files": self.data_files,
            "part_idx": self.part_idx,
            "part_token_count": self.part_token_count,
            "current_tmp_file": (
                self.current_path.name if self.current_path is not None else None
            ),
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
    parser.add_argument("--pattern", default="*.jsonl.zst")
    parser.add_argument("--text-field", default="text")
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

    if metadata_path.exists():
        raise ValueError(
            f"Output directory {output_dir} already has metadata.json. "
            "It looks complete; pass --overwrite to replace it."
        )

    if progress is None:
        existing = []
        if output_dir.exists():
            existing.extend(output_dir.glob("*.bin"))
            existing.extend(output_dir.glob("*.bin.tmp"))
        if existing:
            raise ValueError(
                f"Output directory {output_dir} contains bin files but no "
                "progress.json. Pass --overwrite to replace them."
            )
        return

    kept_bins = {entry["data_file"] for entry in progress.get("data_files", [])}
    current_tmp_file = progress.get("current_tmp_file")
    for path in output_dir.glob("*.bin"):
        if path.name not in kept_bins:
            path.unlink()
    for path in output_dir.glob("*.bin.tmp"):
        if path.name != current_tmp_file:
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
        "format_version": 1,
        "dtype": "uint32",
        "num_tokens": num_tokens,
        "num_documents": num_documents,
        "tokens_per_bin": args.tokens_per_bin,
        "next_source_file_idx": next_source_file_idx,
        "num_source_files": len(source_files),
        "source_name": args.input_dir.name,
        "source_dir": str(args.input_dir),
        "source_files": [str(path) for path in source_files],
        "tokenizer_path": str(args.tokenizer_path),
        "text_field": args.text_field,
        "add_bos": args.add_bos,
        "add_eos": args.add_eos,
    }
    payload.update(writer.checkpoint_state())
    return payload


def main() -> None:
    args = parse_args()

    if not args.input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {args.input_dir}")
    if shutil.which("zstdcat") is None:
        raise RuntimeError("zstdcat is required to read .zst files")
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
    progress = None if args.overwrite else _load_progress(progress_path)
    _cleanup_for_resume(output_dir, progress=progress, overwrite=args.overwrite)

    source_files = sorted(args.input_dir.glob(args.pattern))
    if not source_files:
        raise ValueError(f"No files matched {args.pattern!r} under {args.input_dir}")

    if progress is not None:
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

    tokenizer = HuggingFaceTokenizer(tokenizer_path=str(args.tokenizer_path))

    start_source_file_idx = int(progress.get("next_source_file_idx", 0)) if progress else 0
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
        print(
            f"Resuming {args.input_dir.name} from source file "
            f"{start_source_file_idx}/{len(source_files)} with "
            f"{num_documents} documents and {num_tokens} tokens",
            flush=True,
        )

    stop_after_current_file = False
    for source_file_idx in range(start_source_file_idx, len(source_files)):
        source_file = source_files[source_file_idx]
        for record in _iter_records(source_file):
            text = record.get(args.text_field)
            if not isinstance(text, str):
                continue

            token_ids = tokenizer.encode(
                text,
                add_bos=args.add_bos,
                add_eos=args.add_eos,
            )
            writer.append(token_ids)
            num_documents += 1
            num_tokens += len(token_ids)

            if args.progress_every > 0 and num_documents % args.progress_every == 0:
                print(
                    f"Processed {num_documents} documents and {num_tokens} tokens",
                    flush=True,
                )

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
        "format_version": 1,
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
