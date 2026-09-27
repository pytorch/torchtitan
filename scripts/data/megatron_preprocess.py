#!/usr/bin/env python3

# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

"""Tokenize text into Megatron ``.bin``/``.idx`` files for ``MegatronDataLoader``.

Port of lm-engine's ``tools/data/preprocess_data.py`` without its Ray and
remote-staging paths. Each JSON key gets its own ``<output-prefix>_<key>``
dataset. Input formats: ``.jsonl``, ``.jsonl.zst`` (needs ``zstandard``),
``.json.gz``, ``.parquet``, ``.arrow`` (pre-tokenized ``tokens`` column), or a
Hugging Face dataset name. A directory input is processed file by file, up to
``--max-local-processes`` in parallel.

Example:
    python scripts/data/megatron_preprocess.py --input data/train.jsonl \\
        --tokenizer ./assets/hf/Llama-3.1-8B --append-eod \\
        --output-prefix data/megatron/train
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import os
import subprocess
import sys
from argparse import ArgumentParser, Namespace
from collections.abc import Iterator

import pyarrow as pa
import torch
from datasets import load_dataset

from torchtitan.components.data.megatron.bin import get_bin_path
from torchtitan.components.data.megatron.dtype import DType
from torchtitan.components.data.megatron.indexed_dataset import (
    get_idx_path,
    MMapIndexedDatasetBuilder,
)
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.kernels.utils import is_zstandard_available
from tqdm import tqdm

if is_zstandard_available():
    from zstandard import ZstdDecompressor

logger = logging.getLogger(__name__)


class ArrowIterator:
    def __init__(self, filename: str) -> None:
        self.fin = pa.ipc.open_file(filename)
        self.num_records = self.fin.num_record_batches

    def __iter__(self) -> Iterator[list[int]]:
        for i in range(self.num_records):
            yield self.fin.get_batch(i)["tokens"].to_numpy().tolist()


class Encoder:
    def __init__(
        self, tokenizer: HuggingFaceTokenizer, json_keys: list[str], append_eod: bool
    ) -> None:
        self.tokenizer = tokenizer
        self.json_keys = json_keys
        self.append_eod = append_eod

    def _encode_data(self, data) -> dict:
        ids = {}
        for key in self.json_keys:
            document_ids = self.tokenizer.encode(data[key])
            if len(document_ids) > 0:
                if self.append_eod:
                    document_ids.append(self.tokenizer.eos_id)
                ids[key] = document_ids
        return ids

    def encode(self, json_line) -> dict:
        """Encode one JSONL line; a corrupt line or missing key returns {} so
        the caller counts and skips it instead of aborting the run."""
        try:
            return self._encode_data(json.loads(json_line))
        except (json.JSONDecodeError, KeyError, TypeError):
            return {}

    def encode_jsonl_zstd(self, bytes_obj) -> dict:
        try:
            json_str = bytes_obj.decode("utf-8")
        except UnicodeDecodeError:
            return {}
        return self.encode(json_str)

    def encode_hf(self, sample) -> dict:
        return self._encode_data(sample)

    def convert_fms_arrow_to_megatron(self, sample) -> dict:
        if len(sample) > 0 and self.append_eod:
            sample.append(self.tokenizer.eos_id)
        return {"text": [sample]}


def _zstd_lines(path: str) -> Iterator[bytes]:
    if not is_zstandard_available():
        raise ValueError(f"{path}: reading .jsonl.zst needs the zstandard package")
    with open(path, "rb") as compressed:
        with ZstdDecompressor().stream_reader(compressed) as reader:
            # A 64MB buffer keeps very long lines efficient to read.
            yield from io.BufferedReader(reader, buffer_size=64 * 1024 * 1024)


def convert_file(
    tokenizer: HuggingFaceTokenizer,
    input_file: str,
    output_prefix: str,
    subset: str | None = None,
    json_keys: list[str] = ["text"],
    append_eos_token: bool = True,
) -> int:
    """Tokenize ``input_file`` into ``<output_prefix>_<key>.{bin,idx}`` per key.
    Returns the number of skipped (unparsable) records."""
    encoder = Encoder(tokenizer, json_keys, append_eos_token)

    if subset is not None and not os.path.isdir(input_file):
        if input_file.endswith((".jsonl", ".jsonl.zst", ".json.gz", ".arrow")):
            raise ValueError(f"{input_file}: --subset only applies to HF datasets")

    if input_file.endswith(".jsonl"):
        encoded_docs = map(encoder.encode, open(input_file, "r", encoding="utf-8"))
    elif input_file.endswith(".jsonl.zst"):
        encoded_docs = map(encoder.encode_jsonl_zstd, _zstd_lines(input_file))
    elif input_file.endswith(".json.gz"):
        encoded_docs = map(
            encoder.encode, gzip.open(input_file, "rt", encoding="utf-8")
        )
    elif input_file.endswith(".parquet"):
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(input_file)

        def parquet_iterator():
            for batch in parquet_file.iter_batches(columns=json_keys, batch_size=10000):
                # to_pylist() is much faster than to_pandas() + row iteration.
                yield from batch.to_pylist()

        encoded_docs = map(encoder.encode_hf, parquet_iterator())
    elif input_file.endswith(".arrow"):
        encoded_docs = map(
            encoder.convert_fms_arrow_to_megatron, ArrowIterator(input_file)
        )
    else:
        ds = load_dataset(input_file, streaming=True, split="train", data_dir=subset)
        encoded_docs = map(encoder.encode_hf, ds)

    os.makedirs(os.path.dirname(os.path.abspath(output_prefix)), exist_ok=True)
    builders = {
        key: MMapIndexedDatasetBuilder(
            get_bin_path(f"{output_prefix}_{key}"),
            dtype=DType.optimal_dtype(tokenizer.get_vocab_size()),
        )
        for key in json_keys
    }

    skipped = 0
    for item in encoded_docs:
        if not item:
            skipped += 1
            continue
        for key, document in item.items():
            builders[key].add_item(torch.IntTensor(document))
            builders[key].end_document()

    for key in json_keys:
        builders[key].finalize(get_idx_path(f"{output_prefix}_{key}"))

    return skipped


def collect_files(args: Namespace) -> list[tuple[str, str]]:
    """(input file, output prefix) pairs; a directory input mirrors its layout."""
    if os.path.isfile(args.input):
        return [(args.input, args.output_prefix)]

    files = []
    for root, _, names in os.walk(args.input):
        for name in names:
            if name.startswith("."):
                continue
            output_dir = os.path.join(
                args.output_prefix, root.removeprefix(args.input).lstrip(os.path.sep)
            )
            os.makedirs(output_dir, exist_ok=True)
            output_prefix = os.path.join(output_dir, os.path.splitext(name)[0])
            # .jsonl.zst has a two-part extension.
            if output_prefix.endswith(".jsonl"):
                output_prefix = os.path.splitext(output_prefix)[0]
            files.append((os.path.join(root, name), output_prefix))

    return sorted(files, key=lambda x: x[0])


def process_with_subprocess(args: Namespace, files: list[tuple[str, str]]) -> None:
    """Tokenize each file in its own process, at most ``max_local_processes`` at once."""
    logger.info(
        f"Processing {len(files)} files with subprocesses "
        f"(max {args.max_local_processes} parallel)"
    )
    processes = []
    for input_file, output_prefix in tqdm(files, desc="Tokenizing"):
        cmd = [
            sys.executable,
            __file__,
            "--input",
            input_file,
            "--output-prefix",
            output_prefix,
            "--tokenizer",
            args.tokenizer,
            "--json-keys",
            *args.json_keys,
        ]
        if args.subset:
            cmd += ["--subset", args.subset]
        if args.append_eod:
            cmd += ["--append-eod"]

        while len(processes) >= args.max_local_processes:
            processes = [p for p in processes if p.poll() is None]

        processes.append(subprocess.Popen(cmd))

    failed = [p.args for p in tqdm(processes, desc="Waiting for jobs") if p.wait() != 0]
    if failed:
        raise RuntimeError(f"{len(failed)} tokenization jobs failed: {failed}")


def get_args() -> Namespace:
    parser = ArgumentParser(description=__doc__.splitlines()[0])

    group = parser.add_argument_group(title="input data")
    group.add_argument(
        "--input", type=str, required=True, help="Input file, directory, or HF dataset"
    )
    group.add_argument(
        "--subset", type=str, default=None, help="data_dir of an HF dataset input"
    )
    group.add_argument(
        "--json-keys", nargs="+", default=["text"], help="Keys to extract from JSON"
    )

    group = parser.add_argument_group(title="tokenizer")
    group.add_argument(
        "--tokenizer", type=str, required=True, help="Path to HF tokenizer files"
    )
    group.add_argument(
        "--append-eod", action="store_true", help="Append EOS after each document"
    )

    group = parser.add_argument_group(title="output data")
    group.add_argument(
        "--output-prefix", type=str, required=True, help="Output path without suffix"
    )

    group = parser.add_argument_group(title="runtime")
    group.add_argument(
        "--max-local-processes",
        type=int,
        default=16,
        help="Parallel processes for a directory input",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = get_args()

    if os.path.isfile(args.input) or not os.path.exists(args.input):
        # A single file, or an HF dataset name.
        skipped = convert_file(
            tokenizer=HuggingFaceTokenizer(tokenizer_path=args.tokenizer),
            input_file=args.input,
            output_prefix=args.output_prefix,
            subset=args.subset,
            json_keys=args.json_keys,
            append_eos_token=args.append_eod,
        )
        logger.info(f"Processed {args.input} ({skipped} records skipped)")
        return

    files = collect_files(args)
    if not files:
        logger.info(f"No files found under {args.input}")
        return
    process_with_subprocess(args, files)
    logger.info("All files processed")


if __name__ == "__main__":
    main()
