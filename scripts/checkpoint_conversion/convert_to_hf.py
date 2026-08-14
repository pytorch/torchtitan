# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import importlib
import io
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint import HuggingFaceStorageWriter
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.planner import LoadItemType
from torch.futures import Future
from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.components.checkpoint_utils import canonical_fqn
from torchtitan.config import TORCH_DTYPE_MAP

_EMA_STATE_PREFIX = "ema_optimizer.state."
_EMA_STATE_SUFFIX = ".ema_params"


class ParallelFileSystemReader(FileSystemReader):
    """FileSystemReader that reads shard files concurrently.

    torch's stock FileSystemReader.read_data reads DCP shard files strictly
    one at a time, one tensor read_item at a time, in the calling thread --
    no thread pool. For a checkpoint sharded into hundreds of files (one per
    training rank), that serializes what should be an I/O-bound operation
    onto a single core, even though converting a checkpoint needs no GPU and
    the host typically has many idle cores. This subclass keeps torch's
    exact per-item read/deserialize logic but fans the per-file work out
    across a thread pool, since each file (and each tensor within it) is
    read and committed independently.
    """

    def __init__(self, path, thread_count: int = 16):
        super().__init__(path)
        self.thread_count = max(1, thread_count)

    def read_data(self, plan, planner):
        per_file: dict[str, list] = {}
        for read_item in plan.items:
            item_md = self.storage_data[read_item.storage_index]
            per_file.setdefault(item_md.relative_path, []).append(read_item)

        def _read_one_file(relative_path, reqs) -> None:
            new_path = self.fs.concat_path(self.path, relative_path)
            with self.fs.create_stream(new_path, "rb") as stream:
                for req in reqs:
                    item_md = self.storage_data[req.storage_index]
                    file_slice = self._slice_file(stream, item_md)
                    transform_from = self.transforms.transform_load_stream(
                        req,
                        item_md.transform_descriptors or (),
                        file_slice,
                    )

                    if req.type == LoadItemType.BYTE_IO:
                        read_bytes = io.BytesIO(transform_from.read(-1))
                        read_bytes.seek(0)
                        planner.load_bytes(req, read_bytes)
                    else:
                        if transform_from.seekable():
                            seekable = transform_from
                        else:
                            seekable = io.BytesIO(transform_from.read(-1))
                            seekable.seek(0)

                        tensor = torch.load(
                            seekable, map_location="cpu", weights_only=True
                        )
                        tensor = narrow_tensor_by_index(
                            tensor, req.storage_offsets, req.lengths
                        )
                        target_tensor = planner.resolve_tensor(req).detach()
                        assert target_tensor.size() == tensor.size(), (
                            f"req {req.storage_index} mismatch sizes "
                            f"{target_tensor.size()} vs {tensor.size()}"
                        )
                        target_tensor.copy_(tensor)
                        planner.commit_tensor(req, target_tensor)

        # Clamp to the number of files this plan actually touches -- far more
        # pool threads than files (e.g. 64 threads for a 4-shard checkpoint)
        # has been observed to hang.
        num_workers = min(self.thread_count, len(per_file))
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            list(pool.map(lambda kv: _read_one_file(*kv), per_file.items()))

        fut: Future = Future()
        fut.set_result(None)
        return fut


def _load_ema_state_dict(model_state_dict, trainable_fqns, input_dir, read_threads=16):
    """Load EMA weights from `input_dir` into a copy of `model_state_dict`,
    replacing only the trainable-parameter entries (EMA never tracks
    buffers). Returns None if the checkpoint has no EMA data (EMA was
    disabled during training, or the checkpoint predates EMA support).
    """
    metadata = dcp.FileSystemReader(input_dir).read_metadata()
    if not any(k.startswith(_EMA_STATE_PREFIX) for k in metadata.state_dict_metadata):
        return None

    flat_ema_sd = {
        f"{_EMA_STATE_PREFIX}{fqn}{_EMA_STATE_SUFFIX}": torch.empty_like(
            model_state_dict[fqn]
        )
        for fqn in trainable_fqns
    }
    dcp.load(
        flat_ema_sd,
        storage_reader=ParallelFileSystemReader(input_dir, thread_count=read_threads),
    )

    ema_state_dict = dict(model_state_dict)
    for fqn in trainable_fqns:
        ema_state_dict[fqn] = flat_ema_sd[
            f"{_EMA_STATE_PREFIX}{fqn}{_EMA_STATE_SUFFIX}"
        ]
    return ema_state_dict


def _save_as_hf(state_dict, sd_adapter, target_dtype, output_dir):
    hf_state_dict = sd_adapter.to_hf(state_dict)
    if target_dtype != torch.float32:
        hf_state_dict = {k: v.to(target_dtype) for k, v in hf_state_dict.items()}

    storage_writer = HuggingFaceStorageWriter(
        path=output_dir,
        save_distributed=True,
        fqn_to_index_mapping=sd_adapter.fqn_to_index_mapping,
        enable_consolidation=True,
        thread_count_consolidation=5,
    )
    dcp.save(hf_state_dict, storage_writer=storage_writer)


@torch.inference_mode()
def convert_to_hf(
    input_dir,
    output_dir,
    model_name,
    model_flavor,
    hf_assets_path,
    export_dtype,
    ema_output=None,
    read_threads=16,
):
    # load model and model args so that we can get the state dict shape
    model_module = importlib.import_module(f"torchtitan.models.{model_name}")
    model_spec = model_module.model_registry(model_flavor)
    model_config = model_spec.model

    with torch.device("cpu"):
        raw_model = model_config.build()
    trainable_fqns = [
        canonical_fqn(name)
        for name, p in raw_model.named_parameters()
        if p.requires_grad
    ]
    model = ModelWrapper(raw_model)

    sd_adapter = model_spec.state_dict_adapter(model_config, hf_assets_path)
    assert (
        sd_adapter is not None
    ), "trying to convert checkpoint from DCP to HF safetensors format, but sd_adapter is not provided."

    # allocate state dict memory with empty weights to load checkpoint
    state_dict = model._get_state_dict()
    dcp.load(
        state_dict,
        storage_reader=ParallelFileSystemReader(input_dir, thread_count=read_threads),
    )

    target_dtype = TORCH_DTYPE_MAP[export_dtype]
    _save_as_hf(state_dict, sd_adapter, target_dtype, output_dir)

    if ema_output is not None:
        ema_state_dict = _load_ema_state_dict(
            state_dict, trainable_fqns, input_dir, read_threads
        )
        if ema_state_dict is None:
            print(
                f"[WARNING] --ema_output was given but the checkpoint at {input_dir} "
                "has no EMA weights (EMA was disabled during that training run, or "
                "this checkpoint predates EMA support). Skipping EMA export."
            )
        else:
            _save_as_hf(ema_state_dict, sd_adapter, target_dtype, ema_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DCP weights to HF format.")
    parser.add_argument(
        "input_dir", type=Path, help="Input directory with DCP weights."
    )
    parser.add_argument(
        "output_dir", type=Path, help="Output directory for HF checkpoint."
    )
    parser.add_argument(
        "--hf_assets_path",
        type=Path,
        help="Path to HF assets directory. This is used to get the model.safetensors.index.json mapping",
        default="./assets/hf/Llama-3.1-8B",
    )
    parser.add_argument("--model_name", type=str, nargs="?", default="llama3")
    parser.add_argument("--model_flavor", type=str, nargs="?", default="8B")
    parser.add_argument(
        "--export_dtype",
        type=str,
        nargs="?",
        choices=["float16", "bfloat16", "float32"],
        default="float32",
        help="Export dtype for HF checkpoint (default: float32)",
    )
    parser.add_argument(
        "--ema_output",
        type=Path,
        default=None,
        help="If given and the checkpoint has EMA weights (see "
        "torchtitan.components.ema), also export them as a second HF "
        "checkpoint at this directory.",
    )
    parser.add_argument(
        "--read_threads",
        type=int,
        default=min(32, os.cpu_count() or 16),
        help="Number of threads used to read DCP checkpoint shard files in "
        "parallel (default: min(32, cpu_count)). This step is CPU/disk I/O "
        "bound, not GPU bound; the default torch DCP reader reads shard "
        "files one at a time on a single thread.",
    )
    args = parser.parse_args()

    convert_to_hf(
        args.input_dir,
        args.output_dir,
        args.model_name,
        args.model_flavor,
        args.hf_assets_path,
        args.export_dtype,
        ema_output=args.ema_output,
        read_threads=args.read_threads,
    )
