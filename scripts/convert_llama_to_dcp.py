# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as DCP
from torchtitan.models.llama.model import precompute_freqs_cis
from torchtitan.tools.logging import init_logger, logger


@torch.inference_mode()
def convert_llama_weights(input_dir, output_dir, max_seq_len: int):
    with open(input_dir / "params.json", "r") as f:
        params = json.load(f)
    n_layers = params["n_layers"]
    n_heads = params["n_heads"]
    dim = params["dim"]
    dims_per_head = dim // n_heads

    checkpoint_list = sorted([file for file in input_dir.rglob("*.pth")])
    logger.info(
        f"Loading original Llama weights from {[ckpt.name for ckpt in checkpoint_list]}"
    )
    shards = [
        torch.load(ckpt, map_location="cpu", weights_only=True, mmap=True)
        for ckpt in checkpoint_list
    ]

    if len(shards) == 1:
        state_dict = shards[0]
    else:  # sharded
        state_dict = {}
        n_heads_per_shard = n_heads // len(shards)
        num_key_value_heads = params["n_kv_heads"]
        n_kv_heads_per_shard = num_key_value_heads // len(shards)
        key_value_dim = dims_per_head * num_key_value_heads
        for layer in range(n_layers):
            state_dict[f"layers.{layer}.attention_norm.weight"] = shards[0][
                f"layers.{layer}.attention_norm.weight"
            ]
            for i in range(len(shards)):
                del shards[i][f"layers.{layer}.attention_norm.weight"]
            state_dict[f"layers.{layer}.ffn_norm.weight"] = shards[0][
                f"layers.{layer}.ffn_norm.weight"
            ]
            for i in range(len(shards)):
                del shards[i][f"layers.{layer}.ffn_norm.weight"]

            for wn, nh in [
                ("wq", n_heads_per_shard),
                ("wk", n_kv_heads_per_shard),
                ("wv", n_kv_heads_per_shard),
            ]:
                state_dict[f"layers.{layer}.attention.{wn}.weight"] = torch.cat(
                    [
                        shards[i][f"layers.{layer}.attention.{wn}.weight"].view(
                            nh, dims_per_head, dim
                        )
                        for i in range(len(shards))
                    ],
                    dim=0,
                ).reshape(nh * len(shards) * dims_per_head, dim)
                for i in range(len(shards)):
                    del shards[i][f"layers.{layer}.attention.{wn}.weight"]

            state_dict[f"layers.{layer}.attention.wo.weight"] = torch.cat(
                [
                    shards[i][f"layers.{layer}.attention.wo.weight"]
                    for i in range(len(shards))
                ],
                dim=1,
            )
            for i in range(len(shards)):
                del shards[i][f"layers.{layer}.attention.wo.weight"]

            state_dict[f"layers.{layer}.feed_forward.w1.weight"] = torch.cat(
                [
                    shards[i][f"layers.{layer}.feed_forward.w1.weight"]
                    for i in range(len(shards))
                ],
                dim=0,
            )
            for i in range(len(shards)):
                del shards[i][f"layers.{layer}.feed_forward.w1.weight"]

            state_dict[f"layers.{layer}.feed_forward.w2.weight"] = torch.cat(
                [
                    shards[i][f"layers.{layer}.feed_forward.w2.weight"]
                    for i in range(len(shards))
                ],
                dim=1,
            )
            for i in range(len(shards)):
                del shards[i][f"layers.{layer}.feed_forward.w2.weight"]

            state_dict[f"layers.{layer}.feed_forward.w3.weight"] = torch.cat(
                [
                    shards[i][f"layers.{layer}.feed_forward.w3.weight"]
                    for i in range(len(shards))
                ],
                dim=0,
            )
            for i in range(len(shards)):
                del shards[i][f"layers.{layer}.feed_forward.w3.weight"]

        state_dict["norm.weight"] = shards[0]["norm.weight"]
        for i in range(len(shards)):
            del shards[i]["norm.weight"]
        state_dict["tok_embeddings.weight"] = torch.cat(
            [shards[i]["tok_embeddings.weight"] for i in range(len(shards))], dim=0
        )
        for i in range(len(shards)):
            del shards[i]["tok_embeddings.weight"]
        state_dict["output.weight"] = torch.cat(
            [shards[i]["output.weight"] for i in range(len(shards))], dim=0
        )
        for i in range(len(shards)):
            del shards[i]["output.weight"]

    # NOTE: precompute freqs_cis because must be persisted by default in torchtitan
    state_dict["freqs_cis"] = precompute_freqs_cis(
        dims_per_head,
        max_seq_len,
        params.get("rope_theta", 500000),
    )

    logger.info(f"Writing to DCP at '{output_dir}'")
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_writer = DCP.filesystem.FileSystemWriter(output_dir, thread_count=8)
    DCP.save({"model": state_dict}, storage_writer=storage_writer)


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser(description="Convert Llama weights to DCP format.")
    parser.add_argument(
        "input_dir", type=Path, help="Input directory with original Llama weights."
    )
    parser.add_argument("output_dir", type=Path, help="Output directory for DCP.")
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=131072,
        help="The maximum sequence length of the model.",
    )
    args = parser.parse_args()

    convert_llama_weights(args.input_dir, args.output_dir, max_seq_len=args.max_seq_len)
