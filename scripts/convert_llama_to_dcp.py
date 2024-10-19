import argparse
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as DCP

@torch.inference_mode()
def convert_llama_weights(input_dir, output_dir):
    with open(args.input_dir / "params.json", "r") as f:
        params = json.load(f)
    n_layers = params["n_layers"]
    n_heads = params["n_heads"]
    dim = params["dim"]
    dims_per_head = dim // n_heads

    checkpoint_list = sorted([file for file in input_dir.rglob("*.pth")])
    print(f"Loading from these files: {checkpoint_list}.")
    shards = [torch.load(ckpt, map_location="cpu", weights_only=True) for ckpt in checkpoint_list]

    n_heads_per_shard = n_heads // len(shards)
    num_key_value_heads = params["n_kv_heads"]
    n_kv_heads_per_shard = num_key_value_heads // len(shards)
    key_value_dim = dims_per_head * num_key_value_heads

    if len(shards) == 1:
        state_dict = shards[0]
    else: # sharded
        state_dict = {}
        for layer in range(n_layers):
            state_dict[f"layers.{layer}.attention_norm.weight"] = shards[0][f"layers.{layer}.attention_norm.weight"].clone() # replicated
            state_dict[f"layers.{layer}.ffn_norm.weight"] = shards[0][f"layers.{layer}.ffn_norm.weight"].clone() # replicated

            for wn, nh in [("wq", n_heads_per_shard), 
                           ("wk", n_kv_heads_per_shard), 
                           ("wv", n_kv_heads_per_shard)]:
                state_dict[f"layers.{layer}.attention.{wn}.weight"] = torch.cat(
                    [shards[i][f"layers.{layer}.attention.{wn}.weight"].view(nh, dims_per_head, dim) for i in range(len(shards))], dim=0
                ).reshape(nh * len(shards) * dims_per_head, dim)

            state_dict[f"layers.{layer}.attention.wo.weight"] = torch.cat([shards[i][f"layers.{layer}.attention.wo.weight"] for i in range(len(shards))], dim=1)
            state_dict[f"layers.{layer}.feed_forward.w1.weight"] = torch.cat([shards[i][f"layers.{layer}.feed_forward.w1.weight"] for i in range(len(shards))], dim=0)
            state_dict[f"layers.{layer}.feed_forward.w2.weight"] = torch.cat([shards[i][f"layers.{layer}.feed_forward.w2.weight"] for i in range(len(shards))], dim=1)
            state_dict[f"layers.{layer}.feed_forward.w3.weight"] = torch.cat([shards[i][f"layers.{layer}.feed_forward.w3.weight"] for i in range(len(shards))], dim=0)

        state_dict["norm.weight"] = shards[0]["norm.weight"]
        state_dict["tok_embeddings.weight"] = torch.cat([shards[i]["tok_embeddings.weight"] for i in range(len(shards))], dim=0)
        state_dict["output.weight"] = torch.cat([shards[i]["output.weight"] for i in range(len(shards))], dim=0)

    print("Writing to DCP...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    storage_writer = DCP.filesystem.FileSystemWriter(output_dir)
    DCP.save({"model": state_dict}, storage_writer=storage_writer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Llama weights to DCP format.")
    parser.add_argument(
        "input_dir", type=Path, help="Input directory with original Llama weights."
    )
    parser.add_argument("output_dir", type=Path, help="Output directory for DCP.")
    args = parser.parse_args()

    convert_llama_weights(args.input_dir, args.output_dir)
