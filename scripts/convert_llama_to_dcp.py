import argparse
from pathlib import Path

import torch
import torch.distributed.checkpoint as DCP

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Llama weights to DCP format.")
    parser.add_argument(
        "input_dir", type=Path, help="Input directory with original Llama weights."
    )
    parser.add_argument("output_dir", type=Path, help="Output directory for DCP.")
    args = parser.parse_args()

    loaded = torch.load(
        args.input_dir / "consolidated.00.pth", map_location="cpu", weights_only=True
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    storage_writer = DCP.filesystem.FileSystemWriter(args.output_dir)
    DCP.save({"model": loaded}, storage_writer=storage_writer)
