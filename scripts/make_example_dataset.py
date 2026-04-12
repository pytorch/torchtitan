"""
Create small example local datasets (train + eval) that work with the trainer.

Saves HF Datasets to `./data/example` and `./data/example_eval`.

After running, point your config at them:
    data:
      dataset_path: "./data/example"
    eval:
      dataset_path: "./data/example_eval"
"""

import os

from datasets import Dataset


_LOREM = (
    "The mixture of experts model routes each token through a small subset of "
    "specialist feed-forward networks, letting the total parameter count grow "
    "without proportionally increasing compute per token. Flash attention "
    "computes softmax attention in tiles that fit in SRAM, avoiding the "
    "materialization of the full attention matrix and dramatically cutting "
    "memory traffic. Fully sharded data parallelism shards parameters, "
    "gradients, and optimizer state across ranks and all-gathers them just in "
    "time for forward and backward. Expert parallelism further distributes the "
    "routed experts across ranks so that each rank only holds a slice of the "
    "expert weight matrices. These techniques combine to make training "
    "multi-billion-parameter models on commodity interconnects feasible."
)


def _make_samples(prefix: str, n: int) -> list[dict]:
    return [{"text": f"{prefix} document {i}. " + _LOREM * 5} for i in range(n)]


def main():
    for out_dir, prefix, n in [
        ("./data/example", "Train", 500),
        ("./data/example_eval", "Eval", 200),
    ]:
        ds = Dataset.from_list(_make_samples(prefix, n))
        os.makedirs(out_dir, exist_ok=True)
        ds.save_to_disk(out_dir)
        print(f"Saved {len(ds)} samples to {out_dir}")


if __name__ == "__main__":
    main()
