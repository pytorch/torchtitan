# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Build a SWE-R2E training JSONL from the R2E-Gym HuggingFace datasets.

Each output row is exactly what ``SWER2EDataset`` (data.py) consumes::

    {
      "prompt": <problem_statement>,
      "label":  <instance_id>,
      "metadata": {
        "instance_id", "image" (docker.io/...), "workdir": "/testbed",
        "problem_statement",
        "r2e": {"test_file_names", "test_file_codes", "expected_output_json"}
      }
    }

The R2E-Gym datasets are sorted by repo, so a contiguous ``offset`` slice is a
single project. To get a DIVERSE multi-repo training set we sample evenly across
the whole split (``--num-points`` batches of ``--per-point`` rows spread by a
stride), dedup by instance_id, and drop rows missing a problem_statement or the
grading payload.

Host egress (the HF datasets-server REST API) required -- on a fwdproxy box run
with ``HTTPS_PROXY=http://fwdproxy:8080``. Example::

    HTTPS_PROXY=http://fwdproxy:8080 python -m \
        torchtitan.experiments.rl.examples.swe_r2e.prepare_r2e_data \
        --dataset R2E-Gym/R2E-Gym-Lite --num-points 100 --per-point 8 \
        --out mast_rl/swe_assets/r2e_train.jsonl

Sizes (rows): R2E-Gym-Subset 4578, R2E-Gym-V1 7478, R2E-Gym-Lite 11788.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request

_ROWS_URL = "https://datasets-server.huggingface.co/rows"
_SIZE_URL = "https://datasets-server.huggingface.co/size"


def _total_rows(dataset: str) -> int:
    url = f"{_SIZE_URL}?dataset={urllib.parse.quote(dataset)}"
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)["size"]["dataset"]["num_rows"]


def _fetch(dataset: str, offset: int, length: int) -> list[dict]:
    url = (
        f"{_ROWS_URL}?dataset={urllib.parse.quote(dataset)}"
        f"&config=default&split=train&offset={offset}&length={length}"
    )
    with urllib.request.urlopen(url, timeout=60) as r:
        return [x["row"] for x in json.load(r)["rows"]]


def _to_row(row: dict, image_prefix: str = "docker.io/") -> dict:
    erc = json.loads(row["execution_result_content"])
    instance_id = f"{row['repo_name']}-{row['commit_hash'][:10]}"
    image = row["docker_image"]
    if image_prefix and "/" in image and not image.startswith(image_prefix):
        image = image_prefix + image
    return {
        "prompt": row["problem_statement"],
        "label": instance_id,
        "metadata": {
            "instance_id": instance_id,
            "image": image,
            "workdir": "/testbed",
            "problem_statement": row["problem_statement"],
            "r2e": {
                "test_file_names": erc["test_file_names"],
                "test_file_codes": erc["test_file_codes"],
                "expected_output_json": row["expected_output_json"],
            },
        },
    }


def _is_complete(row: dict) -> bool:
    md = row["metadata"]
    r2e = md.get("r2e") or {}
    return bool(
        (md.get("problem_statement") or "").strip()
        and (md.get("image") or "").strip()
        and (r2e.get("expected_output_json") or "").strip()
        and r2e.get("test_file_codes")
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="R2E-Gym/R2E-Gym-Lite")
    ap.add_argument(
        "--num-points", type=int, default=100, help="evenly-spread sample points"
    )
    ap.add_argument("--per-point", type=int, default=8, help="rows fetched per point")
    ap.add_argument("--image-prefix", default="docker.io/")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n = _total_rows(args.dataset)
    stride = max(1, n // args.num_points)
    by_id: dict[str, dict] = {}
    for i in range(args.num_points):
        offset = min(i * stride, max(0, n - args.per_point))
        try:
            for raw in _fetch(args.dataset, offset, args.per_point):
                row = _to_row(raw, args.image_prefix)
                if _is_complete(row):
                    by_id.setdefault(row["metadata"]["instance_id"], row)
        except Exception as e:  # noqa: BLE001 -- best-effort per point
            print(f"  offset {offset}: {type(e).__name__}: {e}", file=sys.stderr)

    with open(args.out, "w") as f:
        for row in by_id.values():
            f.write(json.dumps(row) + "\n")
    print(
        f"wrote {len(by_id)} unique complete tasks -> {args.out} ({args.dataset}, N={n})"
    )


if __name__ == "__main__":
    main()
