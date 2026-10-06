# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts.dsv3_671b_dist_moe_256gpu.paired_numerics_runtime import (
    parse_capture_steps,
    tensor_sha256,
)
from scripts.dsv3_671b_dist_moe_256gpu.run_paired_numerics import (
    compare_config_contracts,
    compare_gradient_manifests,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _gradient_manifest(scenario: str, digests: dict[str, str]) -> dict:
    return {
        "format_version": 1,
        "scenario": scenario,
        "step": 2,
        "rank": 0,
        "world_size": 1,
        "parameters": [
            {
                "index": index,
                "fqn": fqn,
                "dtype": "torch.bfloat16",
                "global_shape": [2],
                "local_shape": [2],
                "placements": [],
                "gradient_sha256": digest,
            }
            for index, (fqn, digest) in enumerate(digests.items())
        ],
    }


def test_parse_capture_steps() -> None:
    assert parse_capture_steps("4, 1,2") == (1, 2, 4)
    with pytest.raises(ValueError, match="duplicate"):
        parse_capture_steps("1,1")
    with pytest.raises(ValueError, match="positive"):
        parse_capture_steps("0")


def test_tensor_sha256_hashes_bfloat16_storage_bytes() -> None:
    tensor = torch.tensor([1.0, -2.0, 0.5], dtype=torch.bfloat16)
    expected = hashlib.sha256(
        tensor.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()
    assert tensor_sha256(tensor, chunk_size=2) == expected


def test_compare_gradient_manifests_reports_exact_and_mismatch(
    tmp_path: Path,
) -> None:
    eager = _gradient_manifest("eager", {"part0.a": "a", "part0.b": "b"})
    graph = _gradient_manifest(
        "graph_trainer", {"part0.a": "a", "part0.b": "different"}
    )
    for scenario, manifest in (("eager", eager), ("graph_trainer", graph)):
        _write_json(
            tmp_path / "gradient_sha256" / scenario / "step_2" / "rank_0.json",
            manifest,
        )

    result = compare_gradient_manifests(
        tmp_path,
        capture_steps=(2,),
        world_size=1,
    )

    assert not result["equal"]
    assert result["total_parameters"] == 2
    assert result["total_matches"] == 1
    assert result["total_mismatches"] == 1
    mismatch = result["comparisons"][0]["mismatches"][0]
    assert mismatch["fqn"] == "part0.b"
    assert mismatch["differing_fields"] == ["gradient_sha256"]


def test_compare_config_contracts_enforces_rank_and_pair_parity(
    tmp_path: Path,
) -> None:
    common = {
        "debug": {
            "seed": 42,
            "deterministic": True,
            "deterministic_warn_only": False,
        }
    }
    for scenario in ("eager", "graph_trainer"):
        for rank in range(2):
            _write_json(
                tmp_path / "config_contracts" / scenario / f"rank_{rank}.json",
                {
                    "rank": rank,
                    "common": common,
                    "implementation": {"scenario": scenario},
                },
            )
    assert compare_config_contracts(tmp_path, world_size=2)["equal"]

    graph_rank_one = tmp_path / "config_contracts" / "graph_trainer" / "rank_1.json"
    _write_json(
        graph_rank_one,
        {
            "rank": 1,
            "common": {"debug": {"seed": 7}},
            "implementation": {"scenario": "graph_trainer"},
        },
    )
    result = compare_config_contracts(tmp_path, world_size=2)
    assert not result["equal"]
    assert any("rank 1" in error for error in result["errors"])
