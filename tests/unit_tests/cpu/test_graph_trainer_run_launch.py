# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.graph_trainer_runs import launch


@pytest.mark.parametrize("num_nodes", [1, 64])
@pytest.mark.parametrize("topology", ["spmd", "pp"])
def test_training_command(num_nodes: int, topology: str, tmp_path: Path) -> None:
    command = launch.training_command(
        topology, "graph_trainer", tmp_path, num_nodes=num_nodes
    )
    assert f"--nnodes={num_nodes}" in command
    assert "--nproc-per-node=4" in command
    assert command[command.index("--config") + 1] == (
        f"{topology}_{num_nodes * 4}gpu_graph_trainer"
    )
    assert command[command.index("--output-dir") + 1] == str(tmp_path)


def test_spec_copies_only_explicit_input_overrides(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GT_RUN_C4_SOURCE", "/dataset")
    monkeypatch.setenv("GT_RUN_TOKENIZER_PATH", "/tokenizer")
    monkeypatch.setenv("SECRET_TOKEN", "must-not-appear")
    monkeypatch.setattr(
        subprocess, "check_output", lambda *_args, **_kwargs: "abc123\n"
    )
    spec = launch.job_spec("pp", "eager", tmp_path, mode="benchmark", steps=60)
    assert spec["num_nodes"] == 64
    assert spec["gpus_per_node"] == 4
    assert spec["priority"] == "normal"
    assert spec["source_commit"] == "abc123"
    assert spec["env"]["GT_RUN_C4_SOURCE"] == "/dataset"
    assert spec["env"]["GT_RUN_SOURCE_COMMIT"] == "abc123"
    assert "SECRET_TOKEN" not in spec["env"]
    assert "must-not-appear" not in json.dumps(spec)
    script = launch.slurm_script(spec)
    assert "--ntasks=64" in script
    assert "--gpus-per-task=4" in script
    assert "--topology pp --implementation eager" in script


def test_local_dry_run_never_probes_gpu_or_starts_training(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["launch", "local", "spmd", "--output-dir", str(tmp_path), "--dry-run"],
    )

    def unexpected(*_args, **_kwargs):
        pytest.fail("dry run must not probe GPUs or start a process")

    monkeypatch.setattr(launch, "require_idle_gpus", unexpected)
    monkeypatch.setattr(subprocess, "run", unexpected)
    launch.main()
    assert "run_paired_numerics" in capsys.readouterr().out


def test_submit_preview_is_four_plans_without_submission(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["launch", "submit", "all", "--output-dir", str(tmp_path)]
    )
    monkeypatch.setattr(
        subprocess, "check_output", lambda *_args, **_kwargs: "abc123\n"
    )

    def unexpected(*_args, **_kwargs):
        pytest.fail("preview must not submit jobs")

    monkeypatch.setattr(subprocess, "run", unexpected)
    launch.main()
    output = capsys.readouterr().out
    assert output.count('"num_nodes": 64') == 4
    for topology in ("spmd", "pp"):
        for implementation in ("eager", "graph_trainer"):
            assert f'"name": "gt-{topology}-{implementation}-256"' in output


def test_busy_gpus_are_rejected_without_stopping_processes(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")

    def query(command, **_kwargs):
        if "--query-compute-apps=gpu_uuid" in command:
            return "GPU-0\n"
        return "".join(f"GPU-{index}, 0, 0\n" for index in range(4))

    monkeypatch.setattr(subprocess, "check_output", query)
    with pytest.raises(RuntimeError, match="busy"):
        launch.require_idle_gpus(wait=False)
