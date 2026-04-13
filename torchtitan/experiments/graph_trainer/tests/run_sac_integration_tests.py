#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.distributed.checkpoint.state_dict import get_model_state_dict

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.tools.logging import logger
from torchtitan.tools import utils

TB_LOSS_TAG = "loss_metrics/global_avg_loss"
TB_MAX_RESERVED_TAG = "memory/max_reserved(GiB)"
TB_MAX_ACTIVE_TAG = "memory/max_active(GiB)"
FIXED_OPTIONS = (
    "--debug.deterministic --debug.seed=42 "
    "--metrics.enable_tensorboard --metrics.log_freq=1"
)
RUN_TRAIN_SCRIPT = "./run_train.sh"


@dataclass(frozen=True)
class SACTestDefinition:
    test_name: str
    test_descr: str
    baseline_module: str
    baseline_config: str
    baseline_options: str
    test_module: str
    test_config: str
    test_options: str
    ngpu: int = 1
    steps: int = 1
    max_reserved_ratio: float = 1.15
    use_seed_checkpoint: bool = False


@dataclass(frozen=True)
class RunMetrics:
    losses: dict[int, float]
    max_reserved_gib: float
    max_reserved_step: int
    max_active_gib: float
    max_active_step: int


def _run_cmd(cmd: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [cmd],
        shell=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        env=env,
    )


def _single_event_dir(tb_root: Path) -> Path:
    if not tb_root.exists():
        raise FileNotFoundError(f"TensorBoard directory not found: {tb_root}")
    subdirs = sorted(path for path in tb_root.iterdir() if path.is_dir())
    if len(subdirs) != 1:
        raise RuntimeError(
            f"Expected exactly one TensorBoard run directory under {tb_root}, found {subdirs}"
        )
    return subdirs[0]


def _scalar_series(tb_root: Path, tag: str):
    event_dir = _single_event_dir(tb_root)
    event_accumulator = EventAccumulator(str(event_dir))
    event_accumulator.Reload()
    tags = event_accumulator.Tags().get("scalars", [])
    if tag not in tags:
        raise KeyError(f"Scalar tag {tag!r} not found in {event_dir}: {tags}")
    return event_accumulator.Scalars(tag)


def _extract_run_metrics(tb_root: Path) -> RunMetrics:
    loss_scalars = _scalar_series(tb_root, TB_LOSS_TAG)
    reserved_scalars = _scalar_series(tb_root, TB_MAX_RESERVED_TAG)
    active_scalars = _scalar_series(tb_root, TB_MAX_ACTIVE_TAG)

    reserved_peak = max(reserved_scalars, key=lambda scalar: scalar.value)
    active_peak = max(active_scalars, key=lambda scalar: scalar.value)
    return RunMetrics(
        losses={scalar.step: scalar.value for scalar in loss_scalars},
        max_reserved_gib=reserved_peak.value,
        max_reserved_step=reserved_peak.step,
        max_active_gib=active_peak.value,
        max_active_step=active_peak.step,
    )


def _build_train_cmd(
    *,
    module: str,
    config: str,
    options: str,
    dump_folder: Path,
    tb_folder: str,
    steps: int,
    load_seed_checkpoint: bool,
) -> str:
    cmd = (
        f"MODULE='{module}' CONFIG='{config}' {RUN_TRAIN_SCRIPT} "
        f"--dump_folder={dump_folder} "
        f"{FIXED_OPTIONS} "
        f"--training.steps={steps} "
        f"--metrics.save_tb_folder={tb_folder}"
    )
    if load_seed_checkpoint:
        cmd += (
            " --checkpoint.enable --checkpoint.export_dtype=bfloat16"
            " --checkpoint.load_only"
        )
    if options:
        cmd += f" {options}"
    return cmd


def _create_seed_checkpoint(test: SACTestDefinition, dump_folder: Path) -> None:
    if "." in test.baseline_module:
        module_path = (
            f"torchtitan.experiments.{test.baseline_module}.config_registry"
        )
    else:
        module_path = f"torchtitan.models.{test.baseline_module}.config_registry"
    config_module = importlib.import_module(module_path)
    trainer_config_factory = getattr(config_module, test.baseline_config)
    trainer_config = trainer_config_factory()

    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)

    model_spec = deepcopy(trainer_config.model_spec)
    model_config = deepcopy(model_spec.model)
    model_config.update_from_config(trainer_config=trainer_config)
    with (
        torch.device("meta"),
        utils.set_default_dtype(TORCH_DTYPE_MAP[trainer_config.training.dtype]),
    ):
        model = model_config.build()
    model.to_empty(device="cpu")
    with torch.no_grad():
        model.init_weights(buffer_device=None)

    checkpoint_dir = dump_folder / trainer_config.checkpoint.folder / "step-0"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Creating seed checkpoint at {checkpoint_dir}")
    dcp.save(get_model_state_dict(model), checkpoint_id=str(checkpoint_dir))


def _run_training(
    *,
    scenario_name: str,
    module: str,
    config: str,
    options: str,
    dump_folder: Path,
    tb_folder: str,
    ngpu: int,
    steps: int,
    load_seed_checkpoint: bool,
) -> RunMetrics:
    cmd = _build_train_cmd(
        module=module,
        config=config,
        options=options,
        dump_folder=dump_folder,
        tb_folder=tb_folder,
        steps=steps,
        load_seed_checkpoint=load_seed_checkpoint,
    )
    env = os.environ.copy()
    env["NGPU"] = str(ngpu)
    env["LOG_RANK"] = "0"
    logger.info(f"Running {scenario_name}: {cmd}")
    result = _run_cmd(cmd, env)
    if result.stdout:
        logger.info(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(
            f"{scenario_name} failed.\nCommand: {cmd}\n{result.stderr}"
        )
    return _extract_run_metrics(dump_folder / tb_folder)


def _assert_losses_bitwise_equal(
    baseline: RunMetrics,
    test: RunMetrics,
) -> None:
    if baseline.losses.keys() != test.losses.keys():
        raise AssertionError(
            f"Loss step mismatch: baseline={baseline.losses.keys()} test={test.losses.keys()}"
        )
    for step in sorted(baseline.losses):
        baseline_loss = baseline.losses[step]
        test_loss = test.losses[step]
        if baseline_loss != test_loss:
            raise AssertionError(
                f"Loss mismatch at step {step}: "
                f"baseline={baseline_loss:.9f}, test={test_loss:.9f}"
            )


def _assert_peak_memory_reasonable(
    baseline: RunMetrics,
    test: RunMetrics,
    max_reserved_ratio: float,
) -> float:
    reserved_ratio = test.max_reserved_gib / baseline.max_reserved_gib
    if reserved_ratio > max_reserved_ratio:
        raise AssertionError(
            "Peak reserved memory from the CUDA caching allocator exceeded tolerance: "
            f"baseline={baseline.max_reserved_gib:.3f} GiB, "
            f"test={test.max_reserved_gib:.3f} GiB, "
            f"ratio={reserved_ratio:.3f}, limit={max_reserved_ratio:.3f}"
        )
    return reserved_ratio


def _write_summary(
    output_dir: Path,
    test_definition: SACTestDefinition,
    baseline_metrics: RunMetrics,
    test_metrics: RunMetrics,
    reserved_ratio: float,
) -> None:
    summary = {
        "test": asdict(test_definition),
        "baseline": asdict(baseline_metrics),
        "graph_trainer": asdict(test_metrics),
        "reserved_ratio": reserved_ratio,
    }
    summary_path = output_dir / test_definition.test_name / "sac_integration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))


def _build_tests() -> list[SACTestDefinition]:
    return [
        SACTestDefinition(
            test_name="llama3_debugmodel_graph_sac",
            test_descr=(
                "Llama3 debugmodel eager selective AC vs "
                "graph_trainer aot_fx_trace SAC"
            ),
            baseline_module="llama3",
            baseline_config="llama3_debugmodel",
            baseline_options=(
                "--activation_checkpoint.mode selective "
                "--training.local_batch_size 2 "
                "--training.seq_len 8192"
            ),
            test_module="graph_trainer.llama3",
            test_config="graph_trainer_llama3_debugmodel",
            test_options=(
                "--compile.mode aot_fx_trace "
                "--activation_checkpoint.mode selective "
                "--training.local_batch_size 2 "
                "--training.seq_len 8192"
            ),
            ngpu=1,
            steps=1,
            max_reserved_ratio=1.15,
            use_seed_checkpoint=True,
        ),
    ]


def run_sac_integration_test(test: SACTestDefinition, output_dir: Path) -> None:
    dump_folder = output_dir / test.test_name
    dump_folder.mkdir(parents=True, exist_ok=True)

    if test.use_seed_checkpoint:
        _create_seed_checkpoint(test, dump_folder)
    baseline_metrics = _run_training(
        scenario_name="baseline eager selective AC",
        module=test.baseline_module,
        config=test.baseline_config,
        options=test.baseline_options,
        dump_folder=dump_folder,
        tb_folder="tb_baseline",
        ngpu=test.ngpu,
        steps=test.steps,
        load_seed_checkpoint=test.use_seed_checkpoint,
    )
    graph_trainer_metrics = _run_training(
        scenario_name="graph_trainer aot_fx_trace SAC",
        module=test.test_module,
        config=test.test_config,
        options=test.test_options,
        dump_folder=dump_folder,
        tb_folder="tb_graph_trainer",
        ngpu=test.ngpu,
        steps=test.steps,
        load_seed_checkpoint=test.use_seed_checkpoint,
    )

    _assert_losses_bitwise_equal(baseline_metrics, graph_trainer_metrics)
    reserved_ratio = _assert_peak_memory_reasonable(
        baseline_metrics, graph_trainer_metrics, test.max_reserved_ratio
    )
    _write_summary(
        output_dir, test, baseline_metrics, graph_trainer_metrics, reserved_ratio
    )
    logger.info(
        f"{test.test_name} passed. "
        f"baseline max_reserved={baseline_metrics.max_reserved_gib:.3f} GiB, "
        f"graph_trainer max_reserved={graph_trainer_metrics.max_reserved_gib:.3f} GiB, "
        f"ratio={reserved_ratio:.3f}"
    )


def run_sac_integration_tests(
    *,
    output_dir: Path,
    test_name: str = "all",
    ngpu: int = 1,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise RuntimeError("Please provide an empty output directory.")

    tests = _build_tests()
    ran_any = False
    for test in tests:
        if test_name != "all" and test_name != test.test_name:
            continue
        if ngpu < test.ngpu:
            logger.info(
                f"Skipping {test.test_name}: requires {test.ngpu} GPUs, "
                f"--ngpu={ngpu}"
            )
            continue
        ran_any = True
        run_sac_integration_test(test, output_dir)

    if not ran_any:
        available = [test.test_name for test in tests]
        raise RuntimeError(
            f"No SAC integration tests were run for --test_name={test_name}. "
            f"Available tests: {available}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--test_name", default="all")
    parser.add_argument("--ngpu", default=1, type=int)
    args = parser.parse_args()

    run_sac_integration_tests(
        output_dir=Path(args.output_dir),
        test_name=args.test_name,
        ngpu=args.ngpu,
    )


if __name__ == "__main__":
    main()
