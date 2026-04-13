# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
import json
import logging
import os
import subprocess
from functools import lru_cache
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.distributed.checkpoint.state_dict import get_model_state_dict

from tests.utils import hash_gradient
from torchtitan.config import ConfigManager, TORCH_DTYPE_MAP
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger

ENTRY_MODULE = "torchtitan.experiments.graph_trainer.tests.test_compile_mode_compare"
HASH_OUTPUT_ENV = "GRAPH_TRAINER_HASH_OUTPUT"
FIXED_ARGS = (
    "--debug.deterministic",
    "--debug.seed=42",
    "--training.steps=5",
    "--metrics.enable_tensorboard",
    "--metrics.log_freq=1",
    "--metrics.save_tb_folder=tb",
    "--checkpoint.enable",
    "--checkpoint.load_only",
)
TB_LOSS_TAG = "loss_metrics/global_avg_loss"
TB_MAX_RESERVED_TAG = "memory/max_reserved(GiB)"
TB_MAX_ACTIVE_TAG = "memory/max_active(GiB)"
TB_THROUGHPUT_TAG = "throughput(tps)"
TB_END_TO_END_TAG = "time_metrics/end_to_end(s)"
MAX_PEAK_MEMORY_RATIO = 1.10


@dataclass(frozen=True)
class CompareCase:
    test_name: str
    module: str
    config: str
    baseline_name: str
    baseline_args: tuple[str, ...]
    candidate_name: str
    candidate_args: tuple[str, ...]
    ngpu: int = 8
    requires_h100: bool = False


@dataclass(frozen=True)
class RunMetrics:
    losses: dict[int, float]
    grad_hash: str
    max_reserved_gib: float
    max_reserved_step: int
    max_active_gib: float
    max_active_step: int
    throughput_tps: dict[int, float]
    end_to_end_s: dict[int, float]


@dataclass(frozen=True)
class ModelStats:
    total_params: int
    trainable_params: int
    buffers: int


LLAMA3_AOT_VS_AOT_FX_TRACE = CompareCase(
    test_name="llama3_aot_vs_aot_fx_trace",
    module="graph_trainer.llama3",
    config="graph_trainer_llama3_debugmodel",
    baseline_name="aot",
    baseline_args=(
        "--compile.mode",
        "aot",
        "--parallelism.data_parallel_shard_degree",
        "4",
        "--parallelism.tensor_parallel_degree",
        "2",
    ),
    candidate_name="aot_fx_trace",
    candidate_args=(
        "--compile.mode",
        "aot_fx_trace",
        "--parallelism.data_parallel_shard_degree",
        "4",
        "--parallelism.tensor_parallel_degree",
        "2",
    ),
)

LLAMA3_AOT_SAC_VS_AOT_FX_TRACE_SAC = CompareCase(
    test_name="llama3_aot_sac_vs_aot_fx_trace_sac",
    module="graph_trainer.llama3",
    config="graph_trainer_llama3_debugmodel",
    baseline_name="aot_sac",
    baseline_args=(
        "--compile.mode",
        "aot",
        "--activation_checkpoint.mode",
        "selective",
        "--parallelism.data_parallel_shard_degree",
        "4",
        "--parallelism.tensor_parallel_degree",
        "2",
    ),
    candidate_name="aot_fx_trace_sac",
    candidate_args=(
        "--compile.mode",
        "aot_fx_trace",
        "--activation_checkpoint.mode",
        "selective",
        "--parallelism.data_parallel_shard_degree",
        "4",
        "--parallelism.tensor_parallel_degree",
        "2",
    ),
)

DEEPSEEK_V3_AOT_VS_AOT_FX_TRACE = CompareCase(
    test_name="deepseek_v3_aot_vs_aot_fx_trace",
    module="graph_trainer.deepseek_v3",
    config="graph_trainer_deepseek_v3_debugmodel",
    baseline_name="aot",
    baseline_args=(
        "--compile.mode",
        "aot",
        "--parallelism.data_parallel_shard_degree",
        "4",
        "--parallelism.tensor_parallel_degree",
        "2",
        "--parallelism.expert_parallel_degree",
        "4",
        "--parallelism.expert_tensor_parallel_degree",
        "1",
    ),
    candidate_name="aot_fx_trace",
    candidate_args=(
        "--compile.mode",
        "aot_fx_trace",
        "--parallelism.data_parallel_shard_degree",
        "4",
        "--parallelism.tensor_parallel_degree",
        "2",
        "--parallelism.expert_parallel_degree",
        "4",
        "--parallelism.expert_tensor_parallel_degree",
        "1",
    ),
    requires_h100=True,
)

DEEPSEEK_V3_AOT_SAC_VS_AOT_FX_TRACE_SAC = CompareCase(
    test_name="deepseek_v3_aot_sac_vs_aot_fx_trace_sac",
    module="graph_trainer.deepseek_v3",
    config="graph_trainer_deepseek_v3_debugmodel",
    baseline_name="aot_sac",
    baseline_args=(
        "--compile.mode",
        "aot",
        "--activation_checkpoint.mode",
        "selective",
        "--parallelism.data_parallel_shard_degree",
        "4",
        "--parallelism.tensor_parallel_degree",
        "2",
        "--parallelism.expert_parallel_degree",
        "4",
        "--parallelism.expert_tensor_parallel_degree",
        "1",
    ),
    candidate_name="aot_fx_trace_sac",
    candidate_args=(
        "--compile.mode",
        "aot_fx_trace",
        "--activation_checkpoint.mode",
        "selective",
        "--parallelism.data_parallel_shard_degree",
        "4",
        "--parallelism.tensor_parallel_degree",
        "2",
        "--parallelism.expert_parallel_degree",
        "4",
        "--parallelism.expert_tensor_parallel_degree",
        "1",
    ),
    requires_h100=True,
)


def _all_h100() -> bool:
    return torch.cuda.is_available() and all(
        torch.cuda.get_device_capability(i) >= (9, 0)
        for i in range(torch.cuda.device_count())
    )


def _write_run_log(log_path: Path, result: subprocess.CompletedProcess[str]) -> None:
    content = result.stdout
    if result.stderr:
        content += f"\nSTDERR:\n{result.stderr}"
    log_path.write_text(content)


def _run_cmd(
    cmd: list[str],
    env: dict[str, str],
    *,
    log_path: Path,
) -> None:
    result = subprocess.run(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        env=env,
    )
    _write_run_log(log_path, result)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(cmd)}\n"
            f"See log: {log_path}"
        )


def _read_tensorboard_scalars(tb_root: Path) -> dict[str, list]:
    if not tb_root.exists():
        raise FileNotFoundError(f"TensorBoard directory not found: {tb_root}")
    run_dirs = [path for path in tb_root.iterdir() if path.is_dir()]
    if len(run_dirs) != 1:
        raise RuntimeError(
            f"Expected exactly one TensorBoard run directory under {tb_root}, found {run_dirs}"
        )

    tensorboard_logger = logging.getLogger("tensorboard")
    original_level = tensorboard_logger.level
    tensorboard_logger.setLevel(logging.WARNING)
    event_accumulator = EventAccumulator(str(run_dirs[0]))
    try:
        event_accumulator.Reload()
    finally:
        tensorboard_logger.setLevel(original_level)

    scalar_tags = event_accumulator.Tags().get("scalars", [])
    required_tags = [
        TB_LOSS_TAG,
        TB_MAX_RESERVED_TAG,
        TB_MAX_ACTIVE_TAG,
        TB_THROUGHPUT_TAG,
        TB_END_TO_END_TAG,
    ]
    missing_tags = [tag for tag in required_tags if tag not in scalar_tags]
    if missing_tags:
        raise KeyError(
            f"Missing scalar tags in {run_dirs[0]}: {missing_tags}; found {scalar_tags}"
        )

    return {tag: event_accumulator.Scalars(tag) for tag in required_tags}


def _read_run_metrics(run_dir: Path) -> RunMetrics:
    scalars = _read_tensorboard_scalars(run_dir / "tb")
    hash_data = json.loads((run_dir / "hashes.json").read_text())
    reserved_peak = max(scalars[TB_MAX_RESERVED_TAG], key=lambda scalar: scalar.value)
    active_peak = max(scalars[TB_MAX_ACTIVE_TAG], key=lambda scalar: scalar.value)
    return RunMetrics(
        losses={scalar.step: scalar.value for scalar in scalars[TB_LOSS_TAG]},
        grad_hash=hash_data["grad_hash"],
        max_reserved_gib=reserved_peak.value,
        max_reserved_step=reserved_peak.step,
        max_active_gib=active_peak.value,
        max_active_step=active_peak.step,
        throughput_tps={
            scalar.step: scalar.value for scalar in scalars[TB_THROUGHPUT_TAG]
        },
        end_to_end_s={
            scalar.step: scalar.value for scalar in scalars[TB_END_TO_END_TAG]
        },
    )


@lru_cache(maxsize=None)
def _get_model_stats(module: str, config: str) -> ModelStats:
    module_path = f"torchtitan.experiments.{module}.config_registry"
    config_module = importlib.import_module(module_path)
    trainer_config_factory = getattr(config_module, config)
    trainer_config = trainer_config_factory()

    model_spec = deepcopy(trainer_config.model_spec)
    model_config = deepcopy(model_spec.model)
    model_config.update_from_config(trainer_config=trainer_config)
    with (
        torch.device("meta"),
        utils.set_default_dtype(TORCH_DTYPE_MAP[trainer_config.training.dtype]),
    ):
        model = model_config.build()

    return ModelStats(
        total_params=sum(param.numel() for param in model.parameters()),
        trainable_params=sum(
            param.numel() for param in model.parameters() if param.requires_grad
        ),
        buffers=sum(buffer.numel() for buffer in model.buffers()),
    )


def _create_seed_checkpoint(case: CompareCase, dump_folder: Path) -> None:
    module_path = f"torchtitan.experiments.{case.module}.config_registry"
    config_module = importlib.import_module(module_path)
    trainer_config_factory = getattr(config_module, case.config)
    trainer_config = trainer_config_factory()

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
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
    dcp.save(get_model_state_dict(model), checkpoint_id=str(checkpoint_dir))


def _build_train_cmd(
    *,
    case: CompareCase,
    dump_folder: Path,
    mode_args: tuple[str, ...],
) -> list[str]:
    return [
        "torchrun",
        f"--nproc_per_node={case.ngpu}",
        "--rdzv_backend",
        "c10d",
        "--rdzv_endpoint",
        "localhost:0",
        "--local-ranks-filter",
        "0",
        "--role",
        "rank",
        "--tee",
        "3",
        "-m",
        ENTRY_MODULE,
        "--module",
        case.module,
        "--config",
        case.config,
        f"--dump_folder={dump_folder}",
        *FIXED_ARGS,
        *mode_args,
    ]


def _run_training(
    *,
    case: CompareCase,
    scenario_name: str,
    mode_args: tuple[str, ...],
    case_dir: Path,
) -> RunMetrics:
    run_dir = case_dir / scenario_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _create_seed_checkpoint(case, run_dir)
    env = os.environ.copy()
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["GRAPH_TRAINER_HASH_OUTPUT"] = str(run_dir / "hashes.json")
    cmd = _build_train_cmd(case=case, dump_folder=run_dir, mode_args=mode_args)
    _run_cmd(cmd, env, log_path=run_dir / "train.log")
    return _read_run_metrics(run_dir)


def run_compare_case(
    *,
    case: CompareCase,
    tmp_path: Path,
    max_peak_memory_ratio: float = MAX_PEAK_MEMORY_RATIO,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.device_count() < case.ngpu:
        pytest.skip(f"requires {case.ngpu} GPUs")
    if case.requires_h100 and not _all_h100():
        pytest.skip("requires H100 GPUs")

    case_dir = tmp_path / case.test_name
    baseline = _run_training(
        case=case,
        scenario_name=case.baseline_name,
        mode_args=case.baseline_args,
        case_dir=case_dir,
    )
    candidate = _run_training(
        case=case,
        scenario_name=case.candidate_name,
        mode_args=case.candidate_args,
        case_dir=case_dir,
    )
    _assert_runs_match(
        case=case,
        baseline=baseline,
        candidate=candidate,
        max_peak_memory_ratio=max_peak_memory_ratio,
    )
    _print_compare_summary(case=case, baseline=baseline, candidate=candidate)


def _print_compare_summary(
    *,
    case: CompareCase,
    baseline: RunMetrics,
    candidate: RunMetrics,
) -> None:
    def _format_step_map(values: dict[int, float], *, precision: int) -> str:
        return ", ".join(
            f"{step}:{values[step]:.{precision}f}" for step in sorted(values)
        )

    def _average_over_steps(values: dict[int, float], steps: range) -> float:
        selected = [values[step] for step in steps]
        return sum(selected) / len(selected)

    model_stats = _get_model_stats(case.module, case.config)
    reserved_ratio = candidate.max_reserved_gib / baseline.max_reserved_gib
    active_ratio = candidate.max_active_gib / baseline.max_active_gib
    steady_state_steps = range(3, 6)
    baseline_losses = _format_step_map(baseline.losses, precision=9)
    candidate_losses = _format_step_map(candidate.losses, precision=9)
    baseline_times = _format_step_map(baseline.end_to_end_s, precision=3)
    candidate_times = _format_step_map(candidate.end_to_end_s, precision=3)
    baseline_tps = _format_step_map(baseline.throughput_tps, precision=3)
    candidate_tps = _format_step_map(candidate.throughput_tps, precision=3)
    baseline_steady_time = _average_over_steps(baseline.end_to_end_s, steady_state_steps)
    candidate_steady_time = _average_over_steps(
        candidate.end_to_end_s, steady_state_steps
    )
    baseline_steady_tps = _average_over_steps(
        baseline.throughput_tps, steady_state_steps
    )
    candidate_steady_tps = _average_over_steps(
        candidate.throughput_tps, steady_state_steps
    )
    summary = (
        f"{case.test_name} summary\n"
        f"  model={case.module} config={case.config}\n"
        f"  model_stats: total_params={model_stats.total_params:,} "
        f"trainable_params={model_stats.trainable_params:,} "
        f"buffers={model_stats.buffers:,}\n"
        f"  losses ({case.baseline_name}): {baseline_losses}\n"
        f"  losses ({case.candidate_name}): {candidate_losses}\n"
        f"  step_wall_time(s) ({case.baseline_name}): {baseline_times}\n"
        f"  step_wall_time(s) ({case.candidate_name}): {candidate_times}\n"
        f"  throughput(tps) ({case.baseline_name}): {baseline_tps}\n"
        f"  throughput(tps) ({case.candidate_name}): {candidate_tps}\n"
        f"  steady_state steps 3-5 ({case.baseline_name}): "
        f"avg_step_time={baseline_steady_time:.3f} s, "
        f"avg_throughput={baseline_steady_tps:.3f} tps\n"
        f"  steady_state steps 3-5 ({case.candidate_name}): "
        f"avg_step_time={candidate_steady_time:.3f} s, "
        f"avg_throughput={candidate_steady_tps:.3f} tps\n"
        f"  {case.baseline_name}: "
        f"peak_reserved={baseline.max_reserved_gib:.3f} GiB "
        f"(step {baseline.max_reserved_step}), "
        f"peak_active={baseline.max_active_gib:.3f} GiB "
        f"(step {baseline.max_active_step})\n"
        f"  {case.candidate_name}: "
        f"peak_reserved={candidate.max_reserved_gib:.3f} GiB "
        f"(step {candidate.max_reserved_step}), "
        f"peak_active={candidate.max_active_gib:.3f} GiB "
        f"(step {candidate.max_active_step})\n"
        f"  ratios: reserved={reserved_ratio:.3f} active={active_ratio:.3f}"
    )
    print(summary, flush=True)


def _assert_runs_match(
    *,
    case: CompareCase,
    baseline: RunMetrics,
    candidate: RunMetrics,
    max_peak_memory_ratio: float,
) -> None:
    expected_steps = set(range(1, 6))
    assert set(baseline.losses) == expected_steps, baseline.losses
    assert set(candidate.losses) == expected_steps, candidate.losses
    for step in range(1, 6):
        assert baseline.losses[step] == candidate.losses[step], (
            f"{case.test_name}: loss mismatch at step {step}: "
            f"{case.baseline_name}={baseline.losses[step]!r}, "
            f"{case.candidate_name}={candidate.losses[step]!r}"
        )

    assert baseline.grad_hash == candidate.grad_hash, (
        f"{case.test_name}: grad hash mismatch: "
        f"{case.baseline_name}={baseline.grad_hash}, "
        f"{case.candidate_name}={candidate.grad_hash}"
    )

    reserved_ratio = candidate.max_reserved_gib / baseline.max_reserved_gib
    active_ratio = candidate.max_active_gib / baseline.max_active_gib
    assert reserved_ratio <= max_peak_memory_ratio, (
        f"{case.test_name}: reserved peak memory ratio too high: "
        f"{case.candidate_name}={candidate.max_reserved_gib:.3f} GiB, "
        f"{case.baseline_name}={baseline.max_reserved_gib:.3f} GiB, "
        f"ratio={reserved_ratio:.3f}, limit={max_peak_memory_ratio:.3f}"
    )
    assert active_ratio <= max_peak_memory_ratio, (
        f"{case.test_name}: active peak memory ratio too high: "
        f"{case.candidate_name}={candidate.max_active_gib:.3f} GiB, "
        f"{case.baseline_name}={baseline.max_active_gib:.3f} GiB, "
        f"ratio={active_ratio:.3f}, limit={max_peak_memory_ratio:.3f}"
    )


def _maybe_write_hashes(trainer) -> None:
    output_path = os.environ.get(HASH_OUTPUT_ENV)
    if not output_path:
        return
    assert len(trainer.model_parts) == 1, "compile-mode compare does not support PP"
    model = trainer.model_parts[0]
    if not dist.is_initialized() or dist.get_rank() == 0:
        Path(output_path).write_text(
            json.dumps(
                {
                    "grad_hash": hash_gradient(model),
                },
                indent=2,
            )
        )


def _train_entrypoint() -> None:
    init_logger()
    config_manager = ConfigManager()
    config = config_manager.parse_args()
    trainer = None

    try:
        trainer = config.build()
        trainer.train()
        if dist.is_initialized():
            dist.barrier()
        _maybe_write_hashes(trainer)
    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
        if dist.is_initialized():
            dist.destroy_process_group()
        logger.info("Process group destroyed")


def test_llama3_aot_vs_aot_fx_trace(tmp_path: Path) -> None:
    run_compare_case(case=LLAMA3_AOT_VS_AOT_FX_TRACE, tmp_path=tmp_path)


def test_llama3_aot_sac_vs_aot_fx_trace_sac(tmp_path: Path) -> None:
    run_compare_case(case=LLAMA3_AOT_SAC_VS_AOT_FX_TRACE_SAC, tmp_path=tmp_path)


def test_deepseek_v3_aot_vs_aot_fx_trace(tmp_path: Path) -> None:
    run_compare_case(case=DEEPSEEK_V3_AOT_VS_AOT_FX_TRACE, tmp_path=tmp_path)


def test_deepseek_v3_aot_sac_vs_aot_fx_trace_sac(tmp_path: Path) -> None:
    run_compare_case(case=DEEPSEEK_V3_AOT_SAC_VS_AOT_FX_TRACE_SAC, tmp_path=tmp_path)


if __name__ == "__main__":
    _train_entrypoint()
