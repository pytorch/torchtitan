#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import os
import re
import shlex
import subprocess
import time
from pathlib import Path


logger: logging.Logger = logging.getLogger(__name__)

# lm-eval resolves every task's dataset from huggingface.co while it builds the
# task objects, and it does that independently on every rank. The olmo3 task set
# expands to 63 datasets (2 ARC configs + hellaswag + 57 MMLU subjects +
# humaneval + mbpp + math500), so an 8-GPU eval fires 500+ hub API calls in a
# few seconds and trips the anonymous rate limit:
#   429 Too Many Requests: you have reached your 'api' rate limit.
#   (0/500 requests remaining in current 300s window)
# Warming the cache once, single-process, before torchrun starts keeps the call
# count 8x lower and lets the ranks run fully offline afterwards.
_HF_RATE_LIMIT_RETRIES = 8
_HF_RATE_LIMIT_BACKOFF_SECONDS = 10.0

if __package__:
    from scripts.olmo3_harness_tasks import (
        OLMO3_TASK_ALIASES,
        SKIPPED_OLMO3_TASKS,
        prepare_task_configs,
    )
else:
    from olmo3_harness_tasks import (
        OLMO3_TASK_ALIASES,
        SKIPPED_OLMO3_TASKS,
        prepare_task_configs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a TorchTitan DCP checkpoint with LM Evaluation Harness."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--hf-assets-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-flavor", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--export-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--eval-gpus", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-sequence-length", type=int, default=8192)
    parser.add_argument("--attn-backend", default="varlen")
    parser.add_argument("--lm-eval-bin", default="python -m lm_eval")
    parser.add_argument("--lm-eval-model", default="pytorch_dcp")
    parser.add_argument("--lm-eval-extra-args", default="")
    parser.add_argument(
        "--no-prefetch-datasets",
        dest="prefetch_datasets",
        action="store_false",
        help=(
            "Skip the single-process HuggingFace cache warmup and let every "
            "rank download its own copy (will likely hit the hub rate limit)."
        ),
    )
    parser.add_argument(
        "--tb-log-dir",
        default=None,
        help=(
            "TensorBoard directory for eval scalars. Defaults to "
            "<dump_folder>/tb/external_eval, derived from --output-dir."
        ),
    )
    parser.add_argument(
        "--no-tensorboard",
        dest="tensorboard",
        action="store_false",
        help="Do not publish eval metrics to TensorBoard.",
    )
    return parser.parse_args()


def default_tb_log_dir(output_dir: str) -> Path:
    """`<dump_folder>/tb/external_eval`, alongside the trainer's timestamped run.

    ExternalEval writes results to `<dump_folder>/<output_folder>/<name>` with
    output_folder defaulting to "eval", and MetricsProcessor puts TensorBoard
    under `<dump_folder>/<save_tb_folder>/<timestamp>` with save_tb_folder
    defaulting to "tb". The trainer's timestamp is not knowable from here, so
    eval lands in a sibling run directory; pointing TensorBoard at
    `<dump_folder>/tb` shows both.
    """
    return Path(output_dir).resolve().parents[1] / "tb" / "external_eval"


def _read_evaluation_step(output_dir: str) -> int | None:
    manifest_path = Path(output_dir) / "checkpoint_manifest.json"
    if manifest_path.exists():
        try:
            with manifest_path.open() as f:
                step = json.load(f).get("evaluation_step")
            if isinstance(step, int):
                return step
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Could not read %s: %s", manifest_path, e)

    # Fall back to the directory name (e.g. "step-400-merged-4").
    if match := re.search(r"step-(\d+)", Path(output_dir).name):
        return int(match.group(1))
    return None


def _latest_results_file(output_dir: str) -> Path | None:
    """lm-eval appends a timestamp when --output_path ends in .json.

    EvaluationTracker.save_results_aggregated rewrites `results.json` as
    `results_<iso-timestamp>.json`, so glob rather than assuming the exact name.
    """
    candidates = sorted(
        Path(output_dir).glob("results*.json"), key=lambda p: p.stat().st_mtime
    )
    return candidates[-1] if candidates else None


def _flatten_metrics(results_blob: dict) -> dict[str, float]:
    """`{"acc_raw,none": 0.42}` under each task -> `{"<task>/acc_raw": 0.42}`.

    Three shapes in a real results file need care:

    * lm-eval suffixes every real metric with the filter name ("acc_raw,none").
      Bookkeeping entries ("name", "alias", "sample_len", "sample_count") carry
      no filter, so the comma is what separates metrics from metadata --
      "sample_len" is an int and would otherwise be logged as a scalar.
    * Metrics with no stderr are reported as the string "N/A"
      ("bpb_v1_stderr,none"), so non-numeric values must be dropped.
    * Group aggregates appear in *both* "results" and "groups" with identical
      values, so the first section to define a tag wins; writing both would put
      two points on the same tag at the same step.
    """
    flat: dict[str, float] = {}
    for section in ("results", "groups"):
        for task, metrics in (results_blob.get(section) or {}).items():
            if not isinstance(metrics, dict):
                continue
            for key, value in metrics.items():
                if "," not in key:
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                tag = f"{task}/{key.split(',', 1)[0]}"
                flat.setdefault(tag, float(value))
    return flat


def log_results_to_tensorboard(
    output_dir: str, tb_log_dir: Path, step: int | None
) -> None:
    if step is None:
        logger.warning("No evaluation step available; skipping TensorBoard logging.")
        return

    results_file = _latest_results_file(output_dir)
    if results_file is None:
        logger.warning("No results*.json under %s; nothing to log.", output_dir)
        return

    with results_file.open() as f:
        results_blob = json.load(f)

    metrics = _flatten_metrics(results_blob)
    if not metrics:
        logger.warning("No numeric metrics found in %s.", results_file)
        return

    from torch.utils.tensorboard import SummaryWriter

    tb_log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_log_dir), max_queue=1000)
    try:
        for name, value in metrics.items():
            writer.add_scalar(f"eval/{name}", value, step)
        writer.flush()
    finally:
        writer.close()

    logger.info(
        "Logged %d eval metrics from %s at step %d to %s.",
        len(metrics),
        results_file.name,
        step,
        tb_log_dir,
    )


def _parse_task_config_dir(
    task_config_path: Path,
) -> tuple[dict[str, tuple[str, str | None]], dict[str, list[str]]]:
    """Map task name -> (dataset_path, dataset_name) and group name -> members.

    The generated configs carry `!function` tags that yaml.safe_load rejects, and
    the three keys we need are always plain top-level scalars, so scan lines
    instead of pulling in a custom loader.
    """
    datasets_by_task: dict[str, tuple[str, str | None]] = {}
    group_members: dict[str, list[str]] = {}

    for config_file in sorted(task_config_path.glob("*.yaml")):
        task_name: str | None = None
        group_name: str | None = None
        dataset_path: str | None = None
        dataset_name: str | None = None
        members: list[str] = []
        in_task_list = False

        for line in config_file.read_text().splitlines():
            if match := re.fullmatch(r"task:\s*(\S.*)", line):
                task_name = match.group(1).strip()
                in_task_list = False
            elif line.rstrip() == "task:":
                in_task_list = True
            elif match := re.fullmatch(r"group:\s*(\S.*)", line):
                group_name = match.group(1).strip()
                in_task_list = False
            elif match := re.fullmatch(r"dataset_path:\s*(\S.*)", line):
                dataset_path = match.group(1).strip()
                in_task_list = False
            elif match := re.fullmatch(r"dataset_name:\s*(\S.*)", line):
                dataset_name = match.group(1).strip()
                in_task_list = False
            elif in_task_list and (match := re.fullmatch(r"\s+-\s+(\S+)", line)):
                members.append(match.group(1))
            elif line and not line[0].isspace():
                in_task_list = False

        if group_name and members:
            group_members[group_name] = members
        if task_name and dataset_path:
            datasets_by_task[task_name] = (dataset_path, dataset_name)

    return datasets_by_task, group_members


def _resolve_dataset_specs(
    task_names: list[str], task_config_path: Path
) -> list[tuple[str, str | None]]:
    datasets_by_task, group_members = _parse_task_config_dir(task_config_path)

    pending = list(task_names)
    seen_tasks: set[str] = set()
    specs: list[tuple[str, str | None]] = []
    seen_specs: set[tuple[str, str | None]] = set()

    while pending:
        name = pending.pop()
        if name in seen_tasks:
            continue
        seen_tasks.add(name)

        if name in group_members:
            pending.extend(group_members[name])
            continue

        spec = datasets_by_task.get(name)
        if spec is None:
            logger.warning("No dataset found for task %s; skipping prefetch.", name)
            continue
        if spec not in seen_specs:
            seen_specs.add(spec)
            specs.append(spec)

    return specs


def prefetch_datasets(
    task_names: list[str], task_config_path: Path
) -> bool:
    """Populate the local HF cache serially. Returns True if everything landed.

    Only when every dataset resolves can the ranks safely run with
    HF_HUB_OFFLINE=1; a partial cache would turn a slow download into a hard
    failure.
    """
    import datasets

    specs = _resolve_dataset_specs(task_names, task_config_path)
    logger.info("Prefetching %d HuggingFace datasets for lm-eval.", len(specs))

    complete = True
    for dataset_path, dataset_name in specs:
        for attempt in range(_HF_RATE_LIMIT_RETRIES):
            try:
                datasets.load_dataset(dataset_path, dataset_name)
                break
            except Exception as e:  # noqa: BLE001 - hub errors are not a stable type
                retriable = "429" in str(e) or "rate limit" in str(e).lower()
                last_attempt = attempt == _HF_RATE_LIMIT_RETRIES - 1
                if not retriable or last_attempt:
                    logger.warning(
                        "Prefetch failed for %s/%s: %s", dataset_path, dataset_name, e
                    )
                    complete = False
                    break
                delay = _HF_RATE_LIMIT_BACKOFF_SECONDS * (attempt + 1)
                logger.info(
                    "Rate limited on %s/%s; sleeping %.0fs before retry %d/%d.",
                    dataset_path,
                    dataset_name,
                    delay,
                    attempt + 1,
                    _HF_RATE_LIMIT_RETRIES,
                )
                time.sleep(delay)

    return complete


def select_task_names(tasks: str) -> tuple[list[str], dict[str, str]]:
    selected = []
    skipped = {}
    for task_name in (task.strip() for task in tasks.split(",")):
        if not task_name:
            continue
        if task_name in SKIPPED_OLMO3_TASKS:
            skipped[task_name] = SKIPPED_OLMO3_TASKS[task_name]
        else:
            resolved_task_name = OLMO3_TASK_ALIASES.get(task_name, task_name)
            if resolved_task_name not in selected:
                selected.append(resolved_task_name)

    if not selected:
        raise ValueError("No supported external evaluation tasks were requested.")
    return selected, skipped


def lm_eval_prefix(args: argparse.Namespace) -> list[str]:
    if args.eval_gpus < 1:
        raise ValueError("eval-gpus must be at least 1.")
    if args.eval_gpus == 1:
        return shlex.split(args.lm_eval_bin)
    if args.lm_eval_bin != "python -m lm_eval":
        raise ValueError(
            "A custom lm-eval-bin cannot be combined with eval-gpus greater than 1."
        )
    return [
        "torchrun",
        "--nproc_per_node",
        str(args.eval_gpus),
        "--rdzv_backend",
        "c10d",
        "--rdzv_endpoint",
        "localhost:0",
        "-m",
        "lm_eval",
    ]


def build_lm_eval_command(
    args: argparse.Namespace, *, task_config_path: Path
) -> list[str]:
    torchtitan_path = str(Path(__file__).resolve().parents[1])
    model_args = {
        "checkpoint_dir": args.checkpoint_dir,
        "hf_assets_path": args.hf_assets_path,
        "model_name": args.model_name,
        "model_flavor": args.model_flavor,
        "torchtitan_path": torchtitan_path,
        "dtype": args.export_dtype,
        "seq_length": args.max_sequence_length,
        "attn_backend": args.attn_backend,
        "devices": args.eval_gpus,
        "data_parallel_replicate_degree": args.eval_gpus,
        "data_parallel_shard_degree": 1,
    }
    serialized_model_args = ",".join(
        f"{key}={value}" for key, value in model_args.items()
    )
    results_path = os.path.join(args.output_dir, "results.json")
    task_names, _ = select_task_names(args.tasks)

    return [
        *lm_eval_prefix(args),
        "--model",
        args.lm_eval_model,
        "--model_args",
        serialized_model_args,
        "--tasks",
        *task_names,
        "--include_path",
        str(task_config_path),
        "--output_path",
        results_path,
        "--batch_size",
        str(args.batch_size),
        *shlex.split(args.lm_eval_extra_args),
    ]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    task_config_path = prepare_task_configs(Path(args.output_dir) / "task_configs")
    task_names, requested_skips = select_task_names(args.tasks)
    with open(os.path.join(args.output_dir, "skipped_tasks.json"), "w") as f:
        json.dump(
            {
                "requested": requested_skips,
                "known_unsupported": SKIPPED_OLMO3_TASKS,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    command = build_lm_eval_command(args, task_config_path=task_config_path)
    with open(os.path.join(args.output_dir, "eval_command.json"), "w") as f:
        json.dump(command, f, indent=2)

    env = os.environ.copy()
    # hf-xet's Rust client ignores http_proxy/https_proxy, so keep downloads on
    # the plain HTTP path both here and in the ranks.
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    if args.prefetch_datasets:
        if prefetch_datasets(task_names, task_config_path):
            logger.info("HF cache warm; running lm-eval with HF_HUB_OFFLINE=1.")
            env["HF_HUB_OFFLINE"] = "1"
        else:
            logger.warning(
                "HF cache is incomplete; leaving the ranks online. Expect hub "
                "rate-limit (429) errors if many datasets are still missing."
            )

    subprocess.run(command, check=True, env=env)

    if args.tensorboard:
        tb_log_dir = (
            Path(args.tb_log_dir)
            if args.tb_log_dir
            else default_tb_log_dir(args.output_dir)
        )
        try:
            log_results_to_tensorboard(
                args.output_dir, tb_log_dir, _read_evaluation_step(args.output_dir)
            )
        except Exception as e:  # noqa: BLE001 - never fail a good eval on logging
            logger.warning("Failed to publish eval metrics to TensorBoard: %s", e)


if __name__ == "__main__":
    main()
