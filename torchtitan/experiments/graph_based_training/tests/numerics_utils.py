# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared utilities for numerics testing between JIT and AOT modes."""

import glob
import os
import subprocess

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_metrics(event_path, metric_names):
    """Load metrics from tensorboard event files."""
    event_acc = EventAccumulator(event_path)
    event_acc.Reload()

    metrics = {}
    for metric_name in metric_names:
        try:
            scalars = event_acc.Scalars(metric_name)
            metrics[metric_name] = {scalar.step: scalar.value for scalar in scalars}
        except KeyError:
            print(f"Warning: Metric {metric_name!r} not found in event file")
            metrics[metric_name] = {}

    return metrics


def compare_metrics(metrics1, metrics2, label1="Eager", label2="Compiled"):
    """Compare two sets of metrics and verify bitwise equivalence using torch.equal()."""

    all_metrics = set(metrics1.keys()) | set(metrics2.keys())
    all_match = True

    for metric_name in sorted(all_metrics):

        steps1 = set(metrics1[metric_name].keys())
        steps2 = set(metrics2[metric_name].keys())

        if steps1 != steps2:
            print("  ERROR: Step mismatch!")
            print(f"    {label1} steps: {sorted(steps1)}")
            print(f"    {label2} steps: {sorted(steps2)}")
            all_match = False
            continue

        # Convert values to tensors for each step and compare
        values1 = [metrics1[metric_name][step] for step in sorted(steps1)]
        values2 = [metrics2[metric_name][step] for step in sorted(steps2)]

        tensor1 = torch.tensor(values1)
        tensor2 = torch.tensor(values2)

        if torch.equal(tensor1, tensor2):
            print(f"  PASS: All {len(steps1)} steps match exactly (bitwise equivalent)")
        else:
            # Find and report mismatches
            mismatches = []
            for idx, step in enumerate(sorted(steps1)):
                val1 = values1[idx]
                val2 = values2[idx]
                if val1 != val2:
                    mismatches.append((step, val1, val2, abs(val1 - val2)))

            print(
                f"  ERROR: Found {len(mismatches)} mismatches out of {len(steps1)} steps"
            )

    return all_match


def find_latest_event_dir(base_path):
    """Find the latest timestamped directory in the base path."""
    if not os.path.exists(base_path):
        raise ValueError(f"Path does not exist: {base_path}")

    subdirs = [d for d in glob.glob(os.path.join(base_path, "*")) if os.path.isdir(d)]
    if not subdirs:
        return base_path

    latest = max(subdirs, key=os.path.getmtime)
    return latest


def run_training(
    ngpu,
    config_file,
    model_name,
    dp_shard_degree,
    tp_degree,
    cp_degree,
    ep_degree,
    ac_mode,
    steps,
    seed,
    deterministic,
    tb_folder,
    compile_mode=None,
    passes=None,
):
    """Run a training job with the specified configuration."""
    print(f"\nStarting training: {model_name} (mode={compile_mode})")

    env = os.environ.copy()
    env["NGPU"] = str(ngpu)
    env["CONFIG_FILE"] = config_file
    env["TRAIN_FILE"] = "torchtitan.experiments.graph_based_training.train"

    cmd = [
        "./run_train.sh",
        "--model.name",
        model_name,
        "--parallelism.data_parallel_shard_degree",
        str(dp_shard_degree),
        "--parallelism.tensor_parallel_degree",
        str(tp_degree),
    ]

    if cp_degree > 1:
        cmd.extend(["--parallelism.context_parallel_degree", str(cp_degree)])
    if ep_degree > 1:
        cmd.extend(["--parallelism.expert_parallel_degree", str(ep_degree)])

    cmd.extend(
        [
            "--activation_checkpoint.mode",
            ac_mode,
            "--training.steps",
            str(steps),
            "--debug.seed",
            str(seed),
            "--debug.deterministic",
            "--metrics.enable_tensorboard",
            "--metrics.save_tb_folder",
            tb_folder,
        ]
    )

    if compile_mode or passes:
        cmd.extend(
            [
                "--job.custom_config_module",
                "torchtitan.experiments.graph_based_training.job_config",
            ]
        )
        if compile_mode:
            cmd.extend(["--compile.mode", compile_mode])
        if passes:
            cmd.extend(["--compile.passes", passes])

    print(f"Environment: NGPU={env['NGPU']}, CONFIG_FILE={env['CONFIG_FILE']}")
    print(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(f"Training completed: {model_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {model_name}")
        print(f"Error output:\n{e.stdout}")
        return False


def determine_model_names(config_file):
    """Determine model names based on config file."""
    if "deepseek" in config_file:
        model_name = "deepseek_v3"
    elif "llama3" in config_file:
        model_name = "llama3"
    else:
        raise ValueError(
            f"Unable to determine model names from config file: {config_file}"
        )

    # Both eager and compiled use graph_based_training experiment
    return f"graph_based_training.{model_name}"


def run_numerics_test(
    ngpu,
    config_file,
    dp_shard_degree,
    tp_degree,
    cp_degree,
    ep_degree,
    ac_mode,
    steps,
    seed,
    eager_tb_folder,
    compiled_tb_folder,
    metrics,
    passes=None,
):
    """
    Run numerics test by training both JIT and AOT modes and comparing metrics.

    Returns:
        bool: True if all metrics match, False otherwise.
    """
    model_name = determine_model_names(config_file)

    # Run JIT (eager baseline) training
    eager_success = run_training(
        ngpu=ngpu,
        config_file=config_file,
        model_name=model_name,
        dp_shard_degree=dp_shard_degree,
        tp_degree=tp_degree,
        cp_degree=cp_degree,
        ep_degree=ep_degree,
        ac_mode=ac_mode,
        steps=steps,
        seed=seed,
        deterministic=True,
        tb_folder=eager_tb_folder,
        compile_mode="jit",
    )

    if not eager_success:
        print("JIT training failed")
        return False

    # Run AOT (compiled) training
    compiled_success = run_training(
        ngpu=ngpu,
        config_file=config_file,
        model_name=model_name,
        dp_shard_degree=dp_shard_degree,
        tp_degree=tp_degree,
        cp_degree=cp_degree,
        ep_degree=ep_degree,
        ac_mode=ac_mode,
        steps=steps,
        seed=seed,
        deterministic=True,
        tb_folder=compiled_tb_folder,
        compile_mode="aot",
        passes=passes,
    )

    if not compiled_success:
        print("AOT training failed")
        return False

    # Compare metrics
    eager_path = find_latest_event_dir(f"./outputs/{eager_tb_folder}")
    compiled_path = find_latest_event_dir(f"./outputs/{compiled_tb_folder}")

    eager_metrics = load_metrics(eager_path, metrics)
    compiled_metrics = load_metrics(compiled_path, metrics)

    all_match = compare_metrics(eager_metrics, compiled_metrics, "JIT", "AOT")

    if all_match:
        print("SUCCESS: All metrics are bitwise equivalent")
    else:
        print("FAILURE: Metrics differ between runs")

    return all_match
