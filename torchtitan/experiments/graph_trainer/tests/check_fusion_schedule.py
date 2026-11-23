# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run the reference and scheduled recipes in fresh processes on the same GPU."""

import argparse
import multiprocessing
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from tempfile import TemporaryDirectory
from unittest.mock import patch


def run_step(recipe: str, config: str, output_dir: str):
    from torchtitan.experiments.graph_trainer.auto_fuser import scheduled_kernels
    from torchtitan.experiments.graph_trainer.auto_fuser.substitute_kernels import (
        VerifiedKernel,
    )
    from torchtitan.observability.metrics import MetricsProcessor
    from torchtitan.train import main as train

    metrics = []
    executed = Counter()
    expected = Counter()
    original_log = MetricsProcessor.log
    original_forward = VerifiedKernel.forward
    original_substitute = scheduled_kernels.substitute_with_kernels

    def record_metrics(
        self, step, global_avg_loss, global_max_loss, grad_norm, *args, **kwargs
    ):
        metrics.append((float(global_avg_loss).hex(), float(grad_norm).hex()))
        return original_log(
            self, step, global_avg_loss, global_max_loss, grad_norm, *args, **kwargs
        )

    def record_execution(self, *args):
        result = original_forward(self, *args)
        executed[self.problem_id] += 1
        return result

    def record_schedule(gm, calls, **kwargs):
        original_substitute(gm, calls, **kwargs)
        for call in calls:
            kernel = gm.get_submodule(call.target)
            if isinstance(kernel, VerifiedKernel):
                expected[kernel.problem_id] += 1
        return gm

    sys.argv = [
        "torchtitan.train",
        "--module",
        f"torchtitan_recipes.graph_trainer.fusion_schedules.{recipe}",
        "--config",
        config,
        "--comm-backend",
        "fake",
        "--output-dir",
        output_dir,
    ]
    with (
        patch.object(MetricsProcessor, "log", record_metrics),
        patch.object(VerifiedKernel, "forward", record_execution),
        patch.object(scheduled_kernels, "substitute_with_kernels", record_schedule),
    ):
        train()
    if len(metrics) != 1:
        raise AssertionError(f"{config} logged {len(metrics)} steps")
    # CUDA graph capture may execute a region multiple times during warmup.
    if config == "scheduled" and (not expected or expected - executed):
        raise AssertionError(
            f"Scheduled kernels did not execute: {expected - executed}"
        )
    return metrics[0], expected.total(), executed.total(), len(executed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recipe",
        choices=["deepseek_v3_debug", "qwen3_30b", "gpt_oss_120b", "deepseek_v3_671b"],
    )
    recipe = parser.parse_args().recipe
    results = []
    for config in ("reference", "scheduled"):
        with (
            TemporaryDirectory() as output,
            ProcessPoolExecutor(
                max_workers=1, mp_context=multiprocessing.get_context("spawn")
            ) as worker,
        ):
            results.append(worker.submit(run_step, recipe, config, output).result())
    reference, scheduled = results
    if reference[0] != scheduled[0]:
        raise AssertionError(
            f"Loss and grad_norm differ: {reference[0]} vs {scheduled[0]}"
        )
    print(
        f"{recipe}: {scheduled[1]} kernel calls in the graph, "
        f"{scheduled[2]} executions across {scheduled[3]} specializations. "
        f"Identical loss {scheduled[0][0]} and grad_norm {scheduled[0][1]}."
    )
