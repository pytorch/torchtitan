# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import queue
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

import torch.distributed as dist

from torchtitan.components.metrics import (
    BaseLogger,
    MetricsProcessor,
    TensorBoardLogger,
)
from torchtitan.config import Configurable
from torchtitan.tools.logging import logger

# torchelastic exports these to every training process. A nested launcher must
# not inherit them, otherwise the eval job joins the training rendezvous instead
# of starting its own.
_ELASTIC_ENV_VARS = (
    "GROUP_RANK",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "ROLE_NAME",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "WORLD_SIZE",
)
_ELASTIC_ENV_PREFIXES = ("TORCHELASTIC_", "TORCH_ELASTIC_")

# How often close() checks on the eval jobs it is waiting for.
_POLL_INTERVAL_SECONDS = 0.5


class _CheckpointCoordinator(Protocol):
    enable: bool
    load_only: bool

    def register_eval_checkpoint_callback(
        self, callback: Callable[[int, str], None]
    ) -> None: ...


@dataclass
class _EvalJob:
    """Bookkeeping for one in-flight eval subprocess."""

    step: int
    process: subprocess.Popen
    result_path: str
    log_path: str


class AsyncEval(Configurable):
    """Run evaluation for a training step in a separate process.

    When an eval checkpoint is fully persisted, a callback hands it to an eval
    runner launched as a subprocess. The runner
    reports its metrics back by writing a JSON file::

        {"step": 100, "metrics": {"loss": 2.31}}

    The trainer picks up finished results on a later step and logs them to its
    own TensorBoard run, so eval curves show up next to the training curves
    without ever blocking the training loop.

    The runner contract is four flags appended to the configured command:
    ``--checkpoint-dir``, ``--output-dir``, ``--step`` and ``--result-path``.
    Any runner implementing them works (e.g. a script wrapping ``lm_eval``);
    ``torchtitan.components.async_eval.example_loss_eval`` is the bundled example
    and computes validation loss.

    NOTE: the eval job needs devices of its own. Point it at spare devices with
    ``cuda_visible_devices``, otherwise it competes with training for memory on
    the devices the trainer already owns.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        enable: bool = False
        """Whether to launch async evaluation jobs during training."""

        launcher: str = (
            "torchrun --nproc_per_node=1 --rdzv_backend=c10d "
            "--rdzv_endpoint=localhost:0"
        )
        """Launcher prefix of the eval command. Set to an empty string to run
        the runner directly instead of under torchrun."""

        runner: str = "-m torchtitan.components.async_eval.example_loss_eval"
        """Eval runner, appended to the launcher. Defaults to the bundled
        validation-loss runner. Anything honoring the runner contract works."""

        forward_train_args: bool = True
        """Whether to forward this job's own command line arguments (i.e.
        ``--module``, ``--config`` and any overrides) to the runner, so that the
        eval job sees the same model and dataset configuration as training.
        Turn this off for runners that do not take torchtitan arguments."""

        extra_args: str = ""
        """Additional shell-style arguments for the eval command, e.g. the
        validation dataset to evaluate on. They are placed *before* the
        forwarded training arguments, because tyro rejects options that follow
        a subcommand, so an option passed to the training job as well wins."""

        cuda_visible_devices: str = ""
        """``CUDA_VISIBLE_DEVICES`` for the eval subprocess. Empty inherits the
        trainer's value, which means eval shares devices with training."""

        output_folder: str = "async_eval"
        """Folder under dump_folder for eval outputs (logs and results)."""

        exit_timeout: float = 300.0
        """Seconds to wait at the end of training for in-flight eval jobs, so
        that their metrics still get logged. 0 means do not wait."""

        raise_on_failure: bool = False
        """Whether an eval job that fails, or reports no metrics, should bring
        down the training job. Off by default: eval runs beside training and a
        broken eval job is usually not worth losing training progress over."""

        def __post_init__(self):
            if self.enable and not self.runner.strip():
                raise ValueError("Async eval runner cannot be empty when enabled.")
            if self.exit_timeout < 0:
                raise ValueError("Async eval exit_timeout cannot be negative.")

    def __init__(
        self,
        config: Config,
        *,
        dump_folder: str = "./outputs",
        metrics_config: MetricsProcessor.Config | None = None,
    ):
        self.config = config
        self.dump_folder = dump_folder
        self.jobs: list[_EvalJob] = []
        self._ready_checkpoints: queue.SimpleQueue[tuple[int, str]] = (
            queue.SimpleQueue()
        )
        # Captured before training mutates anything, and only meaningful on the
        # process that owns the eval subprocesses.
        self.train_args = list(sys.argv[1:])
        self.logger: BaseLogger = BaseLogger()

        if not (config.enable and self._is_launch_rank()):
            return

        if metrics_config is not None and metrics_config.enable_tensorboard:
            log_dir = os.path.join(
                dump_folder,
                metrics_config.save_tb_folder,
                config.output_folder,
                datetime.now().strftime("%Y%m%d-%H%M"),
            )
            self.logger = TensorBoardLogger(log_dir, tag=config.output_folder)

    def register_checkpoint_callback(
        self, checkpointer: _CheckpointCoordinator
    ) -> None:
        """Launch evaluation after an eligible checkpoint is fully persisted."""
        if not self.config.enable:
            return

        if not checkpointer.enable:
            raise ValueError(
                "async_eval requires checkpoint.enable=True so evaluation steps "
                "produce checkpoints."
            )
        if checkpointer.load_only:
            raise ValueError(
                "async_eval requires checkpoint.load_only=False because it runs "
                "only after a training checkpoint is saved."
            )
        checkpointer.register_eval_checkpoint_callback(
            self._on_eval_checkpoint_save_done
        )

    def _on_eval_checkpoint_save_done(self, step: int, checkpoint_dir: str) -> None:
        if self._is_launch_rank():
            self._ready_checkpoints.put((step, checkpoint_dir))

    def _launch(self, step: int, checkpoint_dir: str) -> None:
        output_dir = os.path.join(
            self.dump_folder, self.config.output_folder, f"step-{step}"
        )
        os.makedirs(output_dir, exist_ok=True)
        result_path = os.path.join(output_dir, "result.json")
        log_path = os.path.join(output_dir, "eval.log")

        command = self._build_command(
            step=step,
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            result_path=result_path,
        )
        logger.info("Launching async eval for step %d: %s", step, shlex.join(command))

        with open(log_path, "a") as log_file:
            process = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=self._build_env(),
            )

        self.jobs.append(
            _EvalJob(
                step=step,
                process=process,
                result_path=result_path,
                log_path=log_path,
            )
        )

    def collect(self) -> None:
        """Log the metrics of every eval job that has finished since last call."""

        while True:
            try:
                ready_checkpoint = self._ready_checkpoints.get_nowait()
            except queue.Empty:
                break
            self._launch(*ready_checkpoint)

        running: list[_EvalJob] = []
        finished: list[_EvalJob] = []
        for job in self.jobs:
            (running if job.process.poll() is None else finished).append(job)
        # Drop the finished jobs before handling them, so that a job failing
        # under raise_on_failure is not reported a second time from close().
        self.jobs = running
        for job in finished:
            self._finish_job(job)

    def close(self) -> None:
        """Wait for in-flight eval jobs, log their metrics, and stop logging."""

        self.collect()
        if self.jobs and self.config.exit_timeout > 0:
            logger.info(
                "Waiting up to %.0fs for %d async eval job(s) to finish.",
                self.config.exit_timeout,
                len(self.jobs),
            )
            deadline = time.monotonic() + self.config.exit_timeout
            while self.jobs and time.monotonic() < deadline:
                time.sleep(_POLL_INTERVAL_SECONDS)
                self.collect()

        if self.jobs:
            logger.warning(
                "%d async eval job(s) did not finish in time; their metrics will "
                "not be logged. Results are still written under %s.",
                len(self.jobs),
                os.path.join(self.dump_folder, self.config.output_folder),
            )
        self.logger.close()

    def _is_launch_rank(self) -> bool:
        return (
            not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
        )

    def _build_command(
        self, *, step: int, checkpoint_dir: str, output_dir: str, result_path: str
    ) -> list[str]:
        command = shlex.split(self.config.launcher)
        if command and command[0] == "torchrun":
            # Reuse the interpreter running the trainer rather than whichever
            # torchrun happens to be first on PATH.
            command = [sys.executable, "-m", "torch.distributed.run"] + command[1:]
        command += shlex.split(self.config.runner)
        # Before the training arguments: those may end with a tyro subcommand,
        # which every option has to precede.
        command += shlex.split(self.config.extra_args)
        if self.config.forward_train_args:
            command += self.train_args
        # The runner contract flags are parsed by the runner itself, ahead of
        # any config parsing, so their position does not matter.
        command += [
            "--checkpoint-dir",
            checkpoint_dir,
            "--output-dir",
            output_dir,
            "--step",
            str(step),
            "--result-path",
            result_path,
        ]
        return command

    def _build_env(self) -> dict[str, str]:
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in _ELASTIC_ENV_VARS and not k.startswith(_ELASTIC_ENV_PREFIXES)
        }
        if self.config.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = self.config.cuda_visible_devices
        return env

    def _finish_job(self, job: _EvalJob) -> None:
        returncode = job.process.returncode
        failure = ""
        if returncode != 0:
            failure = (
                f"Async eval for step {job.step} failed with exit code "
                f"{returncode}. See {job.log_path}."
            )
        else:
            metrics = self._read_metrics(job)
            if metrics:
                self.logger.log(metrics, job.step)
                summary = "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
                logger.info("async eval step: %d  %s", job.step, summary)
            else:
                failure = (
                    f"Async eval for step {job.step} reported no metrics. "
                    f"See {job.log_path}."
                )

        if failure:
            if self.config.raise_on_failure:
                raise RuntimeError(failure)
            logger.warning(failure)

    def _read_metrics(self, job: _EvalJob) -> dict[str, float]:
        if not os.path.isfile(job.result_path):
            logger.warning(
                "Async eval for step %d wrote no result to %s.",
                job.step,
                job.result_path,
            )
            return {}

        try:
            with open(job.result_path) as f:
                result: dict[str, Any] = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(
                "Async eval result %s is not valid JSON: %s.", job.result_path, e
            )
            return {}

        raw_metrics = result.get("metrics", {})
        if not isinstance(raw_metrics, dict):
            logger.warning(
                "Async eval result %s has no 'metrics' dict.", job.result_path
            )
            return {}

        metrics = {
            k: float(v) for k, v in raw_metrics.items() if isinstance(v, (int, float))
        }
        skipped = raw_metrics.keys() - metrics.keys()
        if skipped:
            logger.warning(
                "Skipping non-numeric async eval metrics %s from %s.",
                sorted(skipped),
                job.result_path,
            )
        return metrics
