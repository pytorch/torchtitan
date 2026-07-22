# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import torch.distributed as dist

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.config import Configurable
from torchtitan.tools.logging import logger


class ExternalEval(Configurable):
    """Launch an external evaluation job from the training loop."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        enable: bool = False
        """Whether to launch external evaluation jobs."""

        freq: int = 100
        """Launch frequency in training steps."""

        path: str = "/home/ruisizhang123/scaling-ladder/eval/run_eval.py"
        """Path to the external evaluation runner."""

        tasks: str = "mmlu,wikitext2"
        """Comma-separated external evaluation task string."""

        launch_async: bool = True
        """If True, launch evaluation and continue training without waiting."""

        eval_cuda_visible_devices: str = ""
        """CUDA_VISIBLE_DEVICES value for the eval subprocess. Empty means inherit."""

        export_dtype: str = "bfloat16"
        """Dtype used by the eval runner when exporting DCP weights to HF."""

        extra_args: str = ""
        """Additional shell-style arguments appended to the eval runner command."""

        output_folder: str = "eval"
        """Folder under dump_folder where eval outputs are written."""

        def __post_init__(self):
            if self.freq < 1:
                raise ValueError("External eval frequency needs to be at least 1 step.")
            if self.enable and not self.path.strip():
                raise ValueError("External eval path cannot be empty when enabled.")
            if self.enable and not self.tasks.strip():
                raise ValueError("External eval tasks cannot be empty when enabled.")

    def __init__(self, config: Config):
        self.config = config
        self.processes: list[subprocess.Popen] = []

    def _reap_finished(self) -> None:
        live_processes = []
        for proc in self.processes:
            if proc.poll() is None:
                live_processes.append(proc)
            else:
                proc.wait()
        self.processes = live_processes

    def should_eval(self, step: int) -> bool:
        return self.config.enable and step % self.config.freq == 0

    def launch(
        self,
        *,
        step: int,
        trainer_config: Any,
        checkpointer: CheckpointManager,
    ) -> None:
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return

        self._reap_finished()

        if not getattr(checkpointer, "enable", False):
            raise ValueError(
                "external_eval requires checkpoint.enable=True so eval can export "
                "a model checkpoint."
            )

        model_spec = trainer_config.model_spec
        if model_spec is None:
            raise ValueError("external_eval requires trainer_config.model_spec.")

        checkpoint_dir = checkpointer.save_for_external_eval(step)
        output_dir = os.path.join(
            trainer_config.dump_folder,
            self.config.output_folder,
            f"step-{step}",
        )
        os.makedirs(output_dir, exist_ok=True)

        command = [
            sys.executable,
            self.config.path,
            "--checkpoint-dir",
            checkpoint_dir,
            "--hf-assets-path",
            trainer_config.hf_assets_path,
            "--model-name",
            model_spec.name,
            "--model-flavor",
            model_spec.flavor,
            "--tasks",
            self.config.tasks,
            "--output-dir",
            output_dir,
            "--export-dtype",
            self.config.export_dtype,
        ]
        if self.config.extra_args:
            command.extend(shlex.split(self.config.extra_args))

        env = os.environ.copy()
        if self.config.eval_cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = self.config.eval_cuda_visible_devices

        log_path = os.path.join(output_dir, "launch.log")
        logger.info("Launching external eval at step %s: %s", step, shlex.join(command))
        log_file = open(log_path, "a")
        if self.config.launch_async:
            proc = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
            )
            log_file.close()
            self.processes.append(proc)
        else:
            try:
                subprocess.run(
                    command,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                    check=True,
                )
            finally:
                log_file.close()

    def close(self) -> None:
        self._reap_finished()
        running = [proc for proc in self.processes if proc.poll() is None]
        if running:
            logger.info("%d external eval job(s) still running.", len(running))
