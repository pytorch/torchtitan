# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import torch.distributed as dist

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

        eval_raw: bool = True
        """Evaluate the raw checkpoint produced at each evaluation step."""

        launch_async: bool = True
        """If True, launch evaluation and continue training without waiting."""

        request_only: bool = False
        """Write an eval request for a dedicated worker instead of running locally."""

        wait_on_close: bool = True
        """Wait for asynchronous evaluations before the training process exits."""

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
        self.request_root: str | None = None

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
        checkpoint_dir: str,
        output_name: str,
        source_steps: list[int] | tuple[int, ...],
        trainer_config: Any,
    ) -> None:
        if trainer_config.model_spec is None:
            raise ValueError("external_eval requires trainer_config.model_spec.")
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return

        self._reap_finished()
        self._launch_checkpoint(
            step=step,
            checkpoint_dir=checkpoint_dir,
            output_name=output_name,
            source_steps=list(source_steps),
            trainer_config=trainer_config,
        )

    def _launch_checkpoint(
        self,
        *,
        step: int,
        checkpoint_dir: str,
        output_name: str,
        source_steps: list[int],
        trainer_config: Any,
    ) -> None:
        model_spec = trainer_config.model_spec
        assert model_spec is not None
        output_dir = os.path.join(
            trainer_config.dump_folder,
            self.config.output_folder,
            output_name,
        )
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "checkpoint_manifest.json"), "w") as f:
            json.dump(
                {
                    "evaluation_step": step,
                    "source_steps": source_steps,
                    "checkpoint_dir": checkpoint_dir,
                },
                f,
                indent=2,
            )

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
            # Publish eval scalars next to the trainer's TensorBoard run. Pass
            # this explicitly rather than letting run_external_eval.py infer it
            # from --output-dir, so a customized output_folder/save_tb_folder
            # still lands in the right place.
            "--tb-log-dir",
            os.path.join(
                trainer_config.dump_folder,
                getattr(
                    getattr(trainer_config, "metrics", None), "save_tb_folder", "tb"
                ),
                "external_eval",
            ),
        ]
        if self.config.extra_args:
            command.extend(shlex.split(self.config.extra_args))

        env = os.environ.copy()
        if self.config.eval_cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = self.config.eval_cuda_visible_devices

        if self.config.request_only:
            self.request_root = os.path.dirname(output_dir)
            request_path = os.path.join(output_dir, "eval_request.json")
            temporary_request_path = f"{request_path}.tmp"
            with open(temporary_request_path, "w") as f:
                json.dump(
                    {
                        "command": command,
                        "env": {
                            "CUDA_VISIBLE_DEVICES": env.get(
                                "CUDA_VISIBLE_DEVICES", ""
                            )
                        },
                    },
                    f,
                    indent=2,
                )
            os.replace(temporary_request_path, request_path)
            logger.info(
                "Queued external eval at step %s from checkpoints %s for a "
                "dedicated evaluator.",
                step,
                source_steps,
            )
            return

        log_path = os.path.join(output_dir, "launch.log")
        logger.info(
            "Launching external eval at step %s from checkpoints %s: %s",
            step,
            source_steps,
            shlex.join(command),
        )
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
        if (
            self.config.request_only
            and self.request_root
            and (
                not dist.is_available()
                or not dist.is_initialized()
                or dist.get_rank() == 0
            )
        ):
            completion_path = os.path.join(self.request_root, "_TRAINING_COMPLETE")
            with open(completion_path, "w"):
                pass

        self._reap_finished()
        running = [proc for proc in self.processes if proc.poll() is None]
        if running:
            logger.info("%d external eval job(s) still running.", len(running))
        if running and self.config.wait_on_close:
            logger.info("Waiting for external eval jobs to write their results.")
            for proc in running:
                return_code = proc.wait()
                if return_code != 0:
                    logger.error(
                        "External eval process %s exited with code %s.",
                        proc.pid,
                        return_code,
                    )
            self.processes = []
