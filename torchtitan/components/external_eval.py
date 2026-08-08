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

        eval_raw: bool = True
        """Evaluate each raw checkpoint in addition to merged checkpoints."""

        merge_checkpoint_count: int = 1
        """Number of equally spaced checkpoints to average; one disables merging."""

        merge_checkpoint_interval: int = 1000
        """Step spacing between checkpoints included in each merged model."""

        merge_start_step: int = -1
        """First merge/eval step, or -1 to use count times checkpoint interval."""

        merge_decay: float = 1.0
        """Exponential decay applied *within* one merge window.

        Checkpoint at age ``k`` (counting back from the eval step) is weighted
        ``merge_decay ** k`` and the result is normalized. ``1.0`` is the uniform
        model soup; values below 1 weight recent checkpoints more heavily and
        approximate an EMA truncated to ``merge_checkpoint_count`` samples, which
        retains ``1 - merge_decay ** count`` of the true-EMA mass. Because the
        window is bounded by ``checkpoint.keep_latest_k``, a small decay paired
        with a small count throws away most of the tail -- raise the count
        alongside lowering the decay.

        Ignored when ``ema_decay > 0``; that path carries history in the on-disk
        EMA rather than inside a fixed window.
        """

        ema_decay: float = 0.0
        """Decay for a stateful on-disk EMA. ``0.0`` disables it.

        When positive, each merge reads back the previous merged checkpoint and
        applies ``W_ema <- ema_decay * W_ema + (1 - ema_decay) * avg(new)``,
        where ``new`` is the set of regular checkpoints written since the last
        merge. Because the running average lives on disk, this gives an
        unbounded effective window without holding a shadow copy of the model in
        memory, and without being capped by ``checkpoint.keep_latest_k`` the way
        the windowed ``merge_decay`` path is.

        The effective window is roughly ``freq / (1 - ema_decay)`` steps, so at
        ``freq=1000`` a decay of 0.9 averages over ~10k steps.

        Note this is a per-merge decay, not per-step: to match a per-step EMA
        with coefficient beta, use ``ema_decay = beta ** freq``.
        """

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
            if self.merge_checkpoint_count < 1:
                raise ValueError("merge_checkpoint_count must be at least 1.")
            if self.merge_checkpoint_interval < 1:
                raise ValueError("merge_checkpoint_interval must be at least 1.")
            if self.merge_start_step == 0 or self.merge_start_step < -1:
                raise ValueError("merge_start_step must be positive or -1.")
            if not 0.0 < self.merge_decay <= 1.0:
                raise ValueError(
                    f"merge_decay must be in (0, 1], got {self.merge_decay}."
                )
            if not 0.0 <= self.ema_decay < 1.0:
                raise ValueError(
                    f"ema_decay must be in [0, 1), got {self.ema_decay}."
                )
            if self.merge_checkpoint_count > 1:
                first_merge_step = (
                    self.merge_checkpoint_count * self.merge_checkpoint_interval
                    if self.merge_start_step == -1
                    else self.merge_start_step
                )
                first_source_step = (
                    first_merge_step
                    - (self.merge_checkpoint_count - 1) * self.merge_checkpoint_interval
                )
                if first_source_step < 1:
                    raise ValueError(
                        "merge_start_step does not leave enough positive source steps."
                    )
            if self.enable and not self.path.strip():
                raise ValueError("External eval path cannot be empty when enabled.")
            if self.enable and not self.tasks.strip():
                raise ValueError("External eval tasks cannot be empty when enabled.")

    def __init__(self, config: Config):
        self.config = config
        self.processes: list[subprocess.Popen] = []
        self.request_root: str | None = None
        self._warned_merge_window_warmup = False

    def _warn_if_merge_window_spans_warmup(
        self, merge_steps: list[int], trainer_config: Any
    ) -> None:
        """Warn when a merge window reaches back into LR warmup.

        OLMo-core's ModelMergeCallback documents the same constraint: merges
        should sit inside a stable LR regime, otherwise the average mixes
        checkpoints trained under very different learning rates. Warmup is the
        sharpest such transition (LR sweeps from 0 to peak), so a window that
        straddles it yields a soup that is not comparable to later ones.
        """
        if self._warned_merge_window_warmup:
            return
        warmup_steps = getattr(
            getattr(trainer_config, "lr_scheduler", None), "warmup_steps", 0
        )
        if not warmup_steps or merge_steps[0] >= warmup_steps:
            return

        self._warned_merge_window_warmup = True
        suggested_start = (
            warmup_steps
            + (self.config.merge_checkpoint_count - 1)
            * self.config.merge_checkpoint_interval
        )
        logger.warning(
            "External eval merge window %s reaches back before LR warmup ends at "
            "step %s, so this average mixes warmup and post-warmup checkpoints. "
            "Set external_eval.merge_start_step >= %s to keep every merge inside "
            "a stable LR regime.",
            merge_steps,
            warmup_steps,
            suggested_start,
        )

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

    def _first_merge_step(self) -> int:
        return (
            self.config.merge_checkpoint_count * self.config.merge_checkpoint_interval
            if self.config.merge_start_step == -1
            else self.config.merge_start_step
        )

    def merge_checkpoint_steps(self, step: int) -> list[int]:
        if not self.should_eval(step) or self.config.merge_checkpoint_count == 1:
            return []

        first_merge_step = self._first_merge_step()
        if step < first_merge_step:
            return []

        first_source_step = (
            step
            - (self.config.merge_checkpoint_count - 1)
            * self.config.merge_checkpoint_interval
        )
        return [
            first_source_step + i * self.config.merge_checkpoint_interval
            for i in range(self.config.merge_checkpoint_count)
        ]

    def ema_source_steps(self, step: int) -> list[int]:
        """Regular checkpoints written since the previous merge, oldest first.

        Folding only the *new* checkpoints -- rather than the overlapping
        rolling window ``merge_checkpoint_steps`` returns -- keeps each one
        entering the running average exactly once, so the effective per-merge
        decay is exactly ``ema_decay``. With ``freq == merge_checkpoint_interval``
        this is a single checkpoint, i.e. the textbook EMA update.
        """
        if not self.should_eval(step) or self.config.ema_decay <= 0.0:
            return []
        if step < self._first_merge_step():
            return []

        interval = self.config.merge_checkpoint_interval
        previous_merge_step = step - self.config.freq
        first_step = max(((previous_merge_step // interval) + 1) * interval, interval)
        return list(range(first_step, step + 1, interval))

    def previous_ema_step(self, step: int) -> int | None:
        """The merge step whose EMA output this update should build on."""
        candidate = step - self.config.freq
        return candidate if candidate >= self._first_merge_step() else None

    def launch(
        self,
        *,
        step: int,
        trainer_config: Any,
        checkpointer: CheckpointManager,
    ) -> None:
        if not getattr(checkpointer, "enable", False):
            raise ValueError(
                "external_eval requires checkpoint.enable=True so eval can export "
                "a model checkpoint."
            )

        model_spec = trainer_config.model_spec
        if model_spec is None:
            raise ValueError("external_eval requires trainer_config.model_spec.")

        launches: list[tuple[str, str, list[int]]] = []
        if self.config.eval_raw:
            checkpoint_dir = checkpointer.save_for_external_eval(step)
            launches.append((checkpoint_dir, f"step-{step}", [step]))

        merge_steps = self.merge_checkpoint_steps(step)
        ema_steps = self.ema_source_steps(step)
        if ema_steps:
            self._warn_if_merge_window_spans_warmup(ema_steps, trainer_config)
            ema_checkpoint_dir = checkpointer.save_ema_model_checkpoint(
                new_steps=ema_steps,
                curr_step=step,
                decay=self.config.ema_decay,
                prev_ema_step=self.previous_ema_step(step),
            )
            launches.append(
                (
                    ema_checkpoint_dir,
                    f"step-{step}-ema{self.config.ema_decay:g}",
                    ema_steps,
                )
            )
        elif merge_steps:
            self._warn_if_merge_window_spans_warmup(merge_steps, trainer_config)
            merged_checkpoint_dir = checkpointer.save_averaged_model_checkpoint(
                source_steps=merge_steps,
                curr_step=step,
                decay=self.config.merge_decay,
            )
            merged_name = f"step-{step}-merged-{len(merge_steps)}"
            if self.config.merge_decay != 1.0:
                merged_name += f"-ema{self.config.merge_decay:g}"
            launches.append(
                (
                    merged_checkpoint_dir,
                    merged_name,
                    merge_steps,
                )
            )

        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return

        self._reap_finished()
        for checkpoint_dir, output_name, source_steps in launches:
            self._launch_checkpoint(
                step=step,
                checkpoint_dir=checkpoint_dir,
                output_name=output_name,
                source_steps=source_steps,
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
