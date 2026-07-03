# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import copy
import gc
import importlib
import io
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import Any, cast, Protocol

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader

from torchtitan.components import fs
from torchtitan.components.checkpoint import CheckpointManager, MODEL
from torchtitan.observability import structured_logger as sl
from torchtitan.tools.logging import init_logger, logger


os.environ.setdefault("NCCL_P2P_DISABLE", "1")

DEFAULT_RECIPE_STATE_FILE = "_torchpackage_recipe_state.pt"
DEFAULT_STRUCTURED_LOG_DIR = os.getenv(
    "TORCHTITAN_STRUCTURED_LOG_DIR", "./outputs/torchpackage_checkpoint"
)
DEFAULT_WORKER_MODULE = "torchtitan.components.torchpackage_checkpoint"


class TorchPackageRecipe(Protocol):
    """Experiment-owned torch package recipe used by the generic manager."""

    def build_empty_state_dict(self, state: Any) -> dict[str, torch.Tensor]:
        """Build CPU tensors matching the DCP keys this recipe needs."""
        ...

    def build_package(
        self,
        *,
        state: Any,
        state_dict: dict[str, torch.Tensor],
        step: int,
    ) -> bytes:
        """Return the final package bytes."""
        ...


def load_torch_package_recipe(recipe_path: str) -> TorchPackageRecipe:
    """Load a recipe object from ``module:qualname``."""

    if ":" not in recipe_path:
        raise ValueError(
            "torch package recipe must be a module path in the form "
            f"'module:qualname', got {recipe_path!r}."
        )

    module_name, qualname = recipe_path.split(":", 1)
    if not module_name or not qualname:
        raise ValueError(
            "torch package recipe must be a module path in the form "
            f"'module:qualname', got {recipe_path!r}."
        )

    obj: Any = importlib.import_module(module_name)
    for attr in qualname.split("."):
        obj = getattr(obj, attr)

    recipe = obj() if isinstance(obj, type) else obj
    for method_name in (
        "build_empty_state_dict",
        "build_package",
    ):
        if not callable(getattr(recipe, method_name, None)):
            raise TypeError(f"{recipe_path} is missing {method_name}().")
    return cast(TorchPackageRecipe, recipe)


def save_recipe_state(path: str, recipe_state: Any) -> None:
    buffer = io.BytesIO()
    torch.save(recipe_state, buffer)
    data = buffer.getvalue()
    del buffer
    with fs.open_file(path, "wb") as handle:
        handle.write(data)
    del data


def load_recipe_state(path: str) -> Any:
    with fs.open_file(path, "rb") as handle:
        data = handle.read()
    try:
        return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    finally:
        del data


def export_torch_package(
    *,
    recipe: TorchPackageRecipe | str,
    checkpoint_path: str,
    output_path: str,
    recipe_state: Any,
    step: int,
    recipe_state_path: str | None = None,
) -> None:
    """Load a DCP checkpoint into recipe buffers and write package bytes."""

    if isinstance(recipe, str):
        recipe = load_torch_package_recipe(recipe)

    sl.set_step(step)
    logger.info("Packaging torch checkpoint step=%s", step)
    logger.info("DCP checkpoint path: %s", checkpoint_path)
    if recipe_state_path is not None:
        logger.info("Torch package recipe state path: %s", recipe_state_path)
    logger.info("Torch package output path: %s", output_path)

    with sl.log_trace_span("torch_package_load_dcp"):
        state_dict = recipe.build_empty_state_dict(recipe_state)
        dcp.load(
            state_dict,
            storage_reader=FsspecReader(checkpoint_path),
            checkpoint_id=checkpoint_path,
        )

    try:
        package = recipe.build_package(
            state=recipe_state,
            state_dict=state_dict,
            step=step,
        )
    finally:
        state_dict.clear()
        del state_dict
        gc.collect()

    package_bytes = len(package)
    with sl.log_trace_span("torch_package_write"):
        with fs.open_file(output_path, "wb") as handle:
            handle.write(package)
    del package
    gc.collect()
    logger.info(
        "Saved %.2f GiB torch package to %s", package_bytes / (1024**3), output_path
    )


class TorchPackageCheckpointManager(CheckpointManager):
    """Checkpoint manager that writes recipe-defined torch packages after DCP."""

    @dataclass(kw_only=True, slots=True)
    class Config(CheckpointManager.Config):
        export_torch_package: bool = False
        """Export a rank-0 torch package artifact alongside written checkpoints."""

        torch_package_recipe: str = ""
        """Recipe object path in ``module:qualname`` form."""

        torch_package_file: str = ""
        """File name for the torch package artifact."""

        torch_package_recipe_state_file: str = DEFAULT_RECIPE_STATE_FILE
        """File name for the serialized recipe state used by the packager."""

        torch_package_async: bool = True
        """Run packaging in a rank-0 subprocess after DCP has completed."""

        torch_package_wait_on_close: bool = True
        """Wait for active torch package subprocesses when closing the manager."""

        torch_package_max_concurrent: int = 1
        """Maximum rank-0 torch package subprocesses to run at once."""

        torch_package_structured_log_dir: str = DEFAULT_STRUCTURED_LOG_DIR
        """Structured log directory used by the package worker."""

        def __post_init__(self) -> None:
            CheckpointManager.Config.__post_init__(self)
            if self.export_torch_package and not self.torch_package_recipe:
                raise ValueError("torch_package_recipe cannot be empty.")
            if self.export_torch_package and not self.torch_package_file:
                raise ValueError("torch_package_file cannot be empty.")
            if self.export_torch_package and not self.torch_package_recipe_state_file:
                raise ValueError("torch_package_recipe_state_file cannot be empty.")
            if self.torch_package_max_concurrent < 1:
                raise ValueError("torch_package_max_concurrent must be at least 1.")

    def __init__(self, config: Config, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.export_torch_package_enabled = config.export_torch_package
        self.torch_package_recipe = config.torch_package_recipe
        self.torch_package_file = config.torch_package_file
        self.torch_package_recipe_state_file = config.torch_package_recipe_state_file
        self.torch_package_async = config.torch_package_async
        self.torch_package_wait_on_close = config.torch_package_wait_on_close
        self.torch_package_max_concurrent = config.torch_package_max_concurrent
        self.torch_package_structured_log_dir = config.torch_package_structured_log_dir
        self._torch_package_processes: list[subprocess.Popen] = []
        self._torch_package_wait_threads: list[threading.Thread] = []
        self._torch_package_lock = threading.Lock()

    @torch.no_grad()
    def save(self, curr_step: int, last_step: bool = False) -> bool:
        saved = super().save(curr_step, last_step)
        if saved and self.export_torch_package_enabled:
            self._schedule_torch_package(curr_step)
        return saved

    def close(self) -> None:
        if not hasattr(self, "_torch_package_lock"):
            super().close()
            return
        if getattr(self, "torch_package_wait_on_close", False):
            self._wait_for_torch_package_wait_threads()
            self._wait_for_torch_package_processes()
        else:
            self._reap_torch_package_wait_threads()
            self._reap_torch_package_processes()
        super().close()

    def _schedule_torch_package(self, curr_step: int) -> None:
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return

        checkpoint_path = self._create_checkpoint_id(curr_step)
        output_path = fs.join_path(checkpoint_path, self.torch_package_file)
        recipe_state_path = fs.join_path(
            checkpoint_path, self.torch_package_recipe_state_file
        )

        with sl.log_trace_span("torch_package_save_recipe_state"):
            recipe_state = self._get_torch_package_state()
            try:
                save_recipe_state(recipe_state_path, recipe_state)
            finally:
                del recipe_state
                gc.collect()

        job = {
            "checkpoint_path": checkpoint_path,
            "output_path": output_path,
            "recipe_state_path": recipe_state_path,
            "step": curr_step,
        }
        save_future = self.save_future
        if save_future is None:
            self._start_torch_package(**job)
            return

        self._start_torch_package_after_dcp(save_future, **job)
        logger.info(
            "Queued torch package export for %s after async DCP completion.",
            output_path,
        )

    def _get_torch_package_state(self) -> Any:
        model_parts = self.states[MODEL].model
        if len(model_parts) != 1:
            raise ValueError("Torch package export does not support PP.")

        model = model_parts[0]
        if hasattr(model, "package_config"):
            return copy.deepcopy(model.package_config)
        if hasattr(model, "config"):
            return copy.deepcopy(model.config)
        raise AttributeError(
            "Torch package export requires the model to expose package_config "
            "or config."
        )

    def _start_torch_package_after_dcp(
        self,
        save_future: Any,
        *,
        checkpoint_path: str,
        output_path: str,
        recipe_state_path: str,
        step: int,
    ) -> None:
        def wait_and_start() -> None:
            try:
                save_future.result()
            except Exception:
                logger.exception(
                    "Skipping torch package export for %s because DCP save failed.",
                    checkpoint_path,
                )
                return
            self._start_torch_package(
                checkpoint_path=checkpoint_path,
                output_path=output_path,
                recipe_state_path=recipe_state_path,
                step=step,
            )

        thread = threading.Thread(
            target=wait_and_start,
            name=f"torch-package-after-dcp-step-{step}",
        )
        with self._torch_package_lock:
            self._torch_package_wait_threads.append(thread)
        thread.start()

    def _start_torch_package(
        self,
        *,
        checkpoint_path: str,
        output_path: str,
        recipe_state_path: str,
        step: int,
    ) -> None:
        if not self.torch_package_async:
            recipe_state = load_recipe_state(recipe_state_path)
            try:
                export_torch_package(
                    recipe=self.torch_package_recipe,
                    checkpoint_path=checkpoint_path,
                    output_path=output_path,
                    recipe_state=recipe_state,
                    step=step,
                    recipe_state_path=recipe_state_path,
                )
            finally:
                del recipe_state
                gc.collect()
            return

        self._reap_torch_package_processes()
        with self._torch_package_lock:
            active_processes = len(self._torch_package_processes)
        if active_processes >= self.torch_package_max_concurrent:
            logger.warning(
                "Skipping torch package export for %s because %s package worker(s) "
                "are already running.",
                output_path,
                active_processes,
            )
            return

        cmd = [
            sys.executable,
            "-m",
            DEFAULT_WORKER_MODULE,
            "--recipe",
            self.torch_package_recipe,
            "--checkpoint-path",
            checkpoint_path,
            "--output-path",
            output_path,
            "--recipe-state-path",
            recipe_state_path,
            "--step",
            str(step),
            "--structured-log-dir",
            self.torch_package_structured_log_dir,
        ]
        env = os.environ.copy()
        env.setdefault("NCCL_P2P_DISABLE", "1")
        process = subprocess.Popen(cmd, env=env, start_new_session=True)
        with self._torch_package_lock:
            self._torch_package_processes.append(process)
        logger.info(
            "Started torch package export pid=%s for %s", process.pid, output_path
        )

    def _wait_for_torch_package_wait_threads(self) -> None:
        while True:
            with self._torch_package_lock:
                if not self._torch_package_wait_threads:
                    return
                thread = self._torch_package_wait_threads.pop(0)
            thread.join()

    def _reap_torch_package_wait_threads(self) -> None:
        with self._torch_package_lock:
            self._torch_package_wait_threads = [
                thread
                for thread in self._torch_package_wait_threads
                if thread.is_alive()
            ]

    def _wait_for_torch_package_processes(self) -> None:
        while True:
            with self._torch_package_lock:
                if not self._torch_package_processes:
                    return
                process = self._torch_package_processes.pop(0)
            return_code = process.wait()
            self._log_torch_package_process_return(process, return_code)

    def _reap_torch_package_processes(self) -> None:
        with self._torch_package_lock:
            active = []
            for process in self._torch_package_processes:
                return_code = process.poll()
                if return_code is None:
                    active.append(process)
                else:
                    self._log_torch_package_process_return(process, return_code)
            self._torch_package_processes = active

    @staticmethod
    def _log_torch_package_process_return(
        process: subprocess.Popen, return_code: int
    ) -> None:
        if return_code == 0:
            logger.info("Torch package export pid=%s completed.", process.pid)
        else:
            logger.error(
                "Torch package export pid=%s failed with return code %s.",
                process.pid,
                return_code,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package a DCP checkpoint with a recipe-defined torch package."
    )
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--recipe-state-path", required=True)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--structured-log-dir", default=DEFAULT_STRUCTURED_LOG_DIR)
    args = parser.parse_args()

    init_logger()
    sl.init_structured_logger(
        source="torchpackage_checkpoint",
        output_dir=args.structured_log_dir,
    )
    recipe_state = load_recipe_state(args.recipe_state_path)
    try:
        with sl.log_trace_span("torch_package_total"):
            export_torch_package(
                recipe=args.recipe,
                checkpoint_path=args.checkpoint_path,
                output_path=args.output_path,
                recipe_state=recipe_state,
                step=args.step,
                recipe_state_path=args.recipe_state_path,
            )
    finally:
        del recipe_state
        gc.collect()


if __name__ == "__main__":
    main()
