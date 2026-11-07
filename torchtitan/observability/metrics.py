# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Training metrics collection and export."""

import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar

import torch
from tensorboard.compat.proto.summary_pb2 import (  # generated protobuf class
    Summary,  # pyrefly: ignore [missing-module-attribute]
)
from torch.distributed.pipelining.schedules import (
    get_schedule_class,
    ScheduleDualPipeV,
    ScheduleZBVZeroBubble,
)
from torch.utils.tensorboard import FileWriter

from torchtitan.components.optim import OptimizersContainer
from torchtitan.config import Configurable
from torchtitan.distributed import ParallelismContext
from torchtitan.observability.logging import Color, NoColor
from torchtitan.tools import utils
from torchtitan.tools.utils import device_module, device_type

# named tuple for passing device memory stats for logging
logger = logging.getLogger(__name__)


DeviceMemStats = namedtuple(
    "DeviceMemStats",
    [
        "max_active_gib",
        "max_active_pct",
        "max_reserved_gib",
        "max_reserved_pct",
        "num_alloc_retries",
        "num_ooms",
    ],
)


class DeviceMemoryMonitor:
    def __init__(self, device: str = f"{device_type}:0"):
        # pyrefly: ignore [read-only]
        self.device = torch.device(device)  # device object
        self.device_name = device_module.get_device_name(self.device)
        self.device_index = device_module.current_device()
        self.device_capacity = device_module.get_device_properties(
            self.device
        ).total_memory
        self.device_capacity_gib = self._to_gib(self.device_capacity)

        device_module.reset_peak_memory_stats()
        device_module.empty_cache()

    def _to_gib(self, memory_in_bytes):
        # NOTE: GiB (gibibyte) is 1024, vs GB is 1000
        _gib_in_bytes = 1024 * 1024 * 1024
        memory_in_gib = memory_in_bytes / _gib_in_bytes
        return memory_in_gib

    def _to_pct(self, memory):
        return 100 * memory / self.device_capacity

    def get_peak_stats(self):
        device_info = device_module.memory_stats(self.device)

        max_active = device_info.get("active_bytes.all.peak", -1)
        max_active_gib = self._to_gib(max_active)
        max_active_pct = self._to_pct(max_active)

        max_reserved = device_info.get("reserved_bytes.all.peak", -1)
        max_reserved_gib = self._to_gib(max_reserved)
        max_reserved_pct = self._to_pct(max_reserved)

        num_retries = device_info.get("num_alloc_retries", -1)
        num_ooms = device_info.get("num_ooms", -1)

        if num_retries > 0:
            logger.warning(
                f"{num_retries} {device_type.upper()} memory allocation retries."
            )
        if num_ooms > 0:
            logger.warning(f"{num_ooms} {device_type.upper()} OOM errors thrown.")

        return DeviceMemStats(
            max_active_gib,
            max_active_pct,
            max_reserved_gib,
            max_reserved_pct,
            num_retries,
            num_ooms,
        )

    def reset_peak_stats(self):
        device_module.reset_peak_memory_stats()


def build_device_memory_monitor():
    device_memory_monitor = DeviceMemoryMonitor(device_type)
    logger.info(
        f"{device_type.upper()} capacity: {device_memory_monitor.device_name} "
        f"with {device_memory_monitor.device_capacity_gib:.2f}GiB memory"
    )
    return device_memory_monitor


def compute_training_performance_metrics(
    *,
    num_tokens: int,
    elapsed_time: float,
    non_data_parallel_size: int,
    num_flops: float,
    gpu_peak_flops: float,
    has_quantization: bool,
) -> dict[str, float]:
    """Compute per-device throughput, TFLOPS, and optional MFU."""
    tokens_per_second = num_tokens / (elapsed_time * non_data_parallel_size)
    device_flops_per_second = num_flops / (elapsed_time * non_data_parallel_size)
    tflops = device_flops_per_second / 1e12
    metrics = {
        "tokens_per_second": tokens_per_second,
        "tflops": tflops,
    }
    if not has_quantization:
        metrics["mfu_percent"] = 100 * device_flops_per_second / gpu_peak_flops
    return metrics


class BaseLogger(Configurable, ABC):
    """Exports metrics through a configurable backend."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        metrics_rank_only: ClassVar[bool] = False

    @abstractmethod
    def log(self, metrics: dict[str, Any], step: int) -> None:
        pass

    def close(self) -> None:
        pass


class TensorBoardLogger(BaseLogger):
    """Logger implementation for TensorBoard."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLogger.Config):
        pass

    def __init__(
        self,
        config: Config,
        *,
        log_dir: str,
        config_dict: dict[str, Any] | None = None,
        tag: str | None = None,
    ):
        self.tag = tag
        self.writer = FileWriter(log_dir, max_queue=1000)
        # Without TensorFlow, TensorBoard opens and closes the events file per event: slow on
        # FUSE/NFS. Keep one unbuffered file open; safe because EventFileWriter.__init__ has
        # already flushed its file_version event. TODO: drop this private-field swap (tensorboard
        # is unpinned) when its file layer keeps files open: https://github.com/tensorflow/tensorboard/issues/5286
        event_writer = self.writer.event_writer
        record_writer = event_writer._async_writer._writer
        # If TensorBoard renames RecordWriter._writer, the assignment would silently do nothing.
        if not hasattr(record_writer, "_writer"):
            raise RuntimeError(
                f"TensorBoard's {type(record_writer)} has no _writer field"
            )
        record_writer._writer = open(event_writer._file_name, "ab", buffering=0)
        logger.info(f"TensorBoard logging enabled. Logs will be saved at {log_dir}")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        # One event per step, so each slot of the writer queue (max_queue) holds a whole step.
        summary = Summary(
            value=[
                Summary.Value(
                    tag=k if self.tag is None else f"{self.tag}/{k}",
                    simple_value=float(v),
                )
                for k, v in metrics.items()
            ]
        )
        self.writer.add_summary(summary, step)

    def close(self) -> None:
        self.writer.close()


class WandBLogger(BaseLogger):
    """Logger implementation for Weights & Biases: each `log` call is one row, with `trainer/step`.

    Example:
        log({"loss": 2.1}, step=10)                       -> row _step=0, trainer/step=10
        log({"validation_metrics/loss": 2.3}, step=10)    -> row _step=1, trainer/step=10
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLogger.Config):
        pass

    def __init__(
        self,
        config: Config,
        *,
        log_dir: str,
        config_dict: dict[str, Any] | None = None,
        tag: str | None = None,
    ):
        # Import wandb here to avoid startup import
        import wandb

        self.wandb = wandb
        self.tag = tag

        # Create logging directory
        os.makedirs(log_dir, exist_ok=True)

        # WANDB_RUN_TAGS is comma-separated; wandb expects a sequence of tags.
        tags = None
        if tags_env := os.getenv("WANDB_RUN_TAGS"):
            tags = [t.strip() for t in tags_env.split(",") if t.strip()] or None

        self.wandb.init(
            entity=os.getenv("WANDB_TEAM", None),
            project=os.getenv("WANDB_PROJECT", "torchtitan"),
            name=os.getenv("WANDB_RUN_NAME", None),
            id=os.getenv("WANDB_RUN_ID", None),
            notes=os.getenv("WANDB_RUN_NOTES", None),
            tags=tags,
            group=os.getenv("WANDB_RUN_GROUP", None),
            job_type=os.getenv("WANDB_RUN_JOB_TYPE", None),
            resume_from=os.getenv("WANDB_RESUME_FROM", None),
            fork_from=os.getenv("WANDB_FORK_FROM", None),
            dir=log_dir,
            config=config_dict,
        )
        # Generated charts plot `trainer/step` on the x-axis.
        self.wandb.define_metric("trainer/step")
        self.wandb.define_metric("*", step_metric="trainer/step")
        logger.info("WandB logging enabled")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        wandb_metrics = {
            (k if self.tag is None else f"{self.tag}/{k}"): v
            for k, v in metrics.items()
        }
        # W&B drops a log to a step it already sent, e.g. validation logged after training at
        # the same step. So the step goes in the `trainer/step` key.
        self.wandb.log({**wandb_metrics, "trainer/step": step})

    def close(self) -> None:
        if self.wandb.run is not None:
            self.wandb.finish()


class MLFlowLogger(BaseLogger):
    """Logger implementation for MLflow."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLogger.Config):
        # MLflow uses a shared tracking server, so only the metrics rank logs.
        metrics_rank_only: ClassVar[bool] = True

        def __post_init__(self) -> None:
            # Validate on every rank before selecting the rank that starts a run.
            try:
                import mlflow  # noqa: F401
            except ModuleNotFoundError as error:
                if error.name != "mlflow":
                    raise
                raise ModuleNotFoundError(
                    "MLflow logging requires the optional mlflow package. "
                    "Install it with pip install mlflow-skinny.",
                    name="mlflow",
                ) from error

    def __init__(
        self,
        config: Config,
        *,
        log_dir: str,
        config_dict: dict[str, Any] | None = None,
        tag: str | None = None,
    ):
        import mlflow

        self.mlflow = mlflow
        self.tag = tag
        run = mlflow.active_run()
        self.owns_run = run is None
        if run is None:
            run = mlflow.start_run()
        if config_dict is not None:
            mlflow.log_dict(config_dict, "torchtitan_config.json")
        logger.info("MLflow logging enabled for run %s", run.info.run_id)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        # MLflow does not accept parentheses or percent signs in metric names.
        mlflow_metrics = {
            (key if self.tag is None else f"{self.tag}/{key}")
            .replace("(", "_")
            .replace(")", "")
            .replace("%", "percent"): value
            for key, value in metrics.items()
        }
        if len(mlflow_metrics) != len(metrics):
            raise ValueError("MLflow metric names collide after normalization")
        self.mlflow.log_metrics(mlflow_metrics, step=step)

    def close(self) -> None:
        if self.owns_run:
            status = "FAILED" if sys.exc_info()[0] is not None else "FINISHED"
            self.mlflow.end_run(status=status)
            self.owns_run = False


def _is_v_schedule(pp_schedule: str) -> bool:
    # V schedules put the last stage, which computes the loss, on pp rank 0.
    # Keep in sync with _get_pp_rank_to_stage_indices_mapping in pipeline_parallel.py.
    return get_schedule_class(pp_schedule) in (ScheduleZBVZeroBubble, ScheduleDualPipeV)


def ensure_pp_loss_visible(
    *, parallelism_context: ParallelismContext, pp_schedule: str, color: Color | NoColor
) -> None:
    """
    Ensures that the loss is visible on the console for pipeline-parallel training.

    For pipeline-parallel training, the loss is only visible on the last pipeline stage.
    This function checks if the appropriate rank is included in the LOG_RANK environment
    variable and warns if it's not.
    """

    # V Block Schedules return loss on rank 0
    if _is_v_schedule(pp_schedule):
        return

    # This hint is about run_train.sh, which exports LOG_RANK and passes it to
    # torchrun as --local-ranks-filter. Launchers that leave LOG_RANK unset, such
    # as scripts/multinode_trainer.slurm, do not filter console output.
    env_log_rank = os.environ.get("LOG_RANK", "")
    if not env_log_rank:
        return

    # Calculate the rank where loss is visible (first rank of the last pipeline stage)
    world_size = parallelism_context.world_size
    pp_size = parallelism_context.pp
    loss_visible_rank = (world_size // pp_size) * (pp_size - 1)

    if str(loss_visible_rank) not in env_log_rank.split(","):
        logger.warning(
            f"{color.red}Pipeline Parallel loss is not visible. "
            f"Please add {color.yellow}rank {loss_visible_rank}{color.red} "
            f"to LOG_RANK environment variable in run_train.sh.{color.reset}"
        )


def _get_metrics_rank(
    *,
    parallelism_context: ParallelismContext,
    pp_schedule: str,
) -> int:
    """
    Determines which rank should log metrics.

    Returns:
       int: The rank responsible for logging metrics:
            - Rank 0 for non-pipeline-parallel configs
            - Rank 0 for pipeline-parallel V schedules (ZBVZeroBubble, DualPipeV)
            - The first rank of the last pipeline stage for other pipeline-parallel schedules
    """
    # Early return for non-pipeline-parallel configurations
    if not parallelism_context.pp_enabled:
        return 0

    # V Block Schedules return loss on rank 0
    if _is_v_schedule(pp_schedule):
        return 0

    # Calculate first rank of the last pipeline stage
    world_size = parallelism_context.world_size
    pp_size = parallelism_context.pp
    return (world_size // pp_size) * (pp_size - 1)


class MetricsProcessor(Configurable):
    """Metrics processor to processes the metrics and log metrics.

    The current MetricsProcessor log some metrics to STDOUT and some metrics to
    TensorBoard, WandB, or MLflow.

    Args:
        config (Config): Metrics configuration.
        parallelism_context (ParallelismContext): Parallel dimensions.
        device_memory_monitor (DeviceMemoryMonitor): Monitor supplied by the
            execution component that owns the device.
        dump_folder (str): Base folder for log output.
        pp_schedule (str): Pipeline parallel schedule name.
        ft_enable (bool): Whether fault tolerance is enabled.
        ft_replica_id (int): Fault tolerance replica ID.
        config_dict (dict | None): Full job config dict for logging.
        tag (str | None): Prefix for metric names. Defaults to None.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        log_freq: int = 10
        """How often to log metrics to TensorBoard, in iterations"""

        loggers: list[BaseLogger.Config] = field(default_factory=list)
        """Metric exporters, built in list order. An empty list disables export."""

        disable_color_printing: bool = False
        """Whether to disable color printing in logs"""

        save_tb_folder: str = "tb"
        """Folder to dump TensorBoard states"""

        save_for_all_ranks: bool = False
        """
        Whether to save TensorBoard/Wandb metrics only for rank 0 or for all ranks.
        When this option is False and pipeline_parallel_degree is > 1, the metrics
        component uses the 0th rank of the last stage pipeline group, which is the
        only stage that computes loss metrics.
        """

        def __post_init__(self) -> None:
            if self.log_freq <= 0:
                raise ValueError("metrics.log_freq must be greater than 0.")

    config: Config
    loggers: list[BaseLogger]
    parallelism_context: ParallelismContext
    device_memory_monitor: DeviceMemoryMonitor
    color: NoColor | Color

    gpu_peak_flops: float
    ntokens_since_last_log: int
    data_loading_times: list[float]
    time_last_log: float
    step_last_log: int | None

    has_quantization: bool
    optimizers: OptimizersContainer | None
    model_parts: list[torch.nn.Module] | None

    def __init__(
        self,
        config: Config,
        *,
        parallelism_context: ParallelismContext,
        device_memory_monitor: DeviceMemoryMonitor,
        dump_folder: str = "./outputs",
        pp_schedule: str = "1F1B",
        ft_enable: bool = False,
        ft_replica_id: int = 0,
        config_dict: dict[str, Any] | None = None,
        tag: str | None = None,
        has_quantization: bool = False,
    ):
        self.loggers = []
        if config.loggers:
            rank = torch.distributed.get_rank()
            metrics_rank = _get_metrics_rank(
                parallelism_context=parallelism_context, pp_schedule=pp_schedule
            )
            log_dir = os.path.join(
                dump_folder,
                config.save_tb_folder,
                datetime.now().strftime("%Y%m%d-%H%M"),
            )
            if ft_enable:
                log_dir = os.path.join(log_dir, f"replica_{ft_replica_id}")
            if config.save_for_all_ranks:
                log_dir = os.path.join(log_dir, f"rank_{rank}")
            for logger_config in config.loggers:
                if rank != metrics_rank and (
                    not config.save_for_all_ranks or logger_config.metrics_rank_only
                ):
                    continue
                self.loggers.append(
                    logger_config.build(
                        log_dir=log_dir, config_dict=config_dict, tag=tag
                    )
                )
        self.parallelism_context = parallelism_context
        self.config = config
        self.device_memory_monitor = device_memory_monitor
        # used for colorful printing
        self.color = NoColor() if config.disable_color_printing else Color()

        self.gpu_peak_flops = utils.get_peak_flops(
            self.device_memory_monitor.device_name
        )
        # Tokens, data-loading times and the timer cover the window since the
        # last reset(), not only since the last log.
        self.ntokens_since_last_log = 0
        self.data_loading_times = []
        self.time_last_log = time.perf_counter()
        self.step_last_log = None
        self.device_memory_monitor.reset_peak_stats()

        self.has_quantization = has_quantization

        # These variables have to be set later as they depend on other components or model.
        self.optimizers = None
        self.model_parts = None

    def should_log(self, step: int) -> bool:
        if self.step_last_log is None:
            self.step_last_log = step - 1
        return step == 1 or step % self.config.log_freq == 0

    def log(
        self,
        step: int,
        global_avg_loss: float,
        global_max_loss: float,
        grad_norm: float,
        extra_metrics: dict[str, Any] | None = None,
        *,
        num_flops: float,
    ):
        """
        Log training metrics including loss, throughput, and memory statistics.

        Args:
            step: Current training step
            global_avg_loss: Global average loss across all valid tokens on all ranks
                Defined as global_loss_sum / global_valid_tokens
            global_max_loss: Maximum local loss across all ranks
                Defined as max(local_loss_sum / local_valid_tokens)
            grad_norm: Gradient norm after clipping
            extra_metrics: Optional additional metrics to log
            num_flops: Mean model-wide logical FLOPs per DP replica in this interval

        """

        time_delta = time.perf_counter() - self.time_last_log

        performance = compute_training_performance_metrics(
            num_tokens=self.ntokens_since_last_log,
            elapsed_time=time_delta,
            non_data_parallel_size=self.parallelism_context.non_data_parallel_size,
            num_flops=num_flops,
            gpu_peak_flops=self.gpu_peak_flops,
            has_quantization=self.has_quantization,
        )
        tps = performance["tokens_per_second"]
        tflops = performance["tflops"]
        mfu = performance.get("mfu_percent")

        assert self.step_last_log is not None
        time_end_to_end = time_delta / (step - self.step_last_log)
        time_data_loading = sum(self.data_loading_times) / len(self.data_loading_times)
        time_data_loading_pct = 100 * sum(self.data_loading_times) / time_delta

        device_mem_stats = self.device_memory_monitor.get_peak_stats()

        metrics = {
            "loss_metrics/global_avg_loss": global_avg_loss,
            "loss_metrics/global_max_loss": global_max_loss,
            "grad_norm": grad_norm,
            "throughput(tps)": tps,
            "tflops": tflops,
            "time_metrics/end_to_end(s)": time_end_to_end,
            "time_metrics/data_loading(s)": time_data_loading,
            "time_metrics/data_loading(%)": time_data_loading_pct,
            "memory/max_active(GiB)": device_mem_stats.max_active_gib,
            "memory/max_active(%)": device_mem_stats.max_active_pct,
            "memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
            "memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
            "memory/num_alloc_retries": device_mem_stats.num_alloc_retries,
            "memory/num_ooms": device_mem_stats.num_ooms,
        }
        if mfu is not None:
            metrics["mfu(%)"] = mfu

        if extra_metrics:
            metrics.update(extra_metrics)

        for metric_logger in self.loggers:
            metric_logger.log(metrics, step)

        color = self.color
        mfu_str = f"{mfu:.2f}%" if mfu is not None else "N/A"
        logger.info(
            f"{color.red}step: {step:2}  "
            f"{color.green}loss: {global_avg_loss:8.5f}  "
            f"{color.orange}grad_norm: {grad_norm:7.4f}  "
            f"{color.turquoise}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)  "
            f"{color.blue}tps: {round(tps):,}  "
            f"{color.cyan}tflops: {tflops:,.2f}  "
            f"{color.magenta}mfu: {mfu_str}{color.reset}"
        )

        self.step_last_log = step

    def reset(self) -> None:
        """Start a new window for throughput, data-loading time and peak memory.

        The trainer calls this right before training resumes after a log or a
        validation, and validators call it before validating, so neither window
        includes the other or a checkpoint saved at that step.
        """
        self.ntokens_since_last_log = 0
        self.data_loading_times.clear()
        self.time_last_log = time.perf_counter()
        self.device_memory_monitor.reset_peak_stats()

    def log_validation(
        self, loss: float, step: int, extra_metrics: dict[str, Any] | None = None
    ):
        time_delta = time.perf_counter() - self.time_last_log

        device_mem_stats = self.device_memory_monitor.get_peak_stats()

        # tokens per second per device, abbreviated as tps
        tps = self.ntokens_since_last_log / (
            time_delta * self.parallelism_context.non_data_parallel_size
        )

        metrics = {
            "validation_metrics/loss": loss,
            "validation_metrics/throughput(tps)": tps,
            "validation_metrics/memory/max_active(GiB)": device_mem_stats.max_active_gib,
            "validation_metrics/memory/max_active(%)": device_mem_stats.max_active_pct,
            "validation_metrics/memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
            "validation_metrics/memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
        }

        if extra_metrics:
            metrics.update(extra_metrics)

        for metric_logger in self.loggers:
            metric_logger.log(metrics, step)

        color = self.color
        logger.info(
            f"{color.yellow}validate step: {step:2}  "
            f"{color.green}loss: {loss:7.4f}  "
            f"{color.turquoise}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)  "
            f"{color.blue}tps: {round(tps):,}{color.reset}"
        )

        self.step_last_log = step

    def close(self):
        for metric_logger in self.loggers:
            metric_logger.close()
