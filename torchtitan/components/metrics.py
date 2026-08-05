# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import os
import time
from collections import namedtuple
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter
from torchtitan.components.moe_metrics import (
    MoEMetricCollector,
    MoEMetricsConfig,
    set_active_moe_metric_collector,
)
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Configurable
from torchtitan.distributed import ParallelDims
from torchtitan.tools import utils
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import Color, device_module, device_type, NoColor


def _extract_model_metadata(model_config: Any | None) -> dict[str, Any]:
    """Best-effort architecture hyperparameters for the metrics manifest.

    Reads only model-agnostic fields exposed by every ``Decoder.Config``: the
    top-level ``dim`` and layer count, plus the scalar attention hyperparameters
    (``n_heads``, head dims, LoRA ranks, mask type, ...) from the first
    transformer block's ``attention`` sub-config. Nested sub-configs (Linear /
    RMSNorm projections) are skipped. Returns an empty dict when the config does
    not expose these attributes, so it is safe for any model.
    """
    if model_config is None:
        return {}
    meta: dict[str, Any] = {}
    dim = getattr(model_config, "dim", None)
    if isinstance(dim, int):
        meta["dim"] = dim
    layers = getattr(model_config, "layers", None)
    if isinstance(layers, (list, tuple)) and layers:
        meta["num_layers"] = len(layers)
        attention = getattr(layers[0], "attention", None)
        if attention is not None and dataclasses.is_dataclass(attention):
            attn = {
                f.name: getattr(attention, f.name)
                for f in dataclasses.fields(attention)
                if isinstance(getattr(attention, f.name), (bool, int, float, str))
            }
            if attn:
                meta["attention"] = attn
    return meta


# named tuple for passing device memory stats for logging
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


class BaseLogger:
    """Logger that does nothing, used when logging is disabled."""

    def log(self, metrics: dict[str, Any], step: int) -> None:
        pass

    def close(self) -> None:
        pass


class TensorBoardLogger(BaseLogger):
    """Logger implementation for TensorBoard."""

    def __init__(self, log_dir: str, tag: str | None = None):
        self.tag = tag
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir, max_queue=1000)
        logger.info(f"TensorBoard logging enabled. Logs will be saved at {log_dir}")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        for k, v in metrics.items():
            tag = k if self.tag is None else f"{self.tag}/{k}"
            self.writer.add_scalar(tag, v, step)

    def close(self) -> None:
        self.writer.close()


class WandBLogger(BaseLogger):
    """Logger implementation for Weights & Biases."""

    def __init__(
        self,
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

        self.wandb.init(
            entity=os.getenv("WANDB_TEAM", None),
            project=os.getenv("WANDB_PROJECT", "torchtitan"),
            name=os.getenv("WANDB_RUN_NAME", None),
            id=os.getenv("WANDB_RUN_ID", None),
            notes=os.getenv("WANDB_RUN_NOTES", None),
            tags=os.getenv("WANDB_RUN_TAGS", None),
            group=os.getenv("WANDB_RUN_GROUP", None),
            job_type=os.getenv("WANDB_RUN_JOB_TYPE", None),
            resume_from=os.getenv("WANDB_RESUME_FROM", None),
            fork_from=os.getenv("WANDB_FORK_FROM", None),
            dir=log_dir,
            config=config_dict,
        )
        logger.info("WandB logging enabled")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        wandb_metrics = {
            (k if self.tag is None else f"{self.tag}/{k}"): v
            for k, v in metrics.items()
        }
        self.wandb.log(wandb_metrics, step=step)

    def close(self) -> None:
        if self.wandb.run is not None:
            self.wandb.finish()


class LoggerContainer(BaseLogger):
    """Container to call all loggers enabled in the job config."""

    def __init__(self) -> None:
        self._loggers: list[BaseLogger] = []

    def add_logger(self, logger_instance: BaseLogger) -> None:
        self._loggers.append(logger_instance)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        for logger_instance in self._loggers:
            logger_instance.log(metrics, step)

    @property
    def number_of_loggers(self) -> int:
        return len(self._loggers)

    @property
    def tb_log_dir(self) -> str | None:
        """Log directory of the TensorBoard logger, if one is enabled."""
        for logger_instance in self._loggers:
            if isinstance(logger_instance, TensorBoardLogger):
                return logger_instance.log_dir
        return None

    def close(self) -> None:
        for logger_instance in self._loggers:
            logger_instance.close()


def ensure_pp_loss_visible(
    *, parallel_dims: ParallelDims, pp_schedule: str, color: Color | NoColor
) -> None:
    """
    Ensures that the loss is visible on the console for pipeline-parallel training.

    For pipeline-parallel training, the loss is only visible on the last pipeline stage.
    This function checks if the appropriate rank is included in the LOG_RANK environment
    variable and warns if it's not.
    """

    # V Block Schedules return loss on rank 0
    if pp_schedule == "ZBVZeroBubble":
        return

    # Calculate the rank where loss is visible (first rank of the last pipeline stage)
    world_size = parallel_dims.world_size
    pp_size = parallel_dims.pp
    loss_visible_rank = (world_size // pp_size) * (pp_size - 1)

    # Check if the loss-visible rank is included in LOG_RANK environment variable
    env_logged_ranks = os.environ.get("LOG_RANK", "").split(",")
    if env_logged_ranks == [""]:
        env_logged_ranks = []

    if str(loss_visible_rank) not in env_logged_ranks:
        logger.warning(
            f"{color.red}Pipeline Parallel loss is not visible. "
            f"Please add {color.yellow}rank {loss_visible_rank}{color.red} "
            f"to LOG_RANK environment variable in run_train.sh.{color.reset}"
        )


def _get_metrics_rank(
    *,
    parallel_dims: ParallelDims,
    pp_schedule: str,
) -> int:
    """
    Determines which rank should log metrics.

    Returns:
       int: The rank responsible for logging metrics:
            - Rank 0 for non-pipeline-parallel configs
            - Rank 0 for pipeline-parallel 'ZBVZeroBubble' schedule
            - The first rank of the last pipeline stage for other pipeline-parallel schedules
    """
    # Early return for non-pipeline-parallel configurations
    if not parallel_dims.pp_enabled:
        return 0

    # V Block Schedules return loss on rank 0
    if pp_schedule == "ZBVZeroBubble":
        return 0

    # Calculate first rank of the last pipeline stage
    world_size = parallel_dims.world_size
    pp_size = parallel_dims.pp
    return (world_size // pp_size) * (pp_size - 1)


class MetricsProcessor(Configurable):
    """Metrics processor to processes the metrics and log metrics.

    The current MetricsProcessor log some metrics to STDOUT and some metrics to
    TensorBoard or WandB.

    Args:
        config (Config): Metrics configuration.
        parallel_dims (ParallelDims): Parallel dimensions.
        dump_folder (str): Base folder for log output.
        pp_schedule (str): Pipeline parallel schedule name.
        ft_enable (bool): Whether fault tolerance is enabled.
        ft_replica_id (int): Fault tolerance replica ID.
        config_dict (dict | None): Full job config dict for WandB logging.
        tag (str | None): Tag to use for TensorBoard or WandB. Defaults to None.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        log_freq: int = 10
        """How often to log metrics to TensorBoard, in iterations"""

        enable_tensorboard: bool = False
        """Whether to log metrics to TensorBoard"""

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

        enable_wandb: bool = False
        """Whether to log metrics to Weights & Biases"""

        moe: MoEMetricsConfig = field(default_factory=MoEMetricsConfig)
        """MoE grouped-GEMM metrics collection settings."""

    config: Config
    logger: BaseLogger
    parallel_dims: ParallelDims
    device_memory_monitor: DeviceMemoryMonitor
    color: utils.NoColor | utils.Color

    gpu_peak_flops: float
    ntokens_since_last_log: int
    data_loading_times: list[float]
    time_last_log: float

    num_flops_per_token: int
    has_quantization: bool
    optimizers: OptimizersContainer | None
    model_parts: list[torch.nn.Module] | None

    def __init__(
        self,
        config: Config,
        *,
        parallel_dims: ParallelDims,
        dump_folder: str = "./outputs",
        pp_schedule: str = "1F1B",
        ft_enable: bool = False,
        ft_replica_id: int = 0,
        config_dict: dict[str, Any] | None = None,
        tag: str | None = None,
        has_quantization: bool = False,
        model_config: Any | None = None,
    ):
        self.logger = self._build_metric_logger(
            config=config,
            parallel_dims=parallel_dims,
            dump_folder=dump_folder,
            pp_schedule=pp_schedule,
            ft_enable=ft_enable,
            ft_replica_id=ft_replica_id,
            config_dict=config_dict,
            tag=tag,
        )
        self.parallel_dims = parallel_dims
        self.config = config
        self.device_memory_monitor = build_device_memory_monitor()
        # used for colorful printing
        self.color = utils.NoColor() if config.disable_color_printing else utils.Color()
        self.gpu_peak_flops = utils.get_peak_flops(
            self.device_memory_monitor.device_name
        )
        self.ntokens_since_last_log = 0
        self.data_loading_times = []
        self.time_last_log = time.perf_counter()
        self.device_memory_monitor.reset_peak_stats()

        self.has_quantization = has_quantization

        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        # These variables have to be set later as they depend on other components or model.
        self.num_flops_per_token = -1
        self.optimizers = None
        self.model_parts = None
        training_cfg = (config_dict or {}).get("training", {})
        model_spec_cfg = (config_dict or {}).get("model_spec", {})
        # Per-rank (per-microbatch) batch size is local_batch_size; the global
        # batch is split into `gradient_accumulation_steps` such microbatches.
        batch_degree = parallel_dims.dp_replicate * parallel_dims.dp_shard
        local_batch_size = training_cfg.get("local_batch_size")
        global_batch_size = training_cfg.get("global_batch_size")
        gradient_accumulation_steps = None
        if local_batch_size:
            if not global_batch_size or global_batch_size < 0:
                global_batch_size = local_batch_size * batch_degree
            gradient_accumulation_steps = global_batch_size // (
                local_batch_size * batch_degree
            )
        run_metadata = {
            "model_name": model_spec_cfg.get("name"),
            "model_flavor": model_spec_cfg.get("flavor"),
            "seq_len": training_cfg.get("seq_len"),
            "local_batch_size": local_batch_size,
            "global_batch_size": global_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "tp": parallel_dims.tp,
            "ep": parallel_dims.ep,
            "cp": parallel_dims.cp,
            "pp": parallel_dims.pp,
            "dp_shard": parallel_dims.dp_shard,
            "dp_replicate": parallel_dims.dp_replicate,
            "mixed_precision_param": training_cfg.get("mixed_precision_param"),
            **_extract_model_metadata(model_config),
        }
        # The TensorBoard SummaryWriter lives only on the metrics rank (rank 0
        # without PP, the last-stage rank otherwise). The MoE collector writes
        # from rank 0, so broadcast the log dir to every rank to keep the MoE
        # event file in the same TB run as the training scalars.
        moe_tb_log_dir = self._resolve_tb_log_dir(
            parallel_dims=parallel_dims, pp_schedule=pp_schedule
        )
        # The grouped-GEMM metrics hook runs inside GroupedExperts._experts_forward,
        # which is compiled with fullgraph=True when the model is compiled. Its
        # active-collector body (ContextVar read, record construction) is not
        # traceable, so an enabled collector is incompatible with model compile.
        # The disabled path is compile-safe (dead-code-eliminated), so this only
        # blocks the both-on combination.
        compile_cfg = (config_dict or {}).get("compile", {})
        compiles_model = compile_cfg.get("enable", False) and "model" in (
            compile_cfg.get("components", ["model", "loss"])
        )
        if config.moe.enabled and compiles_model:
            raise ValueError(
                "MoE metrics collection (metrics.moe.enabled=true) is incompatible "
                "with model compilation (compile.enable=true with 'model' in "
                "compile.components): the grouped-GEMM metrics hook runs inside the "
                "fullgraph-compiled expert forward and cannot be traced. To collect "
                "metrics, set compile.enable=false (or remove 'model' from "
                "compile.components); otherwise set metrics.moe.enabled=false."
            )
        self.moe_metric_collector = MoEMetricCollector(
            config=config.moe,
            dump_folder=dump_folder,
            rank=rank,
            world_size=world_size,
            run_metadata=run_metadata,
            tb_log_dir=moe_tb_log_dir,
        )

    def _resolve_tb_log_dir(
        self, *, parallel_dims: ParallelDims, pp_schedule: str
    ) -> str | None:
        """Resolve the TensorBoard log dir and make it visible on every rank.

        ``self.logger`` only holds a ``TensorBoardLogger`` on the metrics rank
        (rank 0 without PP, the last pipeline stage's first rank otherwise);
        non-metrics ranks get a bare ``BaseLogger`` with no log dir. The MoE
        collector writes from rank 0 and the ``tb`` sink runs a collective
        ``gather_object`` on every rank, so all ranks must agree on whether the
        sink is built. Broadcast the metrics rank's log dir to all ranks so the
        decision (and the target directory) is identical everywhere. Returns
        ``None`` when TensorBoard is disabled.
        """
        local_dir = (
            self.logger.tb_log_dir if isinstance(self.logger, LoggerContainer) else None
        )
        # When the MoE collector is disabled it never builds the tb sink, so
        # ranks do not need to agree on the log dir. Skip the broadcast to keep
        # MetricsProcessor construction collective-free when the feature is off.
        # config.moe.enabled is identical on every rank, so all ranks either
        # skip together or broadcast together.
        if not self.config.moe.enabled:
            return local_dir
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            return local_dir
        src = _get_metrics_rank(parallel_dims=parallel_dims, pp_schedule=pp_schedule)
        payload = [local_dir]
        torch.distributed.broadcast_object_list(payload, src=src)
        return payload[0]

    def begin_step(self, step: int) -> None:
        """Mark the beginning of a train step for step-scoped collectors."""
        self.moe_metric_collector.begin_step(step)
        if self.moe_metric_collector.is_enabled():
            set_active_moe_metric_collector(self.moe_metric_collector)

    def end_step(self) -> None:
        """Flush step-scoped collectors after a completed train step."""
        self.moe_metric_collector.flush()
        set_active_moe_metric_collector(None)

    def should_log(self, step: int) -> bool:
        return step == 1 or step % self.config.log_freq == 0

    def _build_metric_logger(
        self,
        *,
        config: Config,
        parallel_dims: ParallelDims,
        dump_folder: str,
        pp_schedule: str,
        ft_enable: bool = False,
        ft_replica_id: int = 0,
        config_dict: dict[str, Any] | None = None,
        tag: str | None = None,
    ) -> BaseLogger:
        """
        Build an appropriate metric logger based on configuration.
        """
        # Log initial config state
        logger.debug(
            f"Building logger with config: wandb={config.enable_wandb}, "
            f"tensorboard={config.enable_tensorboard}"
        )

        # Check if any logging backend is enabled
        has_logging_enabled = config.enable_tensorboard or config.enable_wandb

        # Determine if this rank should log
        should_log = has_logging_enabled
        if (not config.save_for_all_ranks) and should_log:
            metrics_rank = _get_metrics_rank(
                parallel_dims=parallel_dims, pp_schedule=pp_schedule
            )
            should_log = torch.distributed.get_rank() == metrics_rank

        logger.debug(
            f"Logging decision: has_logging_enabled={has_logging_enabled}, should_log={should_log}"
        )

        if not should_log:
            logger.debug("Returning BaseLogger due to should_log=False")
            return BaseLogger()

        # Setup logging directory
        base_log_dir = os.path.join(
            dump_folder,
            config.save_tb_folder,
            datetime.now().strftime("%Y%m%d-%H%M"),
        )

        if ft_enable:
            base_log_dir = os.path.join(
                base_log_dir,
                f"replica_{ft_replica_id}",
            )

        if config.save_for_all_ranks:
            base_log_dir = os.path.join(
                base_log_dir, f"rank_{torch.distributed.get_rank()}"
            )

        # Create logger container
        logger_container = LoggerContainer()

        # Create loggers in priority order
        if config.enable_wandb:
            logger.debug("Attempting to create WandB logger")
            try:
                wandb_logger = WandBLogger(
                    base_log_dir, config_dict=config_dict, tag=tag
                )
                logger_container.add_logger(wandb_logger)
            except Exception as e:
                if "No module named 'wandb'" in str(e):
                    logger.error(
                        "Failed to create WandB logger: No module named 'wandb'. Please install it using 'pip install wandb'."
                    )
                else:
                    logger.error(f"Failed to create WandB logger: {e}")

        if config.enable_tensorboard:
            logger.debug("Creating TensorBoard logger")
            tensorboard_logger = TensorBoardLogger(base_log_dir, tag)
            logger_container.add_logger(tensorboard_logger)

        if logger_container.number_of_loggers == 0:
            logger.debug("No loggers enabled, returning an empty LoggerContainer")
        return logger_container

    def log(
        self,
        step: int,
        global_avg_loss: float,
        global_max_loss: float,
        grad_norm: float,
        extra_metrics: dict[str, Any] | None = None,
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

        """
        assert self.num_flops_per_token > 0, "num_flops_per_token must be set"

        time_delta = time.perf_counter() - self.time_last_log

        # tokens per second per device, abbreviated as tps
        tps = self.ntokens_since_last_log / (
            time_delta * self.parallel_dims.non_data_parallel_size
        )
        # model FLOPS utilization
        # For its definition and calculation, please refer to the PaLM paper:
        # https://arxiv.org/abs/2204.02311
        # MFU is based on BF16 peak FLOPS which is misleading when quantization
        # (FP8/MX) is active, so we skip it in that case.
        tflops = self.num_flops_per_token * tps / 1e12
        if self.has_quantization:
            mfu = None
        else:
            mfu = 100 * self.num_flops_per_token * tps / self.gpu_peak_flops

        time_end_to_end = time_delta / self.config.log_freq
        # Feed the per-step wall time to the MoE collector so its TB sink can
        # overlay end_to_end(s) against the summed per-layer peak expert load.
        # No-op unless MoE metrics collection is enabled.
        self.moe_metric_collector.record_step_end_to_end(step, time_end_to_end)
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

        self.logger.log(metrics, step)

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
            time_delta * self.parallel_dims.non_data_parallel_size
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

        self.logger.log(metrics, step)

        color = self.color
        logger.info(
            f"{color.yellow}validate step: {step:2}  "
            f"{color.green}loss: {loss:7.4f}  "
            f"{color.turquoise}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)  "
            f"{color.blue}tps: {round(tps):,}{color.reset}"
        )

        self.ntokens_since_last_log = 0
        self.time_last_log = time.perf_counter()
        self.device_memory_monitor.reset_peak_stats()

    def close(self):
        set_active_moe_metric_collector(None)
        self.moe_metric_collector.close()
        self.logger.close()
