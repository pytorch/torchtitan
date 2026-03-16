# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Aggregation: reduce experiment metrics from JSONL + logging subprocess."""

import glob
import json
import logging
import multiprocessing
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Any

from torchtitan.components.metrics import (
    BaseLogger,
    LoggerContainer,
    TensorBoardLogger,
    WandBLogger,
)
from torchtitan.observability.metrics import REDUCE_REGISTRY
from torchtitan.observability.structured_logging import init_observability
from torchtitan.tools.utils import Color, NoColor

logger = logging.getLogger(__name__)

_QUEUE_TIMEOUT_S = 600  # 10 minutes — if no signal, assume training crashed


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(entries: list[dict]) -> dict[str, float]:
    """Reduce a list of metric entries to a single dict.

    Groups entries by key and delegates to REDUCE_REGISTRY.

    Example:

        entries = [
            {"key": "loss", "reduce": "MeanMetric", "sum": 6.0, "weight": 3.0},
            {"key": "loss", "reduce": "MeanMetric", "sum": 4.0, "weight": 2.0},
        ]
        aggregate(entries)  # {"loss": 2.0}
    """
    if not entries:
        return {}

    # Group entries by metric key
    by_key: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        by_key[entry["key"]].append(entry)

    # Reduce each key using its registered reduce type
    result: dict[str, float] = {}
    for key, key_entries in by_key.items():
        cls = REDUCE_REGISTRY[key_entries[0]["reduce"]]
        result[key] = cls.get_reduced_value_from_states(key_entries)
    return result


# ---------------------------------------------------------------------------
# JSONL reader
# ---------------------------------------------------------------------------


def _read_new_lines(
    log_dir: str,
    offsets: dict[str, int],
    buffer: dict[int, list[dict]],
) -> None:
    """Read new JSONL lines into buffer, grouped by step.

    Tracks file offsets to avoid re-reading old lines.
    """
    for fp in sorted(glob.glob(os.path.join(log_dir, "*.jsonl"))):
        with open(fp) as f:
            f.seek(offsets.get(fp, 0))
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                step = entry.get("step")
                if step is not None:
                    buffer[step].append(entry)
                else:
                    logger.warning(
                        "Skipping experiment entry without step: %s",
                        entry.get("key", "?"),
                    )
            offsets[fp] = f.tell()


# ---------------------------------------------------------------------------
# Flush: aggregate + write to backends + console
# ---------------------------------------------------------------------------


def _flush_step(
    step: int,
    buffer: dict[int, list[dict]],
    is_validation: bool,
    logger_backend: BaseLogger,
    console_log_metric_keys: list[str],
    console_log_validation_keys: list[str],
) -> tuple[dict[str, float], int]:
    """Aggregate entries for ``step``, write to backends, print to console.

    Also purges entries for older steps to prevent memory leaks.
    """
    # Pop this step's entries and discard older steps
    entries = buffer.pop(step, [])
    for s in [s for s in buffer if s < step]:
        del buffer[s]

    aggregated = aggregate(entries)

    if aggregated:
        # Write all metrics to WandB/TensorBoard
        logger_backend.log(aggregated, step)

        # Log training console line
        if console_log_metric_keys:
            _log_to_console(step, aggregated, console_log_metric_keys)

        # Log validation console line (only on validation steps)
        if is_validation and console_log_validation_keys:
            _log_to_console(
                step, aggregated, console_log_validation_keys, prefix="validate "
            )

    return aggregated, len(entries)


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

# Available colors from Color, excluding reset/black/white.
_COLORS = [
    k
    for k in vars(Color)
    if not k.startswith("_") and k not in ("reset", "black", "white")
]


def _fmt_value(value: Any) -> str:
    """Format a value for console display.

    Numbers: show at least 2 non-zero decimals, up to 5 decimal places.
    Other types (bool, str): converted to string as-is.

    Examples:

        _fmt_value(3.67060)   → '3.67'
        _fmt_value(0.00123)   → '0.0012'
        _fmt_value(1234.5)    → '1234.5'
        _fmt_value(True)      → 'True'
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return str(value)
    if value == 0:
        return "0"
    if isinstance(value, int) or abs(value) >= 100:
        return f"{value:.1f}"

    # e.g. value=3.67 → frac=0.67
    frac = abs(value) - int(abs(value))
    if frac == 0:
        return f"{value:.1f}"

    # Walk decimal digits until we've seen 2 non-zero ones.
    # e.g. 0.00123: digit 1→0, digit 2→0, digit 3→1(first!), digit 4→2(second!) → 4 decimals
    n_decimals, n_nonzero = 0, 0
    temp = frac
    while n_decimals < 5 and n_nonzero < 2:
        n_decimals += 1
        temp *= 10  # e.g. 0.00123 → 0.0123 → 0.123 → 1.23
        if int(temp) % 10 != 0 or n_nonzero > 0:
            n_nonzero += 1
    return f"{value:.{max(n_decimals, 2)}f}"


def _log_to_console(
    step: int,
    aggregated: dict[str, float],
    keys: list[str],
    prefix: str = "",
) -> None:
    """Log one console line with the configured metric keys.

    Colors cycle by position in the key list. Missing metrics show '--'.
    Color is auto-detected (disabled when stdout is piped to file/CI).

    Args:
        step: Training step number.
        aggregated: All aggregated metrics for this step.
        keys: Which metric keys to print (in order).
        prefix: Optional label before "step:" (e.g., "validate ").

    Example:

        aggregated = {"training/loss": 2.5, "training/lr": 0.001, "memory/peak": 14.2}
        _log_to_console(step=5, aggregated=aggregated,
                        keys=["training/loss", "memory/peak"])
        # Output: "step:  5  training/loss: 2.5  memory/peak: 14.2"
    """
    color = Color() if sys.stdout.isatty() else NoColor()
    parts = [f"{color.red}{prefix}step: {step:2}"]
    for i, key in enumerate(keys):
        c = getattr(color, _COLORS[i % len(_COLORS)])
        val = aggregated.get(key)
        if val is None:
            parts.append(f"{c}{key}: --")
        else:
            parts.append(f"{c}{key}: {_fmt_value(val)}")
    parts.append(color.reset)
    logger.info("  ".join(parts))


# ---------------------------------------------------------------------------
# Backend logger builder
# ---------------------------------------------------------------------------


def _build_metric_logger(
    dump_folder: str,
    *,
    enable_wandb: bool = False,
    enable_tensorboard: bool = False,
    save_tb_folder: str = "tb",
    config_dict: dict[str, Any] | None = None,
    tag: str | None = None,
) -> BaseLogger:
    """Build WandB/TB logger."""
    container = LoggerContainer()
    if enable_tensorboard:
        tb_dir = os.path.join(
            dump_folder, save_tb_folder, datetime.now().strftime("%Y%m%d-%H%M")
        )
        container.add_logger(TensorBoardLogger(log_dir=tb_dir, tag=tag))
    if enable_wandb:
        container.add_logger(
            WandBLogger(log_dir=dump_folder, config_dict=config_dict, tag=tag)
        )
    return container


# ---------------------------------------------------------------------------
# Logging subprocess
# ---------------------------------------------------------------------------


def logging_worker(
    queue: multiprocessing.Queue,
    dump_folder: str,
    *,
    enable_wandb: bool = False,
    enable_tensorboard: bool = False,
    save_tb_folder: str = "tb",
    config_dict: dict[str, Any] | None = None,
    tag: str | None = None,
    console_log_metric_keys: list[str] | None = None,
    console_log_validation_keys: list[str] | None = None,
    queue_timeout_s: float = _QUEUE_TIMEOUT_S,
) -> None:
    """Background process that reads experiment JSONL, aggregates across
    ranks, and writes to WandB/TB/console. Shuts down on ``None`` sentinel
    or queue timeout.

    Example:

        queue = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=logging_worker,
            args=(queue, "./outputs"),
            kwargs={"enable_wandb": True, "console_log_metric_keys": ["training/loss"]},
        )
        p.start()
        queue.put((step, is_validation))  # signal to read + aggregate + flush
        queue.put(None)                   # shutdown

    Args:
        queue: Receives ``(step, is_validation)`` tuple, or ``None`` to
            shut down.
        dump_folder: Root output directory containing ``experiment_logs/``.
        enable_wandb: Whether to log to WandB.
        enable_tensorboard: Whether to log to TensorBoard.
        save_tb_folder: Subfolder for TensorBoard files.
        config_dict: Full config for ``wandb.init(config=...)``.
        tag: Prefix for TB/WandB scalar keys.
        console_log_metric_keys: Training metric keys for console each step.
        console_log_validation_keys: Validation metric keys for console
            (only printed on validation steps).
        queue_timeout_s: Seconds to wait for a signal before assuming
            training crashed. Default 600 (10 min).
    """
    init_observability(source="logging_worker", output_dir=dump_folder, rank=0)

    log_dir = os.path.join(dump_folder, "experiment_logs")
    buffer: dict[int, list[dict]] = defaultdict(list)

    # Skip historical data from previous runs / checkpoint resume.
    offsets: dict[str, int] = {}
    for fp in glob.glob(os.path.join(log_dir, "*.jsonl")):
        offsets[fp] = os.path.getsize(fp)

    # Build backend loggers (WandB, TensorBoard)
    logger_backend = _build_metric_logger(
        dump_folder,
        enable_wandb=enable_wandb,
        enable_tensorboard=enable_tensorboard,
        save_tb_folder=save_tb_folder,
        config_dict=config_dict,
        tag=tag,
    )
    logger.info("[logging process] started, reading from %s", log_dir)

    # Main loop: wait for signals, read JSONL, aggregate, flush
    while True:
        try:
            msg = queue.get(timeout=queue_timeout_s)
        except Exception:
            logger.warning(
                "[logging process] no signal in %ds, assuming training crashed",
                queue_timeout_s,
            )
            break

        if msg is None:
            break

        # Training process sends (step, is_validation) after barrier
        step, is_validation = msg
        time.sleep(0.02)  # let filesystem propagate writes

        # Read new JSONL lines from all rank files
        t_read_start = time.perf_counter()
        _read_new_lines(log_dir, offsets, buffer)
        t_read_end = time.perf_counter()

        # Aggregate and flush to backends + console
        aggregated, num_entries = _flush_step(
            step,
            buffer,
            is_validation,
            logger_backend,
            console_log_metric_keys or [],
            console_log_validation_keys or [],
        )

        logger.debug(
            "[obs] step %d: read=%.1fms entries=%d",
            step,
            (t_read_end - t_read_start) * 1000,
            num_entries,
        )

    logger.info("[logging process] shutting down")
    logger_backend.close()
