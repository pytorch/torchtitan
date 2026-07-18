# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import contextvars

import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import torch

from torchtitan.tools.logging import logger


@dataclass(kw_only=True, slots=True)
class MoEMetricsConfig:
    enabled: bool = False
    """Master switch for MoE grouped-GEMM metrics collection."""

    sample_every: int = 1
    """Record once every N train steps when enabled."""

    sinks: list[Literal["jsonl", "tb", "histogram"]] = field(
        default_factory=lambda: ["histogram"]
    )
    """Enabled sinks. The default ``histogram`` sink accumulates a per-expert
    token-count (grouped-GEMM M) histogram keyed by layer instead of one record
    per step, so its storage stays bounded over arbitrarily long runs. The
    ``jsonl`` sink instead writes one full-fidelity record per step, which is
    useful for debugging an exact shape sequence but grows with step count. The
    ``tb`` sink logs expert-load heatmaps, histograms, and the run manifest into
    TorchTitan's TensorBoard run (requires ``metrics.enable_tensorboard``)."""

    output_dir: str = "moe_metrics"
    """Directory under dump_folder where MoE metric artifacts are written."""

    max_records_per_rank: int = 100_000
    """Upper bound on buffered records per rank before drops start."""

    ranks: str = "rank0"
    """Rank filter: 'all', 'rank0', or comma-separated global rank list."""


@dataclass(frozen=True, kw_only=True, slots=True)
class GroupedGemmRecord:
    step: int
    layer_id: int
    micro_batch_id: int
    rank: int
    ep_rank: int
    ep_size: int
    num_local_experts: int
    top_k: int
    tokens_per_local_expert: tuple[int, ...]
    padded_tokens_per_local_expert: tuple[int, ...]
    gemm_w1: tuple[int, int, int]
    gemm_w3: tuple[int, int, int]
    gemm_w2: tuple[int, int, int]
    dtype: str
    dispatcher: str


@dataclass(kw_only=True, slots=True)
class _PendingGroupedGemm:
    """Buffered grouped-GEMM record that still holds the token-count tensors.

    The hook builds this without running any dispatched tensor ops (only
    Python-level ``.shape``/``.dtype`` reads and reference holding), so it is
    safe to call inside an activation-checkpointed region. Numeric extraction
    (``.tolist()``) is deferred to :meth:`MoEMetricCollector.flush`, which runs
    outside any checkpoint recompute and therefore cannot perturb the
    non-reentrant checkpoint's saved-tensor matching.
    """

    step: int
    layer_id: int
    micro_batch_id: int
    rank: int
    ep_rank: int
    ep_size: int
    top_k: int
    gemm_w1: tuple[int, int, int]
    gemm_w3: tuple[int, int, int]
    gemm_w2: tuple[int, int, int]
    dtype: str
    dispatcher: str
    tokens_tensor: torch.Tensor
    padded_tensor: torch.Tensor | None


class MoEMetricSink(Protocol):
    def write_record(self, record: GroupedGemmRecord) -> None:
        ...

    def flush(self) -> None:
        ...

    def close(self) -> None:
        ...


class JsonlSink:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8")

    def write_record(self, record: GroupedGemmRecord) -> None:
        self._file.write(json.dumps(asdict(record), ensure_ascii=True) + "\n")

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def _weighted_m_stats(m_counts: Counter[int]) -> tuple[int, int, int, float, int]:
    """Compute (count, min, max, mean, median) over an ``M`` histogram.

    ``m_counts`` maps each ``M`` value to the number of times it was observed.
    Stats are weighted by those counts so they describe the full sampled
    population. The median is the weighted median (lower of the two middle
    values for an even total count).
    """
    total = sum(m_counts.values())
    m_min = min(m_counts)
    m_max = max(m_counts)
    mean = sum(m * c for m, c in m_counts.items()) / total
    # Weighted median: walk sorted M values until the cumulative count crosses
    # the midpoint.
    midpoint = (total + 1) // 2
    cumulative = 0
    median: int = m_max
    for m in sorted(m_counts):
        cumulative += m_counts[m]
        if cumulative >= midpoint:
            median = m
            break
    return total, m_min, m_max, mean, median


@dataclass(slots=True)
class _ImbalanceAccum:
    """Running per-(step, layer) accumulator over per-expert ``M`` values on a
    single rank.

    Tracks streaming moments (count, sum, sum-of-squares) plus min/max and the
    expert identities achieving those extrema, so intra-rank stats
    (min/max/mean/std/CV over the local experts) can be derived without
    retaining every sample. ``m_sum`` doubles as this rank's total expert load
    ``L_r`` for the (step, layer), which is the quantity compared across ranks
    for inter-rank imbalance.

    An expert is identified by the pair ``(rank_id, local_ep_id)``. ``rank_id``
    is globally unique across the job, so the pair pins down the physical GPU
    and the expert slot within it without any further EP-coordinate bookkeeping.
    """

    count: int = 0
    m_min: int = 0
    m_max: int = 0
    m_sum: int = 0
    m_sumsq: int = 0
    min_expert: tuple[int, int] = (-1, -1)
    max_expert: tuple[int, int] = (-1, -1)
    _initialized: bool = False

    def add(self, values: tuple[int, ...], rank: int) -> None:
        if not values:
            return
        for local_ep_id, v in enumerate(values):
            expert = (rank, local_ep_id)
            if not self._initialized:
                self.m_min = v
                self.m_max = v
                self.min_expert = expert
                self.max_expert = expert
                self._initialized = True
            else:
                if v < self.m_min:
                    self.m_min = v
                    self.min_expert = expert
                if v > self.m_max:
                    self.m_max = v
                    self.max_expert = expert
            self.count += 1
            self.m_sum += v
            self.m_sumsq += v * v


# Expert identity pair: (rank_id, local_ep_id).
_ExpertId = tuple[int, int]

# Per (step, layer) inter-rank gather payload from one rank:
# (load L_r, per-expert min, min expert id, per-expert max, max expert id).
_ImbalancePayload = tuple[int, int, _ExpertId, int, _ExpertId]


def _fmt_expert(expert: _ExpertId) -> str:
    """Format an expert identity pair as ``rank/local`` for CSV.

    A ``/`` separator is used (not ``,``) so the pair stays in one CSV field.
    """
    rank, local_ep_id = expert
    return f"{rank}/{local_ep_id}"


def _moments_to_stats(
    accum: _ImbalanceAccum,
) -> tuple[int, int, int, float, float, _ExpertId, _ExpertId]:
    """Reduce a streaming accumulator to
    (count, min, max, mean, std, min_expert, max_expert).

    The standard deviation is the population std derived from the first two
    moments; ``var`` is clamped at 0 to absorb floating-point round-off.
    ``min_expert``/``max_expert`` are the expert identity pairs achieving the
    extrema.
    """
    mean = accum.m_sum / accum.count
    var = max(0.0, accum.m_sumsq / accum.count - mean * mean)
    return (
        accum.count,
        accum.m_min,
        accum.m_max,
        mean,
        math.sqrt(var),
        accum.min_expert,
        accum.max_expert,
    )


def _r2(value: float) -> float:
    """Round a float to 2 decimal places for compact CSV output."""
    return round(float(value), 2)


def _fmt_manifest_value(key: str, value: object) -> object:
    """Format a manifest value for the markdown table.

    ``top_k`` carries a ``-1`` sentinel when the MoE was built outside a
    ``Decoder`` (e.g. tests/experiments), so render that as ``unknown`` instead
    of a misleading numeric ``-1``.
    """
    if key == "top_k" and value == -1:
        return "unknown"
    return value


def _load_stats(
    entries: list[tuple[int, int]],
) -> tuple[int, int, int, int, int, float, float, float]:
    """Compute inter-rank stats over per-rank loads ``{L_r}`` for one
    (step, layer): (num_ranks, min, min_rank, max, max_rank, mean, std,
    cv, max_over_mean).

    ``entries`` is a list of ``(rank, load)`` pairs. ``min_rank``/``max_rank``
    are the ranks holding the least/most total tokens for the (step, layer).
    ``max_over_mean`` is the imbalance factor that predicts the synchronized
    expert-parallel stall (the grouped GEMM finishes with the busiest rank,
    while useful work scales with the mean). ``cv`` is the scale-free spread.
    """
    n = len(entries)
    loads = [load for _, load in entries]
    total = sum(loads)
    mean = total / n
    var = max(0.0, sum(v * v for v in loads) / n - mean * mean)
    std = math.sqrt(var)
    cv = std / mean if mean > 0 else 0.0
    max_over_mean = max(loads) / mean if mean > 0 else 0.0
    min_rank, l_min = min(entries, key=lambda e: e[1])
    max_rank, l_max = max(entries, key=lambda e: e[1])
    return n, l_min, min_rank, l_max, max_rank, mean, std, cv, max_over_mean


class HistogramMoESink:
    """Accumulate a histogram of per-expert token counts (the per-expert
    grouped-GEMM ``M`` dimension), keyed by ``(layer_id, M)``.

    Storage is ``O(unique (layer_id, M))`` rather than ``O(records)``, so this
    stays bounded over arbitrarily long runs and never trips
    ``max_records_per_rank``. The invariant ``N``/``K`` dims (fixed by the
    model) are recorded once in the run manifest rather than on every line.
    The CSV is written only at :meth:`close` to avoid partial files.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._hist: Counter[tuple[int, int]] = Counter()

    def write_record(self, record: GroupedGemmRecord) -> None:
        layer_id = record.layer_id
        for m in record.tokens_per_local_expert:
            self._hist[(layer_id, m)] += 1

    def flush(self) -> None:
        return

    def close(self) -> None:
        with self._path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["layer_id", "M", "count"])
            for layer_id, m in sorted(self._hist):
                writer.writerow([layer_id, m, self._hist[(layer_id, m)]])
        self._write_summary()

    def _write_summary(self) -> None:
        """Emit per-rank summary stats of the per-expert ``M`` distribution.

        Writes one row per layer plus an aggregate ``all`` row covering every
        layer on this rank. ``M`` values are weighted by their observed counts
        (each ``(layer_id, M)`` bin contributes ``count`` samples), so the
        stats reflect the full sampled population, not just the unique bins.
        """
        per_layer_counts: dict[int, Counter[int]] = {}
        overall: Counter[int] = Counter()
        for (layer_id, m), count in self._hist.items():
            per_layer_counts.setdefault(layer_id, Counter())[m] += count
            overall[m] += count

        summary_path = self._path.with_name(
            self._path.name.replace("m_histogram", "m_summary")
        )
        with summary_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["layer_id", "count", "min", "max", "mean", "median"])
            for layer_id in sorted(per_layer_counts):
                count, m_min, m_max, mean, median = _weighted_m_stats(
                    per_layer_counts[layer_id]
                )
                writer.writerow([layer_id, count, m_min, m_max, _r2(mean), median])
            if overall:
                count, m_min, m_max, mean, median = _weighted_m_stats(overall)
                writer.writerow(["all", count, m_min, m_max, _r2(mean), median])


class TensorBoardMoESink:
    """Log MoE expert-load metrics into TorchTitan's TensorBoard run.

    This folds the offline ``plot_expert_load_heatmap`` / ``plot_layer_load_heatmap``
    post-processing into the training session. Each rank accumulates the
    per-expert grouped-GEMM ``M`` dimension keyed by ``(step, layer, local
    expert)`` as records stream in (bounded by sampled steps x MoE layers x
    local experts). At :meth:`close` the per-rank slices are gathered onto rank
    0, assembled into a ``(row, layer, step)`` cube whose rows are laid out by
    WORLD rank (``rank * num_local_experts + local``), and logged to the shared
    TB ``log_dir`` as:

      * ``moe_expert_load/layer_{L}`` - a rank/local-expert x step heatmap image,
        plus a per-step ``per_expert_tokens`` histogram and per-step spread
        scalars (``tokens_max``, ``tokens_min``, ``tokens_range`` = max - min,
        ``tokens_mean``, ``max_over_mean``).
      * ``moe_layer_load/expert_{E}`` - a layer x step heatmap image for the
        first and last row, plus per-step ``per_layer_tokens`` histograms and
        ``max_over_mean`` scalars.
      * ``moe_layer_spread/{tokens_max,tokens_min,tokens_range,tokens_mean}`` -
        one line-chart image per statistic, overlaying every MoE layer's
        per-step value so all layers are comparable in a single chart.
      * ``run_metadata`` - the run manifest rendered as a markdown table. This
        is logged for any network (MoE or dense); MoE-specific charts above are
        only emitted when grouped-GEMM records were collected. The manifest
        includes ``gemm_templates`` (grouped-GEMM shapes, MoE-only) and
        ``dense_gemm_templates`` (dense ``nn.Linear`` shapes) when available.

    Rows are keyed by world rank rather than ``ep_rank`` so multiple EP groups
    (``world_size > ep_size``) keep distinct rows instead of overwriting each
    other.

    Heatmap rendering needs matplotlib; it is imported lazily inside
    :meth:`close` so a missing install degrades to native histograms/scalars/
    text instead of failing the run.
    """

    def __init__(
        self,
        *,
        log_dir: str | None,
        rank: int,
        world_size: int,
        cross_rank_gather: bool,
        run_metadata: dict | None = None,
    ) -> None:
        self._log_dir = log_dir
        self._rank = rank
        self._world_size = world_size
        # Whether every rank built this sink (i.e. ranks == 'all'), so the
        # cross-rank gather has a participant on every rank. When False the sink
        # only exists on rank 0 and must NOT call the collective, or it would
        # hang waiting for ranks that never join.
        self._cross_rank_gather = cross_rank_gather
        self._run_metadata = run_metadata or {}
        # (step, layer_id, local_expert_id) -> tokens (per-expert M), this rank.
        self._loads: dict[tuple[int, int, int], int] = {}
        # This rank's WORLD rank, captured from the first record. We key cube
        # rows by world rank (globally unique) rather than ep_rank, which
        # repeats once per EP group when world_size > ep_size (e.g. TP=4/EP=4
        # on 8 GPUs => 2 EP groups, ep_rank in 0..3 shared by two ranks). Keying
        # by ep_rank would collapse both groups onto the same rows.
        self._world_rank: int | None = None
        self._num_local_experts: int | None = None
        self._gemm_templates: dict | None = None
        self._dense_gemm_templates: dict | None = None

    def set_gemm_templates(self, templates: dict) -> None:
        """Provide the invariant (N, K) GEMM templates for the manifest text."""
        self._gemm_templates = templates

    def set_dense_gemm_templates(self, templates: dict) -> None:
        """Provide the dense (per-``nn.Linear``) (N, K) templates for the text."""
        self._dense_gemm_templates = templates

    def write_record(self, record: GroupedGemmRecord) -> None:
        if self._world_rank is None:
            self._world_rank = record.rank
            self._num_local_experts = record.num_local_experts
        step = record.step
        layer_id = record.layer_id
        for local_idx, m in enumerate(record.tokens_per_local_expert):
            self._loads[(step, layer_id, local_idx)] = m

    def flush(self) -> None:
        return

    def close(self) -> None:
        # Per-rank payload: this rank's loads plus its world rank / local count.
        local_payload = (self._world_rank, self._num_local_experts, self._loads)
        if self._cross_rank_gather and torch.distributed.is_initialized():
            gathered: list | None = (
                [None] * self._world_size if self._rank == 0 else None
            )
            torch.distributed.gather_object(local_payload, gathered, dst=0)
            if self._rank != 0:
                return
            assert gathered is not None
            payloads = [p for p in gathered if p is not None]
        else:
            # Single process, or only rank 0 built this sink (ranks != 'all'):
            # render just the experts this rank owns, with no collective.
            if self._rank != 0:
                return
            payloads = [local_payload]

        cube, steps, layers, num_local_experts = self._assemble_cube(payloads)
        # Note: cube is None for dense / non-MoE networks (no records). We still
        # log the run-metadata text below; only the cube-derived charts are
        # skipped in that case.
        self._log_to_tensorboard(cube, steps, layers, num_local_experts)

    def _assemble_cube(
        self, payloads: list
    ) -> tuple["object | None", list[int], list[int], int]:
        """Build a ``(num_rows, num_layers, num_steps)`` NaN-filled cube.

        Rows are laid out by WORLD rank: ``rank * num_local_experts + local``.
        This keeps every rank's experts on distinct rows even when there are
        multiple EP groups (``world_size > ep_size``), where ``ep_rank`` repeats.
        ``num_rows`` therefore spans every participating rank's local experts;
        missing cells stay NaN so they render as gaps. Returns
        ``(None, [], [], 0)`` when no records were collected (e.g. metrics
        enabled but no MoE layers executed).
        """
        import numpy as np

        steps_set: set[int] = set()
        layers_set: set[int] = set()
        num_local_experts = 0
        max_rank = 0
        for world_rank, n_local, loads in payloads:
            if not loads:
                continue
            num_local_experts = n_local or num_local_experts
            max_rank = max(max_rank, world_rank or 0)
            for (step, layer_id, _local) in loads:
                steps_set.add(step)
                layers_set.add(layer_id)
        if not steps_set or num_local_experts == 0:
            return None, [], [], 0

        steps = sorted(steps_set)
        layers = sorted(layers_set)
        step_idx = {s: i for i, s in enumerate(steps)}
        layer_idx = {layer: i for i, layer in enumerate(layers)}
        num_rows = (max_rank + 1) * num_local_experts
        cube = np.full((num_rows, len(layers), len(steps)), np.nan, dtype=float)
        for world_rank, n_local, loads in payloads:
            base = (world_rank or 0) * num_local_experts
            for (step, layer_id, local_idx), m in loads.items():
                cube[base + local_idx, layer_idx[layer_id], step_idx[step]] = m
        return cube, steps, layers, num_local_experts

    @staticmethod
    def _zscore_per_step(matrix) -> "object":
        import numpy as np

        mean = np.nanmean(matrix, axis=0, keepdims=True)
        std = np.nanstd(matrix, axis=0, keepdims=True)
        std = np.where(std > 0, std, 1.0)
        return (matrix - mean) / std

    def _manifest_markdown(self) -> str:
        manifest = {
            "world_size": self._world_size,
            **self._run_metadata,
        }
        if self._gemm_templates is not None:
            manifest["gemm_templates"] = self._gemm_templates
        if self._dense_gemm_templates is not None:
            manifest["dense_gemm_templates"] = self._dense_gemm_templates
        scalars = [(k, v) for k, v in manifest.items() if not isinstance(v, dict)]
        sections = [(k, v) for k, v in manifest.items() if isinstance(v, dict)]
        lines = ["### Run metadata", "", "| field | value |", "| --- | --- |"]
        lines += [f"| `{k}` | {_fmt_manifest_value(k, v)} |" for k, v in scalars]
        for name, sub in sections:
            lines += ["", f"#### {name}", "", "| field | value |", "| --- | --- |"]
            lines += [
                f"| `{k}` | {_fmt_manifest_value(k, v)} |" for k, v in sub.items()
            ]
        return "\n".join(lines)

    def _log_to_tensorboard(self, cube, steps, layers, num_local_experts) -> None:
        import numpy as np
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=self._log_dir)
        try:
            # Run metadata applies to any network (MoE or dense), so always log
            # it even when no MoE records were collected (cube is None).
            writer.add_text("run_metadata", self._manifest_markdown(), 0)
            if cube is None:
                return
            self._log_layer_slices(writer, cube, steps, layers, num_local_experts, np)
            self._log_expert_slices(writer, cube, steps, layers, np)
            self._log_layer_spread_curves(writer, cube, steps, layers, np)
        finally:
            writer.close()

    def _log_layer_slices(
        self, writer, cube, steps, layers, num_local_experts, np
    ) -> None:
        """One expert x step heatmap (+ histogram/scalars) per MoE layer."""
        for li, layer_id in enumerate(layers):
            matrix = cube[:, li, :]  # (num_experts, num_steps)
            fig = self._render_heatmap(
                matrix,
                steps,
                title=f"Per-expert token load (layer {layer_id})",
                ylabel="rank / local expert",
                row_block=num_local_experts,
                np=np,
            )
            if fig is not None:
                self._add_figure(writer, f"moe_expert_load/layer_{layer_id}", fig)
            tag = f"moe_expert_load/layer_{layer_id}"
            for ci, step in enumerate(steps):
                valid = matrix[:, ci][~np.isnan(matrix[:, ci])]
                if valid.size == 0:
                    continue
                writer.add_histogram(
                    f"{tag}/per_expert_tokens", valid, global_step=step
                )
                # Per-step spread of the per-expert load across experts: the
                # hottest/coldest expert, their gap, and the mean give a quick
                # imbalance read-out per layer over training.
                tokens_max = float(valid.max())
                tokens_min = float(valid.min())
                mean = float(valid.mean())
                writer.add_scalar(f"{tag}/tokens_max", tokens_max, global_step=step)
                writer.add_scalar(f"{tag}/tokens_min", tokens_min, global_step=step)
                writer.add_scalar(
                    f"{tag}/tokens_range", tokens_max - tokens_min, global_step=step
                )
                writer.add_scalar(f"{tag}/tokens_mean", mean, global_step=step)
                if mean > 0:
                    writer.add_scalar(
                        f"{tag}/max_over_mean", tokens_max / mean, global_step=step
                    )

    def _log_expert_slices(self, writer, cube, steps, layers, np) -> None:
        """One layer x step heatmap (+ histogram/scalar) for first/last expert."""
        num_experts = cube.shape[0]
        for expert_id in sorted({0, num_experts - 1}):
            matrix = cube[expert_id]  # (num_layers, num_steps)
            fig = self._render_heatmap(
                matrix,
                steps,
                title=f"Per-layer token load (expert {expert_id})",
                ylabel="MoE layer id",
                row_labels=[f"L{layer}" for layer in layers],
                np=np,
            )
            if fig is not None:
                self._add_figure(writer, f"moe_layer_load/expert_{expert_id}", fig)
            for ci, step in enumerate(steps):
                valid = matrix[:, ci][~np.isnan(matrix[:, ci])]
                if valid.size == 0:
                    continue
                writer.add_histogram(
                    f"moe_layer_load/expert_{expert_id}/per_layer_tokens",
                    valid,
                    global_step=step,
                )
                mean = float(valid.mean())
                if mean > 0:
                    writer.add_scalar(
                        f"moe_layer_load/expert_{expert_id}/max_over_mean",
                        float(valid.max()) / mean,
                        global_step=step,
                    )

    def _log_layer_spread_curves(self, writer, cube, steps, layers, np) -> None:
        """Overlay every layer's per-step load stat in one line chart per stat.

        For each step the per-expert load is reduced across experts (cube rows)
        to its max, min, range (max - min) and mean; each statistic gets a
        single image with one line per MoE layer vs training step, so all layers
        are comparable at a glance. Needs matplotlib; skipped (with a warning) if
        it is unavailable.
        """
        maxes = np.nanmax(cube, axis=0)  # (num_layers, num_steps)
        mins = np.nanmin(cube, axis=0)
        means = np.nanmean(cube, axis=0)
        ranges = maxes - mins
        for tag, values, ylabel in (
            ("tokens_max", maxes, "max tokens over experts"),
            ("tokens_min", mins, "min tokens over experts"),
            ("tokens_range", ranges, "max - min tokens over experts"),
            ("tokens_mean", means, "mean tokens over experts"),
        ):
            fig = self._render_layer_curves(
                values, steps, layers, title=tag, ylabel=ylabel, np=np
            )
            if fig is not None:
                self._add_figure(writer, f"moe_layer_spread/{tag}", fig)

    def _render_layer_curves(
        self, values, steps, layers, *, title, ylabel, np
    ) -> "object | None":
        """Render one line per layer (rows of ``values``) vs step, or None."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.ticker import MaxNLocator
        except ImportError:
            logger.warning(
                "matplotlib not installed; skipping MoE layer-spread curves "
                "(per-layer scalars are still logged)."
            )
            return None

        fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
        cmap = plt.get_cmap("viridis")
        denom = max(len(layers) - 1, 1)
        for li, layer_id in enumerate(layers):
            ax.plot(
                steps,
                values[li],
                color=cmap(li / denom),
                linewidth=1.0,
                label=f"L{layer_id}",
            )
        ax.set_title(f"Per-layer {title} vs step")
        ax.set_xlabel("training step")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
        # One legend entry per layer is a lot; show a colorbar proxy instead.
        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=layers[0], vmax=layers[-1])
        )
        fig.colorbar(sm, ax=ax, label="MoE layer id")
        return fig

    def _render_heatmap(
        self,
        matrix,
        steps,
        *,
        title,
        ylabel,
        np,
        row_block: int | None = None,
        row_labels: list[str] | None = None,
    ) -> "object | None":
        """Render a raw + per-step z-score heatmap pair, or None if no matplotlib."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.ticker import MaxNLocator
        except ImportError:
            logger.warning(
                "matplotlib not installed; skipping MoE heatmap images "
                "(histograms and scalars are still logged)."
            )
            return None

        num_rows = matrix.shape[0]
        extent = (steps[0] - 0.5, steps[-1] + 0.5, -0.5, num_rows - 0.5)
        fig, (ax_load, ax_z) = plt.subplots(
            1, 2, figsize=(16, 8), constrained_layout=True
        )
        im0 = ax_load.imshow(
            matrix,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
            extent=extent,
            origin="lower",
        )
        ax_load.set_title(title)
        ax_load.set_xlabel("training step")
        ax_load.set_ylabel(ylabel)
        fig.colorbar(im0, ax=ax_load, label="tokens routed to expert (M)")

        zscore = self._zscore_per_step(matrix)
        vmax = float(np.nanmax(np.abs(zscore))) if np.isfinite(zscore).any() else 1.0
        im1 = ax_z.imshow(
            zscore,
            aspect="auto",
            cmap="coolwarm",
            interpolation="nearest",
            extent=extent,
            origin="lower",
            vmin=-vmax,
            vmax=vmax,
        )
        ax_z.set_title("Per-step z-score of load")
        ax_z.set_xlabel("training step")
        ax_z.set_ylabel(ylabel)
        fig.colorbar(im1, ax=ax_z, label="(load - step mean) / step std")

        for ax in (ax_load, ax_z):
            # Training steps are integers; the continuous ``extent`` otherwise
            # lets matplotlib pick fractional ticks (e.g. 2.5, 17.5).
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if row_block:
                num_ranks = num_rows // row_block
                ticks = [r * row_block for r in range(num_ranks + 1)]
                ax.set_yticks(ticks)
                ax.set_yticklabels(
                    [f"rank {r}" if r < num_ranks else "" for r in range(len(ticks))]
                )
            elif row_labels:
                stride = max(1, len(row_labels) // 13)
                idx = list(range(0, len(row_labels), stride))
                ax.set_yticks(idx)
                ax.set_yticklabels([row_labels[i] for i in idx])
        return fig

    @staticmethod
    def _add_figure(writer, tag: str, fig) -> None:
        import numpy as np

        fig.canvas.draw()
        image_hwc = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.add_image(tag, image_hwc, global_step=0, dataformats="HWC")
        import matplotlib.pyplot as plt

        plt.close(fig)


class MoEMetricCollector:
    def __init__(
        self,
        *,
        config: MoEMetricsConfig,
        dump_folder: str,
        rank: int,
        world_size: int,
        run_metadata: dict | None = None,
        tb_log_dir: str | None = None,
    ) -> None:
        self._config = config
        self._rank = rank
        self._world_size = world_size
        self._run_metadata = run_metadata or {}
        self._tb_log_dir = tb_log_dir
        self._step: int = 0
        self._records_dropped: int = 0
        self._records_seen: int = 0
        self._records_buffer: list[GroupedGemmRecord] = []
        self._pending_buffer: list[_PendingGroupedGemm] = []
        # Invariant grouped-GEMM templates (N, K dims, dtype, dispatcher),
        # captured once from the first record so they are recorded in the
        # manifest rather than duplicated on every line.
        self._gemm_templates: dict | None = None
        # Dense (per-``nn.Linear``) (N, K) shape templates, captured once from
        # the model module tree (see ``capture_dense_gemm_templates``). These
        # cover the dense GEMMs (attention/FFN projections, router, shared
        # experts, output head) that the grouped-GEMM hook does not see.
        self._dense_gemm_templates: dict | None = None
        # Per-(step, layer) intra-rank accumulators over per-expert M values,
        # populated in flush(). Storage is O(sampled steps * MoE layers).
        self._imbalance: dict[tuple[int, int], _ImbalanceAccum] = {}

        self._enabled = config.enabled and self._is_rank_selected(config.ranks)
        self._output_dir = Path(dump_folder) / config.output_dir
        self._sinks: list[MoEMetricSink] = []
        if self._enabled:
            self._sinks = self._build_sinks()

    @property
    def records_dropped(self) -> int:
        return self._records_dropped

    @property
    def records_seen(self) -> int:
        return self._records_seen

    @property
    def current_step(self) -> int:
        return self._step

    def is_enabled(self) -> bool:
        return self._enabled

    def capture_dense_gemm_templates(self, model_parts: list) -> None:
        """Capture static dense ``nn.Linear`` (N, K) templates from the model.

        Linear weight shapes are fixed by the architecture, so this walks the
        module tree once (outside any forward) and records each linear's
        ``(N=out_features, K=in_features)``. This covers the dense GEMMs that
        the grouped-GEMM hook never sees (attention/FFN projections, router,
        shared experts, output head), so dense models (e.g. llama3) and the
        dense portions of MoE models (e.g. DeepSeek V3 layer 0 / shared
        experts) both surface their shapes in the run manifest.

        Under pipeline parallelism each rank only holds its stage's modules, so
        the rank-0 manifest reflects stage 0's linears; without PP, rank 0 sees
        the full model.
        """
        if not self._enabled:
            return
        compute_dtype = self._run_metadata.get("mixed_precision_param")
        self._dense_gemm_templates = collect_dense_gemm_templates(
            model_parts, compute_dtype=compute_dtype
        )

    def begin_step(self, step: int) -> None:
        self._step = step

    def should_record_current_step(self) -> bool:
        if not self._enabled:
            return False
        sample_every = max(1, self._config.sample_every)
        return self._step > 0 and self._step % sample_every == 0

    def record(self, record: GroupedGemmRecord) -> None:
        if not self.should_record_current_step():
            return
        self._records_seen += 1
        if self._buffered_count() >= self._config.max_records_per_rank:
            self._records_dropped += 1
            return
        self._records_buffer.append(record)

    def record_pending(self, pending: _PendingGroupedGemm) -> None:
        """Buffer a record that still holds its token-count tensors.

        Used by the grouped-GEMM hook so that no dispatched tensor ops run in
        the (possibly activation-checkpointed) forward path. The tensors are
        materialized later in :meth:`flush`.
        """
        if not self.should_record_current_step():
            return
        self._records_seen += 1
        if self._buffered_count() >= self._config.max_records_per_rank:
            self._records_dropped += 1
            return
        self._pending_buffer.append(pending)

    def _buffered_count(self) -> int:
        return len(self._records_buffer) + len(self._pending_buffer)

    def _record_from_pending(
        self,
        pending: _PendingGroupedGemm,
        tokens: tuple[int, ...],
        padded: tuple[int, ...],
    ) -> GroupedGemmRecord:
        return GroupedGemmRecord(
            step=pending.step,
            layer_id=pending.layer_id,
            micro_batch_id=pending.micro_batch_id,
            rank=pending.rank,
            ep_rank=pending.ep_rank,
            ep_size=pending.ep_size,
            num_local_experts=len(tokens),
            top_k=pending.top_k,
            tokens_per_local_expert=tokens,
            padded_tokens_per_local_expert=padded,
            gemm_w1=pending.gemm_w1,
            gemm_w3=pending.gemm_w3,
            gemm_w2=pending.gemm_w2,
            dtype=pending.dtype,
            dispatcher=pending.dispatcher,
        )

    def _materialize_pending(
        self, pendings: list[_PendingGroupedGemm]
    ) -> list[GroupedGemmRecord]:
        """Materialize buffered pending records with a single batched D2H copy.

        Each pending still holds its small 1-D int token-count tensors (length
        ``num_local_experts``) on device. Copying them one at a time would force
        a separate device->host sync per tensor (up to 2 per record); since the
        payload is only a few ints, that cost is sync latency, not bandwidth.
        Concatenate every pending's token (and padded) tensor into one 1-D
        tensor, run a single ``.to("cpu")``, then split the result host-side.
        This turns ~2N syncs per flush into one. Runs in flush() at end-of-step,
        never on the (activation-checkpointed) forward path.
        """
        # Flatten every device tensor into one list, recording per-pending how
        # many token / padded entries it contributes (padded_len is None when
        # the pending has no separate padded tensor).
        flat: list[torch.Tensor] = []
        layout: list[tuple[int, int | None]] = []
        for pending in pendings:
            tokens_t = pending.tokens_tensor.detach()
            flat.append(tokens_t)
            if pending.padded_tensor is None:
                layout.append((tokens_t.shape[0], None))
            else:
                padded_t = pending.padded_tensor.detach()
                flat.append(padded_t)
                layout.append((tokens_t.shape[0], padded_t.shape[0]))

        # Single device->host sync for every pending tensor at once.
        combined = torch.cat(flat).to("cpu").tolist()

        records: list[GroupedGemmRecord] = []
        cursor = 0
        for pending, (tokens_len, padded_len) in zip(pendings, layout):
            tokens = tuple(int(v) for v in combined[cursor : cursor + tokens_len])
            cursor += tokens_len
            if padded_len is None:
                padded = tokens
            else:
                padded = tuple(int(v) for v in combined[cursor : cursor + padded_len])
                cursor += padded_len
            records.append(self._record_from_pending(pending, tokens, padded))
        return records

    def flush(self) -> None:
        if not self._enabled:
            return
        if self._pending_buffer:
            self._records_buffer.extend(self._materialize_pending(self._pending_buffer))
            self._pending_buffer.clear()
        if not self._records_buffer:
            return
        if self._gemm_templates is None:
            self._capture_templates(self._records_buffer[0])
        for record in self._records_buffer:
            self._accumulate_imbalance(record)
            for sink in self._sinks:
                sink.write_record(record)
        for sink in self._sinks:
            sink.flush()
        self._records_buffer.clear()

    def close(self) -> None:
        self.flush()
        self._write_manifest()
        self._write_imbalance_local()
        self._write_imbalance_global()
        # Make the invariant GEMM templates (captured during flush) available to
        # the TB sink's manifest text before it renders.
        if self._gemm_templates is not None:
            for sink in self._sinks:
                if isinstance(sink, TensorBoardMoESink):
                    sink.set_gemm_templates(self._gemm_templates)
        if self._dense_gemm_templates is not None:
            for sink in self._sinks:
                if isinstance(sink, TensorBoardMoESink):
                    sink.set_dense_gemm_templates(self._dense_gemm_templates)
        for sink in self._sinks:
            sink.close()

    def _capture_templates(self, record: GroupedGemmRecord) -> None:
        """Record the invariant (N, K) grouped-GEMM dims once.

        N and K are fixed by the model architecture, so they are stored in the
        manifest rather than duplicated on every record / histogram bin. Only
        the ``M`` dimension (per-expert token count) varies at runtime.

        Each ``gemm_*`` record tuple is ``(M, in_features, out_features)``. In
        the standard GEMM form ``C[M, N] = A[M, K] @ B[K, N]`` the contraction
        dim is ``K = in_features`` (tuple index 1) and the output dim is
        ``N = out_features`` (tuple index 2).
        """
        self._gemm_templates = {
            "w1": {"N": record.gemm_w1[2], "K": record.gemm_w1[1]},
            "w3": {"N": record.gemm_w3[2], "K": record.gemm_w3[1]},
            "w2": {"N": record.gemm_w2[2], "K": record.gemm_w2[1]},
            "dtype": record.dtype,
            "dispatcher": record.dispatcher,
            "num_local_experts": record.num_local_experts,
            "num_experts": record.num_local_experts * record.ep_size,
            "top_k": record.top_k,
        }

    def _write_manifest(self) -> None:
        """Write a single run manifest of invariant run/shape metadata.

        This captures everything that does NOT vary per record: run config
        (seq_len, batch size, parallelism) and the fixed grouped-GEMM (N, K)
        templates. These values are identical on every rank, so only rank 0
        writes the shared ``manifest.json``. Consumers reconstruct every GEMM
        shape from this manifest plus the per-expert ``M`` values in the
        histogram / record sinks.
        """
        if not self._enabled or self._rank != 0:
            return
        manifest = {
            "world_size": self._world_size,
            "records_seen": self._records_seen,
            "records_dropped": self._records_dropped,
            **self._run_metadata,
            "gemm_templates": self._gemm_templates,
            "dense_gemm_templates": self._dense_gemm_templates,
        }
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / "manifest.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=True, indent=2)

    def _accumulate_imbalance(self, record: GroupedGemmRecord) -> None:
        """Fold one record's per-expert M values into the (step, layer) accum.

        Records for the same (step, layer) but different micro-batches are
        merged, so each accumulator describes the full per-expert population a
        rank processed for that layer during that step. Each per-expert
        position maps to the identity pair ``(rank_id, local_ep_id)``.
        """
        key = (record.step, record.layer_id)
        accum = self._imbalance.get(key)
        if accum is None:
            accum = _ImbalanceAccum()
            self._imbalance[key] = accum
        accum.add(record.tokens_per_local_expert, record.rank)

    def _write_imbalance_local(self) -> None:
        """Write this rank's intra-rank per-(step, layer) imbalance stats.

        Columns: step, layer_id, count, min_expert, max_expert, min, max, mean,
        std, cv, load — where the stats range over the rank's local experts,
        ``load`` is the rank's total expert tokens ``L_r`` for that (step,
        layer), and ``min_expert``/``max_expert`` are the identity pairs
        ``rank/local`` of the experts with the fewest/most tokens. Float columns
        are rounded to 2 dp.
        """
        if not self._enabled:
            return
        if not self._imbalance:
            return
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / f"m_imbalance_rank_{self._rank}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "step",
                    "layer_id",
                    "count",
                    "min_expert",
                    "max_expert",
                    "min",
                    "max",
                    "mean",
                    "std",
                    "cv",
                    "load",
                ]
            )
            for step, layer_id in sorted(self._imbalance):
                accum = self._imbalance[(step, layer_id)]
                count, m_min, m_max, mean, std, min_e, max_e = _moments_to_stats(accum)
                cv = std / mean if mean > 0 else 0.0
                writer.writerow(
                    [
                        step,
                        layer_id,
                        count,
                        _fmt_expert(min_e),
                        _fmt_expert(max_e),
                        m_min,
                        m_max,
                        _r2(mean),
                        _r2(std),
                        _r2(cv),
                        accum.m_sum,
                    ]
                )

    def _write_imbalance_global(self) -> None:
        """Gather per-rank loads to rank 0 and write inter-rank imbalance stats.

        Across-rank reduction needs every rank to participate in the collective,
        which is only guaranteed when ``ranks == 'all'``; otherwise it is skipped
        to avoid a deadlock. The gather uses ``torch.distributed.gather_object``
        on the default (WORLD) process group — no extra dependency. Only rank 0
        writes ``m_imbalance_global.csv``.

        Columns: step, layer_id, num_ranks, min_expert, min_expert_m,
        max_expert, max_expert_m, load_min, load_min_rank, load_max,
        load_max_rank, load_mean, load_std, load_cv, max_over_mean — where
        ``min_expert``/``max_expert`` are the identity pairs ``rank/local`` of
        the experts with the fewest/most tokens across all ranks,
        ``min_expert_m``/``max_expert_m`` are those experts' token counts, and
        ``load_min_rank``/``load_max_rank`` are the ranks holding the
        least/most total tokens. Float columns are rounded to 2 dp.
        """
        if not self._config.enabled:
            return

        distributed = torch.distributed.is_initialized() and self._world_size > 1
        if distributed and self._config.ranks != "all":
            if self._rank == 0:
                logger.warning(
                    "Inter-rank MoE imbalance aggregation requires "
                    "metrics.moe.ranks='all'; skipping m_imbalance_global.csv."
                )
            return

        # Per (step, layer) payload from this rank: total load plus the
        # per-expert extrema and the expert identity pairs achieving them.
        local_stats: dict[tuple[int, int], _ImbalancePayload] = {
            key: (
                accum.m_sum,
                accum.m_min,
                accum.min_expert,
                accum.m_max,
                accum.max_expert,
            )
            for key, accum in self._imbalance.items()
        }

        if distributed:
            gathered: list[dict[tuple[int, int], _ImbalancePayload] | None] | None = (
                [None] * self._world_size if self._rank == 0 else None
            )
            torch.distributed.gather_object(local_stats, gathered, dst=0)
            if self._rank != 0:
                return
            assert gathered is not None
            rank_payloads = [rs for rs in gathered if rs]
        else:
            if self._rank != 0:
                return
            rank_payloads = [local_stats]

        # Reduce per (step, layer) across ranks: collect each rank's (rank,
        # load) and track the single coldest/hottest expert globally. The
        # owning rank is read off the expert pair (all experts on a rank share
        # the same rank_id).
        per_key: dict[tuple[int, int], list[tuple[int, int]]] = {}
        global_min: dict[tuple[int, int], tuple[int, _ExpertId]] = {}
        global_max: dict[tuple[int, int], tuple[int, _ExpertId]] = {}
        for rank_stats in rank_payloads:
            for key, (load, m_min, min_e, m_max, max_e) in rank_stats.items():
                per_key.setdefault(key, []).append((min_e[0], load))
                cur_min = global_min.get(key)
                if cur_min is None or m_min < cur_min[0]:
                    global_min[key] = (m_min, min_e)
                cur_max = global_max.get(key)
                if cur_max is None or m_max > cur_max[0]:
                    global_max[key] = (m_max, max_e)

        if not per_key:
            return
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / "m_imbalance_global.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "step",
                    "layer_id",
                    "num_ranks",
                    "min_expert",
                    "min_expert_m",
                    "max_expert",
                    "max_expert_m",
                    "load_min",
                    "load_min_rank",
                    "load_max",
                    "load_max_rank",
                    "load_mean",
                    "load_std",
                    "load_cv",
                    "max_over_mean",
                ]
            )
            for step, layer_id in sorted(per_key):
                n, l_min, min_rank, l_max, max_rank, mean, std, cv, mom = _load_stats(
                    per_key[(step, layer_id)]
                )
                m_min, min_e = global_min[(step, layer_id)]
                m_max, max_e = global_max[(step, layer_id)]
                writer.writerow(
                    [
                        step,
                        layer_id,
                        n,
                        _fmt_expert(min_e),
                        m_min,
                        _fmt_expert(max_e),
                        m_max,
                        l_min,
                        min_rank,
                        l_max,
                        max_rank,
                        _r2(mean),
                        _r2(std),
                        _r2(cv),
                        _r2(mom),
                    ]
                )

    def _is_rank_selected(self, ranks: str) -> bool:
        if ranks == "all":
            return True
        if ranks == "rank0":
            return self._rank == 0

        selected: set[int] = set()
        for token in ranks.split(","):
            stripped = token.strip()
            if not stripped:
                continue
            selected.add(int(stripped))
        return self._rank in selected

    def _build_sinks(self) -> list[MoEMetricSink]:
        output_dir = self._output_dir
        sinks: list[MoEMetricSink] = []
        for sink_name in self._config.sinks:
            if sink_name == "jsonl":
                sinks.append(JsonlSink(output_dir / f"rank_{self._rank}.jsonl"))
            elif sink_name == "histogram":
                sinks.append(
                    HistogramMoESink(output_dir / f"m_histogram_rank_{self._rank}.csv")
                )
            elif sink_name == "tb":
                if self._tb_log_dir is None:
                    logger.warning(
                        "MoE 'tb' sink requested but TensorBoard is not enabled "
                        "(metrics.enable_tensorboard=False); skipping it."
                    )
                    continue
                sinks.append(
                    TensorBoardMoESink(
                        log_dir=self._tb_log_dir,
                        rank=self._rank,
                        world_size=self._world_size,
                        cross_rank_gather=(
                            self._config.ranks == "all" and self._world_size > 1
                        ),
                        run_metadata=self._run_metadata,
                    )
                )
        return sinks


_ACTIVE_COLLECTOR: contextvars.ContextVar[
    MoEMetricCollector | None
] = contextvars.ContextVar("active_moe_metric_collector", default=None)

# Plain module-global mirror of "is any collector installed", read by
# ``maybe_record_grouped_gemm`` as its first-line fast path. Unlike the
# ContextVar, a module global is a compile-time constant Dynamo can specialize
# on, so the hook body is dead-code-eliminated under ``fullgraph=True`` when no
# collector is installed (metrics-off + compile-on, the common case).
_COLLECTOR_INSTALLED: bool = False


def set_active_moe_metric_collector(collector: MoEMetricCollector | None) -> None:
    global _COLLECTOR_INSTALLED
    _ACTIVE_COLLECTOR.set(collector)
    _COLLECTOR_INSTALLED = collector is not None


def get_active_moe_metric_collector() -> MoEMetricCollector | None:
    return _ACTIVE_COLLECTOR.get()


def _normalize_module_name(name: str) -> str:
    """Collapse numeric path segments (layer/expert indices) to ``*``.

    ``layers.0.attention.wq`` and ``layers.31.attention.wq`` both normalize to
    ``layers.*.attention.wq`` so repeated layers collapse into one template.
    Wrapper segments injected by activation checkpointing, FSDP, DDP, and
    ``torch.compile`` are dropped so the names stay architecture-readable.
    """
    wrappers = {
        "_checkpoint_wrapped_module",
        "_fsdp_wrapped_module",
        "_orig_mod",
        "module",
    }
    segs = [
        "*" if seg.isdigit() else seg for seg in name.split(".") if seg not in wrappers
    ]
    return ".".join(segs)


def collect_dense_gemm_templates(
    model_parts: list, compute_dtype: str | None = None
) -> dict:
    """Collect static (N, K) shape templates for every dense ``nn.Linear``.

    Walks the module tree of each model part and records each linear's
    ``(N=out_features, K=in_features)`` plus two dtypes:

    * ``param_dtype`` - the linear's stored ``weight`` dtype. TorchTitan keeps
      master weights in ``float32`` by default, so this is usually ``float32``.
    * ``compute_dtype`` - the dtype the matmul actually runs in. Under mixed
      precision (FSDP ``MixedPrecisionPolicy`` / ``torch.autocast``) the fp32
      weight is cast to ``training.mixed_precision_param`` (``bfloat16`` by
      default) for the all-gather and GEMM, so this matches the grouped-GEMM
      hook's runtime activation dtype. It falls back to ``param_dtype`` when no
      mixed-precision dtype is supplied (e.g. full-precision training).

    Module names are normalized (numeric segments -> ``*``, wrapper segments
    stripped) so per-layer repeats collapse into a single template carrying a
    ``count`` of how many concrete linears matched. Weight shapes are read as
    logical (global) shapes, so the result is identical across ranks regardless
    of TP/FSDP sharding.

    Grouped MoE experts are stored as raw parameters (not ``nn.Linear``), so
    they are skipped here and remain covered by the grouped-GEMM hook.

    Returns a dict keyed by normalized module name; each value is
    ``{"N", "K", "param_dtype", "compute_dtype", "count"}``. A ``#<n>`` suffix
    disambiguates the rare case where one normalized name maps to multiple
    distinct shapes.
    """
    templates: dict[str, dict] = {}
    for part in model_parts:
        for name, module in part.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            weight = getattr(module, "weight", None)
            if weight is None or weight.ndim != 2:
                continue
            out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
            param_dtype = str(weight.dtype).replace("torch.", "")
            comp_dtype = compute_dtype or param_dtype
            norm = _normalize_module_name(name)
            key = norm
            existing = templates.get(key)
            # Disambiguate when the same normalized name maps to different shapes.
            suffix = 1
            while existing is not None and (
                existing["N"] != out_features
                or existing["K"] != in_features
                or existing["param_dtype"] != param_dtype
            ):
                suffix += 1
                key = f"{norm}#{suffix}"
                existing = templates.get(key)
            if existing is None:
                templates[key] = {
                    "N": out_features,
                    "K": in_features,
                    "param_dtype": param_dtype,
                    "compute_dtype": comp_dtype,
                    "count": 1,
                }
            else:
                existing["count"] += 1
    return templates


def maybe_record_grouped_gemm(
    *,
    x_RD: torch.Tensor,
    w1_EFD: torch.Tensor,
    w2_EDF: torch.Tensor,
    w3_EFD: torch.Tensor,
    num_tokens_per_expert_E: torch.Tensor,
    token_dispatcher: object | None = None,
    dispatch_metadata: object | None = None,
    layer_id: int = -1,
    micro_batch_id: int = 0,
    top_k: int = -1,
) -> None:
    """Record one grouped-GEMM observation for the active MoE collector.

    Called from the shared ``GroupedExperts._experts_forward`` path, so it
    covers every model that routes through it (llama4, qwen3, deepseek_v3,
    etc.). It is a no-op unless a collector is active and the current step is
    sampled.

    ``_COLLECTOR_INSTALLED`` is a plain module-global bool checked before any
    other work. ``GroupedExperts._experts_forward`` is compiled with
    ``fullgraph=True`` (see ``apply_compile``), and the ``ContextVar.get`` below
    is an unsupported graph op. When no collector is installed (the default),
    Dynamo specializes on the ``False`` global and compiles away the whole body,
    so metrics-off + compile-on (the common case) has zero graph impact. When a
    collector IS installed, the body is incompatible with ``fullgraph`` compile;
    metrics collection therefore requires ``compile.enable = false`` (enforced
    at setup and documented in the RFC).

    Not supported at this point: gpt_oss. ``GptOssGroupedExperts`` subclasses
    ``nn.Module`` (not ``GroupedExperts``) and implements its own
    ``_experts_forward`` that calls ``torch._grouped_mm`` directly without
    invoking this hook, so gpt_oss grouped-GEMM shapes and per-expert load are
    not captured. Wiring gpt_oss up is left as follow-up work.
    """
    if not _COLLECTOR_INSTALLED:
        return
    collector = get_active_moe_metric_collector()
    if collector is None or not collector.should_record_current_step():
        return

    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    # Prefer the dispatcher's own dispatch-time metadata (actual per-expert
    # token counts and reported dispatcher name) when available, else fall back
    # to the pre-dispatch counts and the dispatcher class name. These reads are
    # pure Python (getattr/isinstance/type name) with no dispatched tensor ops,
    # so they are safe on the checkpointed path (see the IMPORTANT note below).
    metadata_tokens = getattr(dispatch_metadata, "num_tokens_per_local_expert_e", None)
    tokens_tensor = (
        metadata_tokens
        if isinstance(metadata_tokens, torch.Tensor)
        else num_tokens_per_expert_E
    )
    metadata_padded = getattr(
        dispatch_metadata, "padded_num_tokens_per_local_expert_e", None
    )
    padded_tensor = (
        metadata_padded
        if isinstance(metadata_padded, torch.Tensor)
        else num_tokens_per_expert_E
    )
    metadata_dispatcher = getattr(dispatch_metadata, "dispatcher", None)
    dispatcher = (
        metadata_dispatcher
        if isinstance(metadata_dispatcher, str)
        else type(token_dispatcher).__name__.replace("TokenDispatcher", "").lower()
    )
    ep_rank = getattr(token_dispatcher, "ep_rank", 0)
    ep_size = getattr(token_dispatcher, "ep_size", 1)

    # IMPORTANT: do not run any dispatched tensor ops here (no ``.to``,
    # ``.tolist``, ``.detach``, arithmetic, etc.). This hook executes inside the
    # (possibly activation-checkpointed) expert forward, but it is gated on a
    # contextvar that is NOT propagated to autograd's backward/recompute thread.
    # Running dispatched ops here would therefore appear during forward but not
    # during recompute, desyncing non-reentrant checkpoint's positional
    # saved-tensor matching. Reading ``.shape``/``.dtype`` and holding tensor
    # references are pure Python and safe; the actual ``.tolist()`` extraction
    # is deferred to ``MoEMetricCollector.flush`` (outside any checkpoint).
    m_total = int(x_RD.shape[0])
    pending = _PendingGroupedGemm(
        step=collector.current_step,
        layer_id=layer_id,
        micro_batch_id=micro_batch_id,
        rank=rank,
        ep_rank=ep_rank,
        ep_size=ep_size,
        top_k=top_k,
        gemm_w1=(m_total, int(w1_EFD.shape[-1]), int(w1_EFD.shape[-2])),
        gemm_w3=(m_total, int(w3_EFD.shape[-1]), int(w3_EFD.shape[-2])),
        gemm_w2=(m_total, int(w2_EDF.shape[-1]), int(w2_EDF.shape[-2])),
        dtype=str(x_RD.dtype).replace("torch.", ""),
        dispatcher=dispatcher,
        tokens_tensor=tokens_tensor,
        padded_tensor=padded_tensor,
    )
    collector.record_pending(pending)
