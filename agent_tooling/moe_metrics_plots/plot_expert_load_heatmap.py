#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Plot a per-expert token-load heatmap (rank/local expert vs. training step).

Reads the per-rank ``rank_*.jsonl`` records emitted by the MoE metrics
``jsonl`` sink and, for one MoE layer, builds a ``num_rows x num_steps``
matrix of the per-expert grouped-GEMM ``M`` dimension (tokens routed to each
expert). Rows are laid out by WORLD rank as ``rank * num_local_experts +
local_index``, so every rank's local experts map to a unique row even when
there are multiple EP groups (``world_size > ep_size``), where ``ep_rank``
repeats.

Two heatmaps are produced from the same matrix:
  * ``load``  - raw token count per (expert, step); the coldest/hottest cells
    are the per-step min/max experts.
  * ``zscore`` - per-step z-score (load minus the step mean, over the step std),
    which highlights *relative* imbalance independent of the step's total token
    count.

Usage:
    python plot_expert_load_heatmap.py --metrics-dir <dir> --layer 1 \\
        --out expert_load_layer1.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def _load_layer_matrix(
    metrics_dir: Path, layer_id: int
) -> tuple[np.ndarray, list[int], int]:
    """Build a (num_rows, num_steps) matrix of per-expert token counts.

    Rows are laid out by WORLD rank (``rank * num_local_experts + local``) so
    every rank's local experts map to a unique row even when there are multiple
    EP groups (``world_size > ep_size``), where ``ep_rank`` repeats. Returns the
    matrix, the sorted list of step indices its columns map to, and
    ``num_local_experts`` (the number of experts per rank, i.e. the rank
    boundary spacing on the row axis). Missing (row, step) cells stay NaN so
    they render as gaps.
    """
    # (step -> {row -> tokens}); row = world_rank * num_local_experts + local
    by_step: dict[int, dict[int, int]] = {}
    num_rows = 0
    num_local_experts = 0
    for path in sorted(metrics_dir.glob("rank_*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec["layer_id"] != layer_id:
                    continue
                world_rank = rec["rank"]
                n_local = rec["num_local_experts"]
                num_local_experts = n_local
                num_rows = max(num_rows, (world_rank + 1) * n_local)
                base = world_rank * n_local
                step = rec["step"]
                slot = by_step.setdefault(step, {})
                for local_idx, m in enumerate(rec["tokens_per_local_expert"]):
                    slot[base + local_idx] = m

    if not by_step:
        raise SystemExit(f"No records found for layer {layer_id} in {metrics_dir}")

    steps = sorted(by_step)
    matrix = np.full((num_rows, len(steps)), np.nan, dtype=float)
    for col, step in enumerate(steps):
        for row, m in by_step[step].items():
            matrix[row, col] = m
    return matrix, steps, num_local_experts


def _zscore_per_step(matrix: np.ndarray) -> np.ndarray:
    """Column-wise z-score so each step is centered on its own mean/std."""
    mean = np.nanmean(matrix, axis=0, keepdims=True)
    std = np.nanstd(matrix, axis=0, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    return (matrix - mean) / std


def _load_manifest(metrics_dir: Path) -> dict:
    """Load the run manifest if present, else return an empty dict."""
    path = metrics_dir / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_to_markdown(manifest: dict) -> str:
    """Render the run manifest as markdown for TensorBoard's Text tab.

    Scalar fields go in a top-level table; nested dicts (e.g. ``attention`` and
    ``gemm_templates``) each get their own sub-table.
    """
    if not manifest:
        return "_(no manifest.json found)_"
    scalars = [(k, v) for k, v in manifest.items() if not isinstance(v, dict)]
    sections = [(k, v) for k, v in manifest.items() if isinstance(v, dict)]
    lines = ["### Run metadata", "", "| field | value |", "| --- | --- |"]
    lines += [f"| `{k}` | {v} |" for k, v in scalars]
    for name, sub in sections:
        lines += ["", f"#### {name}", "", "| field | value |", "| --- | --- |"]
        lines += [f"| `{k}` | {v} |" for k, v in sub.items()]
    return "\n".join(lines)


def _log_manifest_to_tensorboard(logdir: Path, manifest: dict) -> None:
    """Log the run manifest to TensorBoard's Text tab under ``run_metadata``."""
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=str(logdir))
    try:
        writer.add_text("run_metadata", _manifest_to_markdown(manifest), global_step=0)
    finally:
        writer.close()


def _log_to_tensorboard(
    logdir: Path,
    fig: "plt.Figure",
    matrix: np.ndarray,
    steps: list[int],
    layer_id: int,
    tag: str,
) -> None:
    """Log the heatmap figure and per-step imbalance stats to TensorBoard.

    TensorBoard has no native 2D-heatmap widget, so the rendered matplotlib
    figure is logged as an image (visible under the Images tab). In addition,
    each step's per-expert load distribution is logged via ``add_histogram``
    (Histograms/Distributions tabs) and the per-step max/mean load ratio as a
    scalar (Scalars tab), so the imbalance trend over training steps is visible
    natively. Pointing ``logdir`` at TorchTitan's ``tb/<timestamp>`` folder
    merges these into the same TensorBoard run as the training scalars.
    """
    from torch.utils.tensorboard import SummaryWriter

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    image_hwc = rgba[..., :3]  # drop alpha -> H x W x 3

    writer = SummaryWriter(log_dir=str(logdir))
    try:
        writer.add_image(
            f"moe_expert_load/layer_{layer_id}",
            image_hwc,
            global_step=layer_id,
            dataformats="HWC",
        )
        # Per-step imbalance factor: hottest expert / mean expert load, plus the
        # full per-expert load distribution as a histogram (TB Histograms tab).
        for col, step in enumerate(steps):
            column = matrix[:, col]
            valid = column[~np.isnan(column)]
            if valid.size == 0:
                continue
            writer.add_histogram(
                f"moe_expert_load/layer_{layer_id}/per_expert_tokens",
                valid,
                global_step=step,
            )
            mean = float(valid.mean())
            if mean > 0:
                writer.add_scalar(
                    f"moe_expert_load/layer_{layer_id}/max_over_mean",
                    float(valid.max()) / mean,
                    global_step=step,
                )
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-dir",
        required=True,
        type=Path,
        help="Directory containing rank_*.jsonl record files.",
    )
    parser.add_argument(
        "--layer", type=int, default=1, help="MoE layer id to plot (default: 1)."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("expert_load_heatmap.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes (N) for the run, used in the title (default: 1).",
    )
    parser.add_argument(
        "--ppn",
        type=int,
        default=None,
        help="Processes per node (PPN); defaults to world_size // nodes.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Model label for the title; defaults to model_name_flavor.",
    )
    parser.add_argument(
        "--tensorboard",
        type=Path,
        default=None,
        help=(
            "If set, also log the heatmap figure (and per-step load scalars) to "
            "this TensorBoard log directory."
        ),
    )
    args = parser.parse_args()

    manifest = _load_manifest(args.metrics_dir)
    matrix, steps, num_local_experts = _load_layer_matrix(args.metrics_dir, args.layer)
    num_experts = matrix.shape[0]
    zscore = _zscore_per_step(matrix)

    # Build the run-config suptitle from the manifest (with CLI overrides for
    # node topology, which the manifest does not record).
    model = manifest.get("model_name", "model")
    flavor = manifest.get("model_flavor", "")
    label = args.title if args.title is not None else f"{model}_{flavor}"
    world_size = manifest.get("world_size", num_experts)
    ppn = args.ppn if args.ppn is not None else world_size // max(args.nodes, 1)
    tp = manifest.get("tp", "?")
    ep = manifest.get("ep", "?")
    suptitle = f"{label}  |  N={args.nodes}, PPN={ppn}, TP={tp}, EP={ep}"

    fig, (ax_load, ax_z) = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    fig.suptitle(suptitle, fontsize=15, fontweight="bold")
    # origin="lower" puts expert 0 at the bottom and increases upward.
    extent = (steps[0] - 0.5, steps[-1] + 0.5, -0.5, num_experts - 0.5)

    # Tick the expert axis at per-rank boundaries (every num_local_experts),
    # labeled by world rank, so each band of rows is one rank's local experts.
    num_ranks = num_experts // num_local_experts if num_local_experts else 0
    rank_boundaries = [r * num_local_experts for r in range(num_ranks + 1)]

    def _style_expert_axis(ax: "plt.Axes") -> None:
        # Training steps are integers; avoid fractional x-ticks (e.g. 17.5).
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_yticks(rank_boundaries)
        ax.set_yticklabels(
            [f"rank {r}" if r < num_ranks else "" for r in range(num_ranks + 1)]
        )
        # Faint lines delimiting each rank's expert band.
        for boundary in rank_boundaries[1:-1]:
            ax.axhline(boundary - 0.5, color="white", linewidth=0.5, alpha=0.4)

    im0 = ax_load.imshow(
        matrix,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
        extent=extent,
        origin="lower",
    )
    ax_load.set_title(f"Per-expert token load (layer {args.layer})")
    ax_load.set_xlabel("training step")
    ax_load.set_ylabel("rank / local expert")
    _style_expert_axis(ax_load)
    fig.colorbar(im0, ax=ax_load, label="tokens routed to expert (M)")

    vmax = float(np.nanmax(np.abs(zscore)))
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
    ax_z.set_title(f"Per-step z-score of load (layer {args.layer})")
    ax_z.set_xlabel("training step")
    ax_z.set_ylabel("rank / local expert")
    _style_expert_axis(ax_z)
    fig.colorbar(im1, ax=ax_z, label="(load - step mean) / step std")

    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}  (experts={num_experts}, steps={len(steps)})")

    if args.tensorboard is not None:
        _log_to_tensorboard(args.tensorboard, fig, matrix, steps, args.layer, suptitle)
        _log_manifest_to_tensorboard(args.tensorboard, manifest)
        print(f"Logged heatmap to TensorBoard at {args.tensorboard}")

    # Report the globally coldest/hottest expert-step cells for a quick sanity
    # check against m_imbalance_global.csv.
    flat_min = np.unravel_index(np.nanargmin(matrix), matrix.shape)
    flat_max = np.unravel_index(np.nanargmax(matrix), matrix.shape)
    print(
        f"coldest cell: expert {flat_min[0]} @ step {steps[flat_min[1]]} "
        f"= {matrix[flat_min]:.0f} tokens"
    )
    print(
        f"hottest cell: expert {flat_max[0]} @ step {steps[flat_max[1]]} "
        f"= {matrix[flat_max]:.0f} tokens"
    )


if __name__ == "__main__":
    main()
