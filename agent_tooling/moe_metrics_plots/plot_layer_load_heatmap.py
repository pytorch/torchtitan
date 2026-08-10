#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Plot a per-layer token-load heatmap for a fixed expert (layer id vs. step).

The MoE metrics form a 4D cube indexed by ``(step, layer, row)`` holding the
per-expert grouped-GEMM ``M`` dimension (tokens routed to each expert). The
companion ``plot_expert_load_heatmap.py`` slices the cube along the *layer*
dimension (one layer -> a rank/local-expert x step heatmap). This script slices
along the *row* dimension instead: for a fixed row it builds a
``num_layers x num_steps`` matrix, so you can see how one expert's load moves
across MoE layers as training proceeds.

Rows are laid out by WORLD rank as ``rank * num_local_experts + local_index``,
so every rank's local experts map to a unique row even when there are multiple
EP groups (``world_size > ep_size``), where ``ep_rank`` repeats; only the rank
that owns an expert emits records for it. By default the first (row 0) and last
(row ``num_rows - 1``) rows are plotted.

Two heatmaps are produced per expert:
  * ``load``  - raw token count per (layer, step).
  * ``zscore`` - per-step z-score (load minus the step mean over the layers,
    divided by the step std), highlighting which layers are relatively hot at
    each step independent of the step's total token count.

Usage:
    python plot_layer_load_heatmap.py --metrics-dir <dir> --experts 0 63 \\
        --out expert_layer_load.png
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


def _load_cube(
    metrics_dir: Path,
) -> tuple[np.ndarray, list[int], list[int], int]:
    """Build the full ``(num_rows, num_layers, num_steps)`` token-load cube.

    Rows are laid out by WORLD rank (``rank * num_local_experts + local``) so
    every rank's local experts map to a unique row even with multiple EP groups
    (``world_size > ep_size``). Returns the cube plus the sorted step ids,
    sorted layer ids, and ``num_local_experts``. Missing cells stay NaN so they
    render as gaps. Layer ids are taken from the records themselves (the dense
    layer emits no MoE records, so the layer axis only spans MoE layers).
    """
    # (step, layer, row) -> tokens; row = world_rank * num_local_experts + local
    data: dict[tuple[int, int, int], int] = {}
    steps_set: set[int] = set()
    layers_set: set[int] = set()
    num_rows = 0
    num_local_experts = 0
    for path in sorted(metrics_dir.glob("rank_*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                n_local = rec["num_local_experts"]
                num_local_experts = n_local
                world_rank = rec["rank"]
                num_rows = max(num_rows, (world_rank + 1) * n_local)
                base = world_rank * n_local
                step = rec["step"]
                layer = rec["layer_id"]
                steps_set.add(step)
                layers_set.add(layer)
                for local_idx, m in enumerate(rec["tokens_per_local_expert"]):
                    data[(step, layer, base + local_idx)] = m

    if not data:
        raise SystemExit(f"No MoE records found in {metrics_dir}")

    steps = sorted(steps_set)
    layers = sorted(layers_set)
    step_idx = {s: i for i, s in enumerate(steps)}
    layer_idx = {layer: i for i, layer in enumerate(layers)}
    cube = np.full((num_rows, len(layers), len(steps)), np.nan, dtype=float)
    for (step, layer, row), m in data.items():
        cube[row, layer_idx[layer], step_idx[step]] = m
    return cube, steps, layers, num_local_experts


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
    matrices: dict[int, np.ndarray],
    steps: list[int],
) -> None:
    """Log the per-expert layer-load figure and stats to TensorBoard.

    The rendered matplotlib figure is logged as an image (Images tab). For each
    plotted expert, every step's per-layer load distribution is logged via
    ``add_histogram`` (Histograms/Distributions tabs) and the per-step max/mean
    load ratio across layers as a scalar (Scalars tab). Pointing ``logdir`` at
    TorchTitan's ``tb/<timestamp>`` folder merges these into the same run as the
    training scalars.
    """
    from torch.utils.tensorboard import SummaryWriter

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    image_hwc = rgba[..., :3]  # drop alpha -> H x W x 3

    writer = SummaryWriter(log_dir=str(logdir))
    try:
        writer.add_image(
            "moe_layer_load/expert_slice",
            image_hwc,
            global_step=0,
            dataformats="HWC",
        )
        for expert_id, matrix in matrices.items():
            for col, step in enumerate(steps):
                column = matrix[:, col]
                valid = column[~np.isnan(column)]
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
        "--experts",
        type=int,
        nargs="+",
        default=None,
        help="Global expert ids to plot (default: first and last expert).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("expert_layer_load.png"),
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
            "If set, also log the figure (and per-step per-layer load stats) to "
            "this TensorBoard log directory."
        ),
    )
    args = parser.parse_args()

    manifest = _load_manifest(args.metrics_dir)
    cube, steps, layers, num_local_experts = _load_cube(args.metrics_dir)
    num_experts = cube.shape[0]

    if args.experts is not None:
        experts = args.experts
    else:
        experts = [0, num_experts - 1]
    for expert_id in experts:
        if not 0 <= expert_id < num_experts:
            raise SystemExit(
                f"Expert id {expert_id} out of range [0, {num_experts - 1}]"
            )

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

    # One row per requested expert; columns = raw load and per-step z-score.
    n_rows = len(experts)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(16, 4.5 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(suptitle, fontsize=15, fontweight="bold")
    # origin="lower" puts the first layer at the bottom and increases upward.
    extent = (steps[0] - 0.5, steps[-1] + 0.5, -0.5, len(layers) - 0.5)

    # Tick the layer axis with the actual MoE layer ids (thinned if dense).
    tick_stride = max(1, len(layers) // 13)
    layer_ticks = list(range(0, len(layers), tick_stride))

    def _style_layer_axis(ax: "plt.Axes") -> None:
        # Training steps are integers; avoid fractional x-ticks (e.g. 17.5).
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_yticks(layer_ticks)
        ax.set_yticklabels([f"L{layers[i]}" for i in layer_ticks])

    matrices: dict[int, np.ndarray] = {}
    for row, expert_id in enumerate(experts):
        matrix = cube[expert_id]
        matrices[expert_id] = matrix
        zscore = _zscore_per_step(matrix)
        rank = expert_id // num_local_experts
        local = expert_id % num_local_experts
        expert_label = f"expert {expert_id} (rank {rank}/local {local})"

        ax_load = axes[row][0]
        im0 = ax_load.imshow(
            matrix,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
            extent=extent,
            origin="lower",
        )
        ax_load.set_title(f"Per-layer token load - {expert_label}")
        ax_load.set_xlabel("training step")
        ax_load.set_ylabel("MoE layer id")
        _style_layer_axis(ax_load)
        fig.colorbar(im0, ax=ax_load, label="tokens routed to expert (M)")

        ax_z = axes[row][1]
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
        ax_z.set_title(f"Per-step z-score of load - {expert_label}")
        ax_z.set_xlabel("training step")
        ax_z.set_ylabel("MoE layer id")
        _style_layer_axis(ax_z)
        fig.colorbar(im1, ax=ax_z, label="(load - step mean) / step std")

    fig.savefig(args.out, dpi=150)
    print(
        f"Wrote {args.out}  (experts={experts}, layers={len(layers)}, "
        f"steps={len(steps)})"
    )

    if args.tensorboard is not None:
        _log_to_tensorboard(args.tensorboard, fig, matrices, steps)
        _log_manifest_to_tensorboard(args.tensorboard, manifest)
        print(f"Logged expert-slice heatmap to TensorBoard at {args.tensorboard}")

    # Quick sanity check: hottest (layer, step) cell per plotted expert.
    for expert_id in experts:
        matrix = matrices[expert_id]
        flat_max = np.unravel_index(np.nanargmax(matrix), matrix.shape)
        print(
            f"expert {expert_id}: hottest layer {layers[flat_max[0]]} @ "
            f"step {steps[flat_max[1]]} = {matrix[flat_max]:.0f} tokens"
        )


if __name__ == "__main__":
    main()
