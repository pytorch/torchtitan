# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from dataclasses import replace

from xx.common.basedir import XX_BASEDIR

import torch

from .routine import render_report
from .spec import TRAIN_DATASET_LABEL

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VALSET_DIR = os.path.join(XX_BASEDIR, "projects/prune_10m")
ATOMIC_VALSETS = {
    "day_straight": "val_day_straight.txt",
    "night_straight": "val_night_straight.txt",
    "left_lane_change": "val_left_lane_change.txt",
    "right_lane_change": "val_right_lane_change.txt",
}


def valset_path(name: str) -> str:
    return os.path.join(VALSET_DIR, ATOMIC_VALSETS[name])


def valset_len(name: str) -> int:
    with open(valset_path(name)) as fh:
        return sum(1 for line in fh if line.strip())


def valset_dataloader_config(name: str):
    from ..path.config_registry import _vit_dataloader_config

    return replace(_vit_dataloader_config(split="val"), dataset=valset_path(name))


@torch.no_grad()
def accumulate(model, loss_fn, batches, device, max_steps=-1):
    total_loss = torch.zeros((), device=device)
    total_samples = torch.zeros((), device=device)
    metric_sums: dict[str, torch.Tensor] = {}
    for step, (inputs, targets) in enumerate(batches):
        if max_steps != -1 and step >= max_steps:
            break
        inputs = {k: v.to(device) for k, v in inputs.items()}
        targets = {k: v.to(device) for k, v in targets.items()}
        loss_vec, metrics = loss_fn(model(inputs), targets)
        total_loss += loss_vec.float().sum()
        total_samples += next(iter(inputs.values())).shape[0]
        for k, v in metrics.items():
            if k == "loss":
                continue
            metric_sums[k] = (
                metric_sums.get(k, torch.zeros((), device=device)) + v.float().sum()
            )
    samples = total_samples.clamp(min=1.0)
    out = {"loss": float((total_loss / samples).item())}
    for k, v in metric_sums.items():
        out[k] = float((v / samples).item())
    out["n_samples"] = int(total_samples.item())
    return out


def evaluate_valset(model, loss_fn, name, *, steps=None, local_batch_size=16):
    if steps is None:
        steps = (valset_len(name) + local_batch_size - 1) // local_batch_size
    loader = valset_dataloader_config(name).build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=None,
        seq_len=1,
        local_batch_size=local_batch_size,
        validation_steps=steps,
    )
    try:
        return accumulate(model, loss_fn, loader, DEVICE, steps)
    finally:
        loader.close()


def load_model(flavor, checkpoint_dir, *, mup):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        set_model_state_dict,
    )

    from ..path.config_registry import _vit_model_config

    model = _vit_model_config(flavor, mup=mup).build()
    model.to_empty(device=DEVICE)
    state = get_model_state_dict(model)
    dcp.load(state, checkpoint_id=checkpoint_dir)
    set_model_state_dict(model, state)
    return model.eval()


def load_loss():
    from ..path.vit import PlanViTLoss

    return PlanViTLoss(PlanViTLoss.Config()).to(torch.device(DEVICE))


def metric_table(results):
    keys = sorted({k for per in results.values() for m in per.values() for k in m})
    cols = ["run", "valset", *keys]
    rows = []
    for label, per in results.items():
        for name in ATOMIC_VALSETS:
            if name not in per:
                continue
            m = per[name]
            row = [label, name]
            for k in keys:
                v = m.get(k)
                row.append(
                    "-" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v)
                )
            rows.append(row)
    return cols, rows


def _metric_figs(results):
    import plotly.graph_objects as go

    keys = sorted(
        {
            k
            for per in results.values()
            for m in per.values()
            for k in m
            if k != "n_samples"
        }
    )
    figs = []
    for key in keys:
        fig = go.Figure()
        for label, per in results.items():
            names = [n for n in ATOMIC_VALSETS if n in per and key in per[n]]
            fig.add_trace(go.Bar(x=names, y=[per[n][key] for n in names], name=label))
        fig.update_layout(barmode="group", title_text=key, width=1000, height=380)
        figs.append(fig)
    return figs


def build_report(results, *, report_dir, report_name="valset_metrics", read_only=False):
    from xx.release_tests.lib.utils import MarkdownWriter

    cols, rows = metric_table(results)
    figs = _metric_figs(results)

    def make():
        mw = MarkdownWriter(report_name=report_name)
        mw.filename = f"{report_dir}/{report_name}.html"
        mw.print("prune10m atomic valset metrics", heading=1)
        mw.print(
            "per-run validation loss and driving metrics over the atomic valsets "
            "(day / night straight, left / right lane change), each scored on its "
            f"own list; runs to date were trained on the {TRAIN_DATASET_LABEL} "
            "(a checkpoint's own training dataset is named in its run label)."
        )
        for fig in figs:
            mw.add_plot(fig, xscale=None, yscale=None)
        with mw.html_table(cols, borders=True) as t:
            for row in rows:
                t.row(row)
        return mw

    render_report(report_name, report_dir, make, read_only=read_only)
    return f"{report_dir}/{report_name}.html"
