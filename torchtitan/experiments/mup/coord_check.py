# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from xx.ml_tools.constants.model import (
    frame_constants_from_fps,
    FRAME_TYPE,
    INPUT_FRAMES_NAMES,
    ModelInputs,
    N_FRAMES,
    TEMPORAL_INPUTS,
)

from .spec import MODES, SPECS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS, STEPS, BS = 3, 10, 4
TEMPORAL_KEYS = (ModelInputs.DESIRE, ModelInputs.TRAFFIC, ModelInputs.ACTION_T)


def _registry(spec):
    return importlib.import_module(
        f"torchtitan.experiments.{spec.module}.config_registry"
    )


def _config(spec, mode: str, width: int):
    return getattr(_registry(spec), spec.config_name(mode, width))()


def _scaled_pattern(spec) -> re.Pattern:
    width = next(w for w in spec.widths if w != spec.base_width)
    groups = _config(spec, "mup", width).optimizer.param_groups
    scaled = [g for g in groups if g.lr_mult not in (None, 1.0)]
    return re.compile(scaled[0].pattern)


def _blocks(model: nn.Module, pattern: re.Pattern):
    scaled = [n for n, _ in model.named_parameters() if pattern.search(n)]

    def nested(prefix: str) -> bool:
        tail = len(prefix) + 1
        return any(p.startswith(prefix + ".") and "." in p[tail:] for p in scaled)

    containers = [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, (nn.ModuleList, nn.Sequential))
        and any(nested(f"{name}.{i}") for i in range(len(mod)))
    ]
    blocks = []
    for name, mod in containers:
        if any(o != name and o.startswith(name + ".") for o, _ in containers):
            continue
        blocks += [
            (f"{name}.{i}", child)
            for i, child in enumerate(mod)
            if nested(f"{name}.{i}")
        ]
    return blocks


def _optimizer(cfg, model: nn.Module) -> torch.optim.Optimizer:
    base_lr = cfg.optimizer.lr
    groups = [
        (re.compile(g.pattern), base_lr * (1.0 if g.lr_mult is None else g.lr_mult))
        for g in cfg.optimizer.param_groups
    ]

    def lr_of(name: str) -> float:
        return next((lr for rx, lr in groups if rx.search(name)), base_lr)

    params = [{"params": [p], "lr": lr_of(n)} for n, p in model.named_parameters()]
    return torch.optim.Adam(params, betas=(0.9, 0.95))


def _inputs(bs: int) -> dict:
    frame = frame_constants_from_fps(n_frames=N_FRAMES, frame_type=FRAME_TYPE)
    shapes = frame["frame_shapes"]
    t_img = len(frame["history_idxs"])
    temporal_len = frame["temporal_len"]
    inputs = {
        name: torch.randn(bs, t_img, *shapes[name], device=DEVICE)
        for name in INPUT_FRAMES_NAMES
    }
    for key in TEMPORAL_KEYS:
        inputs[key] = torch.randn(
            bs, temporal_len, TEMPORAL_INPUTS[key][0], device=DEVICE
        )
    return inputs


def _model(cfg, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    model = cfg.model_spec.model.build()
    model.to_empty(device=DEVICE)
    with torch.no_grad():
        model.init_weights(buffer_device=torch.device(DEVICE))
    return model


def _record(store: list, j: int):
    def hook(_mod, _inp, out):
        store[j] = out.detach().abs().mean().item()

    return hook


def _spectral(m: torch.Tensor) -> float:
    return torch.linalg.matrix_norm(m.reshape(m.shape[0], -1).float(), ord=2).item()


def _wkey(name: str) -> str:
    return re.sub(r"\.\d+\.", ".", name).removesuffix(".weight")


def _by_key(dicts: list) -> dict:
    keys = {}
    for d in dicts:
        for n, v in d.items():
            keys.setdefault(_wkey(n), []).append(v)
    return {k: float(np.mean(v)) for k, v in keys.items()}


def coord_check(spec, mode: str, pattern: re.Pattern, widths, seeds=SEEDS, steps=STEPS):
    init, trained, wnorm, dwnorm = {}, {}, {}, {}
    for width in widths:
        cfg = _config(spec, mode, width)
        at_init, at_end, w_seed, dw_seed = [], [], [], []
        for seed in range(seeds):
            model = _model(cfg, seed)
            blocks = _blocks(model, pattern)
            scaled = [
                (n, p)
                for n, p in model.named_parameters()
                if pattern.search(n) and p.ndim >= 2
            ]
            w0 = {n: p.detach().clone() for n, p in scaled}
            acts = [0.0] * (len(blocks) + 1)
            hooks = [
                b.register_forward_hook(_record(acts, j))
                for j, (_, b) in enumerate(blocks)
            ]
            opt = _optimizer(cfg, model)
            x = _inputs(BS)
            targets, snapshot = None, None
            for step in range(steps):
                out = model(x)
                if targets is None:
                    targets = {k: torch.randn_like(v) for k, v in out.items()}
                acts[-1] = float(
                    np.mean([v.detach().abs().mean().item() for v in out.values()])
                )
                if snapshot is None:
                    snapshot = list(acts)
                loss = sum(
                    F.mse_loss(v.float(), targets[k].float()) for k, v in out.items()
                )
                loss.backward()
                opt.step()
                opt.zero_grad()
            at_init.append(snapshot)
            at_end.append(list(acts))
            w_seed.append({n: _spectral(p.detach()) for n, p in scaled})
            dw_seed.append({n: _spectral(p.detach() - w0[n]) for n, p in scaled})
            for hook in hooks:
                hook.remove()
        init[width] = np.mean(at_init, axis=0).tolist()
        trained[width] = np.mean(at_end, axis=0).tolist()
        wnorm[width] = _by_key(w_seed)
        dwnorm[width] = _by_key(dw_seed)
    return init, trained, wnorm, dwnorm


def _layers(res: dict):
    n = len(next(iter(res.values()))) - 1
    return [f"block{i}" for i in range(n)] + ["output"]


def show(name: str, res: dict, widths):
    layers = _layers(res)
    print(f"\n{name}: mean activation size by width (flat = stable)", flush=True)
    print("layer   " + "".join(f"{wd:>10}" for wd in widths))
    for i, layer in enumerate(layers):
        row = [res[wd][i] for wd in widths]
        print(
            f"{layer:8}"
            + "".join(f"{v:>10.3f}" for v in row)
            + f"   x{row[-1] / row[0]:.1f}"
        )


def show_spec(name: str, res: dict, widths):
    keys = sorted(next(iter(res.values())))
    print(f"\n{name}: spectral norm by width (flat = maximal, decay = lazy)", flush=True)
    print("layer".ljust(24) + "".join(f"{wd:>10}" for wd in widths))
    for k in keys:
        row = [res[wd][k] for wd in widths]
        print(
            f"{k:24}"
            + "".join(f"{v:>10.3f}" for v in row)
            + f"   x{row[-1] / row[0]:.2f}"
        )


def write_report(spec, results: dict, widths):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from xx.release_tests.lib.base_report import (
        BaseReport,
        BaseReportConfig,
        ReportFormat,
    )
    from xx.release_tests.lib.utils import MarkdownWriter

    layers = _layers(results[MODES[0]][0])
    report_dir = spec.report_dir
    report_url = spec.report_url.replace("mutransfer.html", "coord_check.html")

    def fig(which: int, label: str):
        f = make_subplots(rows=1, cols=2, subplot_titles=list(MODES), shared_yaxes=True)
        for col, mode in enumerate(MODES, 1):
            acts = results[mode][which]
            for li, layer in enumerate(layers):
                f.add_trace(
                    go.Scatter(
                        x=widths,
                        y=[acts[wd][li] for wd in widths],
                        name=layer,
                        mode="lines+markers",
                        legendgroup=layer,
                        showlegend=(col == 1),
                    ),
                    row=1,
                    col=col,
                )
        f.update_xaxes(type="log", title_text="width")
        f.update_yaxes(type="log", title_text="mean abs activation")
        f.update_layout(
            title_text=f"{spec.name} coord check {label}: standard fans out with width, muP stays flat",
            width=1100,
            height=460,
        )
        return f

    def sfig(idx: int, label: str, ytitle: str):
        f = make_subplots(rows=1, cols=2, subplot_titles=list(MODES), shared_yaxes=True)
        for col, mode in enumerate(MODES, 1):
            data = results[mode][idx]
            for k in sorted(next(iter(data.values()))):
                f.add_trace(
                    go.Scatter(
                        x=widths,
                        y=[data[wd][k] for wd in widths],
                        name=k,
                        mode="lines+markers",
                        legendgroup=k,
                        showlegend=(col == 1),
                    ),
                    row=1,
                    col=col,
                )
        f.update_xaxes(type="log", title_text="width")
        f.update_yaxes(type="log", title_text=ytitle)
        f.update_layout(
            title_text=f"{spec.name} spectral coord check {label}: muP flat = maximal update, decay = lazy layer",
            width=1100,
            height=460,
        )
        return f

    class Report(BaseReport):
        def run_data(self):
            return None

        def make_report(self, _):
            mw = MarkdownWriter(report_name=f"{spec.name}_coord_check")
            mw.filename = f"{report_dir}/coord_check.html"
            mw.print(f"{spec.name} muP coord check (torchtitan stack)", heading=1)
            mw.print(
                f"width-scaled blocks of the {spec.name} model over widths {list(widths)}, "
                f"{SEEDS} seeds, {STEPS} steps, random input. standard param vs muP "
                f"(eta/m lr on hidden matmuls, 1/m readout). hidden flat = wired right; "
                f"muP output slopes ~1/sqrt(m) at init, flat trained."
            )
            mw.add_plot(fig(0, "at init"), xscale=None, yscale=None)
            mw.add_plot(fig(1, f"after {STEPS} steps"), xscale=None, yscale=None)
            mw.print(
                "spectral coord check on the width-scaled weights: activation flatness "
                "can pass while a hidden layer is frozen (its init carries the norm). "
                "|dW| flat across width = every layer maximally updating; |dW| decaying "
                "with width = a lazy layer the activation check misses."
            )
            mw.add_plot(sfig(2, "weights", "spectral norm |W|"), xscale=None, yscale=None)
            mw.add_plot(sfig(3, "updates", "spectral norm |dW|"), xscale=None, yscale=None)
            return mw

    os.makedirs(report_dir, exist_ok=True)
    Report(
        BaseReportConfig(
            report_name=f"{spec.name}_coord_check",
            output_dir=report_dir,
            read_only=True,
            format=ReportFormat.FILE,
        )
    ).run_report()
    print(f"\nwrote report -> {report_url}", flush=True)


def run(spec):
    pattern = _scaled_pattern(spec)
    results = {}
    for mode in MODES:
        init, trained, wnorm, dwnorm = coord_check(spec, mode, pattern, spec.widths)
        results[mode] = (init, trained, wnorm, dwnorm)
        show(f"{mode} at init", init, spec.widths)
        show(f"{mode} after {STEPS} steps", trained, spec.widths)
        show_spec(f"{mode} weight |W|", wnorm, spec.widths)
        show_spec(f"{mode} update |dW|", dwnorm, spec.widths)
    write_report(spec, results, spec.widths)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in SPECS:
        sys.exit(
            "usage: python -m torchtitan.experiments.mup.coord_check <model>; "
            f"known: {', '.join(SPECS)}"
        )
    run(SPECS[sys.argv[1]])


if __name__ == "__main__":
    main()
