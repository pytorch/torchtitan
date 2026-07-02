# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from itertools import count
from urllib.request import urlopen

from torchtitan.tools.logging import logger

from .spec import MODES, REPORTERV2_API_URL, SPECS, TRAIN_DATASET_LABEL

PALETTE = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
RUN_PAGE_URL = REPORTERV2_API_URL + "/runs/{run_id}"
SCALING_LR = "1e-2"
TAIL_FRACTION = 0.05


def run_id(array_job_id, task_index):
    h = hashlib.sha1(f"slurm:{array_job_id}:{task_index}".encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def run_config(run):
    url = f"{REPORTERV2_API_URL}/api/runs/{run}"
    try:
        with urlopen(url, timeout=60) as response:
            d = json.load(response)
    except (OSError, ValueError):
        return None
    cmd = d.get("command", "")
    selected = cmd.rsplit("selected_args:", 1)[-1]

    def arg(pattern):
        return re.search(pattern, selected) or re.search(pattern, cmd)

    width = re.search(r"CONFIG=vit_mup_w(\d+)", cmd)
    lr = arg(r"--optimizer\.lr=([0-9.eE+-]+)")
    steps = arg(r"--training\.steps=(\d+)")
    return {
        "width": int(width.group(1)) if width else None,
        "lr": lr.group(1) if lr else None,
        "steps": int(steps.group(1)) if steps else None,
    }


def warn_mixed_steps(steps_seen, source):
    if len(steps_seen) > 1:
        logger.warning(
            "mixed --training.steps %s %s", sorted(steps_seen, key=str), source
        )


def collect_scaling(spec, arrays):
    points = []
    steps_seen = set()
    for want_width, array_job_id in sorted(arrays.items()):
        seen = set()
        by_steps = defaultdict(list)
        for task_index in range(8):
            rid = run_id(array_job_id, task_index)
            cfg = run_config(rid)
            if cfg is None or cfg["lr"] != SCALING_LR:
                continue
            if cfg["width"] is not None:
                seen.add(cfg["width"])
            loss = tail_loss(rid, spec.loss_key)
            if loss is not None:
                by_steps[cfg["steps"]].append((loss, rid))
        width = want_width
        if seen and want_width not in seen:
            width = sorted(seen)[0]
            logger.warning(
                "width mismatch: array %s mapped to w%s but logged config says w%s",
                array_job_id,
                want_width,
                sorted(seen),
            )
        if not by_steps:
            points.append((width, None, None, 0, []))
        for steps, hits in by_steps.items():
            steps_seen.add(steps)
            losses = [x for x, _ in hits]
            points.append(
                (
                    width,
                    steps,
                    sum(losses) / len(losses),
                    len(hits),
                    [r for _, r in hits],
                )
            )
    warn_mixed_steps(steps_seen, f"across arrays {arrays}")
    if not any(n for *_, n, _ in points):
        raise RuntimeError(f"collected zero losses from arrays {arrays}")
    return points


def tail_loss(run, loss_key):
    url = f"{REPORTERV2_API_URL}/api/runs/{run}/metrics"
    try:
        with urlopen(url, timeout=60) as response:
            rows = json.load(response)["metrics"]
    except (OSError, ValueError, KeyError):
        return None
    vals = [row[loss_key] for row in rows if loss_key in row]
    if not vals:
        return None
    tail = vals[-max(1, int(len(vals) * TAIL_FRACTION)) :]
    return sum(tail) / len(tail)


def read_manifest(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"manifest not found at {path}; set MUP_MANIFEST to override the path"
        )
    runs = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 5:
                continue
            param, width, seed, lr_list, array_job_id = parts
            runs.append(
                (param, int(width), int(seed), lr_list.split(","), array_job_id)
            )
    return runs


def _rows(seeded, source):
    results = []
    for (param, width, lr, steps), hits in seeded.items():
        losses = [x for x, _ in hits if x is not None]
        mean = sum(losses) / len(losses) if losses else None
        results.append(
            (param, width, lr, steps, mean, len(losses), [r for _, r in hits])
        )
    if not any(n for *_, n, _ in results):
        raise RuntimeError(f"collected zero losses {source}")
    return results


def collect(spec, manifest_path=None):
    path = manifest_path or spec.manifest_path
    declared = defaultdict(set)
    for param, width, _seed, lrs, array_job_id in read_manifest(path):
        declared[(param, width, array_job_id)].update(lrs)
    seeded = defaultdict(list)
    steps_seen = set()
    for (param, width, array_job_id), lrs in declared.items():
        seen = set()
        for task_index in count():
            rid = run_id(array_job_id, task_index)
            cfg = run_config(rid)
            if cfg is None or cfg["lr"] is None:
                break
            seen.add(cfg["lr"])
            steps_seen.add(cfg["steps"])
            seeded[(param, width, cfg["lr"], cfg["steps"])].append(
                (tail_loss(rid, spec.loss_key), rid)
            )
        if seen != lrs:
            logger.warning(
                "job %s ran lrs %s but manifest %s declares %s",
                array_job_id,
                sorted(seen),
                path,
                sorted(lrs),
            )
    warn_mixed_steps(steps_seen, f"reading manifest {path}")
    return _rows(seeded, f"reading manifest {path}")


def collect_arrays(spec, arrays):
    seeded = defaultdict(list)
    steps_seen = set()
    for want_width, array_job_id in sorted(arrays.items()):
        for task_index in range(8):
            rid = run_id(array_job_id, task_index)
            cfg = run_config(rid)
            if cfg is None or cfg["lr"] is None:
                continue
            width = cfg["width"] or want_width
            steps_seen.add(cfg["steps"])
            seeded[("mup", width, cfg["lr"], cfg["steps"])].append(
                (tail_loss(rid, spec.loss_key), rid)
            )
    warn_mixed_steps(steps_seen, f"across arrays {arrays}")
    return _rows(seeded, f"from arrays {arrays}")


def hp_table(spec, results):
    per_width = {}
    for param, width, lr, _steps, loss, _n, _runs in results:
        if param != "mup" or loss is None:
            continue
        cur = per_width.get(width)
        if cur is None or loss < cur[1]:
            per_width[width] = (lr, loss)
    big = [lr for w, (lr, _) in per_width.items() if w >= spec.base_width]
    transferred = max(set(big), key=big.count) if big else None
    return per_width, transferred


def _steps_text(steps_seen):
    return "/".join(str(s) for s in sorted(steps_seen)) or "?"


def _mutransfer_fig(curves, widths, lr_ticks):
    import plotly.graph_objects as go

    fig = go.Figure()
    for wi, width in enumerate(widths):
        pts = curves.get("mup", {}).get(width, [])
        if not pts:
            continue
        fig.add_trace(
            go.Scatter(
                x=[lr for lr, _ in pts],
                y=[loss for _, loss in pts],
                name=f"w{width}",
                mode="lines+markers",
                line={"color": PALETTE[wi % len(PALETTE)]},
                marker={"color": PALETTE[wi % len(PALETTE)]},
            )
        )
    fig.update_xaxes(
        type="log",
        title_text="learning rate",
        tickvals=[v for v, _ in lr_ticks],
        ticktext=[s for _, s in lr_ticks],
    )
    fig.update_yaxes(title_text="train loss (tail mean)")
    fig.update_layout(
        title_text="muTransfer (mup): train loss vs lr per width",
        width=1000,
        height=460,
    )
    return fig


def _scaling_fig(basins):
    import plotly.graph_objects as go

    ws = sorted(basins)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ws,
            y=[basins[w] for w in ws],
            mode="lines+markers",
            name="measured basin",
            marker={"size": 10},
        )
    )
    fig.update_xaxes(type="log", title_text="width")
    fig.update_yaxes(title_text="muP basin loss (at optimal lr)")
    fig.update_layout(title_text="width-scaling loss (measured points, no fit)", width=1000, height=460)
    return fig


def links(runs):
    return " ".join(
        f'<a href="{RUN_PAGE_URL.format(run_id=r)}">{r[:8]}</a>' for r in runs
    )


def loss_cell(loss):
    return f"{loss:.6f}" if loss is not None else "N/A"


def render_report(report_name, report_dir, make, read_only=True):
    from xx.release_tests.lib.base_report import (
        BaseReport,
        BaseReportConfig,
        ReportFormat,
    )

    class Report(BaseReport):
        def run_data(self):
            return None

        def make_report(self, _):
            return make()

    Report(
        BaseReportConfig(
            report_name=report_name,
            output_dir=report_dir,
            read_only=read_only,
            format=ReportFormat.FILE,
        )
    ).run_report()


def build_report(spec, results=None):
    from xx.release_tests.lib.utils import MarkdownWriter

    if results is None:
        results = collect(spec)
    per_width, transferred = hp_table(spec, results)

    os.makedirs(spec.report_dir, exist_ok=True)
    hp = {
        "model": spec.name,
        "param": "mup",
        "base_width": spec.base_width,
        "optimal_lr": transferred,
        "optimal_lr_per_width": {w: per_width[w][0] for w in sorted(per_width)},
    }
    with open(f"{spec.report_dir}/mup_hparams.json", "w") as fh:
        json.dump(hp, fh, indent=2)

    curves = {m: {} for m in MODES}
    for param, width, lr, _steps, loss, _n, _runs in results:
        if param not in curves or loss is None:
            continue
        curves[param].setdefault(width, []).append((float(lr), loss))
    for mode in curves:
        for width in curves[mode]:
            curves[mode][width].sort()

    widths = sorted({w for mode in curves for w in curves[mode]})
    lr_ticks = sorted(
        {(float(lr), lr) for _, _, lr, _, loss, _, _ in results if loss is not None}
    )
    steps = _steps_text({s for _, _, _, s, _, _, _ in results if s is not None})

    def make():
        mw = MarkdownWriter(report_name=f"{spec.name}_mutransfer")
        mw.filename = f"{spec.report_dir}/mutransfer.html"
        mw.print(f"{spec.name} muP routine", heading=1)
        mw.print("1. muTransfer sweep", heading=2)
        mw.print(
            "under muP the loss-minimising lr is width-stable (panel minima line up "
            f"across width). loss is the mean over each run's trailing "
            f"{TAIL_FRACTION:.0%} of logged {spec.loss_key}, averaged over seeds."
        )
        if widths:
            mw.add_plot(
                _mutransfer_fig(curves, widths, lr_ticks), xscale=None, yscale=None
            )

        mw.print(f"2. transferred muP lr = {transferred}", heading=2)
        mw.print(
            f"conditions: {steps}-step runs, warmup 51, cosine decay over the "
            f"final 80%, trained on the {TRAIN_DATASET_LABEL}; the transferred lr "
            "is specific to this horizon and schedule."
        )
        with mw.html_table(["width", "optimal lr", "basin"], borders=True) as t:
            for w in sorted(per_width):
                t.row([w, per_width[w][0], f"{per_width[w][1]:.3f}"])

        mw.print("runs", heading=2)
        with mw.html_table(
            ["param", "width", "lr", "steps", "seeds", "loss", "reporters"],
            borders=True,
        ) as t:
            for param, width, lr, run_steps, loss, n, runs in sorted(results):
                t.row([param, width, lr, run_steps, n, loss_cell(loss), links(runs)])
        return mw

    render_report(f"{spec.name}_mutransfer", spec.report_dir, make)
    return spec.report_url()


def scaling_report(spec, points):
    from xx.release_tests.lib.utils import MarkdownWriter

    basins = {w: loss for w, _s, loss, _n, _runs in points if loss is not None}
    steps = _steps_text({s for _, s, loss, _, _ in points if loss is not None})

    os.makedirs(spec.report_dir, exist_ok=True)

    def make():
        mw = MarkdownWriter(report_name=f"{spec.name}_scaling")
        mw.filename = f"{spec.report_dir}/scaling.html"
        mw.print(f"{spec.name} width-scaling", heading=1)
        mw.print(
            f"loss(w) at the fixed muP lr after {steps} steps on the "
            f"{TRAIN_DATASET_LABEL}, averaged over seeds. loss is the mean over "
            f"each run's trailing {TAIL_FRACTION:.0%} of logged {spec.loss_key}. "
            "width and steps are read from each run's logged command."
        )
        if basins:
            mw.add_plot(_scaling_fig(basins), xscale=None, yscale=None)
        else:
            mw.print("no width points with a measured loss yet.")

        with mw.html_table(
            ["width", "steps", "loss", "seeds", "reporters"], borders=True
        ) as t:
            for width, run_steps, loss, n, runs in sorted(points):
                t.row([width, run_steps, loss_cell(loss), n, links(runs)])
        return mw

    render_report(f"{spec.name}_scaling", spec.report_dir, make)
    return spec.report_url("scaling")


USAGE = (
    "usage: python -m torchtitan.experiments.mup.routine {collect|report} <model> [manifest]\n"
    "       python -m torchtitan.experiments.mup.routine scaling <model> <w256=job,...>\n"
    "       python -m torchtitan.experiments.mup.routine val <model> --ckpt <dcp_dir> [--label L] [--smoke]\n"
    "       python -m torchtitan.experiments.mup.routine suite <model> [--arrays w256=J,...] [--ckpt DIR]"
)


def _arrays(text):
    out = {}
    for part in text.split(","):
        key, value = part.split("=")
        out[int(key.lstrip("w"))] = int(value)
    return out


def _flags(argv):
    flags = {}
    i = 0
    while i < len(argv):
        token = argv[i]
        if not token.startswith("--"):
            i += 1
        elif i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            flags[token[2:]] = argv[i + 1]
            i += 2
        else:
            flags[token[2:]] = True
            i += 1
    return flags


def _coord_check(spec):
    import torch

    cmd = f"python -m torchtitan.experiments.mup.coord_check {spec.name}"
    if not torch.cuda.is_available():
        return "coord_check", f"needs a GPU; run: {cmd}", False
    from . import coord_check as cc

    try:
        cc.run(spec)
    except Exception as exc:
        return "coord_check", f"{type(exc).__name__}: {exc}; run: {cmd}", False
    return "coord_check", spec.report_url("coord_check"), True


def _mutransfer(spec, arrays):
    if arrays:
        results = collect_arrays(spec, arrays)
        source = "arrays"
    elif os.path.exists(spec.manifest_path):
        results = collect(spec)
        source = spec.manifest_path
    else:
        return (
            "mutransfer",
            f"no --arrays and no manifest at {spec.manifest_path}",
            False,
        )
    per_width, transferred = hp_table(spec, results)
    print(f"muTransfer from {source}:")
    for w in sorted(per_width):
        lr, basin = per_width[w]
        print(f"  w{w}: lr*={lr} basin={basin:.4f}")
    print(f"  transferred muP lr = {transferred}")
    return "mutransfer", build_report(spec, results=results), True


def _scaling(spec, arrays):
    if not arrays:
        return "scaling", "needs --arrays w256=job,...", False
    points = collect_scaling(spec, arrays)
    print("scaling points:")
    for width, steps, loss, n, _runs in sorted(points):
        print(
            f"  w{width}: loss={'N/A' if loss is None else round(loss, 4)} "
            f"seeds={n} steps={steps}"
        )
    return "scaling", scaling_report(spec, points), True


def _ckpt_flavor(ckpt):
    from ..path.config_registry import VIT_WIDTHS

    for part in reversed(ckpt.rstrip("/").split(os.sep)):
        if part in VIT_WIDTHS:
            return part
    sys.exit(
        f"cannot infer flavor from {ckpt}; expected a path segment in "
        f"{sorted(VIT_WIDTHS)}"
    )


def run_val(spec, argv):
    from . import valset_report as vr

    flags = _flags(argv)
    ckpt = flags.get("ckpt")
    if not isinstance(ckpt, str) or not os.path.isdir(ckpt):
        sys.exit("val needs --ckpt <dcp checkpoint dir>")
    flavor = _ckpt_flavor(ckpt)
    label = flags.get("label") or "/".join(ckpt.rstrip("/").split(os.sep)[-2:])
    steps = 1 if flags.get("smoke") else None

    model = vr.load_model(flavor, ckpt, mup=True)
    loss_fn = vr.load_loss()
    per = {
        n: vr.evaluate_valset(model, loss_fn, n, steps=steps) for n in vr.ATOMIC_VALSETS
    }

    os.makedirs(spec.report_dir, exist_ok=True)
    metrics_path = f"{spec.report_dir}/valset_metrics.json"
    results = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as fh:
            results = json.load(fh)
    results[label] = per
    with open(metrics_path, "w") as fh:
        json.dump(results, fh, indent=2)

    for name, m in per.items():
        print(f"{name}: loss={m['loss']:.4f} n_samples={m['n_samples']}")
    out = vr.build_report(results, report_dir=spec.report_dir, read_only=True)
    print(f"report -> {out}")


def run_suite(spec, argv):
    flags = _flags(argv)
    arrays = _arrays(flags["arrays"]) if "arrays" in flags else None
    ckpt = flags.get("ckpt")

    os.makedirs(spec.report_dir, exist_ok=True)
    outcomes = [
        _coord_check(spec),
        _mutransfer(spec, arrays),
        _scaling(spec, arrays),
        (
            "val",
            "GPU job; run: python -m torchtitan.experiments.mup.routine val "
            f"{spec.name} --ckpt {ckpt or 'DIR'}",
            False,
        ),
    ]
    produced = [(label, url) for label, url, ok in outcomes if ok]
    skipped = [(label, reason) for label, reason, ok in outcomes if not ok]

    print(f"\nsuite index for {spec.name} -> {spec.report_dir}")
    for label, url in produced:
        print(f"  produced {label}: {url}")
    for label, reason in skipped:
        print(f"  skipped  {label}: {reason}")


def main():
    argv = sys.argv[1:]
    if len(argv) < 2 or argv[0] not in ("collect", "report", "scaling", "val", "suite"):
        sys.exit(USAGE)
    verb, model = argv[0], argv[1]
    if model not in SPECS:
        sys.exit(f"unknown model {model!r}; known: {', '.join(SPECS)}")
    spec = SPECS[model]
    if not spec.ready:
        sys.exit(f"{model} has no muP configs landed yet; nothing to collect.")

    if verb == "suite":
        run_suite(spec, argv[2:])
        return

    if verb == "val":
        run_val(spec, argv[2:])
        return

    if verb == "scaling":
        if len(argv) < 3:
            sys.exit(USAGE)
        arrays = _arrays(argv[2])
        points = collect_scaling(spec, arrays)
        for width, steps, loss, n, _runs in sorted(points):
            print(
                f"w{width}: loss={'N/A' if loss is None else round(loss, 4)} "
                f"seeds={n} steps={steps}"
            )
        print(f"report -> {scaling_report(spec, points)}")
        return

    manifest = argv[2] if len(argv) > 2 else None
    results = collect(spec, manifest)
    per_width, transferred = hp_table(spec, results)
    for w in sorted(per_width):
        lr, basin = per_width[w]
        print(f"w{w}: lr*={lr} basin={basin:.4f}")
    print(f"transferred muP lr = {transferred}")

    if verb == "report":
        print(f"report -> {build_report(spec, results=results)}")


if __name__ == "__main__":
    main()
