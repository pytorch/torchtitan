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

from .spec import MODES, REPORTERV2_API_URL, SPECS, TRAIN_LOSS_KEY

PALETTE = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
RUN_PAGE_URL = REPORTERV2_API_URL + "/runs/{run_id}"
SCALING_LR = "1e-2"
TAIL_FRACTION = 0.05


def run_id(array_job_id, task_index):
    h = hashlib.sha1(f"slurm:{array_job_id}:{task_index}".encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def fetch_metrics(run):
    url = f"{REPORTERV2_API_URL}/api/runs/{run}/metrics"
    with urlopen(url, timeout=60) as response:
        return json.load(response)["metrics"]


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
        "status": d.get("status"),
    }


def collect_scaling(spec, arrays, lr=SCALING_LR, max_tasks=8):
    points = []
    steps_seen = set()
    for want_width, array_job_id in sorted(arrays.items()):
        seen = set()
        by_steps = defaultdict(list)
        for task_index in range(max_tasks):
            rid = run_id(array_job_id, task_index)
            cfg = run_config(rid)
            if cfg is None or cfg["lr"] != lr:
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
    if len(steps_seen) > 1:
        logger.warning(
            "mixed --training.steps %s across arrays %s",
            sorted(steps_seen, key=str),
            arrays,
        )
    if not any(n for *_, n, _ in points):
        raise RuntimeError(f"collected zero losses from arrays {arrays}")
    return points


def tail_loss(run, loss_key=TRAIN_LOSS_KEY):
    try:
        rows = fetch_metrics(run)
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
    if len(steps_seen) > 1:
        logger.warning(
            "mixed --training.steps %s reading manifest %s",
            sorted(steps_seen, key=str),
            path,
        )
    return _rows(seeded, f"reading manifest {path}")


def collect_arrays(spec, arrays, max_tasks=8):
    seeded = defaultdict(list)
    steps_seen = set()
    for want_width, array_job_id in sorted(arrays.items()):
        for task_index in range(max_tasks):
            rid = run_id(array_job_id, task_index)
            cfg = run_config(rid)
            if cfg is None or cfg["lr"] is None:
                continue
            width = cfg["width"] or want_width
            steps_seen.add(cfg["steps"])
            seeded[("mup", width, cfg["lr"], cfg["steps"])].append(
                (tail_loss(rid, spec.loss_key), rid)
            )
    if len(steps_seen) > 1:
        logger.warning(
            "mixed --training.steps %s across arrays %s",
            sorted(steps_seen, key=str),
            arrays,
        )
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


def fit_predictor(basins):
    import numpy as np
    from scipy.optimize import curve_fit

    ws = np.array(sorted(basins), dtype=float)
    ys = np.array([basins[int(w)] for w in ws])

    def f(w, linf, a, alpha):
        return linf + a * w ** (-alpha)

    popt, _ = curve_fit(f, ws, ys, p0=[ys.min() - 1, 50.0, 0.5], maxfev=20000)
    pred = f(ws, *popt)
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return popt, r2, (lambda w: float(f(np.array([w], float), *popt)[0]))


def _steps_text(steps_seen):
    return "/".join(str(s) for s in sorted(steps_seen)) or "?"


def _curves(results):
    curves = {m: {} for m in MODES}
    for param, width, lr, _steps, loss, _n, _runs in results:
        if param not in curves or loss is None:
            continue
        curves[param].setdefault(width, []).append((float(lr), loss))
    for mode in curves:
        for width in curves[mode]:
            curves[mode][width].sort()
    return curves


def _mutransfer_fig(curves, widths, lr_ticks):
    import plotly.graph_objects as go

    fig = go.Figure()
    for width in widths:
        pts = curves.get("mup", {}).get(width, [])
        if not pts:
            continue
        wi = widths.index(width)
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


def _predictor_fig(per_width, popt, preds):
    import numpy as np
    import plotly.graph_objects as go

    linf, a, alpha = popt
    ws = sorted(per_width)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ws,
            y=[per_width[w][1] for w in ws],
            mode="markers",
            name="measured basin",
            marker={"size": 10},
        )
    )
    xs = np.geomspace(min(ws), max(p[0] for p in preds), 60)
    fig.add_trace(
        go.Scatter(
            x=xs, y=[linf + a * x ** (-alpha) for x in xs], mode="lines", name="fit"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[w for w, _ in preds],
            y=[v for _, v in preds],
            mode="markers",
            name="prediction",
            marker={"size": 12, "symbol": "x"},
        )
    )
    fig.update_xaxes(type="log", title_text="width")
    fig.update_yaxes(title_text="muP basin loss (at optimal lr)")
    fig.update_layout(title_text="width-scaling loss predictor", width=1000, height=460)
    return fig


def build_report(spec, results=None):
    from xx.release_tests.lib.base_report import (
        BaseReport,
        BaseReportConfig,
        ReportFormat,
    )
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

    curves = _curves(results)
    widths = sorted({w for mode in curves for w in curves[mode]})
    lr_ticks = sorted(
        {(float(lr), lr) for _, _, lr, _, loss, _, _ in results if loss is not None}
    )
    steps = _steps_text({s for _, _, _, s, _, _, _ in results if s is not None})

    class Report(BaseReport):
        def run_data(self):
            return None

        def make_report(self, _):
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
                "final 80%; the transferred lr is specific to this horizon and "
                "schedule."
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
                    loss_s = f"{loss:.6f}" if loss is not None else "N/A"
                    links = " ".join(
                        f'<a href="{RUN_PAGE_URL.format(run_id=r)}">{r[:8]}</a>'
                        for r in runs
                    )
                    t.row([param, width, lr, run_steps, n, loss_s, links])
            return mw

    Report(
        BaseReportConfig(
            report_name=f"{spec.name}_mutransfer",
            output_dir=spec.report_dir,
            read_only=True,
            format=ReportFormat.FILE,
        )
    ).run_report()
    return spec.report_url


def scaling_report(spec, points, bigvit_width=None):
    from xx.release_tests.lib.base_report import (
        BaseReport,
        BaseReportConfig,
        ReportFormat,
    )
    from xx.release_tests.lib.utils import MarkdownWriter

    basins = {w: loss for w, _s, loss, _n, _runs in points if loss is not None}
    steps = _steps_text({s for _, s, loss, _, _ in points if loss is not None})
    predictor = None
    if len(basins) >= 3:
        try:
            predictor = fit_predictor(basins)
        except ImportError:
            logger.warning("scipy unavailable; skipping scaling fit")

    os.makedirs(spec.report_dir, exist_ok=True)

    class Report(BaseReport):
        def run_data(self):
            return None

        def make_report(self, _):
            mw = MarkdownWriter(report_name=f"{spec.name}_scaling")
            mw.filename = f"{spec.report_dir}/scaling.html"
            mw.print(f"{spec.name} width-scaling", heading=1)
            mw.print(
                f"loss(w) at the fixed muP lr after {steps} steps, averaged over "
                f"seeds. loss is the mean over each run's trailing "
                f"{TAIL_FRACTION:.0%} of logged {spec.loss_key}. width and steps are "
                "read from each run's logged command."
            )
            if predictor is not None:
                popt, r2, predict = predictor
                linf, a, alpha = popt
                preds = [
                    (bigvit_width, predict(bigvit_width))
                    if bigvit_width is not None
                    else (max(basins), basins[max(basins)])
                ]
                per_width = {w: (None, loss) for w, loss in basins.items()}
                mw.print(
                    f"fit loss(w) = L_inf + A*w^(-alpha): L_inf={linf:.3f}, A={a:.3f}, "
                    f"alpha={alpha:.3f}, R2={r2:.3f}"
                )
                if bigvit_width is not None:
                    mw.print(
                        f"predicted bigvit w{bigvit_width} loss = "
                        f"{predict(bigvit_width):.4f}"
                    )
                mw.add_plot(
                    _predictor_fig(per_width, popt, preds), xscale=None, yscale=None
                )
            else:
                mw.print(
                    "need >=3 width points (and scipy) to fit; showing raw points only."
                )

            with mw.html_table(
                ["width", "steps", "loss", "seeds", "reporters"], borders=True
            ) as t:
                for width, run_steps, loss, n, runs in sorted(points):
                    loss_s = f"{loss:.6f}" if loss is not None else "N/A"
                    links = " ".join(
                        f'<a href="{RUN_PAGE_URL.format(run_id=r)}">{r[:8]}</a>'
                        for r in runs
                    )
                    t.row([width, run_steps, loss_s, n, links])
            return mw

    Report(
        BaseReportConfig(
            report_name=f"{spec.name}_scaling",
            output_dir=spec.report_dir,
            read_only=True,
            format=ReportFormat.FILE,
        )
    ).run_report()
    return spec.report_url.replace("mutransfer.html", "scaling.html")


USAGE = (
    "usage: python -m torchtitan.experiments.mup.routine {collect|report} <model> [manifest]\n"
    "       python -m torchtitan.experiments.mup.routine scaling <model> <w256=job,...> [bigvit_width]\n"
    "       python -m torchtitan.experiments.mup.routine suite <model> [--arrays w256=J,...] [--big-array J] [--ckpt DIR]"
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
        if token.startswith("--") and i + 1 < len(argv):
            flags[token[2:]] = argv[i + 1]
            i += 2
        else:
            i += 1
    return flags


def _coord_check(spec, produced, skipped):
    import torch

    cmd = f"python -m torchtitan.experiments.mup.coord_check {spec.name}"
    if not torch.cuda.is_available():
        skipped.append(("coord_check", f"needs a GPU; run: {cmd}"))
        return
    from . import coord_check as cc

    try:
        cc.run(spec)
    except Exception as exc:
        skipped.append(("coord_check", f"{type(exc).__name__}: {exc}; run: {cmd}"))
        return
    produced.append(
        ("coord_check", spec.report_url.replace("mutransfer.html", "coord_check.html"))
    )


def _mutransfer(spec, arrays, produced, skipped):
    if arrays:
        results = collect_arrays(spec, arrays)
        source = "arrays"
    elif os.path.exists(spec.manifest_path):
        results = collect(spec)
        source = spec.manifest_path
    else:
        skipped.append(
            ("mutransfer", f"no --arrays and no manifest at {spec.manifest_path}")
        )
        return
    per_width, transferred = hp_table(spec, results)
    print(f"muTransfer from {source}:")
    for w in sorted(per_width):
        lr, basin = per_width[w]
        print(f"  w{w}: lr*={lr} basin={basin:.4f}")
    print(f"  transferred muP lr = {transferred}")
    produced.append(("mutransfer", build_report(spec, results=results)))


def _scaling(spec, arrays, big_width, produced, skipped):
    if not arrays:
        skipped.append(("scaling", "needs --arrays w256=job,..."))
        return
    points = collect_scaling(spec, arrays)
    print("scaling points:")
    for width, steps, loss, n, _runs in sorted(points):
        print(
            f"  w{width}: loss={'N/A' if loss is None else round(loss, 4)} "
            f"seeds={n} steps={steps}"
        )
    produced.append(("scaling", scaling_report(spec, points, big_width)))


def _val(spec, ckpt, skipped):
    try:
        from . import valset_report as vr

        names = ", ".join(vr.ATOMIC_VALSETS)
    except Exception:
        names = "day_straight, night_straight, left_lane_change, right_lane_change"
    if ckpt and os.path.isdir(ckpt):
        skipped.append(
            (
                "val",
                f"checkpoint {ckpt} present; scoring the 4 atomic valsets [{names}] "
                "is a GPU job via valset_report.evaluate_all",
            )
        )
    else:
        skipped.append(
            (
                "val",
                f"needs a checkpoint dir (--ckpt DIR) to score the 4 atomic valsets [{names}]",
            )
        )


def _suite(spec, argv):
    flags = _flags(argv)
    arrays = _arrays(flags["arrays"]) if "arrays" in flags else None
    big_width = int(flags["big-array"]) if "big-array" in flags else None
    ckpt = flags.get("ckpt")

    os.makedirs(spec.report_dir, exist_ok=True)
    produced, skipped = [], []
    _coord_check(spec, produced, skipped)
    _mutransfer(spec, arrays, produced, skipped)
    _scaling(spec, arrays, big_width, produced, skipped)
    _val(spec, ckpt, skipped)

    print(f"\nsuite index for {spec.name} -> {spec.report_dir}")
    for label, url in produced:
        print(f"  produced {label}: {url}")
    for label, reason in skipped:
        print(f"  skipped  {label}: {reason}")


def main():
    argv = sys.argv[1:]
    if len(argv) < 2 or argv[0] not in ("collect", "report", "scaling", "suite"):
        sys.exit(USAGE)
    verb, model = argv[0], argv[1]
    if model not in SPECS:
        sys.exit(f"unknown model {model!r}; known: {', '.join(SPECS)}")
    spec = SPECS[model]
    if not spec.ready:
        sys.exit(f"{model} has no muP configs landed yet; nothing to collect.")

    if verb == "suite":
        _suite(spec, argv[2:])
        return

    if verb == "scaling":
        if len(argv) < 3:
            sys.exit(USAGE)
        arrays = _arrays(argv[2])
        big_width = int(argv[3]) if len(argv) > 3 else None
        points = collect_scaling(spec, arrays)
        for width, steps, loss, n, _runs in sorted(points):
            print(
                f"w{width}: loss={'N/A' if loss is None else round(loss, 4)} "
                f"seeds={n} steps={steps}"
            )
        print(f"report -> {scaling_report(spec, points, big_width)}")
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
