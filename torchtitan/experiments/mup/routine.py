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
from collections import defaultdict
from urllib.request import urlopen

from torchtitan.tools.logging import logger

from .spec import MODES, REPORTERV2_API_URL, TRAIN_LOSS_KEY

PALETTE = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
RUN_PAGE_URL = REPORTERV2_API_URL + "/runs/{run_id}"
SCALING_LR = "1e-2"


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
    width = re.search(r"CONFIG=vit_mup_w(\d+)", cmd)
    lr = re.search(r"--optimizer\.lr=([0-9.eE+-]+)", cmd)
    return {
        "width": int(width.group(1)) if width else None,
        "lr": lr.group(1) if lr else None,
        "status": d.get("status"),
    }


def collect_scaling(spec, arrays, lr=SCALING_LR, max_tasks=8):
    points = []
    for want_width, array_job_id in sorted(arrays.items()):
        losses, runs, seen = [], [], set()
        for task_index in range(max_tasks):
            rid = run_id(array_job_id, task_index)
            cfg = run_config(rid)
            if cfg is None or cfg["lr"] != lr:
                continue
            if cfg["width"] is not None:
                seen.add(cfg["width"])
            loss = final_loss(rid, spec.loss_key)
            if loss is not None:
                losses.append(loss)
                runs.append(rid)
        width = want_width
        if seen and want_width not in seen:
            width = sorted(seen)[0]
            logger.warning(
                "width mismatch: array %s mapped to w%s but logged config says w%s",
                array_job_id,
                want_width,
                sorted(seen),
            )
        mean = sum(losses) / len(losses) if losses else None
        points.append((width, mean, len(losses), runs))
    if not any(n for _, _, n, _ in points):
        raise RuntimeError(f"collected zero losses from arrays {arrays}")
    return points


def final_loss(run, loss_key=TRAIN_LOSS_KEY):
    try:
        rows = fetch_metrics(run)
    except (OSError, ValueError, KeyError):
        return None
    vals = [row[loss_key] for row in rows if loss_key in row]
    return float(vals[-1]) if vals else None


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


def collect(spec, manifest_path=None):
    path = manifest_path or spec.manifest_path
    seeded = defaultdict(list)
    for param, width, _seed, lrs, array_job_id in read_manifest(path):
        for task_index, lr in enumerate(lrs):
            rid = run_id(array_job_id, task_index)
            seeded[(param, width, lr)].append((final_loss(rid, spec.loss_key), rid))
    results = []
    for (param, width, lr), hits in seeded.items():
        losses = [x for x, _ in hits if x is not None]
        mean = sum(losses) / len(losses) if losses else None
        results.append((param, width, lr, mean, len(losses), [r for _, r in hits]))
    if not any(n for *_, n, _ in results):
        raise RuntimeError(f"collected zero losses reading manifest {path}")
    return results


def collect_arrays(spec, arrays, max_tasks=8):
    seeded = defaultdict(list)
    for want_width, array_job_id in sorted(arrays.items()):
        for task_index in range(max_tasks):
            rid = run_id(array_job_id, task_index)
            cfg = run_config(rid)
            if cfg is None or cfg["lr"] is None:
                continue
            width = cfg["width"] or want_width
            seeded[("mup", width, cfg["lr"])].append(
                (final_loss(rid, spec.loss_key), rid)
            )
    results = []
    for (param, width, lr), hits in seeded.items():
        losses = [x for x, _ in hits if x is not None]
        mean = sum(losses) / len(losses) if losses else None
        results.append((param, width, lr, mean, len(losses), [r for _, r in hits]))
    if not any(n for *_, n, _ in results):
        raise RuntimeError(f"collected zero losses from arrays {arrays}")
    return results


def hp_table(spec, results):
    per_width = {}
    for param, width, lr, loss, _n, _runs in results:
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


def _curves(results):
    curves = {m: {} for m in MODES}
    for param, width, lr, loss, _n, _runs in results:
        if param not in curves or loss is None:
            continue
        curves[param].setdefault(width, []).append((float(lr), loss))
    for mode in curves:
        for width in curves[mode]:
            curves[mode][width].sort()
    return curves


def _mutransfer_fig(curves, widths):
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
    fig.update_xaxes(type="log", title_text="learning rate")
    fig.update_yaxes(title_text="final train loss")
    fig.update_layout(
        title_text="muTransfer (mup): final train loss vs lr per width",
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

    basins = {w: loss for w, (lr, loss) in per_width.items()}
    predictor = None
    if len(basins) >= 3:
        try:
            predictor = fit_predictor(basins)
        except ImportError:
            logger.warning("scipy unavailable; skipping loss-predictor panel")
    preds = []
    if predictor is not None:
        popt, r2, predict = predictor
        top_w = max(basins)
        preds = [(top_w * 2, predict(top_w * 2)), (top_w * 4, predict(top_w * 4))]

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
                "across width). loss is each run's final logged "
                f"{spec.loss_key}, averaged over seeds."
            )
            if widths:
                mw.add_plot(_mutransfer_fig(curves, widths), xscale=None, yscale=None)

            mw.print(f"2. transferred muP lr = {transferred}", heading=2)
            with mw.html_table(["width", "optimal lr", "basin"], borders=True) as t:
                for w in sorted(per_width):
                    t.row([w, per_width[w][0], f"{per_width[w][1]:.3f}"])

            if predictor is not None:
                linf, a, alpha = popt
                mw.print("3. loss predictor (width scaling)", heading=2)
                mw.print(
                    f"fit loss(w) = L_inf + A*w^(-alpha): L_inf={linf:.3f}, A={a:.3f}, "
                    f"alpha={alpha:.3f}, R2={r2:.3f}"
                )
                mw.print(
                    " ".join(f"predicted basin w{int(w)}={v:.3f}." for w, v in preds)
                )
                mw.add_plot(
                    _predictor_fig(per_width, popt, preds), xscale=None, yscale=None
                )
            mw.print(
                "*caveat: few width points at fixed depth/data, a muP-enabled "
                "width-scaling extrapolation, not a compute-optimal scaling law. "
                "indicative only.*"
            )

            mw.print("runs", heading=2)
            with mw.html_table(
                ["param", "width", "lr", "seeds", "loss", "reporters"], borders=True
            ) as t:
                for param, width, lr, loss, n, runs in sorted(results):
                    loss_s = f"{loss:.6f}" if loss is not None else "N/A"
                    links = " ".join(
                        f'<a href="{RUN_PAGE_URL.format(run_id=r)}">{r[:8]}</a>'
                        for r in runs
                    )
                    t.row([param, width, lr, n, loss_s, links])
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

    basins = {w: loss for w, loss, _n, _runs in points if loss is not None}
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
                "loss(w) at the fixed muP lr, averaged over seeds. loss is each run's "
                f"final {spec.loss_key}. width is read from each run's logged CONFIG."
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
                ["width", "loss", "seeds", "reporters"], borders=True
            ) as t:
                for width, loss, n, runs in sorted(points):
                    loss_s = f"{loss:.6f}" if loss is not None else "N/A"
                    links = " ".join(
                        f'<a href="{RUN_PAGE_URL.format(run_id=r)}">{r[:8]}</a>'
                        for r in runs
                    )
                    t.row([width, loss_s, n, links])
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
