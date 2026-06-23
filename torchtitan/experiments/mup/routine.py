# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The model-agnostic muP routine.

Drives a sweep end to end: build the launch grid, collect each run's final train loss from reporterv2,
read off the transferred muP lr (the HP table), fit a width-scaling loss predictor, and write a plain
plotly html report. Decoupled from any project-specific report infra: it writes to the spec's
report_dir (per-user by default, see spec.py). See spec.py for the per-model MuPSweepSpec.
"""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from .spec import REPORTERV2_API_URL, TRAIN_LOSS_KEY

PALETTE = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]  # per-width
MODES = ("standard", "mup")
RUN_PAGE_URL = REPORTERV2_API_URL + "/runs/{training_id}"


# --- collect: reporterv2 -> final train loss per run --------------------------------------------


def fetch_metrics(training_id):
    url = f"{REPORTERV2_API_URL}/api/runs/{training_id}/metrics"
    with urlopen(url, timeout=60) as response:
        return json.load(response)["metrics"]


def final_train_loss(training_id, loss_key=TRAIN_LOSS_KEY):
    # (loss, step) at the last logged step, or (None, None) if the run/key is absent.
    try:
        rows = fetch_metrics(training_id)
    except (HTTPError, URLError):
        return None, None
    steps, losses = [], []
    for row in rows:
        if loss_key in row:
            steps.append(row.get("step", row.get("epoch")))
            losses.append(float(row[loss_key]))
    if not losses:
        return None, None
    last = max(range(len(steps)), key=lambda i: steps[i])
    return losses[last], steps[last]


def grid(spec, params=MODES):
    # the full launch grid: one dict per run. submission (cluster launch) is the caller's job.
    out = []
    for param in params:
        for width in spec.widths:
            for lr in spec.lrs:
                out.append(
                    {
                        "param": param,
                        "width": width,
                        "lr": lr,
                        "config": spec.config_name(param, width),
                        "training_id": spec.training_id(param, width, lr),
                        "env": f"{spec.env_lr_var}={lr}",
                    }
                )
    return out


def collect(spec, params=MODES):
    # Rows: (param, width, lr_str, loss, step, training_id).
    results = []
    for g in grid(spec, params):
        loss, step = final_train_loss(g["training_id"], spec.loss_key)
        results.append((g["param"], g["width"], g["lr"], loss, step, g["training_id"]))
    return results


# --- routine outputs: HP table + width-scaling loss predictor -----------------------------------


def hp_table(spec, results):
    # muP-optimal lr (argmin loss) and basin per width, plus the transferred lr (mode over widths >= base).
    per_width = {}
    for param, width, lr, loss, step, tid in results:
        if param != "mup" or loss is None:
            continue
        cur = per_width.get(width)
        if cur is None or loss < cur[1]:
            per_width[width] = (lr, loss)
    big = [lr for w, (lr, _) in per_width.items() if w >= spec.base_width]
    transferred = max(set(big), key=big.count) if big else None
    return per_width, transferred


def fit_predictor(basins):
    # fit basin_loss(w) = L_inf + A * w**(-alpha) over {width: loss}. Returns (popt, r2, predict).
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


# --- report ------------------------------------------------------------------------------------


def _curves(results, params=MODES):
    curves = {m: {} for m in params}
    for param, width, lr, loss, step, tid in results:
        if param not in curves or loss is None:
            continue
        curves[param].setdefault(width, []).append((float(lr), loss))
    for mode in curves:
        for width in curves[mode]:
            curves[mode][width].sort()
    return curves


def _mutransfer_fig(curves, widths):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, subplot_titles=list(MODES), shared_yaxes=True)
    for col, mode in enumerate(MODES, 1):
        for width in widths:
            pts = curves.get(mode, {}).get(width, [])
            if not pts:
                continue
            wi = widths.index(width)
            fig.add_trace(
                go.Scatter(
                    x=[lr for lr, _ in pts],
                    y=[loss for _, loss in pts],
                    name=f"w{width}",
                    mode="lines+markers",
                    legendgroup=f"w{width}",
                    showlegend=(col == 1),
                    line={"color": PALETTE[wi % len(PALETTE)]},
                    marker={"color": PALETTE[wi % len(PALETTE)]},
                ),
                row=1,
                col=col,
            )
    fig.update_xaxes(type="log", title_text="learning rate")
    fig.update_yaxes(title_text="final train loss")
    fig.update_layout(title_text="muTransfer: final train loss vs lr per width", width=1100, height=460)
    return fig


def _predictor_fig(per_width, popt, predict, preds):
    import numpy as np
    import plotly.graph_objects as go

    linf, a, alpha = popt
    ws = sorted(per_width)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=ws, y=[per_width[w][1] for w in ws], mode="markers", name="measured basin", marker={"size": 10})
    )
    xs = np.geomspace(min(ws), max(p[0] for p in preds), 60)
    fig.add_trace(go.Scatter(x=xs, y=[linf + a * x ** (-alpha) for x in xs], mode="lines", name="fit"))
    fig.add_trace(
        go.Scatter(
            x=[w for w, _ in preds], y=[v for _, v in preds], mode="markers", name="prediction",
            marker={"size": 12, "symbol": "x"},
        )
    )
    fig.update_xaxes(type="log", title_text="width")
    fig.update_yaxes(title_text="muP basin loss (at optimal lr)")
    fig.update_layout(title_text="width-scaling loss predictor", width=1000, height=460)
    return fig


def build_report(spec, params=MODES, results=None):
    # results may be passed in to avoid re-querying reporterv2; otherwise it is collected here.
    if results is None:
        results = collect(spec, params)
    curves = _curves(results, params)
    widths = sorted({w for mode in curves for w in curves[mode]})
    per_width, transferred = hp_table(spec, results)

    basins = {w: loss for w, (lr, loss) in per_width.items()}
    predictor = fit_predictor(basins) if len(basins) >= 3 else None
    preds = []
    if predictor is not None:
        popt, r2, predict = predictor
        top = max(basins)
        preds = [(top * 2, predict(top * 2)), (top * 4, predict(top * 4))]

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

    blocks = [f"<h1>{spec.name} muP routine</h1>"]
    blocks.append("<h2>1. muTransfer sweep</h2>")
    blocks.append(
        "<p>under muP the loss-minimising lr is width-stable (panel minima line up across width); "
        f"standard's optimum drifts. final train loss is the last logged {spec.loss_key} per run.</p>"
    )
    if widths:
        blocks.append(_mutransfer_fig(curves, widths).to_html(full_html=False, include_plotlyjs="cdn"))
    blocks.append(f"<h2>2. transferred muP lr = {transferred}</h2>")
    blocks.append("<table border=1 cellpadding=6><tr><th>width</th><th>optimal lr</th><th>basin</th></tr>")
    for w in sorted(per_width):
        blocks.append(f"<tr><td>{w}</td><td>{per_width[w][0]}</td><td>{per_width[w][1]:.3f}</td></tr>")
    blocks.append("</table>")
    if predictor is not None:
        linf, a, alpha = popt
        blocks.append("<h2>3. loss predictor (width scaling)</h2>")
        blocks.append(
            f"<p>fit loss(w) = L_inf + A*w^(-alpha): L_inf={linf:.3f}, A={a:.3f}, alpha={alpha:.3f}, R2={r2:.3f}</p>"
        )
        blocks.append("<p>" + " ".join(f"predicted basin w{int(w)}={v:.3f}." for w, v in preds) + "</p>")
        blocks.append(_predictor_fig(per_width, popt, predict, preds).to_html(full_html=False, include_plotlyjs="cdn"))
    blocks.append(
        "<p><i>caveat: few width points at fixed depth/data/steps, a muP-enabled width-scaling "
        "extrapolation, not a compute-optimal scaling law. indicative only.</i></p>"
    )
    blocks.append(
        "<h2>runs</h2><table border=1 cellpadding=6><tr><th>param</th><th>width</th><th>lr</th>"
        "<th>final loss</th><th>step</th><th>reporter</th></tr>"
    )
    for param, width, lr, loss, step, tid in results:
        loss_s = f"{loss:.6f}" if loss is not None else "N/A"
        link = f'<a href="{RUN_PAGE_URL.format(training_id=tid)}">{tid}</a>'
        blocks.append(
            f"<tr><td>{param}</td><td>{width}</td><td>{lr}</td><td>{loss_s}</td>"
            f"<td>{step if step is not None else '-'}</td><td>{link}</td></tr>"
        )
    blocks.append("</table>")

    html = (
        '<html><head><meta charset="utf-8"></head><body style="font-family:sans-serif">'
        + "".join(blocks)
        + "</body></html>"
    )
    with open(f"{spec.report_dir}/mutransfer.html", "w") as fh:
        fh.write(html)
    return spec.report_url
