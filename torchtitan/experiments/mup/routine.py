# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The model-agnostic muP routine: build the launch grid, collect each run's final train loss from
reporterv2, read off the transferred muP lr, and write a plotly html report."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from urllib.request import urlopen

from .spec import REPORTERV2_API_URL, SPECS, TRAIN_LOSS_KEY

PALETTE = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]  # per-width
MODES = ("standard", "mup")
RUN_PAGE_URL = REPORTERV2_API_URL + "/runs/{training_id}"


# --- collect: reporterv2 -> final train loss per run --------------------------------------------


def fetch_metrics(training_id):
    url = f"{REPORTERV2_API_URL}/api/runs/{training_id}/metrics"
    with urlopen(url, timeout=60) as response:
        return json.load(response)["metrics"]


def final_train_loss(training_id, loss_key=TRAIN_LOSS_KEY):
    # (loss, step) at the last logged step, or (None, None) if the run is absent or unparsable.
    try:
        rows = fetch_metrics(training_id)
        steps, losses = [], []
        for row in rows:
            if loss_key in row:
                steps.append(row.get("step", row.get("epoch")))
                losses.append(float(row[loss_key]))
        if not losses:
            return None, None
        last = max(
            range(len(steps)), key=lambda i: steps[i] if steps[i] is not None else -1
        )
        return losses[last], steps[last]
    except Exception:
        return None, None


def grid(spec, modes=MODES):
    # the full launch grid: one dict per run. lr ships as the --optimizer.lr cli arg the trainer reads.
    out = []
    for mode in modes:
        for width in spec.widths:
            for lr in spec.lrs:
                out.append(
                    {
                        "mode": mode,
                        "width": width,
                        "lr": lr,
                        "config": spec.config_name(mode, width),
                        "training_id": spec.training_id(mode, width, lr),
                        "cli": f"--optimizer.lr {lr}",
                    }
                )
    return out


def collect(spec, modes=MODES):
    # Rows: (mode, width, lr_str, loss, step, training_id).
    results = []
    for g in grid(spec, modes):
        loss, step = final_train_loss(g["training_id"], spec.loss_key)
        results.append((g["mode"], g["width"], g["lr"], loss, step, g["training_id"]))
    return results


# --- routine outputs: transferred-lr HP table ---------------------------------------------------


def hp_table(spec, results):
    # muP-optimal lr (argmin loss) and basin per width, plus the transferred lr (mode over widths >= base).
    per_width = {}
    for mode, width, lr, loss, step, tid in results:
        if mode != "mup" or loss is None:
            continue
        cur = per_width.get(width)
        if cur is None or loss < cur[1]:
            per_width[width] = (lr, loss)
    big = [lr for w, (lr, _) in per_width.items() if w >= spec.base_width]
    if big:
        best = max(Counter(big).values())
        transferred = min(
            (lr for lr, c in Counter(big).items() if c == best), key=float
        )
    else:
        transferred = None
    return per_width, transferred


# --- report ------------------------------------------------------------------------------------


def _curves(results, modes=MODES):
    curves = {m: {} for m in modes}
    for mode, width, lr, loss, step, tid in results:
        if mode not in curves or loss is None:
            continue
        curves[mode].setdefault(width, []).append((float(lr), loss))
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
    fig.update_layout(
        title_text="muTransfer: final train loss vs lr per width",
        width=1100,
        height=460,
    )
    return fig


def build_report(spec, modes=MODES, results=None):
    # results may be passed in to avoid re-querying reporterv2; otherwise it is collected here.
    if results is None:
        results = collect(spec, modes)
    curves = _curves(results, modes)
    widths = sorted({w for mode in curves for w in curves[mode]})
    per_width, transferred = hp_table(spec, results)

    os.makedirs(spec.report_dir, exist_ok=True)
    hp = {
        "model": spec.name,
        "mode": "mup",
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
        blocks.append(
            _mutransfer_fig(curves, widths).to_html(
                full_html=False, include_plotlyjs="cdn"
            )
        )
    blocks.append(f"<h2>2. transferred muP lr = {transferred}</h2>")
    blocks.append(
        "<table border=1 cellpadding=6><tr><th>width</th><th>optimal lr</th><th>basin</th></tr>"
    )
    for w in sorted(per_width):
        blocks.append(
            f"<tr><td>{w}</td><td>{per_width[w][0]}</td><td>{per_width[w][1]:.3f}</td></tr>"
        )
    blocks.append("</table>")
    blocks.append(
        "<h2>runs</h2><table border=1 cellpadding=6><tr><th>mode</th><th>width</th><th>lr</th>"
        "<th>final loss</th><th>step</th><th>reporter</th></tr>"
    )
    for mode, width, lr, loss, step, tid in results:
        loss_s = f"{loss:.6f}" if loss is not None else "N/A"
        link = f'<a href="{RUN_PAGE_URL.format(training_id=tid)}">{tid}</a>'
        blocks.append(
            f"<tr><td>{mode}</td><td>{width}</td><td>{lr}</td><td>{loss_s}</td>"
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


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("grid", "report"):
        sys.exit(
            "usage: python -m torchtitan.experiments.mup.routine {grid|report} <model>"
        )
    verb, model = sys.argv[1], sys.argv[2]
    if model not in SPECS:
        sys.exit(f"unknown model {model!r}; known: {', '.join(SPECS)}")
    spec = SPECS[model]

    if verb == "grid":
        n = len(spec.widths) * len(spec.lrs) * 2
        print(
            f"# launch grid for {model}: {n} runs (standard + mup). submission is the caller's job."
        )
        print(
            "# wrap each line with your launcher, e.g. training/run.sh torchtitan/run_train.sh N=2 PARTITION=tbox2 ..."
        )
        for g in grid(spec):
            print(
                f"MODULE={spec.module} CONFIG={g['config']} "
                f"REPORTERV2_TRAINING_ID={g['training_id']} {g['cli']}"
            )
        return

    results = collect(spec)
    url = build_report(spec, results=results)
    _, transferred = hp_table(spec, results)
    print(f"transferred muP lr = {transferred}")
    print(f"report -> {url}")


if __name__ == "__main__":
    main()
