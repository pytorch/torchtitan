# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Loss-vs-compute extrapolation showcase (the plan's Stage 1).

Fit a Chinchilla-style curve ``L(C) = E + A * C**(-alpha)`` on cheap small-rung
checkpoints, predict a held-out larger rung's post-decay loss, then run that rung
once to verify. WSD-S saves a post-decay checkpoint at every Chinchilla period,
so a single ``1xC`` run yields two fit points (at 0.5xC and 1xC).

Runs are spawned sequentially via torchrun (each uses the rung's full GPU
allocation); already-complete runs are skipped, so a campaign is resumable.
``C4`` training/validation stream from the HF hub, so the launching environment
must have the fwdproxy ``https_proxy``/``http_proxy`` set.
"""

import argparse
import json

import numpy as np
from scipy.optimize import curve_fit

from . import LADDER
from .ladder import _spawn_run


def _chinchilla(compute, asymptote, scale, exponent):
    return asymptote + scale * np.power(compute, -exponent)


def fit_loss_vs_compute(computes: list[float], losses: list[float]) -> dict:
    """Fit ``L(C) = E + A * (C / c_ref)**(-alpha)``; returns the params and RMSE.

    Compute is normalized by its geometric mean (``c_ref``) before fitting so the
    optimizer sees O(1) values -- the raw ~1e17-1e20 FLOPs are too ill-conditioned
    for ``curve_fit`` and land in a degenerate (E ~ 0) local minimum. ``predict``
    rescales by the same ``c_ref``.
    """
    computes = np.asarray(computes, dtype=float)
    losses = np.asarray(losses, dtype=float)
    c_ref = float(np.exp(np.mean(np.log(computes))))
    normalized = computes / c_ref
    floor = float(losses.min())
    spread = float(losses.max() - floor) or 1.0
    p0 = (0.9 * floor, spread, 0.3)
    bounds = ((0.0, 0.0, 0.0), (floor, np.inf, 2.0))
    (asymptote, scale, exponent), _ = curve_fit(
        _chinchilla, normalized, losses, p0=p0, bounds=bounds, maxfev=200000
    )
    residuals = _chinchilla(normalized, asymptote, scale, exponent) - losses
    return {
        "E": float(asymptote),
        "A": float(scale),
        "alpha": float(exponent),
        "c_ref": c_ref,
        "rmse": float(np.sqrt(np.mean(residuals**2))),
    }


def predict_loss(fit: dict, compute: float) -> float:
    return float(
        _chinchilla(float(compute) / fit["c_ref"], fit["E"], fit["A"], fit["alpha"])
    )


def matched_points(ladder, rung: str, metric: str, overrides: dict) -> list[dict]:
    """One (compute, loss) point per post-decay checkpoint with ``metric`` present.

    ``compute`` is the ~6ND training FLOPs (non-embedding params x tokens).
    """
    plan = ladder._resolve(rung, overrides)
    points = []
    for record in ladder.metrics(rung, **overrides)["checkpoints"]:
        if record["phase"] != "post-decay" or record.get(metric) is None:
            continue
        points.append(
            {
                "rung": rung,
                "chinchilla_multiple": record["chinchilla_multiple"],
                "params": plan.ladder_params,
                "tokens": record["tokens"],
                "compute": 6.0 * plan.ladder_params * record["tokens"],
                "loss": record[metric],
            }
        )
    return points


def run_rungs(
    ladder, rungs: list[str], overrides: dict, *, compile: bool = True
) -> None:
    """Run each rung once (sequentially, skipping complete runs)."""
    extra_flags = ["--compile"] if compile else []
    for rung in rungs:
        plan = ladder._resolve(rung, overrides)
        if plan.steps in ladder.status(rung, **overrides)["checkpoint_steps_present"]:
            continue  # final checkpoint exists -> already complete
        _spawn_run(rung, overrides, ladder.compute_for(rung).world_size, extra_flags)


def extrapolate(
    ladder,
    *,
    fit_rungs: list[str],
    held_out_rungs: list[str],
    chinchilla_multiple: float = 1.0,
    metric: str = "val_loss",
    execute: bool = True,
    compile: bool = True,
) -> dict:
    """Fit on ``fit_rungs``, predict & verify each held-out rung's post-decay loss."""
    overrides = {"chinchilla_multiple": float(chinchilla_multiple)}
    if execute:
        run_rungs(ladder, fit_rungs + held_out_rungs, overrides, compile=compile)

    fit_points = [
        p for r in fit_rungs for p in matched_points(ladder, r, metric, overrides)
    ]
    fit = fit_loss_vs_compute(
        [p["compute"] for p in fit_points], [p["loss"] for p in fit_points]
    )

    validations = []
    for rung in held_out_rungs:
        for point in matched_points(ladder, rung, metric, overrides):
            predicted = predict_loss(fit, point["compute"])
            validations.append(
                {
                    "rung": rung,
                    "chinchilla_multiple": point["chinchilla_multiple"],
                    "compute": point["compute"],
                    "predicted_loss": predicted,
                    "actual_loss": point["loss"],
                    "relative_error": abs(predicted - point["loss"]) / point["loss"],
                }
            )

    return {
        "metric": metric,
        "fit": fit,
        "fit_points": fit_points,
        "validations": validations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Loss-vs-compute extrapolation showcase."
    )
    parser.add_argument("--fit-rungs", default="60M,100M,190M,370M")
    parser.add_argument("--held-out-rungs", default="760M")
    parser.add_argument("--chinchilla-multiple", type=float, default=1.0)
    parser.add_argument("--metric", default="val_loss")
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="Analyze existing runs only; do not launch training.",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile for the spawned training runs.",
    )
    args = parser.parse_args()
    result = extrapolate(
        LADDER,
        fit_rungs=args.fit_rungs.split(","),
        held_out_rungs=args.held_out_rungs.split(","),
        chinchilla_multiple=args.chinchilla_multiple,
        metric=args.metric,
        execute=not args.no_execute,
        compile=not args.no_compile,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
