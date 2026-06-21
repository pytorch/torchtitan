# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Driver: iso-quality throughput campaign for a chosen performance variable.

Trains a baseline arm and a variant arm of the scaling ladder fresh and packed
onto the node (small rungs on 1 GPU, larger ones on 2-4 so spare GPUs cut
wall-clock), fits each arm's loss-vs-compute curve on the small rungs, predicts
and validates the held-out 760M, overlays the curves (the iso-quality check), and
compares throughput.

Both arms run at the SAME per-rung GPU count, so each rung's baseline/variant pair
shares a data-parallel degree (hence global batch) and is cleanly comparable; the
performance knob (attention kernel, fp8) does not change the schedule.

    --variant flex_flash : flex attention vs the FlexAttention FLASH kernel.
    --variant fp8        : bf16 vs rowwise fp8 linears (skips lm_head).

    # run (needs fwdproxy for C4 streaming):
    https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080 \
      python -m torchtitan.experiments.scaling_ladders.run_perf_campaign \
        --variant flex_flash --phase run
    # analyze + plot:
    python -m torchtitan.experiments.scaling_ladders.run_perf_campaign \
        --variant flex_flash --phase analyze
"""

import argparse
from dataclasses import replace

from . import LADDER
from .concurrent import Job, run_jobs_concurrent
from .planner import auto_compute_spec

FIT_RUNGS = ["60M", "100M", "190M", "370M"]
HELD_OUT = ["760M"]
# Per-rung GPU width (same for both arms). Bigger rungs get more GPUs so the
# makespan, set by 760M, stays small; small rungs pack 1-per-GPU.
GPUS = {"60M": 1, "100M": 1, "190M": 2, "370M": 2, "760M": 4}
OVERRIDES = {"chinchilla_multiple": 1.0}  # 0.5xC and 1xC post-decay -> 2 points/run

# Each variant maps to (baseline_label, variant_label) output subdirs and the
# per-job kwargs that select the arm.
VARIANTS = {
    "varlen": {
        "labels": ("flex", "varlen"),
        "baseline": {"attn_backend": "flex"},
        "variant": {"attn_backend": "varlen"},
    },
    "flex_flash": {
        "labels": ("flex", "flex_flash"),
        "baseline": {"attn_backend": "flex"},
        "variant": {"attn_backend": "flex_flash"},
    },
    "fp8": {
        "labels": ("bf16", "fp8"),
        "baseline": {},
        "variant": {"fp8": True},
    },
}


def _root(variant: str) -> str:
    return f"./outputs/scaling_ladders_{variant}campaign"


def _arms(variant: str):
    spec = VARIANTS[variant]
    root = _root(variant)
    base_label, var_label = spec["labels"]
    return (
        (base_label, f"{root}/{base_label}", spec["baseline"]),
        (var_label, f"{root}/{var_label}", spec["variant"]),
    )


def _jobs(variant: str) -> list[Job]:
    jobs = []
    for rung in FIT_RUNGS + HELD_OUT:
        for _label, folder, kw in _arms(variant):
            jobs.append(
                Job(
                    rung,
                    GPUS[rung],
                    overrides=dict(OVERRIDES),
                    base_dump_folder=folder,
                    **kw,
                )
            )
    return jobs


def _arm_ladder(base_dump_folder: str):
    """Ladder whose per-rung ComputeSpec matches the campaign GPU widths.

    The metrics readback keys validation off the resolved post-decay steps, so the
    analysis ladder must resolve each rung at the GPU count it was trained on.
    """
    specs = {
        rung: auto_compute_spec(rung, LADDER.policy, gpus=GPUS[rung])
        for rung in FIT_RUNGS + HELD_OUT
    }
    return replace(LADDER, base_dump_folder=base_dump_folder, compute_specs=specs)


def run(variant: str, total_gpus: int = 8) -> None:
    ladder = replace(LADDER, compile=True)  # both arms compile (fair, and fp8 needs it)
    results = run_jobs_concurrent(
        ladder, _jobs(variant), total_gpus=total_gpus, poll_secs=60
    )
    print(f"\n=== {variant} CAMPAIGN RESULTS ===")
    for r in results:
        print(f"{r['job'].rung:>5} {r['run_dir']:>55} -> {r['status']}")


def analyze(variant: str) -> None:
    from . import showcase

    (base_label, base_dir, _), (var_label, var_dir, _) = _arms(variant)
    base, var = _arm_ladder(base_dir), _arm_ladder(var_dir)
    base_res = showcase.extrapolate(
        base, fit_rungs=FIT_RUNGS, held_out_rungs=HELD_OUT, execute=False
    )
    var_res = showcase.extrapolate(
        var, fit_rungs=FIT_RUNGS, held_out_rungs=HELD_OUT, execute=False
    )
    curves = {base_label: base_res, var_label: var_res}
    root = _root(variant)
    loss_png = showcase.plot_loss_vs_compute(curves, f"{root}/loss_vs_compute.png")
    perf = showcase.compare_perf(base, var, FIT_RUNGS + HELD_OUT, overrides=OVERRIDES)
    perf_png = showcase.plot_perf(curves, perf["perf"], f"{root}/perf.png")

    print("\n=== 760M EXTRAPOLATION (predict from small rungs, validate at 760M) ===")
    for arm, res in curves.items():
        for v in res["validations"]:
            print(
                f"{arm}: {v['rung']} @ {v['chinchilla_multiple']}xC  "
                f"predicted={v['predicted_loss']:.4f}  actual={v['actual_loss']:.4f}  "
                f"rel_err={v['relative_error']:.2%}"
            )
    print(f"\n=== THROUGHPUT ({var_label} vs {base_label}, matched points) ===")
    for p in perf["perf"]:
        print(
            f"{p['rung']:>5}: {base_label}={p['baseline_tps']:.0f}  "
            f"{var_label}={p['variant_tps']:.0f}  speedup={p['speedup']:.2f}x"
        )
    print(
        f"\nmean {var_label} speedup: {perf['mean_speedup']:.3f}x | "
        f"worst quality regression: {perf['worst_quality_regression']} nats "
        f"(eps={perf['quality_eps']}) | iso-quality: {perf['iso_quality']} | "
        f"VERDICT: {perf['verdict']}"
    )
    print(f"\nplots: {loss_png}\n       {perf_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Iso-quality throughput campaign.")
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--phase", choices=["run", "analyze"], required=True)
    parser.add_argument("--total-gpus", type=int, default=8)
    args = parser.parse_args()
    if args.phase == "run":
        run(args.variant, total_gpus=args.total_gpus)
    else:
        analyze(args.variant)


if __name__ == "__main__":
    main()
