# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

from .routine import (
    build_report,
    collect,
    collect_arrays,
    collect_scaling,
    hp_table,
    scaling_report,
)
from .spec import SPECS

USAGE = (
    "usage: python -m torchtitan.experiments.mup {collect|report} <model> [manifest]\n"
    "       python -m torchtitan.experiments.mup scaling <model> <w256=job,...> [bigvit_width]\n"
    "       python -m torchtitan.experiments.mup suite <model> [--arrays w256=J,...] [--big-array J] [--ckpt DIR]"
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
    for width, loss, n, _runs in sorted(points):
        print(f"  w{width}: loss={'N/A' if loss is None else round(loss, 4)} seeds={n}")
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
        for width, loss, n, _runs in sorted(points):
            print(f"w{width}: loss={'N/A' if loss is None else round(loss, 4)} seeds={n}")
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
