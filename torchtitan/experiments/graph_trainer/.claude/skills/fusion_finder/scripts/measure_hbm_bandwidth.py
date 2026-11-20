# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Measure the HBM bandwidth that ideal glue times are computed with.

    CUDA_VISIBLE_DEVICES=<idle GPU> python measure_hbm_bandwidth.py <capture dir>

Times device-to-device copies of 1, 4 and 8 GiB and writes the fastest rate
(bytes read plus bytes written per second) to ``<capture dir>/hbm_bandwidth.json``.
Run it on an idle GPU of the type the capture ran on.
"""

import argparse
import json
import os

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir")
    args = parser.parse_args()
    iters = 10
    best_gbps = 0.0
    for gib in (1, 4, 8):
        src = torch.empty(gib << 30, dtype=torch.uint8, device="cuda")
        dst = torch.empty_like(src)
        dst.copy_(src)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            dst.copy_(src)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / iters
        best_gbps = max(best_gbps, 2 * src.nbytes / ms / 1e6)
        del src, dst
    result = {"device": torch.cuda.get_device_name(), "hbm_gbps": round(best_gbps, 1)}
    with open(os.path.join(args.capture_dir, "hbm_bandwidth.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(result)


if __name__ == "__main__":
    main()
