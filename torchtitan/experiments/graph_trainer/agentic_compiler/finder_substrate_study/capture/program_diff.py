# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Program-difference table: GPU time by category for each model's eager and graph capture.

    python program_diff.py <captures dir> <out.json>

Categories come from the glue-roofline classifier (outputs/agentic_compiler/glue_roofline/roofline.py):
communication (launched under a collective, fake-PG emulation), attention, GEMM, glue subgroups. Also
reports the optimizer's share (multi_tensor_apply) and the number of kernels.
"""

import json
import os
import sys

sys.path.insert(
    0, "/home/bahuang/local/torchtitan/outputs/agentic_compiler/glue_roofline"
)
import roofline  # noqa: E402


def main():
    caps, out = sys.argv[1], sys.argv[2]
    result = {}
    for model in ("llama3_8b", "qwen3_30b", "dsv3"):
        row = {}
        for arm in ("eager", "graph"):
            run = os.path.join(caps, f"{model}_{arm}")
            if not os.path.exists(os.path.join(run, "traces")):
                continue
            cat_ms, _ = roofline.classify(run)
            step, glue = roofline.summary(cat_ms)
            row[arm] = {
                "categories_ms": {k: round(v, 2) for k, v in sorted(cat_ms.items())},
                "step_ms_excl_comm": round(step, 2),
                "glue_ms": round(glue, 2),
                "glue_ms_excl_optimizer": round(
                    glue - cat_ms["glue: multi_tensor_apply"], 2
                ),
            }
        result[model] = row
        print(f"\n== {model}")
        cats = sorted({c for a in row.values() for c in a["categories_ms"]})
        print(f"  {'category':40s} " + " ".join(f"{a:>10s}" for a in row))
        for c in cats:
            print(
                f"  {c:40s} "
                + " ".join(f"{row[a]['categories_ms'].get(c, 0):10.1f}" for a in row)
            )
        for k in ("step_ms_excl_comm", "glue_ms", "glue_ms_excl_optimizer"):
            print(f"  {k:40s} " + " ".join(f"{row[a][k]:10.1f}" for a in row))
    json.dump(result, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
