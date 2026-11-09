# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compare the kernel contracts the eager Functions use for Qwen3 30B-A3B and 235B-A22B (step 5).

    python compare_contracts.py

For each 30B kernel ``graph_kernels.py`` calls and the 235B loop's kernel for the same region:
the inputs' and outputs' dtypes and ranks, and whether ``reference.py`` (op order and argument
wiring) is identical once numeric literals (sizes, constants) are replaced.
"""

import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
K30 = os.path.join(HERE, "../qwen3_30b_a3b_distmoe_mxfp8/kernel_cache")
K235 = os.path.join(HERE, "../qwen3_235b_a22b_distmoe_mxfp8/kernel_cache")
# (Function, region(s) 30B / 235B, 30B pattern, 235B pattern)
PAIRS = [
    ("attention input", "G-41", "kbxm", "xf3v"),
    ("attention input", "G-02 / G-102", "4dzo", "xfyh"),
    ("attention input", "G-01", "qnfe", "bjai"),
    ("attention input", "G-15", "tl5t", "4j6e"),
    ("attention output", "G-28", "vf5c", "vf5c"),
    ("attention output", "G-45", "na2k", "pnge"),
    ("attention output", "G-19", "zfby", "2nu2"),
    ("router", "G-47", "5ifh", "5ifh"),
    ("router", "G-17", "bp6b", "bp6b"),
    ("router", "G-91", "bj3b", "nh2i"),
    ("router", "G-06", "s3of", "llq3"),
    ("loss", "G-90", "bme7", "bme7"),
    ("loss", "G-07", "53aq", "53aq"),
    ("expert weights", "G-33 / G-115", "gri3", "pqle"),
    ("expert weights", "G-42 / G-129", "adoe", "gbfx"),
]


def load(root, prefix):
    (pat,) = glob.glob(os.path.join(root, f"pat_{prefix}*"))
    (spec_path,) = glob.glob(os.path.join(pat, "specialization_*.json"))
    spec = json.load(open(spec_path))
    outputs = (
        spec["outputs"] if isinstance(spec["outputs"], list) else [spec["outputs"]]
    )

    def sig(tensors):
        return [f"{t['dtype']}/{len(t['shape'])}d" for t in tensors]

    reference = open(os.path.join(pat, "reference.py")).read()
    reference = re.sub(r"(?<![\w.])-?\d+(\.\d+)?(e-?\d+)?\b", "N", reference)
    return sig(spec["inputs"]), sig(outputs), reference


def main():
    print(f"{'Function':17s} {'region':13s} {'inputs':8s} {'outputs':8s} reference")
    for function, region, p30, p235 in PAIRS:
        i30, o30, r30 = load(K30, p30)
        i235, o235, r235 = load(K235, p235)
        print(
            f"{function:17s} {region:13s} {'same' if i30 == i235 else 'DIFF':8s} "
            f"{'same' if o30 == o235 else 'DIFF':8s} {'same' if r30 == r235 else 'DIFF'}"
        )
        if i30 != i235 or o30 != o235:
            print(f"    30B  in {i30} out {o30}")
            print(f"    235B in {i235} out {o235}")


if __name__ == "__main__":
    main()
