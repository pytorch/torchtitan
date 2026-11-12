# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Where the glue goes, base vs pack: kernel names grouped by what they do.

    python glue_groups.py <model>=<base run dir>,<pack run dir> [...]
"""
import collections
import re
import sys

from roofline import classify

GROUPS = [
    ("optimizer step / foreach (multi_tensor_apply)", r"multi_tensor_apply"),
    (
        "deterministic index ops + sorts",
        r"indexing_backward|RadixSort|radix_sort|bitonicSort|cub::",
    ),
    (
        "FSDP copy-in / copy-out, bucket packs, cats",
        r"split_with_sizes_copy|chunk_cat|CatArrayBatched|all_gather_copy_in",
    ),
    ("memset / memcpy", r"^Memset|^Memcpy"),
    ("MXFP8 quantize / scale swizzle", r"mxfp8|quantize|swizzle|to_mx"),
    ("Dist-MoE / MoE glue kernels (triton)", r"_triton_"),
    ("RoPE / norm kernels", r"rope|rms|norm"),
    ("loop's fused kernels", r"_substitute_kernels_|_kernel_pack_pat_"),
    ("other elementwise / reduction / copy", r"."),
]


def groups(run):
    _, names = classify(run)
    out = collections.Counter()
    for (cat, name), ms in names.items():
        if not (cat.startswith("glue") or cat == "fused by the loop"):
            continue
        g = next(g for g, p in GROUPS if re.search(p, name))
        out[g] += ms
    return out


for spec in sys.argv[1:]:
    model, runs = spec.split("=", 1)
    b, p = (groups(r) for r in runs.split(","))
    print(f"\n== {model}: glue {sum(b.values()):.1f} -> {sum(p.values()):.1f} ms")
    for g, _ in GROUPS:
        print(f"  {g:48s} {b[g]:8.1f} -> {p[g]:8.1f}  ({p[g] - b[g]:+7.1f})")
