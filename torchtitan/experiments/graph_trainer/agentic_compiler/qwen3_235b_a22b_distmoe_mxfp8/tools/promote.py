# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Promote a verified candidate into the substitute-kernels cache (coordinator only).

    python promote.py <pattern prefix, e.g. pat_5dmq> <candidate.py> <expected sha256 prefix> <tag>

Backs up an existing <kernel_name> into autokernel/run1/rollback/, writes the candidate to a temporary file in
the pattern directory, then os.replace()s it onto <kernel_name>. Appends a line to autokernel/run1/promotions.log.
"""

import glob
import hashlib
import json
import os
import shutil
import sys
import time

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    prefix, cand, sha_prefix, tag = sys.argv[1:5]
    (pat,) = glob.glob(os.path.join(W, "kernel_cache", prefix + "*"))
    (spec,) = glob.glob(os.path.join(pat, "specialization_*.json"))
    name = json.load(open(spec))["kernel_name"]
    data = open(cand, "rb").read()
    sha = hashlib.sha256(data).hexdigest()
    assert sha.startswith(sha_prefix), (sha, sha_prefix)
    dst = os.path.join(pat, name)
    if os.path.exists(dst):
        rb = os.path.join(
            W,
            "autokernel/run1/rollback",
            f"{os.path.basename(pat)[:12]}_{int(time.time())}_{name}",
        )
        os.makedirs(os.path.dirname(rb), exist_ok=True)
        shutil.copy2(dst, rb)
        print("backup:", rb)
    tmp = dst + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dst)
    line = f"{time.strftime('%F %T')} {tag} {os.path.basename(pat)[:12]}/{name} sha256 {sha[:16]} <- {cand}"
    with open(os.path.join(W, "autokernel/run1/promotions.log"), "a") as f:
        f.write(line + "\n")
    print(line)


if __name__ == "__main__":
    main()
