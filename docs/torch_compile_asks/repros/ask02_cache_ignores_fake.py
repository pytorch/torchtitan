# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inductor's on-disk caches (FX graph / AOTAutograd) do not key on a custom op's fake (meta)
implementation. Fixing a fake whose output strides were wrong leaves cached code generated from the
old fake in use until the cache directory changes.

The parent runs three child processes:
  1. fake v1 (wrong strides), fresh cache dir A   -> the compiled code's stride assert fails
  2. fake v2 (fixed),         same cache dir A     -> still fails: code from v1 is reused
  3. fake v2 (fixed),         fresh cache dir B    -> passes

Expected on 1x GB300 (torch 2.15.0.dev20260926+cu130, triton 3.8.0; see ../README.md, ask 2):
fake v1 fresh cache: stride AssertionError; fake v2 same cache: same AssertionError; fake v2 fresh cache: ok.
"""
import os
import subprocess
import sys
import tempfile

if len(sys.argv) > 1:
    import torch

    version = sys.argv[1]

    @torch.library.custom_op("repro::copy_contiguous", mutates_args=())
    def copy_contiguous(x: torch.Tensor) -> torch.Tensor:
        return x.contiguous().clone()  # real kernel: contiguous output

    @copy_contiguous.register_fake
    def _(x):
        # v1 claims the input's (transposed) strides, like a fake built on empty_like(input)
        return (
            torch.empty_like(x)
            if version == "v1"
            else torch.empty(x.shape, dtype=x.dtype, device=x.device)
        )

    def fn(x):
        return copy_contiguous(x * 2) + 1

    x = torch.randn(64, 32, device="cuda").t()
    try:
        torch.compile(fn, fullgraph=True)(x)
        print(f"fake {version}: ok")
    except Exception as e:
        print(f"fake {version}: {type(e).__name__}: {str(e).splitlines()[0][:110]}")
    sys.exit(0)

dir_a, dir_b = tempfile.mkdtemp(), tempfile.mkdtemp()
for version, cache_dir, label in (
    ("v1", dir_a, "fresh cache A"),
    ("v2", dir_a, "same cache A"),
    ("v2", dir_b, "fresh cache B"),
):
    env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=cache_dir)
    out = subprocess.run(
        [sys.executable, __file__, version], env=env, capture_output=True, text=True
    )
    lines = [line for line in out.stdout.splitlines() if line.startswith("fake")]
    print(
        f"{label:14s} -> {lines[-1] if lines else out.stderr.strip().splitlines()[-1][:120]}"
    )
