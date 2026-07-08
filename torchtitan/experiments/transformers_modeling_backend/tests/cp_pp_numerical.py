# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Logit-level numerical equivalence check for CP + PP in the HF backend.

Loss comparison is insufficient to validate CP+PP (confounded by data/batch,
and across world sizes by weight init). This checks logits directly:

  1. Create a seed checkpoint (identical weights for every layout -- from-
     scratch init is world-size dependent, so this is required).
  2. Run CP-only (cp=2, pp=1) loading the seed; dump per-shard logits.
  3. Run CP+PP  (cp=2, pp=2) loading the seed; dump per-microbatch logits.
  4. Assert every CP-only sample matches some CP+PP microbatch (per CP shard).

Runs the flex (ptrr) and SDPA (headtail) paths. Kept fast (small seq, 1 step,
fp32 so CP/PP reduction noise isn't hidden by bf16). Needs 4 GPUs; self-skips
otherwise. Logits are dumped by the ``HF_BACKEND_LOGIT_DUMP`` hook in
HFTransformerModel.forward.

The real CP+PP path can only be exercised through the trainer/pipeline runtime,
so this orchestrates subprocess launches at different GPU counts (1/2/4) -- it
cannot be expressed in the single-``ngpu`` OverrideDefinitions framework.

Usage: python -m torchtitan.experiments.transformers_modeling_backend.tests.cp_pp_numerical [--cases flex sdpa]
"""

import argparse
import glob
import os
import subprocess
import sys
import tempfile

import torch

_MODULE = "transformers_modeling_backend"
# cp=2; seq_len 256 -> 2 flex Q-blocks so ptrr (blocks % cp == 0) holds; 1 step;
# fp32 so CP/PP reduction-order noise isn't masked by bf16. Small on purpose.
_COMMON = (
    "--parallelism.context_parallel_degree 2 "
    "--training.local_batch_size 4 --training.seq_len 256 --training.steps 1 "
    "--training.mixed_precision_param float32 --debug.seed 42 --debug.deterministic"
)
_CASES = {
    # name: (config, cp load balancer)
    "flex": ("transformers_modeling_backend_debugmodel_flex", "ptrr"),
    "sdpa": ("transformers_modeling_backend_debugmodel", "headtail"),
}
_TOL = 2e-2  # bf16/flex reduction-order noise (fp32 run is ~5e-7 in practice)


def _run(cmd: str, env: dict | None = None) -> None:
    full_env = {**os.environ, **(env or {})}
    result = subprocess.run(
        [cmd],
        shell=True,
        env=full_env,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        print(result.stdout)
        raise RuntimeError(f"Command failed (rc={result.returncode}): {cmd}")


def _torchrun(ngpu: int, config: str, extra: str) -> str:
    return (
        f"torchrun --nproc_per_node={ngpu} --role rank -m torchtitan.train "
        f"--module {_MODULE} --config {config} {extra}"
    )


def _load_by_cp_coord(dump_dir: str) -> dict[int, list[torch.Tensor]]:
    by_coord: dict[int, list[torch.Tensor]] = {}
    files = glob.glob(f"{dump_dir}/logits_rank*.pt")
    if not files:
        raise RuntimeError(f"No logit dumps found in {dump_dir}")
    for f in files:
        for cp_coord, logits in torch.load(f, weights_only=True):
            by_coord.setdefault(cp_coord, []).append(logits.float())
    return by_coord


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item() / max(a.abs().max().item(), 1e-9)


def _compare(ref_dir: str, cp_pp_dir: str) -> None:
    """Every CP-only sample must match some CP+PP microbatch, per CP shard.

    Matching by best rel naturally ignores the pipeline shape-inference "dummy"
    forward and any microbatch reordering (no hardcoded skip).
    """
    ref = _load_by_cp_coord(ref_dir)
    cand = _load_by_cp_coord(cp_pp_dir)
    if sorted(ref) != sorted(cand):
        raise RuntimeError(f"CP coords differ: ref={sorted(ref)} cp_pp={sorted(cand)}")

    worst = 0.0
    for c in sorted(ref):
        ref_samples = [s.unsqueeze(0) for t in ref[c] for s in t]
        cand_forwards = cand[c]
        for i, rs in enumerate(ref_samples):
            best = min(_rel(rs, cf) for cf in cand_forwards if cf.shape == rs.shape)
            worst = max(worst, best)
            status = "ok" if best < _TOL else "FAIL"
            print(f"  cp{c} ref-sample{i}: best CP+PP match rel={best:.3e} [{status}]")
            if best >= _TOL:
                raise RuntimeError(
                    f"cp{c} sample{i}: no CP+PP microbatch within tol "
                    f"(best rel={best:.3e}, tol={_TOL})"
                )
    print(f"  PASS (worst rel={worst:.3e}, tol={_TOL})")


def _run_case(name: str, work: str) -> None:
    config, balancer = _CASES[name]
    print(f"\n==== CP+PP numerical: {name} (config={config} balancer={balancer}) ====")
    seed = os.path.join(work, f"seed_{name}")
    co, pp = os.path.join(work, f"co_{name}"), os.path.join(work, f"pp_{name}")
    os.makedirs(co, exist_ok=True)
    os.makedirs(pp, exist_ok=True)

    print("  [1/4] seed checkpoint")
    _run(
        _torchrun(
            1,
            config,
            "--checkpoint.enable --checkpoint.create_seed_checkpoint "
            "--parallelism.data_parallel_shard_degree 1 "
            "--parallelism.tensor_parallel_degree 1 "
            "--parallelism.pipeline_parallel_degree 1 "
            "--parallelism.context_parallel_degree 1 "
            "--parallelism.expert_parallel_degree 1 "
            f"--dump_folder {seed}",
        ),
    )
    load = (
        f"--checkpoint.enable --checkpoint.initial_load_path {seed}/checkpoint/step-0"
    )
    bal = f"--parallelism.context_parallel_load_balancer {balancer}"

    print("  [2/4] CP-only run (cp=2, pp=1)")
    _run(
        _torchrun(
            2,
            config,
            f"{_COMMON} {load} {bal} --parallelism.data_parallel_shard_degree 1 "
            f"--dump_folder {os.path.join(work, f'out_co_{name}')}",
        ),
        env={"HF_BACKEND_LOGIT_DUMP": co},
    )

    print("  [3/4] CP+PP run (cp=2, pp=2)")
    _run(
        _torchrun(
            4,
            config,
            f"{_COMMON} {load} {bal} --parallelism.pipeline_parallel_degree 2 "
            f"--parallelism.pipeline_parallel_schedule 1F1B "
            f"--dump_folder {os.path.join(work, f'out_pp_{name}')}",
        ),
        env={"HF_BACKEND_LOGIT_DUMP": pp},
    )

    print("  [4/4] compare logits")
    _compare(co, pp)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", nargs="+", default=list(_CASES), choices=list(_CASES)
    )
    args = parser.parse_args()

    n = torch.cuda.device_count()
    if n < 4:
        print(f"SKIP: CP+PP numerical test needs 4 GPUs, found {n}")
        return

    with tempfile.TemporaryDirectory() as work:
        for name in args.cases:
            _run_case(name, work)
    print("\nALL CP+PP numerical checks PASSED")


if __name__ == "__main__":
    sys.exit(main())
