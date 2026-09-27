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

Runs the flex (ptrr) attention path. Kept fast (small seq, 1 step, fp32 so
CP/PP reduction noise isn't hidden by bf16). Needs 4 GPUs; self-skips
otherwise. Logits are dumped by the ``HF_BACKEND_LOGIT_DUMP`` hook in
HFTransformerModel.forward.

The real CP+PP path can only be exercised through the trainer/pipeline runtime,
so this orchestrates subprocess launches at different GPU counts (1/2/4) -- it
cannot be expressed in the single-``ngpu`` IntegrationTestDefinition framework.

Usage: python -m torchtitan.experiments.transformers_modeling_backend.tests.cp_pp_numerical
"""

import glob
import os
import subprocess
import sys
import tempfile

import torch

from scripts._checkpoint_test_config import configure_checkpoint

from torchtitan.experiments.transformers_modeling_backend.config_registry import (
    transformers_modeling_backend_debugmodel,
)

_MODULE = "torchtitan.experiments.transformers_modeling_backend.tests.cp_pp_numerical"
_TOL = 2e-2  # bf16/flex reduction-order noise (fp32 run is ~5e-7 in practice)


def _numerics_config():
    # seq_len=256 gives two flex Q blocks, so ptrr is divisible by CP=2.
    config = transformers_modeling_backend_debugmodel(
        seq_len=256,
        deterministic=True,
    )
    config.training.steps = 1
    config.training.mixed_precision_param = "float32"
    config.debug.seed = 42
    return config


def hf_backend_seed_numerics():
    return _numerics_config()


def hf_backend_cp_numerics():
    config = _numerics_config()
    config.training.num_tokens_per_microbatch_per_dp_rank = 1024
    config.parallelism.context_parallel_degree = 2
    config.parallelism.context_parallel_load_balancer = "ptrr"
    return config


def hf_backend_cp_pp_numerics():
    config = _numerics_config()
    config.training.num_tokens_per_microbatch_per_dp_rank = 256
    config.parallelism.context_parallel_degree = 2
    config.parallelism.context_parallel_load_balancer = "ptrr"
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 4
    config.parallelism.pipeline_parallel_schedule = "1F1B"
    return config


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


def _torchrun(ngpu: int, module: str, config: str, output_dir: str) -> str:
    return (
        f"torchrun --nproc_per_node={ngpu} --role rank -m torchtitan.train "
        f"--module {module} --config {config} --output-dir {output_dir}"
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


def _run_case(work: str) -> None:
    print("\n==== CP+PP numerical (balancer=ptrr) ====")
    seed = os.path.join(work, "seed")
    co, pp = os.path.join(work, "co"), os.path.join(work, "pp")
    os.makedirs(co, exist_ok=True)
    os.makedirs(pp, exist_ok=True)

    print("  [1/4] seed checkpoint")
    seed_env: dict[str, str] = {}
    seed_module, seed_config = configure_checkpoint(
        seed_env,
        module=_MODULE,
        config="hf_backend_seed_numerics",
        mode="seed",
    )
    _run(
        _torchrun(
            1,
            seed_module,
            seed_config,
            seed,
        ),
        env=seed_env,
    )
    load_env: dict[str, str] = {}
    load_module, load_config = configure_checkpoint(
        load_env,
        module=_MODULE,
        config="hf_backend_cp_numerics",
        mode="load",
        initial_load_path=f"{seed}/checkpoint/step-0",
    )
    print("  [2/4] CP-only run (cp=2, pp=1)")
    _run(
        _torchrun(
            2,
            load_module,
            load_config,
            os.path.join(work, "out_co"),
        ),
        env={**load_env, "HF_BACKEND_LOGIT_DUMP": co},
    )

    pp_env: dict[str, str] = {}
    pp_module, pp_config = configure_checkpoint(
        pp_env,
        module=_MODULE,
        config="hf_backend_cp_pp_numerics",
        mode="load",
        initial_load_path=f"{seed}/checkpoint/step-0",
    )
    print("  [3/4] CP+PP run (cp=2, pp=2)")
    _run(
        _torchrun(
            4,
            pp_module,
            pp_config,
            os.path.join(work, "out_pp"),
        ),
        env={**pp_env, "HF_BACKEND_LOGIT_DUMP": pp},
    )

    print("  [4/4] compare logits")
    _compare(co, pp)


def main() -> None:
    n = torch.cuda.device_count()
    if n < 4:
        print(f"SKIP: CP+PP numerical test needs 4 GPUs, found {n}")
        return

    with tempfile.TemporaryDirectory() as work:
        _run_case(work)
    print("\nALL CP+PP numerical checks PASSED")


if __name__ == "__main__":
    sys.exit(main())
