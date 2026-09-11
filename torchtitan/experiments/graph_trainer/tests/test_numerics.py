# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import importlib.util
import math
import os
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard
from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module
from torch.distributed.tensor.placement_types import _StridedShard
from torch.testing._internal.common_fsdp import FSDPTest

from torchtitan.components.loss import cross_entropy_loss
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.utils import get_spmd_backend, set_spmd_backend
from torchtitan.experiments.graph_trainer import simple_fsdp
from torchtitan.experiments.graph_trainer.simple_fsdp import data_parallel
from torchtitan.models.common.config_utils import DEFAULT_DEBUG_MODEL_SEQ_LEN


STEPS = 20


def run_loss_compare(
    baseline_module: str,
    baseline_config: str,
    test_module: str,
    test_config: str,
    baseline_options: str = "",
    test_options: str = "",
    baseline_ngpus: int = 8,
    test_ngpus: int = 8,
) -> bool:
    """Run loss_compare.py comparing a baseline module against a graph_trainer module.

    Args:
        baseline_module: Module name for baseline (e.g., "llama3").
        baseline_config: Config name for baseline (e.g., "llama3_debugmodel").
        test_module: Module name for test (e.g., "graph_trainer.llama3").
        test_config: Config name for test (e.g., "graph_trainer_llama3_debugmodel").
        baseline_options: Additional CLI options for the baseline run.
        test_options: Additional CLI options for the test run.
        baseline_ngpus: Number of GPUs for the baseline run.
        test_ngpus: Number of GPUs for the test run.

    Returns:
        True if the assertion passed, False otherwise.
    """
    # Use a temp dump folder instead of loss_compare.py's default ("outputs"),
    # which is created relative to cwd and is not writable in CI containers.
    with tempfile.TemporaryDirectory() as job_dump_folder:
        cmd = [
            sys.executable,
            "scripts/loss_compare.py",
            ".",
            ".",
            f"--baseline-module={baseline_module}",
            f"--baseline-config={baseline_config}",
            f"--test-module={test_module}",
            f"--test-config={test_config}",
            "--assert-equal",
            f"--steps={STEPS}",
            f"--baseline-ngpus={baseline_ngpus}",
            f"--test-ngpus={test_ngpus}",
            f"--job-dump-folder={job_dump_folder}",
        ]
        if baseline_options:
            cmd.append(f"--baseline-options={baseline_options}")
        if test_options:
            cmd.append(f"--test-options={test_options}")

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, text=True)
        if result.returncode != 0:
            print("loss_compare.py failed")
        return result.returncode == 0


def run_loss_compare_close(
    baseline_module: str,
    baseline_config: str,
    test_module: str,
    test_config: str,
    baseline_options: str = "",
    test_options: str = "",
    baseline_ngpus: int = 8,
    test_ngpus: int = 8,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> bool:
    """Run loss_compare.py and assert losses are numerically close.

    AutoParallel can choose a different SPMD graph and collective ordering than
    eager, so this checks tight numerical agreement rather than bitwise identity.
    """
    from scripts.loss_compare import extract_losses_from_tensorboard

    with tempfile.TemporaryDirectory() as job_dump_folder:
        cmd = [
            sys.executable,
            "scripts/loss_compare.py",
            ".",
            ".",
            f"--baseline-module={baseline_module}",
            f"--baseline-config={baseline_config}",
            f"--test-module={test_module}",
            f"--test-config={test_config}",
            f"--steps={STEPS}",
            f"--baseline-ngpus={baseline_ngpus}",
            f"--test-ngpus={test_ngpus}",
            f"--job-dump-folder={job_dump_folder}",
        ]
        if baseline_options:
            cmd.append(f"--baseline-options={baseline_options}")
        if test_options:
            cmd.append(f"--test-options={test_options}")

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, text=True)
        if result.returncode != 0:
            print("loss_compare.py failed")
            return False

        baseline_losses = extract_losses_from_tensorboard(
            job_dump_folder, "tb_baseline"
        )
        test_losses = extract_losses_from_tensorboard(job_dump_folder, "tb_test")
        if baseline_losses.keys() != test_losses.keys():
            return False
        max_step = max(
            baseline_losses,
            key=lambda step: abs(baseline_losses[step] - test_losses[step]),
        )
        max_diff = abs(baseline_losses[max_step] - test_losses[max_step])
        print(
            "Max loss difference: "
            f"step={max_step} baseline={baseline_losses[max_step]!r} "
            f"test={test_losses[max_step]!r} diff={max_diff!r}"
        )
        return all(
            math.isclose(
                baseline_losses[step],
                test_losses[step],
                rel_tol=rtol,
                abs_tol=atol,
            )
            for step in baseline_losses
        )


@contextmanager
def _log_rank(log_rank: int) -> Iterator[None]:
    previous_log_rank = os.environ.get("LOG_RANK")
    os.environ["LOG_RANK"] = str(log_rank)
    try:
        yield
    finally:
        if previous_log_rank is None:
            os.environ.pop("LOG_RANK", None)
        else:
            os.environ["LOG_RANK"] = previous_log_rank


def _losses_are_equal(
    baseline_losses: dict[int, float],
    test_losses: dict[int, float],
) -> bool:
    if baseline_losses.keys() != test_losses.keys():
        print(
            "Step mismatch: "
            f"baseline={sorted(baseline_losses)} test={sorted(test_losses)}"
        )
        return False

    for step in sorted(baseline_losses):
        if baseline_losses[step] != test_losses[step]:
            print(
                "Loss mismatch at "
                f"step={step}: baseline={baseline_losses[step]!r} "
                f"test={test_losses[step]!r}"
            )
            return False
    return True


def _extract_losses_from_rank_tensorboard(
    job_dump_folder: str,
    tb_folder: str,
    rank: int,
) -> dict[int, float]:
    from scripts.loss_compare import TB_LOSS_TAG
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    base_path = os.path.join(job_dump_folder, tb_folder)
    timestamp_dirs = [
        path
        for path in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, path))
    ]
    rank_event_dirs = [
        os.path.join(base_path, timestamp_dir, f"rank_{rank}")
        for timestamp_dir in timestamp_dirs
        if os.path.isdir(os.path.join(base_path, timestamp_dir, f"rank_{rank}"))
    ]
    if len(rank_event_dirs) != 1:
        raise RuntimeError(
            f"Expected one TensorBoard rank_{rank} directory under {base_path}, "
            f"found {rank_event_dirs}."
        )

    event_accumulator = EventAccumulator(rank_event_dirs[0])
    event_accumulator.Reload()
    losses = {
        scalar.step: scalar.value for scalar in event_accumulator.Scalars(TB_LOSS_TAG)
    }
    print(f"Extracted {len(losses)} losses from {rank_event_dirs[0]}")
    return losses


LLAMA3_PARALLELISM = (
    "--parallelism.tensor_parallel_degree=2"
    " --parallelism.data_parallel_shard_degree=4"
)
DEBUGMODEL_TRAINING_OPTIONS = (
    f"--training.max_context_length={DEFAULT_DEBUG_MODEL_SEQ_LEN}"
    " --training.num_tokens_per_microbatch_per_dp_rank=16384"
)


def _run_llama3_loss_compare(test_options_extra: str = "") -> bool:
    """Run loss_compare for llama3 vs graph_trainer.llama3 with FSDP+TP."""
    options = f"{LLAMA3_PARALLELISM} {DEBUGMODEL_TRAINING_OPTIONS}"
    test_options = options
    if test_options_extra:
        test_options += f" {test_options_extra}"
    return run_loss_compare(
        baseline_module="llama3",
        baseline_config="llama3_debugmodel",
        test_module="graph_trainer.llama3",
        test_config="graph_trainer_llama3_debugmodel",
        baseline_options=options,
        test_options=test_options,
    )


DSV3_PARALLELISM = (
    "--parallelism.data_parallel_shard_degree=4"
    " --parallelism.tensor_parallel_degree=2"
    " --parallelism.expert_parallel_degree=2"
)
DSV3_EP_OVERLAP_GRAPH_PARALLELISM = (
    "--training.disable_cuda_graphs"
    " --parallelism.data_parallel_shard_degree=8"
    " --parallelism.tensor_parallel_degree=1"
    " --parallelism.expert_parallel_degree=2"
)
DSV3_EP_OVERLAP_OPTIONS = (
    "--compile.mode aot_fx_trace"
    " --compile.ep_overlap.enabled"
    " --compile.ep_overlap.chunk_dim batch"
    " --compile.ep_overlap.module_fqn layers.*"
)
DSV3_EP_OVERLAP_MOE_SEQ_OPTIONS = (
    "--compile.mode aot_fx_trace"
    " --compile.ep_overlap.enabled"
    " --compile.ep_overlap.chunk_dim seq"
    " --compile.ep_overlap.module_fqn layers.*.moe"
)
DSV3_EP_OVERLAP_MOE_BATCH_OPTIONS = (
    "--compile.mode aot_fx_trace"
    " --compile.ep_overlap.enabled"
    " --compile.ep_overlap.chunk_dim batch"
    " --compile.ep_overlap.module_fqn layers.*.moe"
)
DSV3_EP_OVERLAP_EAGER = " --compile.ep_overlap.strategy eager"
DSV3_EP_OVERLAP_GRAPH = " --compile.ep_overlap.strategy graph"
DSV3_EP_OVERLAP_GRAPH_BITWISE = (
    DSV3_EP_OVERLAP_GRAPH + " --compile.ep_overlap.disable_early_grad_accumulation"
)


def _run_deepseek_v3_loss_compare(
    test_options_extra: str = "",
    *,
    baseline_module: str = "deepseek_v3",
    baseline_config: str = "deepseek_v3_debugmodel",
    test_config: str = "graph_trainer_deepseek_v3_debugmodel",
    parallelism: str = DSV3_PARALLELISM,
    baseline_options_extra: str = "",
) -> bool:
    """Run loss_compare for deepseek_v3 vs graph_trainer.deepseek_v3."""
    options = f"{parallelism} {DEBUGMODEL_TRAINING_OPTIONS}"
    baseline_options = options
    if baseline_options_extra:
        baseline_options += f" {baseline_options_extra}"
    test_options = options
    if test_options_extra:
        test_options += f" {test_options_extra}"
    return run_loss_compare(
        baseline_module=baseline_module,
        baseline_config=baseline_config,
        test_module="graph_trainer.deepseek_v3",
        test_config=test_config,
        baseline_options=baseline_options,
        test_options=test_options,
    )


def _run_deepseek_v3_ep_overlap_loss_compare() -> bool:
    """Run distributed DeepSeek-v3 EP overlap against eager chunking."""
    return _run_deepseek_v3_loss_compare(
        baseline_module="graph_trainer.deepseek_v3",
        baseline_config="graph_trainer_deepseek_v3_debugmodel",
        test_config="graph_trainer_deepseek_v3_debugmodel",
        parallelism=DSV3_EP_OVERLAP_GRAPH_PARALLELISM,
        baseline_options_extra=DSV3_EP_OVERLAP_OPTIONS + DSV3_EP_OVERLAP_EAGER,
        test_options_extra=DSV3_EP_OVERLAP_OPTIONS + DSV3_EP_OVERLAP_GRAPH_BITWISE,
    )


def _run_deepseek_v3_ep_overlap_moe_seq_loss_compare() -> bool:
    """Run distributed DeepSeek-v3 MoE seq overlap against eager chunking."""
    return _run_deepseek_v3_loss_compare(
        baseline_module="graph_trainer.deepseek_v3",
        baseline_config="graph_trainer_deepseek_v3_debugmodel",
        test_config="graph_trainer_deepseek_v3_debugmodel",
        parallelism=DSV3_EP_OVERLAP_GRAPH_PARALLELISM,
        baseline_options_extra=DSV3_EP_OVERLAP_MOE_SEQ_OPTIONS + DSV3_EP_OVERLAP_EAGER,
        test_options_extra=DSV3_EP_OVERLAP_MOE_SEQ_OPTIONS
        + DSV3_EP_OVERLAP_GRAPH_BITWISE,
    )


def _run_deepseek_v3_ep_overlap_moe_batch_loss_compare() -> bool:
    """Run distributed DeepSeek-v3 MoE batch overlap against eager chunking."""
    return _run_deepseek_v3_loss_compare(
        baseline_module="graph_trainer.deepseek_v3",
        baseline_config="graph_trainer_deepseek_v3_debugmodel",
        test_config="graph_trainer_deepseek_v3_debugmodel",
        parallelism=DSV3_EP_OVERLAP_GRAPH_PARALLELISM,
        baseline_options_extra=DSV3_EP_OVERLAP_MOE_BATCH_OPTIONS
        + DSV3_EP_OVERLAP_EAGER,
        test_options_extra=DSV3_EP_OVERLAP_MOE_BATCH_OPTIONS
        + DSV3_EP_OVERLAP_GRAPH_BITWISE,
    )


GRAPH_PP_DSV3_PP_OPTIONS = (
    "--training.disable_cuda_graphs"
    " --parallelism.pipeline_parallel_degree=2"
    " --parallelism.num_pp_microbatches=8"
    " --parallelism.data_parallel_shard_degree=4"
    " --parallelism.expert_parallel_degree=2"
    " --training.num_tokens_per_microbatch_per_dp_rank=2048"
    # Eager PP cannot be the baseline for ZBVZeroBubble or DualPipeV here:
    # FlexAttention needs torch.compile, and torch.compile is incompatible with
    # those eager PP schedules. Compare GraphPP schedules against eager
    # Interleaved1F1B instead. TorchTitan gradient clipping is applied per
    # local rank, so different PP schedules can produce different clip
    # coefficients even when pre-clip grads are bitwise equal. Disable clipping
    # to isolate GraphPP graph execution from that schedule-level effect.
    " --training.max_norm=inf"
)


GRAPH_PP_DSV3_TEST_PARALLELISM = (
    "--compile.mode aot_fx_trace"
    " --compile.inductor_compilation regional"
    f" {GRAPH_PP_DSV3_PP_OPTIONS}"
)


def _run_graph_pp_deepseek_v3_loss_compare(schedule: str) -> bool:
    """Run exact loss_compare for eager Interleaved1F1B PP vs GraphPP."""
    from scripts.loss_compare import create_seed_checkpoint, run_training

    baseline_options = (
        f"{GRAPH_PP_DSV3_PP_OPTIONS}"
        " --parallelism.pipeline_parallel_schedule=Interleaved1F1B"
        " --metrics.save_for_all_ranks"
    )
    test_options = (
        f"{GRAPH_PP_DSV3_TEST_PARALLELISM}"
        f" --parallelism.pipeline_parallel_schedule={schedule}"
        " --metrics.save_for_all_ranks"
    )

    baseline_module = "graph_trainer.deepseek_v3"
    baseline_config = "graph_trainer_deepseek_v3_debugmodel_eager_pp"
    test_module = "graph_trainer.deepseek_v3"
    test_config = "graph_trainer_deepseek_v3_debugmodel"
    baseline_tb_folder = "tb_baseline"
    test_tb_folder = "tb_test"
    baseline_loss_rank = 4
    test_loss_rank = 0 if schedule in {"ZBVZeroBubble", "DualPipeV"} else 4

    # loss_compare.py and core metrics choose one logging rank per run. The
    # eager Interleaved1F1B baseline owns loss on the first rank of the last PP
    # stage, while V-style GraphPP schedules own loss on rank 0. Save TB for
    # all ranks in this experiment-local test, then compare the full-precision
    # scalars from the ranks that actually own loss.
    with tempfile.TemporaryDirectory() as job_dump_folder:
        create_seed_checkpoint(
            True,
            baseline_module,
            baseline_config,
            None,
            job_dump_folder,
        )
        with _log_rank(baseline_loss_rank):
            run_training(
                "baseline",
                baseline_module,
                baseline_config,
                baseline_options,
                STEPS,
                True,
                None,
                job_dump_folder,
                8,
                tb_folder=baseline_tb_folder,
            )
        baseline_losses = _extract_losses_from_rank_tensorboard(
            job_dump_folder,
            baseline_tb_folder,
            baseline_loss_rank,
        )

        with _log_rank(test_loss_rank):
            run_training(
                "test",
                test_module,
                test_config,
                test_options,
                STEPS,
                True,
                None,
                job_dump_folder,
                8,
                tb_folder=test_tb_folder,
            )
        test_losses = _extract_losses_from_rank_tensorboard(
            job_dump_folder,
            test_tb_folder,
            test_loss_rank,
        )

    return _losses_are_equal(baseline_losses, test_losses)


QWEN3_PARALLELISM = (
    "--parallelism.tensor_parallel_degree=2"
    " --parallelism.data_parallel_shard_degree=4"
)


def _run_qwen3_loss_compare(test_options_extra: str = "") -> bool:
    """Run loss_compare for qwen3 vs graph_trainer.qwen3 with FSDP+TP."""
    options = f"{QWEN3_PARALLELISM} {DEBUGMODEL_TRAINING_OPTIONS}"
    test_options = options
    if test_options_extra:
        test_options += f" {test_options_extra}"
    return run_loss_compare(
        baseline_module="qwen3",
        baseline_config="qwen3_debugmodel",
        test_module="graph_trainer.qwen3",
        test_config="graph_trainer_qwen3_debugmodel",
        baseline_options=options,
        test_options=test_options,
    )


QWEN3_MOE_PARALLELISM = (
    "--training.disable_cuda_graphs"
    " --parallelism.data_parallel_shard_degree=4"
    " --parallelism.tensor_parallel_degree=2"
    " --parallelism.expert_parallel_degree=2"
)


def _run_qwen3_moe_loss_compare(test_options_extra: str = "") -> bool:
    """Run loss_compare for qwen3 MoE vs graph_trainer.qwen3 MoE."""
    options = f"{QWEN3_MOE_PARALLELISM} {DEBUGMODEL_TRAINING_OPTIONS}"
    test_options = options
    if test_options_extra:
        test_options += f" {test_options_extra}"
    return run_loss_compare(
        baseline_module="qwen3",
        baseline_config="qwen3_moe_debug",
        test_module="graph_trainer.qwen3",
        test_config="graph_trainer_qwen3_debugmodel_moe",
        baseline_options=options,
        test_options=test_options,
    )


AUTOPARALLEL_LLAMA3_PARALLELISM = (
    "--parallelism.data_parallel_shard_degree=2"
    " --parallelism.tensor_parallel_degree=2"
)


def _run_autoparallel_llama3_loss_compare() -> bool:
    """Run loss_compare for eager SDPA llama3 vs graph_trainer AutoParallel.

    AutoParallel is unsupported on the default FlexAttention backend (dynamo
    export flattens the BlockMask), so both sides use the test-only SDPA backend.
    The eager baseline runs the same SDPA model through GraphTrainer with
    ``mode=None`` (delegates to the core eager path).
    """
    return run_loss_compare_close(
        baseline_module="graph_trainer.llama3",
        baseline_config="graph_trainer_llama3_debugmodel_sdpa_eager",
        test_module="graph_trainer.llama3",
        test_config="graph_trainer_llama3_debugmodel_sdpa_cross_entropy_loss",
        baseline_options=AUTOPARALLEL_LLAMA3_PARALLELISM,
        test_options=(
            f"{AUTOPARALLEL_LLAMA3_PARALLELISM}"
            " --compile.mode aot_fx_trace"
            " --compile.enable_autoparallel"
        ),
        baseline_ngpus=4,
        test_ngpus=4,
    )


AUTOPARALLEL_DSV3_PARALLELISM = (
    "--parallelism.data_parallel_shard_degree=4"
    " --parallelism.expert_parallel_degree=2"
    f" {DEBUGMODEL_TRAINING_OPTIONS}"
)


def _run_autoparallel_deepseek_v3_loss_compare() -> bool:
    """Run loss_compare for eager DeepSeek V3 vs graph_trainer AutoParallel."""
    return run_loss_compare_close(
        baseline_module="deepseek_v3",
        baseline_config="deepseek_v3_debugmodel",
        test_module="graph_trainer.deepseek_v3",
        test_config="graph_trainer_deepseek_v3_debugmodel",
        baseline_options=AUTOPARALLEL_DSV3_PARALLELISM,
        test_options=(
            f"{AUTOPARALLEL_DSV3_PARALLELISM}"
            " --compile.mode aot_fx_trace"
            " --compile.enable_autoparallel"
        ),
        baseline_ngpus=4,
        test_ngpus=4,
        rtol=5e-4,
    )


class TestGraphTrainerNumerics(unittest.TestCase):
    """Test numerics equivalence between graph_trainer and FSDP2 eager."""

    def test_dense_llama3_aot_fx_trace_vs_eager(self):
        self.assertTrue(
            _run_llama3_loss_compare(test_options_extra="--compile.mode aot_fx_trace"),
        )

    @unittest.skip("Disabled: upstream partitioner regression (#2149)")
    def test_dense_llama3_jit_vs_eager(self):
        self.assertTrue(
            _run_llama3_loss_compare(test_options_extra="--compile.mode jit"),
        )

    @unittest.skip("Disabled: upstream partitioner regression (#2149)")
    def test_dense_llama3_auto_bucketing_jit_vs_eager(self):
        self.assertTrue(
            _run_llama3_loss_compare(
                test_options_extra="--compile.mode jit --compile.passes auto_bucketing"
            ),
        )

    @unittest.skip("Disabled: upstream partitioner regression (#2149)")
    def test_dense_llama3_manual_bucketing_jit_vs_eager(self):
        self.assertTrue(
            _run_llama3_loss_compare(
                test_options_extra="--compile.mode jit --compile.passes transformer_block_bucketing"
            ),
        )

    @unittest.skip("Disabled: upstream partitioner regression (#2149)")
    def test_moe_dsv3_jit_vs_eager(self):
        """Test graph_trainer.deepseek_v3 matches deepseek_v3 (JIT)."""
        self.assertTrue(
            _run_deepseek_v3_loss_compare(test_options_extra="--compile.mode jit"),
        )

    @unittest.skip("Disabled: upstream partitioner regression (#2149)")
    def test_moe_dsv3_manual_bucketing_jit_vs_eager(self):
        self.assertTrue(
            _run_deepseek_v3_loss_compare(
                test_options_extra="--compile.mode jit --compile.passes transformer_block_bucketing"
            ),
        )

    @unittest.skip(
        "Disabled: flaky single-rank crash in DSv3 MoE EP all-to-all. Losses "
        "match eager bitwise for ~12 steps, then one EP rank hard-crashes; "
        "root cause unconfirmed (rank traceback isn't captured under "
        "loss_compare's rank-0-only tee). Re-enable once the crash is "
        "diagnosed and fixed."
    )
    def test_moe_dsv3_aot_fx_trace_vs_eager(self):
        self.assertTrue(
            _run_deepseek_v3_loss_compare(
                test_options_extra="--compile.mode aot_fx_trace"
            ),
        )

    # TODO(#4342): Remove transformer-level chunking. After the model batch
    # dimension was folded into the token dimension, splitting `layers.*` in
    # half cuts the packed token stream mid-document, so neither chunk has full
    # attention context and the loss goes non-finite at step 1.
    @unittest.expectedFailure
    def test_moe_dsv3_ep_overlap_aot_fx_trace_vs_eager_chunked(self):
        self.assertTrue(_run_deepseek_v3_ep_overlap_loss_compare())

    def test_moe_dsv3_ep_overlap_moe_seq_aot_fx_trace_vs_eager_chunked(self):
        self.assertTrue(_run_deepseek_v3_ep_overlap_moe_seq_loss_compare())

    def test_moe_dsv3_ep_overlap_moe_batch_aot_fx_trace_vs_eager_chunked(self):
        self.assertTrue(_run_deepseek_v3_ep_overlap_moe_batch_loss_compare())

    @unittest.skip(
        # Flaky on H100 CI: the DSv3 MoE EP all-to-all is not bitwise
        # deterministic under --debug.deterministic (ALLTOALL_BASE reduction
        # order varies across NCCL/driver), so the aot_fx_trace vs eager loss
        # compare diverges by ~2e-5. Passes bitwise locally. See #3874.
        "flaky: DSv3 MoE EP all-to-all nondeterminism under --debug.deterministic (#3874)"
    )
    def test_graph_pp_moe_dsv3_aot_fx_trace_vs_eager(self):
        for schedule in ("Interleaved1F1B", "ZBVZeroBubble", "DualPipeV"):
            with self.subTest(schedule=schedule):
                self.assertTrue(_run_graph_pp_deepseek_v3_loss_compare(schedule))

    def test_dense_qwen3_aot_fx_trace_vs_eager(self):
        self.assertTrue(
            _run_qwen3_loss_compare(test_options_extra="--compile.mode aot_fx_trace"),
        )

    def test_moe_qwen3_aot_fx_trace_vs_eager(self):
        self.assertTrue(
            _run_qwen3_moe_loss_compare(
                test_options_extra="--compile.mode aot_fx_trace"
            ),
        )


@unittest.skipUnless(
    importlib.util.find_spec("autoparallel"),
    "AutoParallel numerics tests require the autoparallel package",
)
class TestGraphTrainerAutoParallelNumerics(unittest.TestCase):
    """Test graph_trainer AutoParallel numerics equivalence against eager."""

    # AutoParallel runs on the test-only SDPA backend (Decoder.forward lists
    # positions before attention_masks so input_fn's (tokens, positions) binds
    # correctly). It is unsupported on the default FlexAttention backend (dynamo
    # export flattens the BlockMask to (Fake)Tensors and flex_attention fails on
    # missing BLOCK_SIZE), so both eager baseline and AutoParallel test use SDPA.
    # TODO: Disabled due to upstream AutoParallel/PyTorch API skew. PyTorch
    # #186754 (2026-06-24) removed propagate_single_input_strategy in favor of
    # propagate_single_input_single_dim_strategy, but AutoParallel's
    # convert_element_type_rule still imports the old name, so the sharding
    # optimizer fails with ImportError. Re-enable once AutoParallel migrates.
    # https://github.com/pytorch/torchtitan/issues/3699
    @unittest.skip(
        "upstream AutoParallel imports removed propagate_single_input_strategy"
    )
    def test_llama3_aot_fx_trace_autoparallel_vs_eager(self):
        self.assertTrue(_run_autoparallel_llama3_loss_compare())

    @unittest.skip("upstream AutoParallel FakeTensor device mismatch regression")
    def test_deepseek_v3_aot_fx_trace_autoparallel_vs_eager(self):
        self.assertTrue(_run_autoparallel_deepseek_v3_loss_compare())


# B: batch size, D: input features, O: output features.


class _SharedProjection(nn.Module):
    def __init__(self, num_outputs, *, bias, device):
        super().__init__()
        self.projection = nn.Linear(
            8, num_outputs, bias=bias, device=device, dtype=torch.float64
        )

    def forward(self, inputs_BD):
        return sum(self.projection(inputs_BD * scale) for scale in (1, 2, 3, 4))


class TestSimpleFSDP(FSDPTest):
    @staticmethod
    def _local_shard(tensor, *, tp_rank, dp_rank, shard_dim, mode):
        local = tensor.chunk(2, dim=0)[tp_rank]
        if mode != "replicate":
            chunks = local.chunk(2, dim=shard_dim)
            local = (
                chunks[dp_rank]
                if dp_rank < len(chunks)
                else local.narrow(shard_dim, 0, 0)
            )
        return local.contiguous()

    @staticmethod
    def _check_layout(actual, expected):
        assert actual.shape == expected.shape
        assert actual.stride() == expected.stride()
        assert len(actual.placements) == len(expected.placements)
        for placement, expected_placement in zip(
            actual.placements, expected.placements, strict=True
        ):
            assert type(placement) is type(expected_placement)
            assert placement == expected_placement
            if isinstance(placement, _StridedShard):
                assert placement.split_factor == expected_placement.split_factor

    @patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    def _test_sharding(
        self,
        mode="fully_shard",
        *,
        num_outputs=10,
        shard_dim=0,
        meta_init=False,
        compiled=False,
        transposed=False,
        spmd_backend="partial_dtensor",
    ):
        required_world_size = 8 if mode == "hybrid_shard" else 4
        if self.world_size < required_world_size or self.world_size % 4:
            self.skipTest(
                f"Requires a multiple of four devices, at least {required_world_size}"
            )
        device_type = "cuda" if dist.get_backend() == "nccl" else "cpu"
        device = (
            torch.device(device_type, self.rank) if device_type == "cuda" else "cpu"
        )
        previous_backend = get_spmd_backend()
        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        try:
            set_spmd_backend(spmd_backend)
            mesh = init_device_mesh(
                device_type,
                (self.world_size // 4, 2, 2),
                mesh_dim_names=("replica", "fsdp", "tp"),
            )
            dp_mesh = (
                mesh["replica", "fsdp"] if mode == "hybrid_shard" else mesh["fsdp"]
            )
            tp_mesh = mesh["tp"]
            tp_rank = tp_mesh.get_local_rank()
            dp_rank = mesh["fsdp"].get_local_rank()
            reference = _SharedProjection(
                num_outputs, bias=shard_dim == 0, device=device
            )
            if transposed:
                reference.projection.weight = nn.Parameter(
                    reference.projection.weight.detach().t().contiguous().t()
                )
            model = (
                _SharedProjection(num_outputs, bias=shard_dim == 0, device="meta")
                if meta_init
                else copy.deepcopy(reference)
            )
            if spmd_backend == "spmd_types":
                for name, parameter in model.projection.named_parameters():
                    local_parameter = nn.Parameter(
                        parameter.detach().chunk(2, dim=0)[tp_rank].clone(),
                        requires_grad=parameter.requires_grad,
                    )
                    spmd.assert_type(local_parameter, {tp_mesh.get_group(): spmd.S(0)})
                    model.projection.register_parameter(name, local_parameter)
            else:
                parallelize_module(model.projection, tp_mesh, ColwiseParallel())

            original_redistribute = simple_fsdp.redistribute_local_tensor

            def check_outer_spec(local, *, current_spec, target_spec):
                for spec in (current_spec, target_spec):
                    assert spec.shape == local.shape
                    assert spec.tensor_meta.stride == local.stride()
                    assert spec.tensor_meta.dtype == local.dtype
                return original_redistribute(
                    local, current_spec=current_spec, target_spec=target_spec
                )

            with patch.object(
                simple_fsdp, "redistribute_local_tensor", check_outer_spec
            ):
                simple_fsdp.data_parallel(
                    model,
                    dp_mesh,
                    mode,
                    shard_dim=shard_dim,
                    non_dp_mesh=tp_mesh if spmd_backend == "spmd_types" else None,
                )

            def expected_local(tensor):
                return self._local_shard(
                    tensor,
                    tp_rank=tp_rank,
                    dp_rank=dp_rank,
                    shard_dim=shard_dim,
                    mode=mode,
                )

            if meta_init:
                model.to_empty(device=device)
                assert all(not parameter.is_meta for parameter in model.parameters())
                with torch.no_grad(), simple_fsdp.disable_active_parametrization():
                    for parameter, original in zip(
                        model.parameters(), reference.parameters(), strict=True
                    ):
                        parameter.to_local().copy_(expected_local(original))

            # Check logical TP metadata even when local values have the right shape.
            if spmd_backend == "partial_dtensor":
                assert (
                    model.projection.weight.shape == reference.projection.weight.shape
                )
                assert (
                    model.projection.weight.stride()
                    == reference.projection.weight.stride()
                )
            else:
                assert model.projection.weight.shape == (num_outputs // 2, 8)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
            reference_optimizer = torch.optim.AdamW(
                reference.parameters(), lr=0.01, foreach=False
            )
            run_model = (
                torch.compile(model, backend="aot_eager", fullgraph=True)
                if compiled
                else model
            )
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                reference_optimizer.zero_grad(set_to_none=True)
                for microbatch in range(2):
                    # Sparse, exactly representable inputs keep the comparison bitwise
                    # across different GEMM shapes while varying the data across DP ranks.
                    batch_start = (self.rank // 2 + microbatch) % 4
                    inputs_BD = torch.eye(8, dtype=torch.float64, device=device)[
                        batch_start : batch_start + 2
                    ]
                    output_BO = run_model(inputs_BD)
                    expected_BO = reference(inputs_BD)
                    local_expected_BO = expected_BO.chunk(2, dim=1)[
                        tp_rank
                    ].contiguous()
                    torch.testing.assert_close(
                        output_BO, local_expected_BO, rtol=0, atol=0
                    )
                    loss = output_BO.sum()
                    torch.testing.assert_close(
                        loss, local_expected_BO.sum(), rtol=0, atol=0
                    )
                    loss.backward()
                    expected_BO.sum().backward()
                for parameter, original in zip(
                    model.parameters(), reference.parameters(), strict=True
                ):
                    # SimpleFSDP sums gradients across every data-parallel axis.
                    for axis_name in dp_mesh.mesh_dim_names:
                        dist.all_reduce(
                            original.grad, group=dp_mesh.get_group(axis_name)
                        )
                    self._check_layout(parameter.grad, parameter)
                    torch.testing.assert_close(
                        parameter.grad.to_local(),
                        expected_local(original.grad),
                        rtol=0,
                        atol=0,
                    )
                optimizer.step()
                reference_optimizer.step()
                for parameter, original in zip(
                    model.parameters(), reference.parameters(), strict=True
                ):
                    torch.testing.assert_close(
                        parameter.to_local(), expected_local(original), rtol=0, atol=0
                    )
                    for key in ("exp_avg", "exp_avg_sq"):
                        state = optimizer.state[parameter][key]
                        self._check_layout(state, parameter)
                        torch.testing.assert_close(
                            state.to_local(),
                            expected_local(reference_optimizer.state[original][key]),
                            rtol=0,
                            atol=0,
                        )
        finally:
            set_spmd_backend(previous_backend)
            torch.use_deterministic_algorithms(previous_deterministic)

    def test_uneven_fsdp_with_tp(self):
        self._test_sharding()

    def test_uneven_fsdp_with_ep_tp(self):
        if self.world_size < 8 or self.world_size % 4:
            self.skipTest("Requires a multiple of four devices, at least eight")
        device_type = "cuda" if dist.get_backend() == "nccl" else "cpu"
        device = (
            torch.device(device_type, self.rank) if device_type == "cuda" else "cpu"
        )
        mesh = init_device_mesh(
            device_type,
            (self.world_size // 4, 2, 2),
            mesh_dim_names=("fsdp", "ep", "tp"),
        )
        previous_backend = get_spmd_backend()
        try:
            set_spmd_backend("partial_dtensor")
            weight = torch.arange(80, dtype=torch.float32, device=device).reshape(10, 8)
            model = nn.Linear(8, 10, bias=False, device=device)
            model.weight = nn.Parameter(
                distribute_tensor(weight, mesh["ep", "tp"], (Shard(0), Shard(0)))
            )
            expected_local = model.weight.to_local().detach().clone()
            data_parallel(model, mesh["fsdp"], "fully_shard")
            output = model.weight
            self.assertEqual(output.shape, weight.shape)
            self.assertEqual(output.stride(), weight.stride())
            torch.testing.assert_close(
                output.to_local(), expected_local, rtol=0, atol=0
            )
            output.to_local().sum().backward()
            parameter = model._parameters["weight"]
            self.assertEqual(parameter.grad.shape, parameter.shape)
            self.assertEqual(parameter.grad.placements, parameter.placements)
            # Nested uneven EP/TP chunks must retain their exact DP shard sizes
            # in backward, not be treated as one flattened EP*TP partition.
            torch.testing.assert_close(
                parameter.grad.to_local(),
                torch.full_like(parameter.to_local(), mesh["fsdp"].size()),
                rtol=0,
                atol=0,
            )
        finally:
            set_spmd_backend(previous_backend)

    def test_uneven_hsdp_with_tp(self):
        self._test_sharding("hybrid_shard")

    def test_uneven_tp_with_fsdp(self):
        self._test_sharding(num_outputs=7, shard_dim=1)

    def test_uneven_tp_with_replicate(self):
        self._test_sharding("replicate", num_outputs=7)

    def test_uneven_sharding_transposed_parameter(self):
        self._test_sharding(transposed=True)

    def test_uneven_sharding_transposed_parameter_compiled(self):
        self._test_sharding(transposed=True, compiled=True)

    def test_uneven_replicate_transposed_parameter(self):
        self._test_sharding("replicate", num_outputs=7, transposed=True)

    def test_uneven_sharding_meta_init(self):
        self._test_sharding(meta_init=True)

    def test_uneven_sharding_compiled(self):
        self._test_sharding(compiled=True)

    def test_even_fsdp_with_tp(self):
        self._test_sharding(num_outputs=12)

    def test_empty_fsdp_shard_with_tp(self):
        self._test_sharding(num_outputs=2)

    def test_uneven_fsdp_with_spmd_types(self):
        self._test_sharding(spmd_backend="spmd_types")

    def test_frozen_parameter_remains_frozen(self):
        device_type = "cuda" if dist.get_backend() == "nccl" else "cpu"
        device = (
            torch.device(device_type, self.rank) if device_type == "cuda" else "cpu"
        )
        mesh = init_device_mesh(
            device_type, (self.world_size,), mesh_dim_names=("fsdp",)
        )
        model = nn.Linear(8, 8, device=device)
        model.weight.requires_grad_(False)
        previous_backend = get_spmd_backend()
        try:
            set_spmd_backend("partial_dtensor")
            data_parallel(model, mesh, "fully_shard")
            weight = model._parameters["weight"]
            bias = model._parameters["bias"]
            self.assertFalse(weight.requires_grad)
            self.assertTrue(bias.requires_grad)
            original_weight = weight.to_local().detach().clone()
            original_bias = bias.to_local().detach().clone()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            model(torch.ones(2, 8, device=device)).sum().backward()
            self.assertIsNone(weight.grad)
            self.assertIsNotNone(bias.grad)
            optimizer.step()
            self.assertTrue(torch.equal(weight.to_local(), original_weight))
            self.assertFalse(torch.equal(bias.to_local(), original_bias))
        finally:
            set_spmd_backend(previous_backend)

    def init_test(self):
        self.optimizer = torch.optim.Adam
        self.loss_fn = cross_entropy_loss
        data_parallel_shard_degree = -1
        if self.mode == "replicate":
            self.dp_mesh_dim_names = ["dp_replicate"]
            data_parallel_replicate_degree = self.world_size
        elif self.mode == "fully_shard":
            self.dp_mesh_dim_names = ["fsdp"]
            data_parallel_replicate_degree = 1
        elif self.mode == "hybrid_shard":
            self.dp_mesh_dim_names = ["dp_replicate", "fsdp"]
            data_parallel_replicate_degree = self.world_size // 2
        else:
            raise ValueError(f"Unsupported mode {self.mode}")

        self.parallel_dims = ParallelDims(
            dp_shard=data_parallel_shard_degree,
            dp_replicate=data_parallel_replicate_degree,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=self.world_size,
            spmd_backend="partial_dtensor",
        )

    def get_input(self):
        inputs = torch.randn(8, 8).cuda()
        labels = torch.randn(8, 8).cuda()
        model = torch.nn.Linear(8, 8)
        return model, inputs, labels

    def run_fsdp2(self, model, inputs, labels, epoch=20):
        fully_shard(model, mesh=self.parallel_dims.get_mesh(self.dp_mesh_dim_names))
        optim = self.optimizer(model.parameters(), lr=1e-4)
        losses = []
        for _ in range(epoch):
            optim.zero_grad()
            out = model(inputs)
            loss = self.loss_fn(out, labels)
            loss.backward()
            optim.step()
            losses.append(loss)
        return losses

    def run_simple_fsdp(self, model, inputs, labels, epoch=20):
        model = data_parallel(
            model,
            device_mesh=self.parallel_dims.get_mesh(self.dp_mesh_dim_names),
            mode=self.mode,
        )
        optim = self.optimizer(model.parameters(), lr=1e-4)
        losses = []
        for _ in range(epoch):
            optim.zero_grad()
            out = model(inputs)
            loss = self.loss_fn(out, labels)
            loss.backward()
            optim.step()
            losses.append(loss)
        return losses

    def run_simple_fsdp_compiled_aot_eager(self, model, inputs, labels, epoch=20):
        model = data_parallel(
            model,
            device_mesh=self.parallel_dims.get_mesh(self.dp_mesh_dim_names),
            mode=self.mode,
        )
        # TODO: Add "inductor" backend when it's numerical issues are fixed
        model = torch.compile(model, backend="aot_eager", fullgraph=True)
        optim = self.optimizer(model.parameters(), lr=1e-4)
        losses = []
        for _ in range(epoch):
            optim.zero_grad()
            out = model(inputs)
            loss = self.loss_fn(out, labels)
            loss.backward()
            optim.step()
            losses.append(loss)
        return losses

    def test_replicate_convergence(self):
        # unit test for replicate mode
        self.mode = "replicate"
        self.init_test()
        model, inputs, labels = self.get_input()

        fsdp2_losses = self.run_fsdp2(copy.deepcopy(model), inputs, labels)
        simple_fsdp_losses = self.run_simple_fsdp(copy.deepcopy(model), inputs, labels)
        simple_fsdp_compiled_aot_eager_losses = self.run_simple_fsdp_compiled_aot_eager(
            copy.deepcopy(model), inputs, labels
        )

        for (fsdp2_loss, simple_fsdp_loss, simple_fsdp_compiled_aot_eager_loss,) in zip(
            fsdp2_losses,
            simple_fsdp_losses,
            simple_fsdp_compiled_aot_eager_losses,
        ):
            assert torch.equal(fsdp2_loss, simple_fsdp_loss)
            assert torch.equal(fsdp2_loss, simple_fsdp_compiled_aot_eager_loss)

    def test_fullyshard_convergence(self):
        # unit test for fully_shard mode
        self.mode = "fully_shard"
        self.init_test()
        model, inputs, labels = self.get_input()

        fsdp2_losses = self.run_fsdp2(copy.deepcopy(model), inputs, labels)
        simple_fsdp_losses = self.run_simple_fsdp(copy.deepcopy(model), inputs, labels)
        simple_fsdp_compiled_aot_eager_losses = self.run_simple_fsdp_compiled_aot_eager(
            copy.deepcopy(model), inputs, labels
        )

        for (fsdp2_loss, simple_fsdp_loss, simple_fsdp_compiled_aot_eager_loss,) in zip(
            fsdp2_losses,
            simple_fsdp_losses,
            simple_fsdp_compiled_aot_eager_losses,
        ):
            assert torch.equal(fsdp2_loss, simple_fsdp_loss)
            assert torch.equal(fsdp2_loss, simple_fsdp_compiled_aot_eager_loss)

    def test_hybridshard_convergence(self):
        # unit test for hybrid_shard mode
        self.mode = "hybrid_shard"
        self.init_test()
        model, inputs, labels = self.get_input()

        fsdp2_losses = self.run_fsdp2(copy.deepcopy(model), inputs, labels)
        simple_fsdp_losses = self.run_simple_fsdp(copy.deepcopy(model), inputs, labels)
        simple_fsdp_compiled_aot_eager_losses = self.run_simple_fsdp_compiled_aot_eager(
            copy.deepcopy(model), inputs, labels
        )

        for (fsdp2_loss, simple_fsdp_loss, simple_fsdp_compiled_aot_eager_loss,) in zip(
            fsdp2_losses,
            simple_fsdp_losses,
            simple_fsdp_compiled_aot_eager_losses,
        ):
            assert torch.equal(fsdp2_loss, simple_fsdp_loss)
            assert torch.equal(fsdp2_loss, simple_fsdp_compiled_aot_eager_loss)


if __name__ == "__main__":
    unittest.main()
