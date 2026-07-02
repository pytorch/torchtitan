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

import torch
from torch.distributed._composable.fsdp import fully_shard
from torch.testing._internal.common_fsdp import FSDPTest

from torchtitan.components.loss import cross_entropy_loss
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.simple_fsdp import data_parallel


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


def _run_llama3_loss_compare(test_options_extra: str = "") -> bool:
    """Run loss_compare for llama3 vs graph_trainer.llama3 with FSDP+TP."""
    test_options = LLAMA3_PARALLELISM
    if test_options_extra:
        test_options += f" {test_options_extra}"
    return run_loss_compare(
        baseline_module="llama3",
        baseline_config="llama3_debugmodel",
        test_module="graph_trainer.llama3",
        test_config="graph_trainer_llama3_debugmodel",
        baseline_options=LLAMA3_PARALLELISM,
        test_options=test_options,
    )


DSV3_PARALLELISM = (
    "--parallelism.data_parallel_shard_degree=4"
    " --parallelism.tensor_parallel_degree=2"
    " --parallelism.expert_parallel_degree=2"
)
DSV3_EP_OVERLAP_GRAPH_PARALLELISM = (
    "--parallelism.data_parallel_shard_degree=8"
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
    baseline_options = parallelism
    if baseline_options_extra:
        baseline_options += f" {baseline_options_extra}"
    test_options = parallelism
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
    "--parallelism.pipeline_parallel_degree=2"
    " --parallelism.data_parallel_shard_degree=4"
    " --parallelism.expert_parallel_degree=2"
    " --training.local_batch_size=8"
    " --training.global_batch_size=32"
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
    test_options = QWEN3_PARALLELISM
    if test_options_extra:
        test_options += f" {test_options_extra}"
    return run_loss_compare(
        baseline_module="qwen3",
        baseline_config="qwen3_debugmodel",
        test_module="graph_trainer.qwen3",
        test_config="graph_trainer_qwen3_debugmodel",
        baseline_options=QWEN3_PARALLELISM,
        test_options=test_options,
    )


QWEN3_MOE_PARALLELISM = (
    "--parallelism.data_parallel_shard_degree=4"
    " --parallelism.tensor_parallel_degree=2"
    " --parallelism.expert_parallel_degree=2"
)


def _run_qwen3_moe_loss_compare(test_options_extra: str = "") -> bool:
    """Run loss_compare for qwen3 MoE vs graph_trainer.qwen3 MoE."""
    test_options = QWEN3_MOE_PARALLELISM
    if test_options_extra:
        test_options += f" {test_options_extra}"
    return run_loss_compare(
        baseline_module="qwen3",
        baseline_config="qwen3_moe_debug",
        test_module="graph_trainer.qwen3",
        test_config="graph_trainer_qwen3_debugmodel_moe",
        baseline_options=QWEN3_MOE_PARALLELISM,
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

    def test_moe_dsv3_ep_overlap_aot_fx_trace_vs_eager_chunked(self):
        self.assertTrue(_run_deepseek_v3_ep_overlap_loss_compare())

    def test_moe_dsv3_ep_overlap_moe_seq_aot_fx_trace_vs_eager_chunked(self):
        self.assertTrue(_run_deepseek_v3_ep_overlap_moe_seq_loss_compare())

    def test_moe_dsv3_ep_overlap_moe_batch_aot_fx_trace_vs_eager_chunked(self):
        self.assertTrue(_run_deepseek_v3_ep_overlap_moe_batch_loss_compare())

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


class TestSimpleFSDP(FSDPTest):
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
