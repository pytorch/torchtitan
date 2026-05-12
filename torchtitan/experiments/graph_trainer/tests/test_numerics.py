# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import importlib.util
import math
import subprocess
import sys
import tempfile
import unittest

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
        baseline_config="llama3_debugmodel_ce_loss",
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


def _run_deepseek_v3_loss_compare(test_options_extra: str = "") -> bool:
    """Run loss_compare for deepseek_v3 vs graph_trainer.deepseek_v3."""
    test_options = DSV3_PARALLELISM
    if test_options_extra:
        test_options += f" {test_options_extra}"
    return run_loss_compare(
        baseline_module="graph_trainer.deepseek_v3",
        baseline_config="deepseek_v3_debugmodel_ep_ce_loss",
        test_module="graph_trainer.deepseek_v3",
        test_config="graph_trainer_deepseek_v3_debugmodel_ep",
        baseline_options=DSV3_PARALLELISM,
        test_options=test_options,
    )


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
        baseline_module="graph_trainer.qwen3",
        baseline_config="qwen3_debugmodel_ce_loss",
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
        baseline_module="graph_trainer.qwen3",
        baseline_config="qwen3_moe_debug_ep_ce_loss",
        test_module="graph_trainer.qwen3",
        test_config="graph_trainer_qwen3_debugmodel_moe_ep",
        baseline_options=QWEN3_MOE_PARALLELISM,
        test_options=test_options,
    )


AUTOPARALLEL_LLAMA3_PARALLELISM = (
    "--parallelism.data_parallel_shard_degree=2"
    " --parallelism.tensor_parallel_degree=2"
)


def _run_autoparallel_llama3_loss_compare() -> bool:
    """Run loss_compare for eager llama3 vs graph_trainer AutoParallel llama3."""
    return run_loss_compare_close(
        baseline_module="llama3",
        baseline_config="llama3_debugmodel_ce_loss",
        test_module="graph_trainer.llama3",
        test_config="graph_trainer_llama3_debugmodel",
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
    " --parallelism.disable_loss_parallel"
)


def _run_autoparallel_deepseek_v3_loss_compare() -> bool:
    """Run loss_compare for eager DeepSeek V3 vs graph_trainer AutoParallel."""
    return run_loss_compare_close(
        baseline_module="graph_trainer.deepseek_v3",
        baseline_config="deepseek_v3_debugmodel_ep_ce_loss",
        test_module="graph_trainer.deepseek_v3",
        test_config="graph_trainer_deepseek_v3_debugmodel_ep",
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


# TODO: JIT and AOT tests are disabled due to an upstream PyTorch
# partitioner regression ("Node tangents_2 was invalid, but is output")
# triggered by the full DTensor change (#2149). Re-enable once resolved.
# See https://www.internalfb.com/intern/paste/P2285018929/


class TestGraphTrainerNumerics(unittest.TestCase):
    """Test numerics equivalence between graph_trainer and FSDP2 eager."""

    @unittest.skip("Disabled: upstream partitioner regression (#2149)")
    def test_dense_llama3_aot_vs_eager(self):
        self.assertTrue(
            _run_llama3_loss_compare(test_options_extra="--compile.mode aot"),
        )

    @unittest.skip("Disabled: upstream partitioner regression (#2149)")
    def test_dense_llama3_auto_bucketing_aot_vs_eager(self):
        self.assertTrue(
            _run_llama3_loss_compare(
                test_options_extra="--compile.mode aot --compile.passes auto_bucketing"
            ),
        )

    @unittest.skip("Disabled: upstream partitioner regression (#2149)")
    def test_dense_llama3_manual_bucketing_aot_vs_eager(self):
        self.assertTrue(
            _run_llama3_loss_compare(
                test_options_extra="--compile.mode aot --compile.passes transformer_block_bucketing"
            ),
        )

    @unittest.skip("Disabled: upstream partitioner regression (#2149)")
    def test_dense_llama3_cudagraph_aot_vs_eager(self):
        self.assertTrue(
            _run_llama3_loss_compare(
                test_options_extra="--compile.mode aot --compile.passes cudagraph"
            ),
        )

    @unittest.skip("Disabled: upstream partitioner regression (#2149)")
    def test_moe_dsv3_aot_vs_eager(self):
        self.assertTrue(
            _run_deepseek_v3_loss_compare(test_options_extra="--compile.mode aot"),
        )

    @unittest.skip("Disabled: upstream partitioner regression (#2149)")
    def test_moe_dsv3_manual_bucketing_aot_vs_eager(self):
        self.assertTrue(
            _run_deepseek_v3_loss_compare(
                test_options_extra="--compile.mode aot --compile.passes transformer_block_bucketing"
            ),
        )

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

    def test_moe_dsv3_aot_fx_trace_vs_eager(self):
        self.assertTrue(
            _run_deepseek_v3_loss_compare(
                test_options_extra="--compile.mode aot_fx_trace"
            ),
        )

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

    def test_llama3_aot_fx_trace_autoparallel_vs_eager(self):
        self.assertTrue(_run_autoparallel_llama3_loss_compare())

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
