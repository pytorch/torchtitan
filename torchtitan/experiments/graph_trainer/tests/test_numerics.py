# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import subprocess
import sys
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
) -> bool:
    """Run loss_compare.py comparing a baseline module against a graph_trainer module.

    Args:
        baseline_module: Module name for baseline (e.g., "llama3").
        baseline_config: Config name for baseline (e.g., "llama3_debugmodel").
        test_module: Module name for test (e.g., "graph_trainer.llama3").
        test_config: Config name for test (e.g., "graph_trainer_llama3_debugmodel").
        baseline_options: Additional CLI options for the baseline run.
        test_options: Additional CLI options for the test run.

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


def _run_deepseek_v3_loss_compare(test_options_extra: str = "") -> bool:
    """Run loss_compare for deepseek_v3 vs graph_trainer.deepseek_v3."""
    test_options = DSV3_PARALLELISM
    if test_options_extra:
        test_options += f" {test_options_extra}"
    return run_loss_compare(
        baseline_module="deepseek_v3",
        baseline_config="deepseek_v3_debugmodel",
        test_module="graph_trainer.deepseek_v3",
        test_config="graph_trainer_deepseek_v3_debugmodel",
        baseline_options=DSV3_PARALLELISM,
        test_options=test_options,
    )


class TestGraphTrainerNumerics(unittest.TestCase):
    """Test numerics equivalence between graph_trainer and FSDP2 eager."""

    def test_llama3_aot_vs_eager(self):
        self.assertTrue(
            _run_llama3_loss_compare(test_options_extra="--compile.mode aot"),
        )

    def test_llama3_auto_bucketing_aot_vs_eager(self):
        self.assertTrue(
            _run_llama3_loss_compare(
                test_options_extra="--compile.mode aot --compile.passes auto_bucketing"
            ),
        )

    def test_llama3_manual_bucketing_aot_vs_eager(self):
        self.assertTrue(
            _run_llama3_loss_compare(
                test_options_extra="--compile.mode aot --compile.passes transformer_block_bucketing"
            ),
        )

    def test_llama3_cudagraph_aot_vs_eager(self):
        self.assertTrue(
            _run_llama3_loss_compare(
                test_options_extra="--compile.mode aot --compile.passes cudagraph"
            ),
        )

    def test_dsv3_aot_vs_eager(self):
        self.assertTrue(
            _run_deepseek_v3_loss_compare(test_options_extra="--compile.mode aot"),
        )

    def test_dsv3_manual_bucketing_aot_vs_eager(self):
        self.assertTrue(
            _run_deepseek_v3_loss_compare(
                test_options_extra="--compile.mode aot --compile.passes transformer_block_bucketing"
            ),
        )

    def test_llama3_aot_fx_trace_vs_eager(self):
        self.assertTrue(
            _run_llama3_loss_compare(test_options_extra="--compile.mode aot_fx_trace"),
        )

    def test_llama3_jit_vs_eager(self):
        self.assertTrue(
            _run_llama3_loss_compare(test_options_extra="--compile.mode jit"),
        )

    def test_llama3_auto_bucketing_jit_vs_eager(self):
        self.assertTrue(
            _run_llama3_loss_compare(
                test_options_extra="--compile.mode jit --compile.passes auto_bucketing"
            ),
        )

    def test_llama3_manual_bucketing_jit_vs_eager(self):
        self.assertTrue(
            _run_llama3_loss_compare(
                test_options_extra="--compile.mode jit --compile.passes transformer_block_bucketing"
            ),
        )

    def test_dsv3_jit_vs_eager(self):
        """Test graph_trainer.deepseek_v3 matches deepseek_v3 (JIT)."""
        self.assertTrue(
            _run_deepseek_v3_loss_compare(test_options_extra="--compile.mode jit"),
        )

    def test_dsv3_manual_bucketing_jit_vs_eager(self):
        self.assertTrue(
            _run_deepseek_v3_loss_compare(
                test_options_extra="--compile.mode jit --compile.passes transformer_block_bucketing"
            ),
        )

    def test_dsv3_aot_fx_trace_vs_eager(self):
        self.assertTrue(
            _run_deepseek_v3_loss_compare(
                test_options_extra="--compile.mode aot_fx_trace"
            ),
        )


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
            etp=1,
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
