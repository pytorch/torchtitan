# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Numerical equivalence tests for FlexShard.

Compares FlexShard against SimpleFSDP, FSDP2, and itself across various
configurations. Uses FSDPTest base class for distributed test infrastructure.

Requires 2+ GPUs. Run with:
    torchrun --standalone --nproc_per_node=2 -m pytest \
        torchtitan/experiments/flex_shard/tests/test_numerics.py -q \
        -k "not FullModel"
"""

import copy
import unittest

import torch
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.fsdp import DataParallelMeshDims
from torch.testing._internal.common_fsdp import FSDPTest

from torchtitan.components.loss import cross_entropy_loss
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.flex_shard import (
    BucketSpec,
    flat_shard_placements,
    flex_shard,
    lift_params_to_global_spmd_mesh,
    MixedPrecisionPolicy as FlexShardMPPolicy,
    Owned,
    per_param_placements,
    RaggedShard,
)
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    data_parallel,
    MixedPrecisionPolicy as SimpleFSDPMPPolicy,
)


STEPS = 20


def _flex_shard_global(model, mesh, **kwargs):
    lift_params_to_global_spmd_mesh(model, mesh)
    kwargs.setdefault("shard_placement_fn", per_param_placements)
    kwargs.setdefault("buckets", [BucketSpec(["*"])])
    return flex_shard(
        model,
        mesh,
        DataParallelMeshDims(shard="fsdp"),
        **kwargs,
    )


def _single_placement_fn(placement):
    def placement_fn(named_params, _mesh):
        return {fqn: (placement,) for fqn, _ in named_params}

    return placement_fn


class TestFlexShardNumerics(FSDPTest):
    """Test numerical equivalence between FlexShard and other FSDP variants."""

    def init_test(self):
        self.parallel_dims = ParallelDims(
            dp_shard=-1,
            dp_replicate=1,
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

    def _train_loop(self, model, inputs, labels, steps=STEPS):
        optim = torch.optim.Adam(model.parameters(), lr=1e-4)
        losses = []
        for _ in range(steps):
            optim.zero_grad()
            out = model(inputs)
            loss = cross_entropy_loss(out, labels)
            loss.backward()
            optim.step()
            losses.append(loss)
        return losses

    def run_fsdp2(self, model, inputs, labels):
        fully_shard(model, mesh=self.parallel_dims.get_mesh("fsdp"))
        return self._train_loop(model, inputs, labels)

    def run_simple_fsdp(self, model, inputs, labels):
        model = data_parallel(
            model,
            device_mesh=self.parallel_dims.get_mesh("fsdp"),
            mode="fully_shard",
        )
        return self._train_loop(model, inputs, labels)

    def run_flex_shard(self, model, inputs, labels, **kwargs):
        _flex_shard_global(
            model,
            self.parallel_dims.get_mesh("fsdp"),
            **kwargs,
        )
        return self._train_loop(model, inputs, labels)

    def run_flex_shard_compiled(self, model, inputs, labels):
        _flex_shard_global(
            model,
            self.parallel_dims.get_mesh("fsdp"),
        )
        model = torch.compile(model, backend="aot_eager", fullgraph=True)
        return self._train_loop(model, inputs, labels)

    def test_flex_shard_vs_simple_fsdp(self):
        """FlexShard Shard(0) vs SimpleFSDP fully_shard: bitwise identical."""
        self.init_test()
        model, inputs, labels = self.get_input()

        simple_fsdp_losses = self.run_simple_fsdp(copy.deepcopy(model), inputs, labels)
        flex_shard_losses = self.run_flex_shard(copy.deepcopy(model), inputs, labels)

        for step, (sf_loss, fs_loss) in enumerate(
            zip(simple_fsdp_losses, flex_shard_losses, strict=True)
        ):
            assert torch.equal(sf_loss, fs_loss), (
                f"Step {step}: SimpleFSDP loss {sf_loss.item()} != "
                f"FlexShard loss {fs_loss.item()}"
            )

    def test_flex_shard_eager_vs_compiled(self):
        """FlexShard eager vs FlexShard + torch.compile: bitwise identical."""
        self.init_test()
        model, inputs, labels = self.get_input()

        eager_losses = self.run_flex_shard(copy.deepcopy(model), inputs, labels)
        compiled_losses = self.run_flex_shard_compiled(
            copy.deepcopy(model), inputs, labels
        )

        for step, (e_loss, c_loss) in enumerate(
            zip(eager_losses, compiled_losses, strict=True)
        ):
            assert torch.equal(e_loss, c_loss), (
                f"Step {step}: eager loss {e_loss.item()} != "
                f"compiled loss {c_loss.item()}"
            )

    def test_flex_shard_reshard_true_vs_false(self):
        """reshard_after_forward=True vs False: bitwise identical."""
        self.init_test()
        model, inputs, labels = self.get_input()

        reshard_true_losses = self.run_flex_shard(
            copy.deepcopy(model),
            inputs,
            labels,
            buckets=[BucketSpec(["*"], reshard_after_forward=True)],
        )
        reshard_false_losses = self.run_flex_shard(
            copy.deepcopy(model),
            inputs,
            labels,
            buckets=[BucketSpec(["*"], reshard_after_forward=False)],
        )

        for step, (rt_loss, rf_loss) in enumerate(
            zip(reshard_true_losses, reshard_false_losses, strict=True)
        ):
            assert torch.equal(rt_loss, rf_loss), (
                f"Step {step}: reshard=True loss {rt_loss.item()} != "
                f"reshard=False loss {rf_loss.item()}"
            )

    def test_flex_shard_mixed_precision_vs_simple_fsdp(self):
        """FlexShard mixed precision vs SimpleFSDP mixed precision."""
        self.init_test()
        model, inputs, labels = self.get_input()

        # Cast inputs to param_dtype — both SimpleFSDP and FlexShard cast
        # weights to bf16 via parametrization, so inputs must match.
        inputs = inputs.to(torch.bfloat16)

        # SimpleFSDP with mixed precision
        sf_model = copy.deepcopy(model)
        sf_model = data_parallel(
            sf_model,
            device_mesh=self.parallel_dims.get_mesh("fsdp"),
            mode="fully_shard",
            mp_policy=SimpleFSDPMPPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
            ),
        )
        sf_losses = self._train_loop(sf_model, inputs, labels)

        # FlexShard with mixed precision
        fs_model = copy.deepcopy(model)
        _flex_shard_global(
            fs_model,
            self.parallel_dims.get_mesh("fsdp"),
            buckets=[
                BucketSpec(
                    ["*"],
                    mp_policy=FlexShardMPPolicy(
                        param_dtype=torch.bfloat16,
                        reduce_dtype=torch.float32,
                    ),
                )
            ],
        )
        fs_losses = self._train_loop(fs_model, inputs, labels)

        for step, (sf_loss, fs_loss) in enumerate(
            zip(sf_losses, fs_losses, strict=True)
        ):
            # Cast ordering may differ slightly between implementations
            assert torch.allclose(sf_loss, fs_loss, atol=1e-5, rtol=1e-4), (
                f"Step {step}: SimpleFSDP mp loss {sf_loss.item()} not close to "
                f"FlexShard mp loss {fs_loss.item()}"
            )

    def test_flex_shard_flat_shard_vs_shard(self):
        """FlexShard Shard(0) vs FlatShard: bitwise identical for nn.Linear."""
        self.init_test()
        model, inputs, labels = self.get_input()

        shard_losses = self.run_flex_shard(copy.deepcopy(model), inputs, labels)

        fs_model = copy.deepcopy(model)
        _flex_shard_global(
            fs_model,
            self.parallel_dims.get_mesh("fsdp"),
            shard_placement_fn=flat_shard_placements,
        )
        flat_losses = self._train_loop(fs_model, inputs, labels)

        for step, (s_loss, f_loss) in enumerate(
            zip(shard_losses, flat_losses, strict=True)
        ):
            assert torch.equal(s_loss, f_loss), (
                f"Step {step}: Shard(0) loss {s_loss.item()} != "
                f"FlatShard loss {f_loss.item()}"
            )

    def test_flex_shard_owned_convergence(self):
        """FlexShard Owned(0) converges: loss decreases over training."""
        self.init_test()
        model, inputs, labels = self.get_input()

        model = copy.deepcopy(model)
        _flex_shard_global(
            model,
            self.parallel_dims.get_mesh("fsdp"),
            shard_placement_fn=_single_placement_fn(Owned(0)),
        )
        losses = self._train_loop(model, inputs, labels)

        # Verify loss decreases (convergence)
        assert losses[-1] < losses[0], (
            f"Owned model did not converge: first loss {losses[0].item()}, "
            f"last loss {losses[-1].item()}"
        )

    def test_flex_shard_owned_compiled_convergence(self):
        """FlexShard Owned(0) + torch.compile converges."""
        self.init_test()
        model, inputs, labels = self.get_input()

        model = copy.deepcopy(model)
        _flex_shard_global(
            model,
            self.parallel_dims.get_mesh("fsdp"),
            shard_placement_fn=_single_placement_fn(Owned(0)),
        )
        model = torch.compile(model, backend="aot_eager", fullgraph=True)
        losses = self._train_loop(model, inputs, labels)

        assert losses[-1] < losses[0], (
            f"Owned compiled model did not converge: first loss {losses[0].item()}, "
            f"last loss {losses[-1].item()}"
        )

    def test_flex_shard_owned_eager_vs_compiled(self):
        """FlexShard Owned(0) eager vs compiled: bitwise identical."""
        self.init_test()
        model, inputs, labels = self.get_input()

        # Eager
        eager_model = copy.deepcopy(model)
        _flex_shard_global(
            eager_model,
            self.parallel_dims.get_mesh("fsdp"),
            shard_placement_fn=_single_placement_fn(Owned(0)),
        )
        eager_losses = self._train_loop(eager_model, inputs, labels)

        # Compiled
        compiled_model = copy.deepcopy(model)
        _flex_shard_global(
            compiled_model,
            self.parallel_dims.get_mesh("fsdp"),
            shard_placement_fn=_single_placement_fn(Owned(0)),
        )
        compiled_model = torch.compile(
            compiled_model, backend="aot_eager", fullgraph=True
        )
        compiled_losses = self._train_loop(compiled_model, inputs, labels)

        for step, (e_loss, c_loss) in enumerate(
            zip(eager_losses, compiled_losses, strict=True)
        ):
            assert torch.equal(e_loss, c_loss), (
                f"Step {step}: Owned eager loss {e_loss.item()} != "
                f"Owned compiled loss {c_loss.item()}"
            )


class TestFlexShardPhase5Numerics(FSDPTest):
    """Phase 5 numerical tests: uneven shard, RaggedShard."""

    def init_test(self):
        self.parallel_dims = ParallelDims(
            dp_shard=-1,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=self.world_size,
        )

    def _train_loop(self, model, inputs, labels, steps=STEPS):
        optim = torch.optim.Adam(model.parameters(), lr=1e-4)
        losses = []
        for _ in range(steps):
            optim.zero_grad()
            out = model(inputs)
            loss = cross_entropy_loss(out, labels)
            loss.backward()
            optim.step()
            losses.append(loss)
        return losses

    def test_flex_shard_uneven_shard_convergence(self):
        """Uneven Shard(0) converges: param dim not divisible by world_size."""
        self.init_test()
        # dim=0 is 7, not divisible by world_size (2)
        model = torch.nn.Linear(8, 7)
        inputs = torch.randn(8, 8).cuda()
        labels = torch.randn(8, 7).cuda()

        _flex_shard_global(
            model,
            self.parallel_dims.get_mesh("fsdp"),
        )
        losses = self._train_loop(model, inputs, labels)

        assert losses[-1] < losses[0], (
            f"Uneven shard model did not converge: "
            f"first loss {losses[0].item()}, last loss {losses[-1].item()}"
        )

    def test_flex_shard_uneven_shard_eager_vs_compiled(self):
        """Uneven Shard(0) eager vs compiled: bitwise identical."""
        self.init_test()
        model = torch.nn.Linear(8, 7)
        inputs = torch.randn(8, 8).cuda()
        labels = torch.randn(8, 7).cuda()

        # Eager
        eager_model = copy.deepcopy(model)
        _flex_shard_global(
            eager_model,
            self.parallel_dims.get_mesh("fsdp"),
        )
        eager_losses = self._train_loop(eager_model, inputs, labels)

        # Compiled
        compiled_model = copy.deepcopy(model)
        _flex_shard_global(
            compiled_model,
            self.parallel_dims.get_mesh("fsdp"),
        )
        compiled_model = torch.compile(
            compiled_model, backend="aot_eager", fullgraph=True
        )
        compiled_losses = self._train_loop(compiled_model, inputs, labels)

        for step, (e_loss, c_loss) in enumerate(
            zip(eager_losses, compiled_losses, strict=True)
        ):
            assert torch.equal(e_loss, c_loss), (
                f"Step {step}: Uneven eager loss {e_loss.item()} != "
                f"Uneven compiled loss {c_loss.item()}"
            )

    def test_flex_shard_ragged_shard_convergence(self):
        """RaggedShard converges: variable per-rank allocation."""
        self.init_test()
        ws = self.world_size
        model = torch.nn.Linear(8, 8)
        inputs = torch.randn(8, 8).cuda()
        labels = torch.randn(8, 8).cuda()

        # local_units assigns unequal portions to each rank
        local_units = tuple(range(1, ws + 1))  # (1, 2) for ws=2
        _flex_shard_global(
            model,
            self.parallel_dims.get_mesh("fsdp"),
            shard_placement_fn=_single_placement_fn(RaggedShard((0,), local_units)),
            buckets=[BucketSpec(["*"], reshard_after_forward=False)],
        )
        losses = self._train_loop(model, inputs, labels)

        assert losses[-1] < losses[0], (
            f"RaggedShard model did not converge: "
            f"first loss {losses[0].item()}, last loss {losses[-1].item()}"
        )

    def test_flex_shard_ragged_shard_eager_vs_compiled(self):
        """RaggedShard eager vs compiled: bitwise identical."""
        self.init_test()
        ws = self.world_size
        model = torch.nn.Linear(8, 8)
        inputs = torch.randn(8, 8).cuda()
        labels = torch.randn(8, 8).cuda()

        local_units = tuple(range(1, ws + 1))

        # Eager
        eager_model = copy.deepcopy(model)
        _flex_shard_global(
            eager_model,
            self.parallel_dims.get_mesh("fsdp"),
            shard_placement_fn=_single_placement_fn(RaggedShard((0,), local_units)),
            buckets=[BucketSpec(["*"], reshard_after_forward=False)],
        )
        eager_losses = self._train_loop(eager_model, inputs, labels)

        # Compiled
        compiled_model = copy.deepcopy(model)
        _flex_shard_global(
            compiled_model,
            self.parallel_dims.get_mesh("fsdp"),
            shard_placement_fn=_single_placement_fn(RaggedShard((0,), local_units)),
            buckets=[BucketSpec(["*"], reshard_after_forward=False)],
        )
        compiled_model = torch.compile(
            compiled_model, backend="aot_eager", fullgraph=True
        )
        compiled_losses = self._train_loop(compiled_model, inputs, labels)

        for step, (e_loss, c_loss) in enumerate(
            zip(eager_losses, compiled_losses, strict=True)
        ):
            assert torch.equal(e_loss, c_loss), (
                f"Step {step}: RaggedShard eager loss {e_loss.item()} != "
                f"RaggedShard compiled loss {c_loss.item()}"
            )


class TestFlexShardDoubleWrapping(FSDPTest):
    """Phase 5: EP-style double-wrapping — flex_shard on sub-module then root."""

    def init_test(self):
        self.parallel_dims = ParallelDims(
            dp_shard=-1,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=self.world_size,
        )

    def _train_loop(self, model, inputs, labels, steps=STEPS):
        optim = torch.optim.Adam(model.parameters(), lr=1e-4)
        losses = []
        for _ in range(steps):
            optim.zero_grad()
            out = model(inputs)
            loss = cross_entropy_loss(out, labels)
            loss.backward()
            optim.step()
            losses.append(loss)
        return losses

    def test_double_wrap_convergence(self):
        """flex_shard on sub-module then root converges (EP-style pattern)."""
        self.init_test()

        # Simple MoE-like model: shared layers + "experts" sub-module
        class MoELikeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.shared = torch.nn.Linear(8, 8)
                self.experts = torch.nn.Linear(8, 8)

            def forward(self, x):
                return self.experts(self.shared(x))

        model = MoELikeModel()
        inputs = torch.randn(8, 8).cuda()
        labels = torch.randn(8, 8).cuda()
        fsdp_mesh = self.parallel_dims.get_mesh("fsdp")
        lift_params_to_global_spmd_mesh(model, fsdp_mesh)

        # Double-wrap: experts first, then root (EP pattern)
        flex_shard(
            model.experts,
            fsdp_mesh,
            DataParallelMeshDims(shard="fsdp"),
            shard_placement_fn=per_param_placements,
            buckets=[BucketSpec(["*"])],
        )
        flex_shard(
            model,
            fsdp_mesh,
            DataParallelMeshDims(shard="fsdp"),
            shard_placement_fn=per_param_placements,
            buckets=[BucketSpec(["*"])],
        )

        losses = self._train_loop(model, inputs, labels)
        assert losses[-1] < losses[0], (
            f"Double-wrapped model did not converge: "
            f"first loss {losses[0].item()}, last loss {losses[-1].item()}"
        )

    def test_double_wrap_expert_params_excluded(self):
        """Root flex_shard skips params already managed by sub-module flex_shard."""
        self.init_test()

        class MoELikeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.shared = torch.nn.Linear(8, 8)
                self.experts = torch.nn.Linear(8, 8)

            def forward(self, x):
                return self.experts(self.shared(x))

        model = MoELikeModel()
        fsdp_mesh = self.parallel_dims.get_mesh("fsdp")
        lift_params_to_global_spmd_mesh(model, fsdp_mesh)

        # Wrap experts first
        flex_shard(
            model.experts,
            fsdp_mesh,
            DataParallelMeshDims(shard="fsdp"),
            shard_placement_fn=per_param_placements,
            buckets=[BucketSpec(["*"])],
        )
        expert_storages = len(model.experts._dstorages)

        # Wrap root — should only manage shared params, not expert params
        flex_shard(
            model,
            fsdp_mesh,
            DataParallelMeshDims(shard="fsdp"),
            shard_placement_fn=per_param_placements,
            buckets=[BucketSpec(["*"])],
        )

        # Root should have its own storage(s) for shared params only
        root_storages = model._dstorages
        # Expert params should still be managed by the expert's storage
        assert expert_storages == len(
            model.experts._dstorages
        ), "Expert DStorage count changed after root wrapping"
        # Root storage should only contain shared params (weight + bias)
        root_param_fqns = set()
        for s in root_storages:
            root_param_fqns.update(s._param_infos.keys())
        assert (
            "experts.weight" not in root_param_fqns
        ), "Expert params should be excluded from root DStorage"
        assert (
            "shared.weight" in root_param_fqns
        ), "Shared params should be in root DStorage"


STEPS_FULL_MODEL = 20
FLEX_SHARD_PARALLELISM = "--parallelism.data_parallel_shard_degree=-1"


def _run_flex_shard_loss_compare(
    test_options_extra: str = "", assert_equal: bool = False
) -> bool:
    """Compare FlexShard vs FSDP2 eager numerics using loss_compare.py.

    FlexShard uses _c10d_functional ops for reduce-scatter, which accumulates
    in a different order than FSDP2's reduce-scatter. This produces a small
    per-step difference (~2e-5) that compounds over training. The unit test
    (test_flex_shard_vs_simple_fsdp) verifies bitwise eager-mode equivalence.
    """
    import subprocess
    import sys

    test_options = FLEX_SHARD_PARALLELISM
    if test_options_extra:
        test_options += f" {test_options_extra}"
    cmd = [
        sys.executable,
        "scripts/loss_compare.py",
        ".",
        ".",
        "--baseline-module=llama3",
        "--baseline-config=llama3_debugmodel_ce_loss",
        "--test-module=graph_trainer.flex_shard_llama3",
        "--test-config=graph_trainer_flex_shard_llama3_debugmodel",
        f"--steps={STEPS_FULL_MODEL}",
        f"--baseline-options={FLEX_SHARD_PARALLELISM}",
        f"--test-options={test_options}",
    ]
    if assert_equal:
        cmd.append("--assert-equal")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print("loss_compare.py failed")
    return result.returncode == 0


class TestFlexShardNumericsFullModel(unittest.TestCase):
    """Full Llama3 model FlexShard vs FSDP2 eager numerics via loss_compare.py.

    Verifies FlexShard completes end-to-end training with the full model.
    Bitwise eager-mode equivalence is verified by the unit test above
    (test_flex_shard_vs_simple_fsdp). The compiled FlexShard reduce-scatter
    uses _c10d_functional ops that accumulate in a different order than FSDP2,
    producing ~2e-5 per-step differences (not a correctness issue).

    Requires 8 GPUs (loss_compare.py default). Run with:
        pytest torchtitan/experiments/flex_shard/tests/test_numerics.py -x \
            -k FullModel
    """

    def test_flex_shard_jit_full_model(self):
        self.assertTrue(
            _run_flex_shard_loss_compare(test_options_extra="--compile.mode jit"),
        )

    def test_flex_shard_aot_full_model(self):
        self.assertTrue(
            _run_flex_shard_loss_compare(test_options_extra="--compile.mode aot"),
        )


if __name__ == "__main__":
    unittest.main()
