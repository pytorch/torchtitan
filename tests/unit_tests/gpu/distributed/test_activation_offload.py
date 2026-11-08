# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import os
import unittest
import unittest.mock
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch_remat as remat
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.elastic.utils.distributed import get_free_port
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard

from torchtitan.distributed.activation_offload import (
    _State,
    ActivationOffloadConfig,
    ActivationOffloader,
)
from torchtitan.distributed.activation_remat import (
    ActivationRematConfig,
    DefaultActivationRematConfig,
    FullActivationRematConfig,
)
from torchtitan.models.base.module import Module, ModuleDict
from torchtitan.models.common.linear import Linear

# Legend: T = tokens, D = model dim, F = hidden dim.

_DIM = 1024
_NUM_TOKENS = 4096
_NUM_LAYERS = 6


class _Block(Module):
    def __init__(self):
        super().__init__()
        self.w1 = Linear.Config(
            in_features=_DIM, out_features=2 * _DIM, bias=False
        ).build()
        self.w2 = Linear.Config(
            in_features=2 * _DIM, out_features=_DIM, bias=False
        ).build()

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        hidden_TF = self.w1(x_TD)
        remat.recompute_needs_tensor(hidden_TF)
        out_TD = self.w2(F.gelu(hidden_TF))
        remat.recompute_needs_tensor(out_TD)
        return x_TD + out_TD


class _InputRetainingBlock(_Block):
    """Holds a reference to its input, so offloading it frees nothing."""

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        self.last_input_TD = x_TD
        return super().forward(x_TD)


class _InternalWeightBlock(Module):
    """A save region that uses its weight internally instead of taking it as
    a region input, so the weight it saves for backward reaches pack. The
    matmul saves a plain-tensor view of the whole (D, D) weight storage (as a
    TP weight's local tensor would be), so only the parameter-storage check
    keeps it on GPU."""

    def __init__(self):
        super().__init__()
        self.weight_DD = torch.nn.Parameter(torch.randn(_DIM, _DIM) / _DIM**0.5)

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        out_TD = remat.region(
            lambda x_TD: x_TD @ self.weight_DD.view(_DIM, _DIM),
            self.remat_region_name("proj"),
            recompute=self.remat_should_recompute("proj"),
        )(x_TD)
        remat.recompute_needs_tensor(out_TD)
        return x_TD + out_TD


class _ViewsBlock(Module):
    """A save region whose ops save two differently shaped views of one storage."""

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        def views(x_TD: torch.Tensor) -> torch.Tensor:
            scaled_TD = x_TD * 2
            # sin saves a (T * D / 16, 16) view of scaled_TD; cos saves scaled_TD.
            return scaled_TD.view(-1, 16).sin().view_as(scaled_TD) + scaled_TD.cos()

        out_TD = remat.region(
            views,
            self.remat_region_name("views"),
            recompute=self.remat_should_recompute("views"),
        )(x_TD)
        remat.recompute_needs_tensor(out_TD)
        return x_TD + out_TD


class _FusedSlicesBlock(Module):
    """A save region that saves strided slices of one fused buffer, like the
    q, k and v slices of a fused QKV projection."""

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        def fused_slices(x_TD: torch.Tensor) -> torch.Tensor:
            qkv_T3D = torch.cat([x_TD * 2, x_TD * 3, x_TD * 4], dim=1)
            q_TD, k_TD, v_TD = qkv_T3D.split(_DIM, dim=1)
            # sin, cos and tanh each save their (non-contiguous) slice.
            return q_TD.sin() + k_TD.cos() + v_TD.tanh()

        out_TD = remat.region(
            fused_slices,
            self.remat_region_name("fused_slices"),
            recompute=self.remat_should_recompute("fused_slices"),
        )(x_TD)
        remat.recompute_needs_tensor(out_TD)
        return x_TD + out_TD


class _ExpandedSaveBlock(Module):
    """A save region that saves a broadcast (stride-0) view of a small tensor,
    whose logical size is far larger than the storage it pins."""

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        def expanded(x_TD: torch.Tensor) -> torch.Tensor:
            scale_D = x_TD.mean(dim=0)
            # mul saves both operands, including the (T, D) view of scale_D.
            return x_TD * scale_D.expand_as(x_TD)

        out_TD = remat.region(
            expanded,
            self.remat_region_name("expanded"),
            recompute=self.remat_should_recompute("expanded"),
        )(x_TD)
        remat.recompute_needs_tensor(out_TD)
        return x_TD + out_TD


class _TwoRegionBlock(Module):
    """Two save regions; "cos" saves the output of "sin" as its input."""

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        # sin saves its input x_TD * 2, an internal tensor of the region.
        sin_TD = remat.region(
            lambda x_TD: (x_TD * 2).sin(),
            self.remat_region_name("sin"),
            recompute=self.remat_should_recompute("sin"),
        )(x_TD)
        # cos saves sin_TD, a storage that "sin" also persists as its output.
        out_TD = remat.region(
            lambda sin_TD: sin_TD.cos(),
            self.remat_region_name("cos"),
            recompute=self.remat_should_recompute("cos"),
        )(sin_TD)
        remat.recompute_needs_tensor(out_TD)
        return x_TD + out_TD


class _Model(Module):
    def __init__(self, block_cls: type[Module] | None = None):
        super().__init__()
        block_cls = block_cls or _Block
        self.layers = ModuleDict({str(i): block_cls() for i in range(_NUM_LAYERS)})

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        h_TD = x_TD
        for layer in self.layers.values():
            h_TD = layer(h_TD)
        return h_TD


def _build(
    remat_config: ActivationRematConfig | None,
    reference: _Model | None = None,
    offload: ActivationOffloadConfig | None = None,
) -> _Model:
    model = deepcopy(reference) if reference is not None else _Model()
    if remat_config is not None:
        remat_config.apply(model)
    if offload is not None:
        offload.apply(model)
    return model.cuda()


def _grads(model: _Model, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    """Run all forwards before any backward, like gradient accumulation."""
    model.zero_grad(set_to_none=True)
    losses = [
        model(x_TD.clone().requires_grad_(True)).square().mean() for x_TD in inputs
    ]
    for loss in losses:
        loss.backward()
    return [p.grad.detach().clone() for p in model.parameters()]


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestActivationOffload(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.reference = _Model()
        self.inputs = [torch.randn(_NUM_TOKENS, _DIM, device="cuda") for _ in range(2)]

    def _assert_bitwise_equal(self, remat_type, offload: ActivationOffloadConfig):
        baseline = _build(remat_type(), self.reference)
        offloaded = _build(remat_type(), self.reference, offload)
        expected = _grads(baseline, self.inputs)
        actual = _grads(offloaded, self.inputs)
        for e, a in zip(expected, actual, strict=True):
            torch.testing.assert_close(a, e, rtol=0, atol=0)
        return offloaded

    def test_full_remat_block_input_matches(self):
        self._assert_bitwise_equal(FullActivationRematConfig, ActivationOffloadConfig())

    def test_saved_regions_match(self):
        self._assert_bitwise_equal(
            DefaultActivationRematConfig,
            ActivationOffloadConfig(offload_regions=["*"]),
        )

    def test_offload_without_remat_matches(self):
        model = self._assert_bitwise_equal(
            lambda: None, ActivationOffloadConfig(offload_saves_outside_regions=True)
        )
        stats = _offloader(model).forward_stats
        # Without remat no save has a region or is a checkpoint input. Blocks
        # 0..4 each offload what their two matmuls and gelu save: the block
        # input (T, D) and the hidden (T, F), saved by w2 and gelu (one transfer
        # per storage), plus gelu's output (T, F) saved by w2. Block 0's input
        # is the model's input, which the caller holds, so it stays on GPU.
        activation_bytes = _NUM_TOKENS * _DIM * 4
        self.assertEqual(
            stats.offloaded_bytes_by_region,
            {"outside any region": (5 * 5 - 1) * activation_bytes},
        )
        self.assertEqual(stats.still_resident_bytes, {})

    def test_offload_without_remat_keeps_unselected_saves(self):
        offload = ActivationOffloadConfig(offload_regions=["*"])
        model = self._assert_bitwise_equal(lambda: None, offload)
        self.assertEqual(_offloader(model).forward_stats.offloaded_bytes, 0)

    def test_offloads_only_selected_regions(self):
        self.reference = _Model(_TwoRegionBlock)
        save_all = lambda **kwargs: ActivationRematConfig(save_regions=["*"], **kwargs)
        activation_bytes = _NUM_TOKENS * _DIM * 4
        # Per offloaded block (0..4): "sin" retains x * 2; "cos" retains
        # sin_TD, which "sin" also persists, and its own output. Block inputs
        # are offloaded for blocks 1..4.
        expected = {
            "sin": {"sin": 5 * activation_bytes, "block input": 4 * activation_bytes},
            "cos": {"cos": 10 * activation_bytes, "block input": 4 * activation_bytes},
        }
        for region, expected_by_region in expected.items():
            with self.subTest(offload_regions=[region]):
                model = self._assert_bitwise_equal(
                    save_all, ActivationOffloadConfig(offload_regions=[region])
                )
                stats = _offloader(model).forward_stats
                self.assertEqual(stats.offloaded_bytes_by_region, expected_by_region)
                self.assertEqual(stats.still_resident_bytes, {})

    def test_differently_shaped_views_share_one_transfer(self):
        self.reference = _Model(_ViewsBlock)
        model = self._assert_bitwise_equal(
            lambda **kwargs: ActivationRematConfig(save_regions=["*"], **kwargs),
            ActivationOffloadConfig(offload_regions=["*"]),
        )
        del model

    def test_partial_views_free_their_storage(self):
        self.reference = _Model(_FusedSlicesBlock)
        save_all = lambda **kwargs: ActivationRematConfig(save_regions=["*"], **kwargs)
        self._assert_bitwise_equal(
            save_all, ActivationOffloadConfig(offload_regions=["*"])
        )
        saved, stats = self._forward_memory_saved(
            save_all, ActivationOffloadConfig(offload_regions=["*"])
        )
        # Blocks 0..4 each offload one fused buffer for its three slices.
        fused_bytes = _NUM_TOKENS * 3 * _DIM * 4
        self.assertGreaterEqual(stats.offloaded_bytes, (_NUM_LAYERS - 1) * fused_bytes)
        self.assertEqual(stats.still_resident_bytes, {})
        self.assertEqual(saved, stats.offloaded_bytes)

    def test_skipped_bytes_count_pinned_storage(self):
        self.reference = _Model(_ExpandedSaveBlock)
        model = _build(
            ActivationRematConfig(save_regions=["*"]),
            self.reference,
            ActivationOffloadConfig(offload_regions=["*"]),
        )
        model(self.inputs[0].clone().requires_grad_(True)).square().mean().backward()
        # Each offloaded block keeps one (D,) storage, not a (T, D) view's worth.
        self.assertEqual(
            _offloader(model).forward_stats.skipped_bytes["below min_bytes"],
            (_NUM_LAYERS - 1) * _DIM * 4,
        )

    def test_never_offloads_parameters(self):
        self.reference = _Model(_InternalWeightBlock)
        model = self._assert_bitwise_equal(
            lambda **kwargs: ActivationRematConfig(save_regions=["*"], **kwargs),
            ActivationOffloadConfig(offload_regions=["*"]),
        )
        # Offloaded per forward: each offloaded block's proj output (blocks
        # 0..4) and block input (blocks 1..4; block 0's is the model input).
        # Offloading the (D, D) weights would add 5 * _DIM * _DIM * 4 bytes.
        activation_bytes = _NUM_TOKENS * _DIM * 4
        self.assertEqual(
            _offloader(model).forward_stats.offloaded_bytes,
            (_NUM_LAYERS - 1 + _NUM_LAYERS - 2) * activation_bytes,
        )

    def test_reloads_on_demand_without_triggers(self):
        # Every decoder group is prefetched by a backward trigger; remove the
        # triggers to exercise unpack's on-demand fallback.
        with unittest.mock.patch.object(ActivationOffloader, "_prefetch_on_grad"):
            model = self._assert_bitwise_equal(
                FullActivationRematConfig, ActivationOffloadConfig()
            )
        # Blocks 1..4 per microbatch, two microbatches.
        self.assertEqual(_offloader(model).num_on_demand_reloads, 2 * (_NUM_LAYERS - 2))

    def _forward_memory_saved(self, remat_type, offload: ActivationOffloadConfig):
        """Return (GPU bytes saved after forward, the offloader's forward stats)."""

        def stashed_bytes(model: _Model) -> int:
            # Models from earlier tests sit in reference cycles (a wrapped
            # block forward refers back to its block); collect them now so
            # they are not freed in the middle of the measurement.
            gc.collect()
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            loss = model(self.inputs[0].clone().requires_grad_(True)).square().mean()
            torch.cuda.synchronize()
            stashed = torch.cuda.memory_allocated() - before
            loss.backward()
            return stashed

        baseline = _build(remat_type(), self.reference)
        offloaded = _build(remat_type(), self.reference, offload)
        saved = stashed_bytes(baseline) - stashed_bytes(offloaded)
        return saved, _offloader(offloaded).forward_stats

    def test_forward_frees_offloaded_block_inputs(self):
        saved, stats = self._forward_memory_saved(
            FullActivationRematConfig, ActivationOffloadConfig()
        )
        # Block 0's input is the model input and the last block is skipped, so the
        # inputs of blocks 1..4 move to CPU.
        input_bytes = _NUM_TOKENS * _DIM * 4
        self.assertEqual(stats.offloaded_bytes, (_NUM_LAYERS - 2) * input_bytes)
        self.assertEqual(stats.still_resident_bytes, {})
        self.assertEqual(saved, stats.offloaded_bytes)

    def test_forward_frees_offloaded_saved_regions(self):
        saved, stats = self._forward_memory_saved(
            DefaultActivationRematConfig,
            ActivationOffloadConfig(offload_regions=["*"]),
        )
        self.assertGreater(
            stats.offloaded_bytes, (_NUM_LAYERS - 2) * _NUM_TOKENS * _DIM * 4
        )
        self.assertEqual(stats.still_resident_bytes, {})
        self.assertEqual(saved, stats.offloaded_bytes)

    def test_reports_offloads_that_stay_resident(self):
        self.reference = _Model(_InputRetainingBlock)
        with self.assertLogs("torchtitan.distributed.activation_offload", "WARNING"):
            saved, stats = self._forward_memory_saved(
                FullActivationRematConfig, ActivationOffloadConfig()
            )
        self.assertGreater(stats.offloaded_bytes, 0)
        self.assertEqual(
            sum(stats.still_resident_bytes.values()), stats.offloaded_bytes
        )
        self.assertEqual(saved, 0)

    def test_backward_releases_reloaded_copies(self):
        """Reloaded GPU copies must not outlive backward while losses are kept."""

        def bytes_after_backward(model: _Model) -> int:
            gc.collect()
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            losses = [
                model(x_TD.clone().requires_grad_(True)).square().mean()
                for x_TD in self.inputs
            ]
            for loss in losses:
                loss.backward()
            torch.cuda.synchronize()
            # `losses` is still alive here, as in a training loop that logs it.
            return torch.cuda.memory_allocated() - before

        baseline = _build(FullActivationRematConfig(), self.reference)
        offloaded = _build(
            FullActivationRematConfig(), self.reference, ActivationOffloadConfig()
        )
        # Warm up so parameter gradients are allocated in both before measuring.
        bytes_after_backward(baseline)
        bytes_after_backward(offloaded)
        self.assertEqual(
            bytes_after_backward(offloaded), bytes_after_backward(baseline)
        )

    def test_cuda_graph_reuses_host_buffers(self):
        """A captured gradient-accumulation step replays bitwise equal to eager,
        reuses one microbatch's pinned buffers across microbatches, and mixes
        with eager steps that share those buffers."""
        expected = _grads(
            _build(FullActivationRematConfig(), self.reference), self.inputs
        )
        model = _build(
            FullActivationRematConfig(), self.reference, ActivationOffloadConfig()
        )
        static_inputs = [x_TD.clone().requires_grad_(True) for x_TD in self.inputs]

        def step() -> None:
            for x_TD in static_inputs:
                model(x_TD).square().mean().backward()

        def check_grads() -> None:
            for e, p in zip(expected, model.parameters(), strict=True):
                torch.testing.assert_close(p.grad, e, rtol=0, atol=0)

        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            model.zero_grad(set_to_none=False)
            step()
        torch.cuda.current_stream().wait_stream(side_stream)
        model.zero_grad(set_to_none=False)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        for _ in range(2):
            model.zero_grad(set_to_none=False)
            graph.replay()
            check_grads()
            model.zero_grad(set_to_none=False)
            step()
            check_grads()
        # Blocks 1..4 offload one input each; the second microbatch reuses
        # the first one's buffers, in eager mode and in the graph alike.
        pool = _offloader(model)._host_buffers._free
        self.assertEqual(
            sum(len(buffers) for buffers in pool.values()), _NUM_LAYERS - 2
        )

    def _run_schedule(
        self, model: _Model, actions: list[str], inputs: list[torch.Tensor], **reload
    ) -> list[torch.Tensor]:
        """Play one pipeline stage's actions ("F<mb>" / "B<mb>") the way a
        schedule drives a deferred offloader: wait_offload one action after
        each forward, reload_next_backward before each backward."""
        offloader = _offloader(model)
        model.zero_grad(set_to_none=True)
        losses: dict[int, torch.Tensor] = {}
        forward_pending = False
        for action in actions:
            kind, mb = action[0], int(action[1:])
            if kind == "F":
                losses[mb] = (
                    model(inputs[mb].clone().requires_grad_(True)).square().mean()
                )
            else:
                offloader.reload_next_backward(**reload)
                losses.pop(mb).backward()
            if forward_pending:
                offloader.wait_offload()
            forward_pending = kind == "F"
        return [p.grad.detach().clone() for p in model.parameters()]

    def _schedule_driven_model(self) -> _Model:
        model = _build(
            FullActivationRematConfig(),
            self.reference,
            ActivationOffloadConfig(skip_last_layers=0),
        )
        _offloader(model).defer_to_schedule()
        # Like a pipeline stage's metadata inference: a forward and backward
        # outside the schedule's actions, which reload_next_backward skips.
        model(self.inputs[0].clone().requires_grad_(True)).square().mean().backward()
        return model

    def test_schedule_driven_offloads_every_block(self):
        inputs = [torch.randn(_NUM_TOKENS, _DIM, device="cuda") for _ in range(4)]
        expected = _grads(_build(FullActivationRematConfig(), self.reference), inputs)
        model = self._schedule_driven_model()
        offloader = _offloader(model)
        actual = self._run_schedule(
            model, ["F0", "F1", "B0", "F2", "B1", "F3", "B2", "B3"], inputs
        )
        for e, a in zip(expected, actual, strict=True):
            torch.testing.assert_close(a, e, rtol=0, atol=0)
        # Blocks 1..5: nothing is skipped, and every reload was prefetched.
        input_bytes = _NUM_TOKENS * _DIM * 4
        self.assertEqual(
            offloader.forward_stats.offloaded_bytes, (_NUM_LAYERS - 1) * input_bytes
        )
        self.assertEqual(offloader.num_on_demand_reloads, 0)
        self.assertEqual(len(offloader._microbatch_groups), 0)

    def test_schedule_driven_wait_offload_frees_last_group(self):
        model = self._schedule_driven_model()
        offloader = _offloader(model)
        gc.collect()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        loss = model(self.inputs[1].clone().requires_grad_(True)).square().mean()
        after_forward = torch.cuda.memory_allocated()
        offloader.wait_offload()
        after_wait = torch.cuda.memory_allocated()
        # The last block's input is held until wait_offload frees it.
        self.assertEqual(after_forward - after_wait, _NUM_TOKENS * _DIM * 4)
        self.assertLess(after_wait - before, after_forward - before)
        last_group = offloader._microbatch_groups[-1][-1]
        (offloaded,) = last_group.storages.values()
        self.assertIs(offloaded.state, _State.OFFLOADED)
        offloader.reload_next_backward()
        # Queued by the schedule call, before backward reaches the trigger.
        self.assertIs(offloaded.state, _State.RELOADING)
        loss.backward()
        self.assertEqual(offloader.num_on_demand_reloads, 0)

    def test_schedule_driven_reloads_whole_stage(self):
        inputs = [torch.randn(_NUM_TOKENS, _DIM, device="cuda") for _ in range(2)]
        expected = _grads(_build(FullActivationRematConfig(), self.reference), inputs)
        model = self._schedule_driven_model()
        actual = self._run_schedule(
            model, ["F0", "F1", "B0", "B1"], inputs, num_groups=_NUM_LAYERS
        )
        for e, a in zip(expected, actual, strict=True):
            torch.testing.assert_close(a, e, rtol=0, atol=0)
        self.assertEqual(_offloader(model).num_on_demand_reloads, 0)

    def test_backward_triggers_prefetch_every_group(self):
        offload = ActivationOffloadConfig()
        model = _build(FullActivationRematConfig(), self.reference, offload)
        _grads(model, self.inputs)
        self.assertEqual(_offloader(model).num_on_demand_reloads, 0)


def _run_fsdp_cpu_offload_composition(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))
        torch.manual_seed(0)
        reference = _Model()
        inputs = [torch.randn(_NUM_TOKENS, _DIM, device="cuda") for _ in range(2)]

        def build(offload: ActivationOffloadConfig | None) -> _Model:
            model = deepcopy(reference)
            DefaultActivationRematConfig().apply(model)
            for block in model.layers.values():
                fully_shard(block, mesh=mesh, offload_policy=CPUOffloadPolicy())
            fully_shard(model, mesh=mesh, offload_policy=CPUOffloadPolicy())
            if offload is not None:
                offload.apply(model)
            return model

        expected = _grads(build(None), inputs)
        for shared in (False, True):
            model = build(
                ActivationOffloadConfig(
                    offload_regions=["*"], reload_on_fsdp_copy_in_stream=shared
                )
            )
            actual = _grads(model, inputs)
            for e, a in zip(expected, actual, strict=True):
                torch.testing.assert_close(a.to_local(), e.to_local(), rtol=0, atol=0)
            offloader = _offloader(model)
            # pyrefly: ignore [missing-attribute]
            comm_ctx = fully_shard.state(model)._comm_ctx
            assert (
                offloader._reload_stream == comm_ctx.all_gather_copy_in_stream
            ) == shared
            assert offloader.num_on_demand_reloads == 0
            assert offloader.forward_stats.offloaded_bytes > 0
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two GPUs")
class TestActivationOffloadWithFSDPCPUOffload(unittest.TestCase):
    """Activation offload composed with FSDP2 CPU parameter offload. Two ranks:
    on one rank FSDP2 skips the all-gather and copies parameters on the
    compute stream, so its copy-in stream carries no parameter copies."""

    def test_reload_on_fsdp_copy_in_stream_matches(self):
        mp.spawn(
            _run_fsdp_cpu_offload_composition,
            args=(2, get_free_port()),
            nprocs=2,
            join=True,
        )

    def test_reload_on_fsdp_copy_in_stream_requires_fsdp(self):
        with self.assertRaisesRegex(ValueError, "fully_shard"):
            ActivationOffloadConfig(reload_on_fsdp_copy_in_stream=True).apply(_Model())


def _offloader(model: _Model) -> ActivationOffloader:
    for hook in model._forward_pre_hooks.values():
        offloader = getattr(hook, "__self__", None)
        if isinstance(offloader, ActivationOffloader):
            return offloader
    raise AssertionError("no ActivationOffloader registered on the model")


if __name__ == "__main__":
    unittest.main()
