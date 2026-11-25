# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import functools
import gc
import os
import unittest
import unittest.mock
from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch_remat as remat
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.elastic.utils.distributed import get_free_port
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard
from torch.testing._internal.two_tensor import TwoTensor

from torchtitan.distributed.activation_offload import ActivationOffloader
from torchtitan.distributed.activation_remat import (
    ActivationRematConfig,
    BlockRematOverride,
    DefaultActivationRematConfig,
    FullActivationRematConfig,
)
from torchtitan.models.base.module import Module, ModuleDict
from torchtitan.models.common.linear import Linear
from torchtitan.quantization._fsdp_tensor import _ShardedFSDPTensor

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


class _SavesTwoTensor(torch.autograd.Function):
    """sin(5 x), saving a tensor subclass (2 x, 3 x) for backward."""

    @staticmethod
    def forward(ctx, x_TD: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(TwoTensor(x_TD * 2, x_TD * 3))
        return (x_TD * 5).sin()

    @staticmethod
    def backward(ctx, grad_TD: torch.Tensor) -> torch.Tensor:
        (pair_TD,) = ctx.saved_tensors
        return grad_TD * 5 * (pair_TD.a + pair_TD.b).cos()


class _SubclassSaveBlock(Module):
    """A save region that saves a tensor subclass, as MXFP8 Dist-MoE saves
    its unsharded weight wrapper."""

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        out_TD = remat.region(
            _SavesTwoTensor.apply,
            self.remat_region_name("pair"),
            recompute=self.remat_should_recompute("pair"),
        )(x_TD)
        remat.recompute_needs_tensor(out_TD)
        return x_TD + out_TD


class _ExternallyFreedSaveBlock(Module):
    """A save region that saves a buffer whose storage is then freed by
    something other than autograd, as FSDP2's reshard frees unsharded
    parameter data."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "table_DD", torch.randn(_DIM, _DIM) / _DIM**0.5, persistent=False
        )

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        # The matmul saves table_DD for x_TD's gradient.
        out_TD = remat.region(
            lambda x_TD: x_TD @ self.table_DD,
            self.remat_region_name("proj"),
            recompute=self.remat_should_recompute("proj"),
        )(x_TD)
        self.table_DD.untyped_storage().resize_(0)
        remat.recompute_needs_tensor(out_TD)
        return x_TD + out_TD


@dataclasses.dataclass(frozen=True)
class _CopyOperands:
    weight: torch.Tensor


class _CopyShardedTensor(_ShardedFSDPTensor):
    """A weight format whose operand is a plain copy of the unsharded weight.
    As with MXFP8's qdata and scales, FSDP2 allocates the operand's storage
    at unshard and frees it at reshard."""

    def _build_operands(
        self, logical_tensor: torch.Tensor, out: _CopyOperands | None = None
    ) -> _CopyOperands:
        if out is None:
            return _CopyOperands(logical_tensor.clone())
        out.weight.copy_(logical_tensor)
        return out


class _SavesWeightOperand(torch.autograd.Function):
    """x @ w^T for an unsharded _CopyShardedTensor weight, saving its
    operand, as MXFP8 Dist-MoE saves the weight's qdata and scales."""

    @staticmethod
    def forward(ctx, x_TD: torch.Tensor, weight_DD: torch.Tensor) -> torch.Tensor:
        operand_DD = weight_DD.operands.weight
        ctx.save_for_backward(x_TD, operand_DD)
        return x_TD @ operand_DD.T

    @staticmethod
    def backward(ctx, grad_TD: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_TD, operand_DD = ctx.saved_tensors
        return grad_TD @ operand_DD, grad_TD.T @ x_TD


class _OperandWeightBlock(Module):
    """A save region that saves its subclass weight's FSDP2-managed operand."""

    def __init__(self):
        super().__init__()
        self.weight_DD = torch.nn.Parameter(
            _CopyShardedTensor(torch.randn(_DIM, _DIM) / _DIM**0.5)
        )

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        out_TD = remat.region(
            lambda x_TD: _SavesWeightOperand.apply(x_TD, self.weight_DD),
            self.remat_region_name("proj"),
            recompute=self.remat_should_recompute("proj"),
        )(x_TD)
        remat.recompute_needs_tensor(out_TD)
        return x_TD + out_TD


class _Model(Module):
    def __init__(self, block_cls: type[Module] | None = None):
        super().__init__()
        block_cls = block_cls or _Block
        self.layers = ModuleDict({str(i): block_cls() for i in range(_NUM_LAYERS)})
        # The offloader ActivationRematConfig.apply installed (see _apply).
        self.offloader: ActivationOffloader | None = None

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        h_TD = x_TD
        for layer in self.layers.values():
            h_TD = layer(h_TD)
        return h_TD


def _apply(
    remat_config: ActivationRematConfig,
    model: _Model,
    **kwargs,
) -> None:
    """Run ``remat_config.apply`` on ``model`` and keep
    the offloader it installs, if any, on ``model.offloader`` for the
    assertions: ``apply`` returns nothing, since its hooks keep it alive."""
    init = ActivationOffloader.__init__
    created: list[ActivationOffloader] = []

    def record(offloader, *args, **init_kwargs):
        init(offloader, *args, **init_kwargs)
        created.append(offloader)

    with unittest.mock.patch.object(ActivationOffloader, "__init__", record):
        remat_config.apply(model, **kwargs)
    if created:
        (model.offloader,) = created


_NUM_ENCODER_LAYERS = 3


class _Encoder(Module):
    def __init__(self):
        super().__init__()
        self.layers = ModuleDict({str(i): _Block() for i in range(_NUM_ENCODER_LAYERS)})

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        for layer in self.layers.values():
            x_TD = layer(x_TD)
        return x_TD


class _EncoderDecoderModel(_Model):
    """An encoder whose output feeds the decoder blocks, as in a multimodal
    model."""

    def __init__(self):
        super().__init__()
        self.vision_encoder = _Encoder()

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return super().forward(self.vision_encoder(x_TD))


def _build_encoder_decoder(
    remat_config: ActivationRematConfig, reference: _EncoderDecoderModel
) -> _EncoderDecoderModel:
    """Apply ``remat_config`` as multimodal parallelize does."""
    model = deepcopy(reference)
    _apply(
        remat_config, model, block_container_fqns=["vision_encoder.layers", "layers"]
    )
    return model.cuda()


def _build(
    remat_config: ActivationRematConfig, reference: _Model | None = None
) -> _Model:
    """Apply ``remat_config``, which also installs offloading if it offloads
    anything."""
    model = deepcopy(reference) if reference is not None else _Model()
    _apply(remat_config, model)
    return model.cuda()


def _without_offload(remat_config: ActivationRematConfig) -> ActivationRematConfig:
    """The same remat policy with every offload field cleared."""
    overrides = [
        dataclasses.replace(o, offload_block_input=None, offload_regions=None)
        for o in remat_config.block_overrides
        if o.save_regions is not None or o.recompute_regions is not None
    ]
    return dataclasses.replace(
        remat_config,
        offload_block_input=False,
        offload_regions=[],
        block_overrides=overrides,
    )


def _full_remat_offloading_inputs() -> ActivationRematConfig:
    return FullActivationRematConfig(offload_block_input=True)


def _save_all(offload_regions: list[str]) -> ActivationRematConfig:
    """Retain every region, offloading block inputs and ``offload_regions``."""
    return ActivationRematConfig(
        save_regions=["*"], offload_block_input=True, offload_regions=offload_regions
    )


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

    def _assert_bitwise_equal(self, remat_config: ActivationRematConfig) -> _Model:
        """Compare against the same remat policy without offloading."""
        baseline = _build(_without_offload(remat_config), self.reference)
        self.assertIsNone(baseline.offloader)
        offloaded = _build(remat_config, self.reference)
        expected = _grads(baseline, self.inputs)
        actual = _grads(offloaded, self.inputs)
        for e, a in zip(expected, actual, strict=True):
            torch.testing.assert_close(a, e, rtol=0, atol=0)
        return offloaded

    def test_full_remat_block_input_matches(self):
        self._assert_bitwise_equal(_full_remat_offloading_inputs())

    def test_saved_regions_match(self):
        self._assert_bitwise_equal(
            DefaultActivationRematConfig(
                offload_block_input=True, offload_regions=["*"]
            )
        )

    def test_offloads_only_selected_regions(self):
        self.reference = _Model(_TwoRegionBlock)
        activation_bytes = _NUM_TOKENS * _DIM * 4
        # Per block (0..5): "sin" retains x * 2; "cos" retains sin_TD, which
        # "sin" also persists, and its own output. Block inputs are offloaded
        # for blocks 1..5 (block 0's is the model input).
        expected = {
            "sin": {"sin": 6 * activation_bytes, "block input": 5 * activation_bytes},
            "cos": {"cos": 12 * activation_bytes, "block input": 5 * activation_bytes},
        }
        for region, expected_by_region in expected.items():
            with self.subTest(offload_regions=[region]):
                model = self._assert_bitwise_equal(_save_all([region]))
                stats = _offloader(model).forward_stats
                self.assertEqual(stats.offloaded_bytes_by_region, expected_by_region)
                self.assertEqual(stats.still_resident_bytes, {})

    def test_differently_shaped_views_share_one_transfer(self):
        self.reference = _Model(_ViewsBlock)
        model = self._assert_bitwise_equal(_save_all(["*"]))
        del model

    def test_partial_views_free_their_storage(self):
        self.reference = _Model(_FusedSlicesBlock)
        self._assert_bitwise_equal(_save_all(["*"]))
        saved, stats = self._forward_memory_saved(_save_all(["*"]))
        # Every block offloads one fused buffer for its three slices.
        fused_bytes = _NUM_TOKENS * 3 * _DIM * 4
        self.assertGreaterEqual(stats.offloaded_bytes, _NUM_LAYERS * fused_bytes)
        self.assertEqual(stats.still_resident_bytes, {})
        self.assertEqual(saved, stats.offloaded_bytes)

    def test_skipped_bytes_count_pinned_storage(self):
        self.reference = _Model(_ExpandedSaveBlock)
        model = _build(_save_all(["*"]), self.reference)
        model(self.inputs[0].clone().requires_grad_(True)).square().mean().backward()
        # Each block keeps one (D,) storage, not a (T, D) view's worth.
        self.assertEqual(
            _offloader(model).forward_stats.skipped_bytes["smaller than 1 MiB"],
            _NUM_LAYERS * _DIM * 4,
        )

    def test_encoder_blocks_chain_into_the_decoder(self):
        """An encoder's blocks, its last included, offload through the model
        part's offloader, chained into the decoder blocks in run order."""
        reference = _EncoderDecoderModel()
        remat_config = _full_remat_offloading_inputs()
        baseline = _build_encoder_decoder(_without_offload(remat_config), reference)
        model = _build_encoder_decoder(remat_config, reference)
        freed = []
        free_gpu_copies = ActivationOffloader._free_gpu_copies

        def record_free(offloader, group):
            freed.append(group.module_fqn)
            free_gpu_copies(offloader, group)

        expected = _grads(baseline, self.inputs)
        with unittest.mock.patch.object(
            ActivationOffloader, "_free_gpu_copies", record_free
        ):
            actual = _grads(model, self.inputs)
        for e, a in zip(expected, actual, strict=True):
            torch.testing.assert_close(a, e, rtol=0, atol=0)
        # Block inputs offloaded per forward: encoder blocks 1..2 (block 0's
        # is the model input) and every decoder block.
        activation_bytes = _NUM_TOKENS * _DIM * 4
        self.assertEqual(
            _offloader(model).forward_stats.offloaded_bytes,
            (_NUM_ENCODER_LAYERS - 1 + _NUM_LAYERS) * activation_bytes,
        )
        # The encoder's last group is freed by decoder block 0's forward.
        blocks = [f"vision_encoder.layers.{i}" for i in range(_NUM_ENCODER_LAYERS)]
        blocks += [f"layers.{i}" for i in range(_NUM_LAYERS)]
        self.assertEqual(freed[: len(blocks)], blocks)

    def test_never_offloads_parameters(self):
        self.reference = _Model(_InternalWeightBlock)
        model = self._assert_bitwise_equal(_save_all(["*"]))
        # Offloaded per forward: each block's proj output (blocks 0..5) and
        # block input (blocks 1..5; block 0's is the model input). Offloading
        # the (D, D) weights would add 6 * _DIM * _DIM * 4 bytes.
        activation_bytes = _NUM_TOKENS * _DIM * 4
        self.assertEqual(
            _offloader(model).forward_stats.offloaded_bytes,
            (_NUM_LAYERS + _NUM_LAYERS - 1) * activation_bytes,
        )

    def test_keeps_tensor_subclass_saves_on_gpu(self):
        self.reference = _Model(_SubclassSaveBlock)
        model = self._assert_bitwise_equal(_save_all(["*"]))
        self.assertIn(
            "not a plain CUDA tensor", _offloader(model).forward_stats.skipped_bytes
        )

    def test_storage_freed_elsewhere_raises(self):
        # The free point finds the storage freed under the in-flight copy.
        self.reference = _Model(_ExternallyFreedSaveBlock)
        model = _build(_save_all(["*"]), self.reference)
        with self.assertRaisesRegex(RuntimeError, "something other than autograd"):
            model(self.inputs[0].clone().requires_grad_(True))

    def test_block_overrides_select_offloads_per_block(self):
        model = self._assert_bitwise_equal(
            FullActivationRematConfig(
                offload_block_input=True,
                block_overrides=[
                    BlockRematOverride(blocks=["layers.2"], offload_block_input=False)
                ],
            )
        )
        # Blocks 1, 3, 4 and 5: block 0's input is the model input, and
        # block 2 is overridden.
        input_bytes = _NUM_TOKENS * _DIM * 4
        self.assertEqual(
            _offloader(model).forward_stats.offloaded_bytes, 4 * input_bytes
        )

    def test_override_keeps_the_last_block_on_gpu(self):
        last_block = f"layers.{_NUM_LAYERS - 1}"
        model = self._assert_bitwise_equal(
            FullActivationRematConfig(
                offload_block_input=True,
                block_overrides=[
                    BlockRematOverride(blocks=[last_block], offload_block_input=False)
                ],
            )
        )
        # Blocks 1..4: block 0's input is the model input, and the override
        # keeps the last block's on GPU.
        input_bytes = _NUM_TOKENS * _DIM * 4
        self.assertEqual(
            _offloader(model).forward_stats.offloaded_bytes,
            (_NUM_LAYERS - 2) * input_bytes,
        )

    def test_detects_in_place_updates_after_the_save(self):
        """Autograd does not check saves that saved-tensor hooks pack; the
        offloader restores its in-place check for offloaded and kept saves."""
        for name, remat_config in (
            ("offloaded block input", _full_remat_offloading_inputs()),
            (
                "kept block input",
                ActivationRematConfig(save_regions=["*"], offload_regions=["*"]),
            ),
        ):
            with self.subTest(name):
                model = _build(remat_config, self.reference)
                # remat's outputs may not be updated in place; a clone may.
                model.layers["1"].register_forward_hook(
                    lambda module, args, output: output.clone()
                )
                block_inputs = []
                model.layers["2"].register_forward_pre_hook(
                    lambda module, args: block_inputs.append(args[0])
                )
                loss = model(self.inputs[0].clone().requires_grad_(True)).sum()
                block_inputs[0].add_(1)
                with self.assertRaisesRegex(RuntimeError, "modified by an inplace"):
                    loss.backward()

    def test_unpack_without_reload_raises(self):
        # Every group is reloaded by a backward trigger; without the triggers,
        # backward reads tensors still on CPU.
        model = _build(_full_remat_offloading_inputs(), self.reference)
        with unittest.mock.patch.object(
            ActivationOffloader, "_prefetch_on_grad"
        ), self.assertRaisesRegex(RuntimeError, "no reload trigger reloaded"):
            _grads(model, self.inputs[:1])

    def _forward_memory_saved(self, remat_config: ActivationRematConfig):
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

        baseline = _build(_without_offload(remat_config), self.reference)
        offloaded = _build(remat_config, self.reference)
        saved = stashed_bytes(baseline) - stashed_bytes(offloaded)
        return saved, _offloader(offloaded).forward_stats

    def test_forward_frees_offloaded_block_inputs(self):
        saved, stats = self._forward_memory_saved(_full_remat_offloading_inputs())
        # Block 0's input is the model input, so the inputs of blocks 1..5
        # move to CPU.
        input_bytes = _NUM_TOKENS * _DIM * 4
        self.assertEqual(stats.offloaded_bytes, (_NUM_LAYERS - 1) * input_bytes)
        self.assertEqual(stats.still_resident_bytes, {})
        self.assertEqual(saved, stats.offloaded_bytes)

    def test_forward_frees_offloaded_saved_regions(self):
        saved, stats = self._forward_memory_saved(
            DefaultActivationRematConfig(
                offload_block_input=True, offload_regions=["*"]
            )
        )
        self.assertGreater(
            stats.offloaded_bytes, (_NUM_LAYERS - 1) * _NUM_TOKENS * _DIM * 4
        )
        self.assertEqual(stats.still_resident_bytes, {})
        self.assertEqual(saved, stats.offloaded_bytes)

    def test_reports_offloads_that_stay_resident(self):
        self.reference = _Model(_InputRetainingBlock)
        with self.assertLogs("torchtitan.distributed.activation_offload", "WARNING"):
            saved, stats = self._forward_memory_saved(_full_remat_offloading_inputs())
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
        offloaded = _build(_full_remat_offloading_inputs(), self.reference)
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
        model = _build(_full_remat_offloading_inputs(), self.reference)
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
        _collect_garbage_before_capture()
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
        # Blocks 1..5 offload one input each; the second microbatch reuses
        # the first one's buffers, in eager mode and in the graph alike.
        pool = _offloader(model)._host_buffers._free
        self.assertEqual(
            sum(len(buffers) for buffers in pool.values()), _NUM_LAYERS - 1
        )

    def test_varying_sizes_reuse_host_buffers(self):
        """Storages whose size changes every step, like the tokens routed to an
        EP rank, reuse the buffers of their power-of-two capacity."""
        baseline = _build(FullActivationRematConfig(), self.reference)
        model = _build(_full_remat_offloading_inputs(), self.reference)
        # Each block input is T * D fp32: 12-16 MiB, all in the 16 MiB class.
        for num_tokens in (_NUM_TOKENS, 3500, 3072, 4000):
            inputs = [torch.randn(num_tokens, _DIM, device="cuda")]
            for e, a in zip(
                _grads(baseline, inputs), _grads(model, inputs), strict=True
            ):
                torch.testing.assert_close(a, e, rtol=0, atol=0)
        pool = _offloader(model)._host_buffers._free
        self.assertEqual(list(pool), [_NUM_TOKENS * _DIM * 4])
        # Blocks 1..5 offload one input each.
        self.assertEqual(len(pool[_NUM_TOKENS * _DIM * 4]), _NUM_LAYERS - 1)

    def test_reload_trigger_fires_after_later_tensor_hooks(self):
        """Each reload is queued after every tensor hook on the triggering
        output, even one registered after the offloader's hooks, as FSDP2's
        pre-backward is: fully_shard runs after apply installs offloading."""
        model = _build(_full_remat_offloading_inputs(), self.reference)
        offloader = _offloader(model)
        events = []

        def register_tensor_hook(block_id, module, args, output):
            output.register_hook(lambda grad: events.append(f"hook {block_id}"))

        for block_id, block in model.layers.items():
            block.register_forward_hook(
                functools.partial(register_tensor_hook, int(block_id))
            )
        reload_group = offloader._reload_group

        def record_reload(group):
            events.append(f"reload {group.module_fqn.removeprefix('layers.')}")
            reload_group(group)

        with unittest.mock.patch.object(
            offloader, "_reload_group", side_effect=record_reload
        ):
            _grads(model, self.inputs[:1])
        # Block i's output triggers the reload of group i - 1; the model
        # part's output, block 5's, triggers the last group's, which backward
        # needs first, so it is queued before group 4's.
        last = _NUM_LAYERS - 1
        self.assertEqual(
            events,
            [f"hook {last}", f"reload {last}"]
            + [
                event
                for i in range(last, 0, -1)
                for event in (f"reload {i - 1}", f"hook {i - 1}")
            ],
        )


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

        remat_config = DefaultActivationRematConfig(
            offload_block_input=True, offload_regions=["*"]
        )

        def build(config: ActivationRematConfig) -> _Model:
            # As in parallelize: remat (and offloading) before fully_shard.
            model = deepcopy(reference)
            _apply(config, model)
            for block in model.layers.values():
                fully_shard(block, mesh=mesh, offload_policy=CPUOffloadPolicy())
            fully_shard(model, mesh=mesh, offload_policy=CPUOffloadPolicy())
            return model

        expected = _grads(build(_without_offload(remat_config)), inputs)
        model = build(remat_config)
        offloader = _offloader(model)
        # Record which streams the reload stream waits on.
        waited_on: list[torch.cuda.Stream] = []
        wait_stream = torch.cuda.Stream.wait_stream

        def recording_wait_stream(self, stream):
            if self == offloader._reload_stream:
                waited_on.append(stream)
            return wait_stream(self, stream)

        with unittest.mock.patch.object(
            torch.cuda.Stream, "wait_stream", recording_wait_stream
        ):
            actual = _grads(model, inputs)
        for e, a in zip(expected, actual, strict=True):
            torch.testing.assert_close(a.to_local(), e.to_local(), rtol=0, atol=0)
        # Every reload waits on FSDP2's copy-in stream, where FSDP2 queued the
        # previous block's parameter copies, so the parameters load first.
        copy_in_stream = model.get_all_gather_copy_in_stream()
        assert waited_on.count(copy_in_stream) > 0
        assert waited_on.count(copy_in_stream) * 2 == len(waited_on)
        assert offloader.forward_stats.offloaded_bytes > 0
    finally:
        dist.destroy_process_group()


@pytest.mark.multi_gpu
@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two GPUs")
class TestActivationOffloadWithFSDPCPUOffload(unittest.TestCase):
    """Activation offload composed with FSDP2 CPU parameter offload. Two ranks:
    on one rank FSDP2 skips the all-gather and copies parameters on the
    compute stream."""

    def test_matches_without_activation_offload(self):
        mp.spawn(
            _run_fsdp_cpu_offload_composition,
            args=(2, get_free_port()),
            nprocs=2,
            join=True,
        )


def _run_fsdp_subclass_parameters(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))
        torch.manual_seed(0)
        inputs = [torch.randn(_NUM_TOKENS, _DIM, device="cuda") for _ in range(2)]
        remat_config = _save_all(["*"])

        def build(config: ActivationRematConfig) -> _Model:
            torch.manual_seed(0)
            model = _Model(_OperandWeightBlock)
            _apply(config, model)
            for block in model.layers.values():
                fully_shard(block, mesh=mesh)
            fully_shard(model, mesh=mesh)
            return model

        expected = _grads(build(_without_offload(remat_config)), inputs)
        model = build(remat_config)
        # Offloading an operand, which reshard frees after the block's
        # forward, would raise at the free point.
        actual = _grads(model, inputs)
        for e, a in zip(expected, actual, strict=True):
            torch.testing.assert_close(a.to_local(), e.to_local(), rtol=0, atol=0)
        assert _offloader(model).forward_stats.offloaded_bytes > 0
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestActivationOffloadWithFSDPSubclassParameters(unittest.TestCase):
    """FSDP2 with a tensor-subclass weight format (as MXFP8): the unsharded
    weight's operands are parameter data, never offloaded. One rank suffices:
    FSDP2 still frees unsharded data at reshard."""

    def test_never_offloads_subclass_parameter_operands(self):
        mp.spawn(
            _run_fsdp_subclass_parameters,
            args=(1, get_free_port()),
            nprocs=1,
            join=True,
        )


def _collect_garbage_before_capture() -> None:
    """Free earlier tests' dead offloaders before a CUDA graph capture.

    Their pinned buffers sit in reference cycles (model -> hooks -> offloader
    -> model) until the garbage collector runs. Freed during a capture, a
    pinned block last used on a stream that the capture joined (PyTorch hands
    out side streams from a small pool, so a new offloader can reuse a dead
    one's) gets its reuse event recorded into the graph, and the next pinned
    allocation of that size fails with "CUDA error: invalid argument".
    torch.cuda.graph no longer collects garbage on entry by default.
    """
    gc.collect()


def _offloader(model: _Model) -> ActivationOffloader:
    offloader = model.offloader
    assert offloader is not None, "no ActivationOffloader installed on the model"
    return offloader


if __name__ == "__main__":
    unittest.main()
