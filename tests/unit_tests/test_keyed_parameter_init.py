# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from collections.abc import Callable
from copy import deepcopy
from typing import cast

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.func._random import StatefulPRNG
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.keyed_parameter_init import (
    capture_parameter_init_registry,
    keyed_parameter_init,
)
from torchtitan.models.common.param_init import skip_param_init
from torchtitan.protocols.module import Module, ModuleList


_SEED = 1234


def _trunc_normal(param: nn.Parameter) -> None:
    nn.init.trunc_normal_(param, mean=0.0, std=1.0, a=-0.2, b=0.2)


def _normal(param: nn.Parameter) -> None:
    nn.init.normal_(param, mean=0.1, std=0.7)


def _trunc_then_uniform(param: nn.Parameter) -> None:
    nn.init.trunc_normal_(param, mean=0.0, std=1.0, a=-0.02, b=0.02)
    nn.init.uniform_(param, -0.5, 0.5)


def _uniform(param: nn.Parameter) -> None:
    param.uniform_(-0.75, 0.5)


def _select_uniform(param: nn.Parameter) -> None:
    for index in range(param.shape[0]):
        param.data[index].uniform_(-0.75, 0.5)


def _set_checkpoint_metadata(
    tensor: torch.Tensor,
    *,
    global_shape,
    global_offsets,
    local_offsets,
    local_sizes,
) -> None:
    setattr(tensor, "global_shape", global_shape)  # noqa: B010
    setattr(tensor, "global_offsets", global_offsets)  # noqa: B010
    setattr(tensor, "local_offsets", local_offsets)  # noqa: B010
    setattr(tensor, "local_sizes", local_sizes)  # noqa: B010


def _advanced_state(seed: int = _SEED) -> torch.Tensor:
    rng = StatefulPRNG(seed)
    rng.take_key()
    return rng.get_state()


class _ParameterModule(Module):
    def __init__(
        self,
        shape: tuple[int, ...],
        initializer: Callable[[nn.Parameter], object],
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(shape))
        self._param_init = {"weight": initializer}


class _Stack(Module):
    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.layers = ModuleList(
            [_ParameterModule((3, 4), _normal) for _ in range(num_layers)]
        )
        self.layers._param_init = {}
        self._param_init = {}


def _run_init(
    models: Module | tuple[Module, ...],
    *,
    seed: int = _SEED,
    registry=None,
) -> StatefulPRNG:
    rng = StatefulPRNG(seed)
    with torch.no_grad(), keyed_parameter_init(models, rng, registry=registry):
        if isinstance(models, tuple):
            for model in models:
                model.init_states()
        else:
            models.init_states()
    return rng


class TestKeyedParameterInit(unittest.TestCase):
    def _assert_model_initializer_contract(self, config):
        with torch.device("meta"):
            model = config.build()
        model.verify_module_protocol()
        registry = capture_parameter_init_registry(model)
        state_dict_keys = tuple(model.state_dict())
        parameter_shapes = {
            name: tuple(parameter.shape) for name, parameter in model.named_parameters()
        }

        model.to_empty(device="cpu")
        _run_init(model, registry=registry)

        self.assertEqual(tuple(model.state_dict()), state_dict_keys)
        self.assertEqual(
            {
                name: tuple(parameter.shape)
                for name, parameter in model.named_parameters()
            },
            parameter_shapes,
        )
        self.assertTrue(
            all(torch.isfinite(param).all() for param in model.parameters())
        )
        return model

    def test_dense_matches_uneven_checkpointable_trunc_normal(self):
        dense = _ParameterModule((7, 5), _trunc_normal)
        shard = _ParameterModule((3, 5), _trunc_normal)
        _set_checkpoint_metadata(
            shard.weight,
            global_shape=(7, 5),
            global_offsets=((2, 0),),
            local_offsets=((0, 0),),
            local_sizes=((3, 5),),
        )
        shard.to_empty(device="cpu")

        dense_rng = _run_init(dense)
        shard_rng = _run_init(shard)

        torch.testing.assert_close(shard.weight, dense.weight[2:5], rtol=0, atol=0)
        self.assertTrue(torch.equal(dense_rng.get_state(), _advanced_state()))
        self.assertTrue(torch.equal(shard_rng.get_state(), _advanced_state()))

    def test_draw_site_is_independent_of_rejection_count(self):
        dense = _ParameterModule((31, 5), _trunc_then_uniform)
        shard = _ParameterModule((1, 5), _trunc_then_uniform)
        _set_checkpoint_metadata(
            shard.weight,
            global_shape=(31, 5),
            global_offsets=((17, 0),),
            local_offsets=((0, 0),),
            local_sizes=((1, 5),),
        )

        _run_init(dense)
        _run_init(shard)

        torch.testing.assert_close(shard.weight, dense.weight[17:18], rtol=0, atol=0)

    def test_padded_multiple_pieces_leave_holes_unchanged(self):
        sentinel = 17.0
        dense = _ParameterModule((5, 6), _uniform)
        padded = _ParameterModule((6, 8), _uniform)
        padded.weight.data.fill_(sentinel)
        _set_checkpoint_metadata(
            padded.weight,
            global_shape=(5, 6),
            global_offsets=((0, 0), (2, 0)),
            local_offsets=((1, 1), (3, 1)),
            local_sizes=((2, 6), (3, 6)),
        )

        _run_init(dense)
        _run_init(padded)

        torch.testing.assert_close(
            padded.weight[1:3, 1:7], dense.weight[:2], rtol=0, atol=0
        )
        torch.testing.assert_close(
            padded.weight[3:6, 1:7], dense.weight[2:], rtol=0, atol=0
        )
        holes = torch.ones_like(padded.weight, dtype=torch.bool)
        holes[1:6, 1:7] = False
        self.assertTrue(torch.all(padded.weight[holes] == sentinel))

    def test_pre_pipeline_module_list_compaction_preserves_fqns(self):
        reference = _Stack(3)
        subject = _Stack(3)
        registry = capture_parameter_init_registry(subject)
        original_layers = tuple(deepcopy(subject).layers)

        first_part = ModuleList([original_layers[2], original_layers[0]])
        second_part = ModuleList([original_layers[1]])
        first_part._param_init = {}
        second_part._param_init = {}

        _run_init(reference)
        rng = _run_init((first_part, second_part), registry=registry)

        self.assertEqual(
            registry.canonical_fqns,
            ("layers.0.weight", "layers.1.weight", "layers.2.weight"),
        )
        for index, layer in enumerate(original_layers):
            torch.testing.assert_close(
                layer.weight, reference.layers[index].weight, rtol=0, atol=0
            )
        self.assertTrue(torch.equal(rng.get_state(), _advanced_state()))

    def test_tied_skip_alias_initializes_once_after_retie(self):
        calls = 0

        def initialize(param: nn.Parameter) -> None:
            nonlocal calls
            calls += 1
            param.normal_(0.0, 1.0)

        class TiedModel(Module):
            def __init__(self) -> None:
                super().__init__()
                self.primary = _ParameterModule((3, 4), initialize)
                self.alias = _ParameterModule((3, 4), skip_param_init)
                self.alias.weight = self.primary.weight
                self._param_init = {}

        model = TiedModel()
        registry = capture_parameter_init_registry(model)
        model.to_empty(device="cpu")
        model.alias.weight = nn.Parameter(torch.empty_like(model.alias.weight))
        self.assertIsNot(model.alias.weight, model.primary.weight)

        _run_init(model, registry=registry)

        self.assertEqual(calls, 1)
        self.assertIs(model.alias.weight, model.primary.weight)

    def test_tied_parameters_reject_multiple_initializers(self):
        class MultipleTiedInitializers(Module):
            def __init__(self) -> None:
                super().__init__()
                self.first = _ParameterModule((3, 4), _normal)
                self.second = _ParameterModule((3, 4), _uniform)
                self.second.weight = self.first.weight

        with self.assertRaisesRegex(ValueError, "exactly one non-skip initializer"):
            capture_parameter_init_registry(MultipleTiedInitializers())

    def test_reset_fallback_rejects_before_state_advance(self):
        class ResetFallback(Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.empty(2, 3))

            def reset_parameters(self) -> None:
                nn.init.normal_(self.weight)

        rng = StatefulPRNG(_SEED)
        initial_state = rng.get_state()
        with self.assertRaisesRegex(ValueError, "explicit param_init"):
            with keyed_parameter_init(ResetFallback(), rng):
                raise AssertionError("preflight must fail before entering the body")
        self.assertTrue(torch.equal(rng.get_state(), initial_state))

    def test_body_failure_rolls_back_state(self):
        def failing_initializer(param: nn.Parameter) -> None:
            param.normal_()
            raise RuntimeError("initializer failed")

        failing = _ParameterModule((2, 3), failing_initializer)
        rng = StatefulPRNG(_SEED)
        initial_state = rng.get_state()
        with self.assertRaisesRegex(RuntimeError, "initializer failed"):
            with torch.no_grad(), keyed_parameter_init(failing, rng):
                failing.init_states()
        self.assertTrue(torch.equal(rng.get_state(), initial_state))

        skipped = _ParameterModule((2, 3), _normal)
        with self.assertRaisesRegex(RuntimeError, "did not run initializers"):
            with keyed_parameter_init(skipped, rng):
                pass
        self.assertTrue(torch.equal(rng.get_state(), initial_state))

    def test_fused_complete_dimension_select_and_param_data_uniform(self):
        dense = _ParameterModule((2, 4, 3), _select_uniform)
        shard = _ParameterModule((2, 2, 3), _select_uniform)
        _set_checkpoint_metadata(
            shard.weight,
            global_shape=(2, 4, 3),
            global_offsets=((0, 1, 0),),
            local_offsets=((0, 0, 0),),
            local_sizes=((2, 2, 3),),
        )

        _run_init(dense)
        _run_init(shard)

        torch.testing.assert_close(
            shard.weight, dense.weight[:, 1:3, :], rtol=0, atol=0
        )

    def test_overlapping_view_is_rejected(self):
        def initialize_view(param: nn.Parameter) -> None:
            param.transpose(0, 1).normal_()

        model = _ParameterModule((3, 3), initialize_view)
        rng = StatefulPRNG(_SEED)
        initial_state = rng.get_state()
        with self.assertRaisesRegex(ValueError, "overlapping tensor view"):
            with torch.no_grad(), keyed_parameter_init(model, rng):
                model.init_states()
        self.assertTrue(torch.equal(rng.get_state(), initial_state))

    def test_unsupported_random_operator_is_rejected(self):
        def initialize_with_randn_like(param: nn.Parameter) -> None:
            param.copy_(torch.randn_like(param))

        model = _ParameterModule((3, 4), initialize_with_randn_like)
        rng = StatefulPRNG(_SEED)
        initial_state = rng.get_state()
        with self.assertRaisesRegex(NotImplementedError, "aten.randn_like.default"):
            with torch.no_grad(), keyed_parameter_init(model, rng):
                model.init_states()
        self.assertTrue(torch.equal(rng.get_state(), initial_state))

    def test_shape_dependent_initializer_rejects_plain_local_shard(self):
        model = _ParameterModule((3, 5), nn.init.xavier_uniform_)
        _set_checkpoint_metadata(
            model.weight,
            global_shape=(7, 5),
            global_offsets=((2, 0),),
            local_offsets=((0, 0),),
            local_sizes=((3, 5),),
        )

        with self.assertRaisesRegex(
            NotImplementedError, "shape-dependent parameter initializers"
        ):
            with keyed_parameter_init(model, StatefulPRNG(_SEED)):
                model.init_states()

    def test_qwen_and_deepseek_model_initializers(self):
        from torchtitan.models.deepseek_v3 import deepseekv3_configs
        from torchtitan.models.qwen3 import qwen3_configs

        qwen_config = qwen3_configs["debugmodel"](attn_backend="flex")
        qwen_config.layers = qwen_config.layers[:1]
        qwen = self._assert_model_initializer_contract(qwen_config)
        self.assertIs(qwen.tok_embeddings.weight, qwen.lm_head.weight)
        self.assertEqual(
            tuple(qwen.layers["0"].attention.qkv_linear.wqkv.weight.shape),
            (4096, 256),
        )

        deepseek_config = deepseekv3_configs["debugmodel"](
            attn_backend="flex", moe_comm_backend="standard"
        )
        deepseek_config.layers = deepseek_config.layers[:2]
        deepseek = self._assert_model_initializer_contract(deepseek_config)
        experts = deepseek.layers["1"].moe.routed_experts.inner_experts
        self.assertEqual(tuple(experts.w1_EFD.shape), (8, 256, 256))
        self.assertEqual(tuple(experts.w2_EDF.shape), (8, 256, 256))
        self.assertEqual(tuple(experts.w3_EFD.shape), (8, 256, 256))


class TestKeyedParameterInitDTensor(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @property
    def device_type(self) -> str:
        return "cpu"

    def _make_dtensor_module(self, shape, mesh, placement, initializer=_normal):
        module = _ParameterModule(tuple(shape), initializer)
        module.weight = nn.Parameter(
            distribute_tensor(torch.empty(shape), mesh, [placement])
        )
        return module

    @with_comms
    def test_dtensor_shard_replicate_and_zero_owner_match_dense(self):
        mesh = init_device_mesh("cpu", (self.world_size,))
        rank = dist.get_rank()

        cases = (
            ((7, 5), Shard(0), (slice(2 * rank, min(2 * rank + 2, 7)), slice(None))),
            ((5, 7), Shard(1), (slice(None), slice(2 * rank, min(2 * rank + 2, 7)))),
            ((3, 5), Replicate(), (slice(None), slice(None))),
            ((2, 5), Shard(0), (slice(rank, min(rank + 1, 2)), slice(None))),
        )
        for shape, placement, expected_slice in cases:
            with self.subTest(shape=shape, placement=placement):
                dense = _ParameterModule(shape, _normal)
                distributed = self._make_dtensor_module(shape, mesh, placement)

                dense_rng = _run_init(dense)
                distributed_rng = _run_init(distributed)

                expected = dense.weight[expected_slice]
                self.assertIsInstance(distributed.weight, DTensor)
                actual = distributed.weight.to_local()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertTrue(
                    torch.equal(dense_rng.get_state(), distributed_rng.get_state())
                )
                self.assertTrue(
                    torch.equal(distributed_rng.get_state(), _advanced_state())
                )

        from torchtitan.models.common.config_utils import _fused_qkv_param_init

        fused_qkv_init = _fused_qkv_param_init(
            {"weight": _normal},
            n_heads=4,
            n_kv_heads=2,
            head_dim=3,
        )["weight"]
        shape = (24, 5)
        dense = _ParameterModule(shape, fused_qkv_init)
        distributed = self._make_dtensor_module(shape, mesh, Shard(0), fused_qkv_init)

        _run_init(dense)
        _run_init(distributed)

        torch.testing.assert_close(
            cast(DTensor, distributed.weight).to_local(),
            dense.weight[rank * 6 : (rank + 1) * 6],
            rtol=0,
            atol=0,
        )

        shape = (8, 5)
        dense = _ParameterModule(shape, nn.init.xavier_uniform_)
        distributed = self._make_dtensor_module(
            shape, mesh, Shard(0), nn.init.xavier_uniform_
        )
        _run_init(dense)
        _run_init(distributed)
        torch.testing.assert_close(
            cast(DTensor, distributed.weight).to_local(),
            dense.weight[rank * 2 : (rank + 1) * 2],
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
