# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model config transforms."""

import copy
import unittest
from dataclasses import dataclass

import spmd_types as spmd
import torchtitan.config.transform as transform_api
from torchtitan.config.transform import (
    apply_transforms,
    AsyncTensorParallelTransform,
    ContextParallelTransform,
    convert_config_type,
    LinearLoRAHandler,
    LoRATransform,
    ModelConfigTransform,
    transform_model_config_,
)
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import _per_axis_types

from torchtitan.models.common.attention import (
    FlexInnerAttention,
    GQAttention,
    QKVLinear,
)
from torchtitan.models.common.cp_attention import KVAllGatherCPFlexInnerAttention
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import (
    ColumnParallelLinear,
    Linear,
    RowParallelLinear,
)
from torchtitan.protocols.module import Module


def _llama3_cp_ready():
    from torchtitan.models.llama3 import model_registry
    from torchtitan.models.llama3.config_registry import llama3_debugmodel

    config = llama3_debugmodel()
    config.model_spec = model_registry("debugmodel", attn_backend="flex")
    config.parallelism.context_parallel_degree = 2
    config.training.max_context_length = 512
    return config


class _Record(ModelConfigTransform):
    order: list[str] = []

    def transform(self, model):
        _Record.order.append(type(self).__qualname__)
        return model


class _First(_Record):
    pass


class _Second(_Record):
    run_after = (_First,)


class _Third(_Record):
    run_after = (_Second,)


class _Rival(_Record):
    conflicts_with = (_First,)


class _SelfConflicting(_Record):
    pass


_SelfConflicting.conflicts_with = (_SelfConflicting,)


class _Loose(_Record):
    pass


class _Boom(ModelConfigTransform):
    def transform(self, model):
        model.layers[0].attention.inner_attention.block_size = (1, 1)
        raise ValueError("boom")


class _FeedForwardSlots(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        feed_forward: FeedForward.Config
        shared_experts: FeedForward.Config

    def __init__(self, config: Config):
        super().__init__()


class _ConvertedLinear(Linear):
    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass


class TestConvertConfigType(unittest.TestCase):
    def test_keeps_the_fields_of_the_config_it_replaces(self):
        existing = FlexInnerAttention.Config()
        existing.block_size = (256, 128)
        existing.kernel_options = {"BACKEND": "FLASH"}

        swapped = convert_config_type(existing, KVAllGatherCPFlexInnerAttention)

        self.assertIsInstance(swapped, KVAllGatherCPFlexInnerAttention.Config)
        self.assertEqual(swapped.block_size, (256, 128))
        self.assertEqual(swapped.kernel_options, {"BACKEND": "FLASH"})

    def test_rejects_a_replacement_that_does_not_inherit_the_current_type(self):
        # A non-subclass would drop fields added by an earlier transform.
        existing = KVAllGatherCPFlexInnerAttention.Config()
        with self.assertRaisesRegex(ValueError, "must inherit"):
            convert_config_type(existing, FlexInnerAttention)


class TestOrdering(unittest.TestCase):
    def setUp(self):
        _Record.order = []

    def test_run_after_decides_the_order_not_the_list(self):
        config = _llama3_cp_ready()
        config.parallelism.context_parallel_degree = 1
        apply_transforms(config, [_Third(), _First(), _Second()])
        self.assertEqual(_Record.order, ["_First", "_Second", "_Third"])

    def test_unrelated_transforms_keep_the_declared_order(self):
        config = _llama3_cp_ready()
        config.parallelism.context_parallel_degree = 1
        apply_transforms(config, [_First(), _Loose()])
        self.assertEqual(_Record.order, ["_First", "_Loose"])

    def test_rejects_a_declared_conflict(self):
        config = _llama3_cp_ready()
        config.parallelism.context_parallel_degree = 1
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            apply_transforms(config, [_First(), _Rival()])

    def test_rejects_the_same_self_conflicting_instance_twice(self):
        config = _llama3_cp_ready()
        config.parallelism.context_parallel_degree = 1
        transform = _SelfConflicting()

        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            apply_transforms(config, [transform, transform])

        self.assertEqual(_Record.order, [])


class TestAtomicApplication(unittest.TestCase):
    def test_a_failure_leaves_the_caller_config_untouched(self):
        config = _llama3_cp_ready()
        config.parallelism.context_parallel_degree = 1
        attention = config.model_spec.model.layers[0].attention
        before = attention.inner_attention.block_size

        with self.assertRaisesRegex(ValueError, "boom"):
            apply_transforms(config, [_Boom()])

        self.assertEqual(attention.inner_attention.block_size, before)

    def test_the_caller_config_is_not_the_returned_one(self):
        config = _llama3_cp_ready()
        result = apply_transforms(
            config,
            [ContextParallelTransform(inner_attention=KVAllGatherCPFlexInnerAttention)],
        )
        self.assertIsNot(result, config)
        original = config.model_spec.model.layers[0].attention.inner_attention
        self.assertNotIsInstance(original, KVAllGatherCPFlexInnerAttention.Config)


class TestTransformModel(unittest.TestCase):
    """The primitive runs on a model config alone, with no trainer config."""

    @staticmethod
    def _spec():
        from torchtitan.models.llama3 import model_registry

        return model_registry("debugmodel", attn_backend="flex")

    def test_rewrites_a_bare_model_spec(self):
        spec = self._spec()
        spec.model = transform_model_config_(
            spec.model,
            [ContextParallelTransform(inner_attention=KVAllGatherCPFlexInnerAttention)],
        )
        inner = spec.model.layers[0].attention.inner_attention
        self.assertIsInstance(inner, KVAllGatherCPFlexInnerAttention.Config)

    def test_does_not_validate(self):
        """A CP kernel without a CP degree passes here and fails in the trainer.

        Validation is the caller's job, so RL and ``model_registry`` can rewrite
        a spec that no ``Trainer.Config`` owns yet.
        """
        spec = self._spec()
        transform_model_config_(
            spec.model,
            [ContextParallelTransform(inner_attention=KVAllGatherCPFlexInnerAttention)],
        )

    def test_orders_transforms(self):
        _Record.order = []
        transform_model_config_(
            self._spec().model,
            [_Third(), _First(), _Second()],
        )
        self.assertEqual(_Record.order, ["_First", "_Second", "_Third"])


class TestContextParallelTransform(unittest.TestCase):
    def test_linear_lora_handler_is_exported(self):
        handler_cls = getattr(transform_api, "LinearLoRAHandler", None)
        self.assertIsNotNone(handler_cls, "LinearLoRAHandler is not exported")

    def test_swap_keeps_the_tuning_of_the_kernel_it_replaces(self):
        config = _llama3_cp_ready()
        tuned = config.model_spec.model.layers[0].attention.inner_attention
        tuned.block_size = (256, 128)
        tuned.kernel_options = {"BACKEND": "FLASH"}

        result = apply_transforms(
            config,
            [ContextParallelTransform(inner_attention=KVAllGatherCPFlexInnerAttention)],
        )

        swapped = result.model_spec.model.layers[0].attention.inner_attention
        self.assertIsInstance(swapped, KVAllGatherCPFlexInnerAttention.Config)
        self.assertEqual(swapped.block_size, (256, 128))
        self.assertEqual(swapped.kernel_options, {"BACKEND": "FLASH"})

    def test_rejects_a_kernel_that_is_not_context_parallel(self):
        with self.assertRaisesRegex(ValueError, "must inherit CPInnerAttention"):
            ContextParallelTransform(inner_attention=FlexInnerAttention)

    def test_lora_runs_after_context_parallelism(self):
        transform_cls = getattr(transform_api, "LoRATransform", None)
        self.assertIsNotNone(transform_cls, "LoRATransform is not exported")
        handler_cls = getattr(transform_api, "LinearLoRAHandler", None)
        self.assertIsNotNone(handler_cls, "LinearLoRAHandler is not exported")
        config = _llama3_cp_ready()

        result = apply_transforms(
            config,
            [
                transform_cls(
                    handlers=(handler_cls(),),
                    rank=2,
                    alpha=4.0,
                    target_modules=["wqkv", "wo"],
                ),
                ContextParallelTransform(
                    inner_attention=KVAllGatherCPFlexInnerAttention
                ),
            ],
        )

        inner = result.model_spec.model.layers[0].attention.inner_attention
        self.assertIsInstance(inner, KVAllGatherCPFlexInnerAttention.Config)

        model = result.model_spec.model.build()
        trainable = {
            name for name, param in model.named_parameters() if param.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(all("lora_a" in name or "lora_b" in name for name in trainable))


class TestTensorParallelModules(unittest.TestCase):
    @staticmethod
    def _config():
        from torchtitan.models.llama3.config_registry import llama3_debugmodel

        config = llama3_debugmodel()
        config.parallelism.tensor_parallel_degree = 2
        return config

    def test_common_attention_and_feed_forward_use_sync_tp_projections(self):
        config = self._config()

        for layer in config.model_spec.model.layers:
            self.assertIs(type(layer.attention.qkv_linear), QKVLinear.Config)
            self.assertIsInstance(
                layer.attention.qkv_linear.wqkv, ColumnParallelLinear.Config
            )
            self.assertIsInstance(layer.attention.wo, RowParallelLinear.Config)
            self.assertIsInstance(layer.feed_forward.w13, ColumnParallelLinear.Config)
            self.assertIsInstance(layer.feed_forward.w2, RowParallelLinear.Config)
            self.assertIsNone(layer.feed_forward.w13.sharding_config)
            self.assertIsNone(layer.feed_forward.w2.sharding_config)

    def test_common_moe_shared_experts_use_sync_tp_projections(self):
        source = self._config().model_spec.model.layers[0].feed_forward
        model = _FeedForwardSlots.Config(
            feed_forward=copy.deepcopy(source),
            shared_experts=copy.deepcopy(source),
        )

        self.assertIs(type(model.feed_forward), FeedForward.Config)
        self.assertIsInstance(model.feed_forward.w13, ColumnParallelLinear.Config)
        self.assertIs(type(model.shared_experts), FeedForward.Config)
        self.assertIsInstance(model.shared_experts.w13, ColumnParallelLinear.Config)
        self.assertIsInstance(model.shared_experts.w2, RowParallelLinear.Config)

    def test_async_does_not_replace_moe_shared_experts(self):
        source = self._config().model_spec.model.layers[0].feed_forward
        model = _FeedForwardSlots.Config(
            feed_forward=copy.deepcopy(source),
            shared_experts=copy.deepcopy(source),
        )

        transformed = AsyncTensorParallelTransform().transform(model)

        self.assertIsInstance(
            transformed.shared_experts.w13, ColumnParallelLinear.Config
        )
        self.assertIsInstance(transformed.shared_experts.w2, RowParallelLinear.Config)

    def test_shared_expert_sharding_uses_projection_boundaries(self):
        from torchtitan.models.common.moe_sharding import (
            set_moe_sharding_config,
            set_shared_experts_sharding_config,
        )
        from torchtitan.models.deepseek_v3 import model_registry

        model = model_registry("debugmodel", seq_len=128).model
        moe = next(layer.moe for layer in model.layers if layer.moe is not None)
        assert moe.shared_experts is not None

        set_moe_sharding_config(
            moe,
            enable_ep=True,
            enable_sp=True,
            expert_param_layout={},
        )
        set_shared_experts_sharding_config(
            moe.shared_experts,
            enable_ep=True,
            enable_sp=True,
        )

        self.assertIsNone(moe.shared_experts.sharding_config.in_dst_shardings)
        self.assertIsNone(moe.shared_experts.w13.sharding_config.in_dst_shardings)
        self.assertIsNone(moe.shared_experts.w2.sharding_config.out_dst_shardings)

    def test_root_attention_uses_sync_tp_projections(self):
        config = copy.deepcopy(self._config().model_spec.model.layers[0].attention)

        self.assertIs(type(config), GQAttention.Config)
        self.assertIsInstance(config.qkv_linear.wqkv, ColumnParallelLinear.Config)
        self.assertIsInstance(config.wo, RowParallelLinear.Config)

    def test_async_transform_rejects_converted_projection(self):
        config = copy.deepcopy(self._config().model_spec.model.layers[0].feed_forward)
        config.w13 = _ConvertedLinear.Config(
            in_features=config.w13.linear.in_features,
            out_features=config.w13.linear.out_features,
            param_init=config.w13.linear.param_init,
        )

        with self.assertRaisesRegex(ValueError, "converted w13 projections"):
            AsyncTensorParallelTransform().transform(config)

    def test_async_transform_conflicts_with_lora(self):
        config = self._config().model_spec.model

        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            transform_model_config_(
                config,
                [
                    AsyncTensorParallelTransform(),
                    LoRATransform(handlers=(LinearLoRAHandler(),)),
                ],
            )

    def test_sharding_leaves_collectives_to_transformed_feed_forward(self):
        from torchtitan.models.llama3.sharding import set_llama3_sharding_config

        config = self._config()
        model = config.model_spec.model
        set_llama3_sharding_config(model, enable_sp=True)
        feed_forward = model.layers[0].feed_forward
        assert feed_forward.sharding_config is not None
        assert feed_forward.w13.sharding_config is not None
        assert feed_forward.w2.sharding_config is not None

        self.assertIsNone(feed_forward.sharding_config.in_dst_shardings)
        self.assertIsNotNone(feed_forward.sharding_config.out_src_shardings)
        self.assertIsNone(feed_forward.w13.sharding_config.in_dst_shardings)
        self.assertIsNotNone(feed_forward.w2.sharding_config.out_src_shardings)
        self.assertIsNone(feed_forward.w2.sharding_config.out_dst_shardings)
        self.assertEqual(
            _per_axis_types(feed_forward.w13.sharding_config.in_src_shardings["input"])[
                MeshAxisName.TP
            ],
            spmd.S(0),
        )
        self.assertEqual(
            _per_axis_types(feed_forward.w2.sharding_config.out_src_shardings)[
                MeshAxisName.TP
            ],
            spmd.S(0),
        )

    def test_sharding_sets_no_sequence_parallel_contract(self):
        from torchtitan.models.llama3.sharding import set_llama3_sharding_config

        config = self._config()
        model = config.model_spec.model
        set_llama3_sharding_config(model, enable_sp=False)

        feed_forward = model.layers[0].feed_forward
        assert feed_forward.w13.sharding_config is not None
        assert feed_forward.w2.sharding_config is not None
        self.assertIsNone(feed_forward.w13.sharding_config.in_dst_shardings)
        self.assertIsNone(feed_forward.w2.sharding_config.out_dst_shardings)
        self.assertEqual(
            _per_axis_types(feed_forward.w13.sharding_config.in_src_shardings["input"])[
                MeshAxisName.TP
            ],
            spmd.I,
        )
        self.assertEqual(
            _per_axis_types(feed_forward.w2.sharding_config.out_src_shardings)[
                MeshAxisName.TP
            ],
            spmd.I,
        )


if __name__ == "__main__":
    unittest.main()
