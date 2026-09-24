# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model config transforms."""

import copy
import unittest
from dataclasses import dataclass

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
from torchtitan.models.common.async_linear import (
    AsyncColumnParallelLinear,
    AsyncRowParallelLinear,
)
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.common.config_utils import make_shared_expert_ffn_config
from torchtitan.models.common.cp_attention import KVAllGatherCPFlexInnerAttention
from torchtitan.models.common.linear import (
    ColumnParallelLinear,
    Linear,
    RowParallelLinear,
)
from torchtitan.models.common.vision_encoder import InvariantRowParallelLinear


def _llama3_cp_ready():
    from torchtitan.models.llama3 import model_registry
    from torchtitan.models.llama3.config_registry import llama3_debugmodel

    config = llama3_debugmodel()
    config.model = model_registry("debugmodel", attn_backend="flex")
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


class _ConvertedLinear(ColumnParallelLinear):
    @dataclass(kw_only=True, slots=True)
    class Config(ColumnParallelLinear.Config):
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
        attention = config.model.layers[0].attention
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
        original = config.model.layers[0].attention.inner_attention
        self.assertNotIsInstance(original, KVAllGatherCPFlexInnerAttention.Config)


class TestTransformModel(unittest.TestCase):
    """The primitive runs on a model config alone, with no trainer config."""

    @staticmethod
    def _spec():
        from torchtitan.models.llama3 import model_registry

        return model_registry("debugmodel", attn_backend="flex")

    def test_rewrites_a_bare_model_config(self):
        model_config = self._spec()
        model_config = transform_model_config_(
            model_config,
            [ContextParallelTransform(inner_attention=KVAllGatherCPFlexInnerAttention)],
        )
        inner = model_config.layers[0].attention.inner_attention
        self.assertIsInstance(inner, KVAllGatherCPFlexInnerAttention.Config)

    def test_does_not_validate(self):
        """A CP kernel without a CP degree passes here and fails in the trainer.

        Validation is the caller's job, so RL and ``model_registry`` can rewrite
        a model config that no ``Trainer.Config`` owns yet.
        """
        model_config = self._spec()
        transform_model_config_(
            model_config,
            [ContextParallelTransform(inner_attention=KVAllGatherCPFlexInnerAttention)],
        )

    def test_orders_transforms(self):
        _Record.order = []
        transform_model_config_(
            self._spec(),
            [_Third(), _First(), _Second()],
        )
        self.assertEqual(_Record.order, ["_First", "_Second", "_Third"])


class TestContextParallelTransform(unittest.TestCase):
    def test_linear_lora_handler_is_exported(self):
        handler_cls = getattr(transform_api, "LinearLoRAHandler", None)
        self.assertIsNotNone(handler_cls, "LinearLoRAHandler is not exported")

    def test_swap_keeps_the_tuning_of_the_kernel_it_replaces(self):
        config = _llama3_cp_ready()
        tuned = config.model.layers[0].attention.inner_attention
        tuned.block_size = (256, 128)
        tuned.kernel_options = {"BACKEND": "FLASH"}

        result = apply_transforms(
            config,
            [ContextParallelTransform(inner_attention=KVAllGatherCPFlexInnerAttention)],
        )

        swapped = result.model.layers[0].attention.inner_attention
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

        inner = result.model.layers[0].attention.inner_attention
        self.assertIsInstance(inner, KVAllGatherCPFlexInnerAttention.Config)

        model = result.model.build()
        trainable = {
            name for name, param in model.named_parameters() if param.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(all("lora_a" in name or "lora_b" in name for name in trainable))


class TestAsyncTensorParallelTransform(unittest.TestCase):
    @staticmethod
    def _model_config():
        from torchtitan.models.llama3 import model_registry

        return model_registry("debugmodel")

    def test_replaces_all_parallel_linear_roles(self):
        model = AsyncTensorParallelTransform(enable_sequence_parallel=True).transform(
            self._model_config()
        )

        for layer in model.layers:
            self.assertIsInstance(
                layer.attention.qkv_linear.wqkv, AsyncColumnParallelLinear.Config
            )
            self.assertIsInstance(layer.attention.wo, AsyncRowParallelLinear.Config)
            self.assertIsInstance(
                layer.feed_forward.w13, AsyncColumnParallelLinear.Config
            )
            self.assertIsInstance(layer.feed_forward.w2, AsyncRowParallelLinear.Config)

    def test_sequence_parallel_disabled_raises(self):
        with self.assertRaisesRegex(
            ValueError,
            "requires sequence parallelism",
        ):
            AsyncTensorParallelTransform(enable_sequence_parallel=False).transform(
                self._model_config()
            )

    def test_shared_expert_transforms_only_collective_owning_projection(self):
        config = make_shared_expert_ffn_config(
            dim=4,
            hidden_dim=8,
            w1_param_init={},
            w2w3_param_init={},
        )

        transformed = AsyncTensorParallelTransform(
            enable_sequence_parallel=True
        ).transform(config)

        self.assertIs(type(transformed.w13), AsyncColumnParallelLinear.Config)
        self.assertIs(type(transformed.w2), Linear.Config)

    def test_muse_glimmer_shared_input_projections_are_plain_linears(self):
        from torchtitan.models.muse_glimmer import muse_glimmer_configs

        build_config, max_context_length = muse_glimmer_configs["debugmodel"]
        model = build_config(attn_backend="flex", seq_len=max_context_length)
        attention = model.layers[0].attention

        self.assertIs(type(attention.qkv_linear.wqkv), Linear.Config)
        self.assertIs(type(attention.o_gate), Linear.Config)
        self.assertIsInstance(attention.wo, RowParallelLinear.Config)

        transformed = AsyncTensorParallelTransform(
            enable_sequence_parallel=True
        ).transform(model)
        attention = transformed.layers[0].attention
        self.assertIs(type(attention.qkv_linear.wqkv), Linear.Config)
        self.assertIs(type(attention.o_gate), Linear.Config)
        self.assertIsInstance(attention.wo, AsyncRowParallelLinear.Config)

    def test_gpt_oss_biased_output_projection_uses_async_row_parallel(self):
        from torchtitan.models.gpt_oss import model_registry

        model = model_registry("debugmodel", seq_len=128, attn_backend="flex")
        self.assertIs(type(model.layers[0].attention.wo), RowParallelLinear.Config)
        self.assertTrue(model.layers[0].attention.wo.bias)

        transformed = AsyncTensorParallelTransform(
            enable_sequence_parallel=True
        ).transform(model)

        self.assertIs(
            type(transformed.layers[0].attention.wo),
            AsyncRowParallelLinear.Config,
        )
        self.assertTrue(transformed.layers[0].attention.wo.bias)

    def test_async_transform_rejects_converted_projection(self):
        config = copy.deepcopy(self._model_config().layers[0].feed_forward)
        config.w13 = _ConvertedLinear.Config(
            in_features=config.w13.in_features,
            out_features=config.w13.out_features,
            num_linears=config.w13.num_linears,
            param_init=config.w13.param_init,
        )

        with self.assertRaisesRegex(ValueError, "converted w13 projections"):
            AsyncTensorParallelTransform(enable_sequence_parallel=True).transform(
                config
            )

    def test_async_transform_skips_invariant_row_parallel_linear(self):
        config = InvariantRowParallelLinear.Config(
            in_features=4,
            out_features=4,
            bias=True,
        )

        transformed = AsyncTensorParallelTransform(
            enable_sequence_parallel=True
        ).transform(config)

        self.assertIs(type(transformed), InvariantRowParallelLinear.Config)

    def test_async_transform_conflicts_with_lora(self):
        config = self._model_config()

        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            transform_model_config_(
                config,
                [
                    AsyncTensorParallelTransform(enable_sequence_parallel=True),
                    LoRATransform(handlers=(LinearLoRAHandler(),)),
                ],
            )


if __name__ == "__main__":
    unittest.main()
