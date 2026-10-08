# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model config transforms."""

import copy
import inspect
import unittest
from dataclasses import dataclass

from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.config.transform import (
    apply_transforms,
    AsyncTensorParallelTransform,
    ContextParallelTransform,
    convert_config_type,
    LoRATransform,
    ModelConfigTransform,
    ModelConfigTransformContext,
    TokenDispatcherTransform,
    transform_model_config_,
    TransformRelations,
)
from torchtitan.models.common.async_linear import (
    AsyncColumnParallelLinear,
    AsyncRowParallelLinear,
)
from torchtitan.models.common.attention import (
    FlexInnerAttention,
    SlidingWindowFlexInnerAttention,
)
from torchtitan.models.common.attention.cp_attention import (
    KVAllGatherCPFlexInnerAttention,
    KVAllGatherCPSlidingWindowFlexInnerAttention,
    UlyssesCPFlexInnerAttention,
    UlyssesCPSlidingWindowFlexInnerAttention,
)
from torchtitan.models.common.linear import (
    ColumnParallelLinear,
    Linear,
    RowParallelLinear,
    SharedExpertRowParallelLinear,
)
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.token_dispatcher import DeepEPTokenDispatcher
from torchtitan.models.common.vision_encoder import InvariantRowParallelLinear

_CONTEXT = ModelConfigTransformContext(
    training=TrainingConfig(), parallelism=ParallelismConfig()
)


def _llama3_cp_ready():
    from torchtitan.models.llama3 import build_model_config
    from torchtitan_recipes.tests.models.llama3 import llama3_debugmodel

    config = llama3_debugmodel()
    config.model = build_model_config("debugmodel", attn_backend="flex", seq_len=512)
    config.parallelism.context_parallel_degree = 2
    config.training.max_context_length = 512
    return config


class _Record(ModelConfigTransform):
    order: list[str] = []

    def transform(self, model, *, context):
        del context
        _Record.order.append(type(self).__qualname__)
        return model


class _First(_Record):
    @classmethod
    def contribute_relations(cls, relations: TransformRelations) -> None:
        relations.add_precedence(before=_Loose, after=cls)
        relations.add_precedence(before=cls, after=_Rival)
        relations.add_precedence(before=_Rival, after=_Loose)


class _Second(_Record):
    @classmethod
    def contribute_relations(cls, relations: TransformRelations) -> None:
        relations.add_precedence(before=_First, after=cls)


class _Third(_Record):
    @classmethod
    def contribute_relations(cls, relations: TransformRelations) -> None:
        relations.add_precedence(before=_Second, after=cls)


class _Rival(_Record):
    @classmethod
    def contribute_relations(cls, relations: TransformRelations) -> None:
        relations.add_conflict(_First, cls)


class _SelfConflicting(_Record):
    @classmethod
    def contribute_relations(cls, relations: TransformRelations) -> None:
        relations.add_conflict(cls, cls)


class _Loose(_Record):
    pass


class _Contributing(_Record):
    num_contributions = 0

    @classmethod
    def contribute_relations(cls, relations: TransformRelations) -> None:
        cls.num_contributions += 1
        relations.add_precedence(before=_First, after=cls)
        relations.add_conflict(cls, _Rival)


class _ContributingSubclass(_Contributing):
    pass


class _CycleFirst(_Record):
    num_contributions = 0

    @classmethod
    def contribute_relations(cls, relations: TransformRelations) -> None:
        cls.num_contributions += 1
        relations.add_precedence(before=cls, after=_CycleSecond)


class _CycleSecond(_Record):
    num_contributions = 0

    @classmethod
    def contribute_relations(cls, relations: TransformRelations) -> None:
        cls.num_contributions += 1
        relations.add_precedence(before=cls, after=_CycleFirst)


class _Boom(ModelConfigTransform):
    def transform(self, model, *, context):
        del context
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


class TestPublicApi(unittest.TestCase):
    def test_relation_graph_is_not_caller_configurable(self):
        self.assertNotIn("relations", inspect.signature(apply_transforms).parameters)
        self.assertNotIn(
            "relations",
            inspect.signature(transform_model_config_).parameters,
        )


class TestOrdering(unittest.TestCase):
    def setUp(self):
        _Record.order = []
        _Contributing.num_contributions = 0
        _ContributingSubclass.num_contributions = 0
        _CycleFirst.num_contributions = 0
        _CycleSecond.num_contributions = 0
        self.config = _llama3_cp_ready()
        self.config.parallelism.context_parallel_degree = 1

    def test_precedence_chain_is_resolved(self):
        apply_transforms(
            self.config,
            [_Third(), _First(), _Second()],
        )
        self.assertEqual(_Record.order, ["_First", "_Second", "_Third"])

    def test_unrelated_transforms_keep_the_declared_order(self):
        apply_transforms(self.config, [_Second(), _Loose()])
        self.assertEqual(_Record.order, ["_Second", "_Loose"])

    def test_global_cycle_allows_an_acyclic_selected_subset(self):
        apply_transforms(
            self.config,
            [_First(), _Loose()],
        )

        self.assertEqual(_Record.order, ["_Loose", "_First"])

    def test_contributed_ordering_applies_to_subclasses(self):
        class _FirstSubclass(_First):
            pass

        apply_transforms(
            self.config,
            [_FirstSubclass(), _Loose()],
        )

        self.assertEqual(_Record.order, ["_Loose", _FirstSubclass.__qualname__])

    def test_rejects_a_contributed_conflict_in_either_order(self):
        for selected in ([_First(), _Rival()], [_Rival(), _First()]):
            with self.subTest(selected=[type(t).__qualname__ for t in selected]):
                with self.assertRaisesRegex(ValueError, "cannot be combined"):
                    apply_transforms(self.config, selected)

    def test_rejects_the_same_self_conflicting_instance_twice(self):
        transform = _SelfConflicting()
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            apply_transforms(
                self.config,
                [transform, transform],
            )

    def test_selected_transform_contributes_precedence_once(self):
        apply_transforms(
            self.config,
            [_Contributing(), _First(), _Contributing()],
        )

        self.assertEqual(
            _Record.order,
            ["_First", "_Contributing", "_Contributing"],
        )
        self.assertEqual(_Contributing.num_contributions, 1)

    def test_subclass_inherits_relation_contribution(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            apply_transforms(
                self.config,
                [_ContributingSubclass(), _Rival()],
            )

        self.assertEqual(_ContributingSubclass.num_contributions, 1)
        self.assertEqual(_Record.order, [])

    def test_contributed_cycle_is_rejected_before_any_transform_runs(self):
        with self.assertRaisesRegex(ValueError, "unresolved"):
            apply_transforms(
                self.config,
                [_Loose(), _CycleFirst(), _CycleSecond()],
            )

        self.assertEqual(_CycleFirst.num_contributions, 1)
        self.assertEqual(_CycleSecond.num_contributions, 1)
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
            [
                ContextParallelTransform(
                    inner_attention_map={
                        FlexInnerAttention: KVAllGatherCPFlexInnerAttention
                    }
                )
            ],
        )
        self.assertIsNot(result, config)
        original = config.model.layers[0].attention.inner_attention
        self.assertNotIsInstance(original, KVAllGatherCPFlexInnerAttention.Config)


class TestTransformModel(unittest.TestCase):
    """The primitive runs on a model config alone, with no trainer config."""

    @staticmethod
    def _spec():
        from torchtitan.models.llama3 import build_model_config

        return build_model_config("debugmodel", attn_backend="flex")

    def test_rewrites_a_bare_model_config(self):
        model_config = self._spec()
        model_config = transform_model_config_(
            model_config,
            [
                ContextParallelTransform(
                    inner_attention_map={
                        FlexInnerAttention: KVAllGatherCPFlexInnerAttention
                    }
                )
            ],
            context=_CONTEXT,
        )
        inner = model_config.layers[0].attention.inner_attention
        self.assertIsInstance(inner, KVAllGatherCPFlexInnerAttention.Config)

    def test_does_not_validate(self):
        """A CP kernel without a CP degree passes here and fails in the trainer.

        Validation is the caller's job, so RL and ``build_model_config`` can rewrite
        a model config that no ``Trainer.Config`` owns yet.
        """
        model_config = self._spec()
        transform_model_config_(
            model_config,
            [
                ContextParallelTransform(
                    inner_attention_map={
                        FlexInnerAttention: KVAllGatherCPFlexInnerAttention
                    }
                )
            ],
            context=_CONTEXT,
        )

    def test_orders_transforms(self):
        _Record.order = []
        transform_model_config_(
            self._spec(),
            [_Third(), _First(), _Second()],
            context=_CONTEXT,
        )
        self.assertEqual(_Record.order, ["_First", "_Second", "_Third"])

    def test_token_dispatcher_transform_uses_training_context(self):
        from torchtitan_recipes.tests.models.deepseek_v3 import deepseek_v3_debugmodel

        config = deepseek_v3_debugmodel()
        config.parallelism.expert_parallel_degree = 2
        config.parallelism.tensor_parallel_degree = 2
        config.training.disable_cuda_graphs = True
        transformed = apply_transforms(
            config,
            [TokenDispatcherTransform(dispatcher=DeepEPTokenDispatcher)],
        )

        dispatchers = [
            routed_experts.token_dispatcher
            for _, routed_experts, _, _ in transformed.model.traverse(
                RoutedExperts.Config
            )
        ]
        self.assertTrue(dispatchers)
        self.assertTrue(
            all(
                isinstance(dispatcher, DeepEPTokenDispatcher.Config)
                for dispatcher in dispatchers
            )
        )
        self.assertTrue(
            all(
                dispatcher.num_max_tokens_per_rank
                == config.training.num_tokens_per_microbatch_per_dp_rank // 2
                for dispatcher in dispatchers
            )
        )


class TestContextParallelTransform(unittest.TestCase):
    def test_swap_keeps_the_tuning_of_the_kernel_it_replaces(self):
        config = _llama3_cp_ready()
        tuned = config.model.layers[0].attention.inner_attention
        tuned.block_size = (256, 128)
        tuned.kernel_options = {"BACKEND": "FLASH"}

        result = apply_transforms(
            config,
            [
                ContextParallelTransform(
                    inner_attention_map={
                        FlexInnerAttention: KVAllGatherCPFlexInnerAttention
                    }
                )
            ],
        )

        swapped = result.model.layers[0].attention.inner_attention
        self.assertIsInstance(swapped, KVAllGatherCPFlexInnerAttention.Config)
        self.assertEqual(swapped.block_size, (256, 128))
        self.assertEqual(swapped.kernel_options, {"BACKEND": "FLASH"})

    def test_rejects_a_kernel_that_is_not_context_parallel(self):
        with self.assertRaisesRegex(ValueError, "must inherit CPInnerAttention"):
            ContextParallelTransform(
                inner_attention_map={FlexInnerAttention: FlexInnerAttention}
            )

    def test_preserves_gpt_oss_sliding_window_backend(self):
        from torchtitan.models.gpt_oss import build_model_config

        model = build_model_config("debugmodel", seq_len=128, attn_backend="flex")
        self.assertIsInstance(
            model.layers[0].attention.inner_attention,
            SlidingWindowFlexInnerAttention.Config,
        )
        self.assertIsInstance(
            model.layers[1].attention.inner_attention,
            FlexInnerAttention.Config,
        )

        ContextParallelTransform(
            inner_attention_map={
                FlexInnerAttention: KVAllGatherCPFlexInnerAttention,
                SlidingWindowFlexInnerAttention: (
                    KVAllGatherCPSlidingWindowFlexInnerAttention
                ),
            }
        ).transform(model)

        self.assertIsInstance(
            model.layers[0].attention.inner_attention,
            KVAllGatherCPSlidingWindowFlexInnerAttention.Config,
        )
        self.assertIsInstance(
            model.layers[1].attention.inner_attention,
            KVAllGatherCPFlexInnerAttention.Config,
        )

    def test_rejects_missing_attention_backend_override(self):
        from torchtitan.models.gpt_oss import build_model_config

        model = build_model_config("debugmodel", seq_len=128, attn_backend="flex")
        transform = ContextParallelTransform(
            inner_attention_map={FlexInnerAttention: KVAllGatherCPFlexInnerAttention}
        )

        with self.assertRaisesRegex(
            ValueError,
            "No CP inner attention configured for SlidingWindowFlexInnerAttention",
        ):
            transform.transform(model)

    def test_preserves_muse_glimmer_sliding_window_backend(self):
        from torchtitan.models.muse_glimmer import build_model_config

        for transform, expected_backends in (
            (
                ContextParallelTransform(
                    inner_attention_map={
                        FlexInnerAttention: KVAllGatherCPFlexInnerAttention,
                        SlidingWindowFlexInnerAttention: (
                            KVAllGatherCPSlidingWindowFlexInnerAttention
                        ),
                    }
                ),
                {
                    KVAllGatherCPFlexInnerAttention.Config,
                    KVAllGatherCPSlidingWindowFlexInnerAttention.Config,
                },
            ),
            (
                ContextParallelTransform(
                    inner_attention_map={
                        FlexInnerAttention: UlyssesCPFlexInnerAttention,
                        SlidingWindowFlexInnerAttention: (
                            UlyssesCPSlidingWindowFlexInnerAttention
                        ),
                    }
                ),
                {
                    UlyssesCPFlexInnerAttention.Config,
                    UlyssesCPSlidingWindowFlexInnerAttention.Config,
                },
            ),
        ):
            model = build_model_config("debugmodel", attn_backend="flex", seq_len=128)
            self.assertEqual(
                {type(layer.attention.inner_attention) for layer in model.layers},
                {FlexInnerAttention.Config, SlidingWindowFlexInnerAttention.Config},
            )

            transform.transform(model)

            self.assertEqual(
                {type(layer.attention.inner_attention) for layer in model.layers},
                expected_backends,
            )

    def test_transforms_nested_kda_backend(self):
        from torchtitan.models.common.attention.cp_kda import ContextParallelInnerKDA
        from torchtitan.models.common.attention.kda import InnerKDA
        from torchtitan.models.kimi_k3 import build_model_config

        model = build_model_config("debugmodel", attn_backend="flex", seq_len=128)

        ContextParallelTransform(
            inner_attention_map={
                FlexInnerAttention: KVAllGatherCPFlexInnerAttention,
                InnerKDA: ContextParallelInnerKDA,
            }
        ).transform(model)

        assert model.vision_encoder is not None
        self.assertIsInstance(
            model.vision_encoder.block.attn.inner_attention,
            FlexInnerAttention.Config,
        )
        for layer in model.layers:
            if layer.attention is not None:
                self.assertIsInstance(
                    layer.attention.inner_attention,
                    KVAllGatherCPFlexInnerAttention.Config,
                )
            else:
                assert layer.delta_attention is not None
                self.assertIsInstance(
                    layer.delta_attention.inner_kda,
                    ContextParallelInnerKDA.Config,
                )

    def test_transforms_mtp_layers(self):
        from torchtitan.models.deepseek_v3.mtp import MTPDecoder
        from torchtitan_recipes.tests.models.deepseek_v3 import (
            deepseek_v3_debugmodel_mtp,
        )

        model = deepseek_v3_debugmodel_mtp().model
        assert isinstance(model, MTPDecoder.Config)

        ContextParallelTransform(
            inner_attention_map={
                FlexInnerAttention: KVAllGatherCPFlexInnerAttention,
            }
        ).transform(model)

        for layers in (model.layers, model.mtp_layers):
            self.assertTrue(layers)
            for layer in layers:
                self.assertIsInstance(
                    layer.attention.inner_attention,
                    KVAllGatherCPFlexInnerAttention.Config,
                )

    def test_lora_runs_after_context_parallelism(self):
        config = _llama3_cp_ready()

        result = apply_transforms(
            config,
            [
                LoRATransform(
                    rank=2,
                    alpha=4.0,
                    target_modules=["wqkv", "wo"],
                ),
                ContextParallelTransform(
                    inner_attention_map={
                        FlexInnerAttention: KVAllGatherCPFlexInnerAttention
                    }
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
        from torchtitan.models.llama3 import build_model_config

        return build_model_config("debugmodel")

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
        from torchtitan.models.deepseek_v3 import build_model_config

        model = build_model_config("debugmodel")
        moe = model.layers[1].moe
        assert moe is not None and moe.shared_experts is not None

        transformed = AsyncTensorParallelTransform(
            enable_sequence_parallel=True
        ).transform(model)
        transformed_moe = transformed.layers[1].moe
        assert transformed_moe is not None
        config = transformed_moe.shared_experts
        assert config is not None

        self.assertIs(type(config.w13), AsyncColumnParallelLinear.Config)
        self.assertIs(type(config.w2), AsyncRowParallelLinear.Config)

    def test_shared_expert_conversion_is_explicit(self):
        config = SharedExpertRowParallelLinear.Config(
            in_features=4,
            out_features=4,
        )

        transformed = AsyncTensorParallelTransform(
            enable_sequence_parallel=True
        ).transform(config, context=_CONTEXT)

        self.assertIs(type(transformed), AsyncRowParallelLinear.Config)

    def test_muse_glimmer_shared_input_projections_are_plain_linears(self):
        from torchtitan.models.muse_glimmer import MODEL_FLAVORS

        build_config, max_context_length = MODEL_FLAVORS["debugmodel"]
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
        from torchtitan.models.gpt_oss import build_model_config

        model = build_model_config("debugmodel", seq_len=128, attn_backend="flex")
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

    def test_async_transform_skips_projection_subclasses(self):
        config = copy.deepcopy(self._model_config().layers[0].feed_forward)
        config.w13 = _ConvertedLinear.Config(
            in_features=config.w13.in_features,
            out_features=config.w13.out_features,
            num_linears=config.w13.num_linears,
            param_init=config.w13.param_init,
        )

        transformed = AsyncTensorParallelTransform(
            enable_sequence_parallel=True
        ).transform(config, context=_CONTEXT)

        self.assertIs(type(transformed.w13), _ConvertedLinear.Config)
        self.assertIs(type(transformed.w2), AsyncRowParallelLinear.Config)

    def test_async_transform_skips_invariant_row_parallel_linear(self):
        config = InvariantRowParallelLinear.Config(
            in_features=4,
            out_features=4,
            bias=True,
        )

        transformed = AsyncTensorParallelTransform(
            enable_sequence_parallel=True
        ).transform(config, context=_CONTEXT)

        self.assertIs(type(transformed), InvariantRowParallelLinear.Config)

    def test_async_transform_conflicts_with_lora(self):
        config = self._model_config()

        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            transform_model_config_(
                config,
                [
                    AsyncTensorParallelTransform(enable_sequence_parallel=True),
                    LoRATransform(),
                ],
                context=_CONTEXT,
            )


if __name__ == "__main__":
    unittest.main()
