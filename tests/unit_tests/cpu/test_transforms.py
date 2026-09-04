# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copied from upstream open PR 4322/4449/4450 (fegin's CP stack) to unblock running; pending rebase and reconcile.

"""Model transform base class, ordering, and the context-parallel transform."""

import unittest
from dataclasses import dataclass

from torchtitan.models.common.attention import FlexAttention
from torchtitan.models.common.cp_attention import AllGatherCPFlexAttention
from torchtitan.transforms import (
    apply_transforms,
    ContextParallelTransform,
    ModelTransform,
    retype_node,
    transform_model,
)


def _llama3_cp_ready():
    from torchtitan.models.llama3 import model_registry
    from torchtitan.models.llama3.config_registry import llama3_debugmodel

    config = llama3_debugmodel()
    config.model_spec = model_registry("debugmodel", attn_backend="flex")
    config.parallelism.spmd_backend = "spmd_types"
    config.parallelism.context_parallel_degree = 2
    config.training.max_context_length = 512
    return config


class _Record(ModelTransform):
    order: list[str] = []

    @dataclass(kw_only=True, slots=True)
    class Config(ModelTransform.Config):
        pass

    def transform(self, model):
        _Record.order.append(type(self).__qualname__)
        return model


class _First(_Record):
    @dataclass(kw_only=True, slots=True)
    class Config(_Record.Config):
        pass


class _Second(_Record):
    run_after = (_First,)

    @dataclass(kw_only=True, slots=True)
    class Config(_Record.Config):
        pass


class _Third(_Record):
    run_after = (_Second,)

    @dataclass(kw_only=True, slots=True)
    class Config(_Record.Config):
        pass


class _Rival(_Record):
    conflicts_with = (_First,)

    @dataclass(kw_only=True, slots=True)
    class Config(_Record.Config):
        pass


class _Loose(_Record):
    @dataclass(kw_only=True, slots=True)
    class Config(_Record.Config):
        pass


class _Boom(ModelTransform):
    @dataclass(kw_only=True, slots=True)
    class Config(ModelTransform.Config):
        pass

    def transform(self, model):
        model.layers[0].attention.inner_attention.block_size = (1, 1)
        raise ValueError("boom")


class TestRetypeNode(unittest.TestCase):
    def test_keeps_the_fields_of_the_config_it_replaces(self):
        existing = FlexAttention.Config()
        existing.block_size = (256, 128)
        existing.kernel_options = {"BACKEND": "FLASH"}

        swapped = retype_node(existing, AllGatherCPFlexAttention)

        self.assertIsInstance(swapped, AllGatherCPFlexAttention.Config)
        self.assertEqual(swapped.block_size, (256, 128))
        self.assertEqual(swapped.kernel_options, {"BACKEND": "FLASH"})

    def test_rejects_a_replacement_that_does_not_inherit_the_current_type(self):
        # A non-subclass would drop fields added by an earlier transform.
        existing = AllGatherCPFlexAttention.Config()
        with self.assertRaisesRegex(ValueError, "must inherit"):
            retype_node(existing, FlexAttention)

    def test_sets_fields_defined_by_the_replacement(self):
        swapped = retype_node(
            FlexAttention.Config(),
            AllGatherCPFlexAttention,
            reduce_dtype="float32",
        )

        self.assertEqual(swapped.reduce_dtype, "float32")


class TestOrdering(unittest.TestCase):
    def setUp(self):
        _Record.order = []

    def test_run_after_decides_the_order_not_the_list(self):
        config = _llama3_cp_ready()
        config.parallelism.context_parallel_degree = 1
        apply_transforms(config, [_Third.Config(), _First.Config(), _Second.Config()])
        self.assertEqual(_Record.order, ["_First", "_Second", "_Third"])

    def test_unrelated_transforms_keep_the_declared_order(self):
        config = _llama3_cp_ready()
        config.parallelism.context_parallel_degree = 1
        apply_transforms(config, [_First.Config(), _Loose.Config()])
        self.assertEqual(_Record.order, ["_First", "_Loose"])

    def test_rejects_a_declared_conflict(self):
        config = _llama3_cp_ready()
        config.parallelism.context_parallel_degree = 1
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            apply_transforms(config, [_First.Config(), _Rival.Config()])


class TestAtomicApplication(unittest.TestCase):
    def test_a_failure_leaves_the_caller_config_untouched(self):
        config = _llama3_cp_ready()
        config.parallelism.context_parallel_degree = 1
        attention = config.model_spec.model.layers[0].attention
        before = attention.inner_attention.block_size

        with self.assertRaisesRegex(ValueError, "boom"):
            apply_transforms(config, [_Boom.Config()])

        self.assertEqual(attention.inner_attention.block_size, before)

    def test_the_caller_config_is_not_the_returned_one(self):
        config = _llama3_cp_ready()
        result = apply_transforms(
            config,
            [ContextParallelTransform.Config(kernel=AllGatherCPFlexAttention)],
        )
        self.assertIsNot(result, config)
        original = config.model_spec.model.layers[0].attention.inner_attention
        self.assertNotIsInstance(original, AllGatherCPFlexAttention.Config)


class TestTransformModel(unittest.TestCase):
    """Copied from upstream open PR 4322/4449/4450 to unblock running; pending rebase and reconcile.

    The primitive runs on a model config alone, with no trainer config."""

    @staticmethod
    def _spec():
        from torchtitan.models.llama3 import model_registry

        return model_registry("debugmodel", attn_backend="flex")

    def test_rewrites_a_bare_model_spec(self):
        spec = self._spec()
        spec.model = transform_model(
            spec.model,
            [ContextParallelTransform.Config(kernel=AllGatherCPFlexAttention)],
        )
        inner = spec.model.layers[0].attention.inner_attention
        self.assertIsInstance(inner, AllGatherCPFlexAttention.Config)

    def test_does_not_validate(self):
        """A CP kernel without a CP degree passes here and fails in the trainer.

        Validation is the caller's job, so RL and ``model_registry`` can rewrite
        a spec that no ``Trainer.Config`` owns yet.
        """
        spec = self._spec()
        transform_model(
            spec.model,
            [ContextParallelTransform.Config(kernel=AllGatherCPFlexAttention)],
        )

    def test_orders_transforms(self):
        _Record.order = []
        transform_model(
            self._spec().model,
            [_Third.Config(), _First.Config(), _Second.Config()],
        )
        self.assertEqual(_Record.order, ["_First", "_Second", "_Third"])


class TestContextParallelTransform(unittest.TestCase):
    def test_sets_cp_kernel_config_fields(self):
        config = _llama3_cp_ready()
        result = apply_transforms(
            config,
            [
                ContextParallelTransform.Config(
                    kernel=AllGatherCPFlexAttention,
                    kernel_config_overrides={"reduce_dtype": "float32"},
                )
            ],
        )

        swapped = result.model_spec.model.layers[0].attention.inner_attention
        self.assertEqual(swapped.reduce_dtype, "float32")

    def test_swap_keeps_the_tuning_of_the_kernel_it_replaces(self):
        config = _llama3_cp_ready()
        tuned = config.model_spec.model.layers[0].attention.inner_attention
        tuned.block_size = (256, 128)
        tuned.kernel_options = {"BACKEND": "FLASH"}

        result = apply_transforms(
            config,
            [ContextParallelTransform.Config(kernel=AllGatherCPFlexAttention)],
        )

        swapped = result.model_spec.model.layers[0].attention.inner_attention
        self.assertIsInstance(swapped, AllGatherCPFlexAttention.Config)
        self.assertEqual(swapped.block_size, (256, 128))
        self.assertEqual(swapped.kernel_options, {"BACKEND": "FLASH"})

    def test_rejects_a_kernel_that_is_not_context_parallel(self):
        config = _llama3_cp_ready()
        with self.assertRaisesRegex(ValueError, "must inherit ContextParallelKernel"):
            apply_transforms(
                config, [ContextParallelTransform.Config(kernel=FlexAttention)]
            )


if __name__ == "__main__":
    unittest.main()
