# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bitwise deterministic guardrail for graph_trainer.

Tests that Trainer (eager) and GraphTrainer (aot_fx_trace) produce bitwise
identical losses and gradients on Llama3 and DeepSeek-v3 debug models.

Requires a CUDA GPU. Run with:
    pytest torchtitan/experiments/graph_trainer/tests/test_bitwise_deterministic.py -x
"""

import copy
import tempfile
import unittest
from collections.abc import Callable

import torch
import torch.nn as nn
from expecttest import assert_expected_inline
from tests.utils import hash_gradient, hash_model
from torch.nn.attention.flex_attention import flex_attention

from torchtitan.components.loss import cross_entropy_loss
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.experiments.graph_trainer.deepseek_v3 import (
    model_registry as dsv3_model_registry,
)
from torchtitan.experiments.graph_trainer.deepseek_v3.parallelize import (
    annotate_deepseekv3,
)
from torchtitan.experiments.graph_trainer.llama3 import (
    model_registry as llama3_model_registry,
)
from torchtitan.experiments.graph_trainer.llama3.parallelize import annotate_llama
from torchtitan.experiments.graph_trainer.tests._trainer_test_utils import (
    build_minimal_trainer,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.models.common.attention import FlexAttention
from torchtitan.tools.utils import has_cuda_capability
from torchtitan.trainer import Trainer

SEED = 42
NUM_STEPS = 5
BATCH_SIZE = 4
SEQ_LEN = 128


def _set_deterministic(seed: int = SEED) -> None:
    """Set all random seeds and enable deterministic mode."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


_TOKENIZER_PATH = "./tests/assets/tokenizer"


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class BitwiseDeterministicBase(unittest.TestCase):
    """Base class for bitwise determinism tests.

    Subclasses must set `model_registry` to the appropriate model registry function.
    """

    model_registry: Callable
    annotate_model: Callable
    model_flavor: str

    def setUp(self):
        # Disable max_autotune for FlexAttention to ensure bitwise-identical
        # results between eager (torch.compile) and traced (regional_inductor)
        # paths. max_autotune causes kernel config divergence between the two.
        self._orig_inductor_configs = FlexAttention.inductor_configs
        self._orig_compiled_flex_attn = FlexAttention._compiled_flex_attn
        FlexAttention.inductor_configs = {
            **self._orig_inductor_configs,
            "max_autotune": False,
            "coordinate_descent_tuning": False,
        }
        FlexAttention._compiled_flex_attn = torch.compile(
            flex_attention,
            options=FlexAttention.inductor_configs,
        )

        _set_deterministic()
        model_spec = self.model_registry(self.model_flavor)
        self.model_config = model_spec.model
        vocab_size = self.model_config.vocab_size
        with torch.device("meta"):
            model = self.model_config.build()
        model.to_empty(device="cuda")
        with torch.no_grad():
            model.init_states(buffer_device=None)
        model.train()
        self.model = model
        self.inputs = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN), device="cuda")
        self.labels = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN), device="cuda")

    def tearDown(self):
        FlexAttention.inductor_configs = self._orig_inductor_configs
        FlexAttention._compiled_flex_attn = self._orig_compiled_flex_attn

    def _run_steps(
        self, model: nn.Module, trainer_cls: type, *, enable_passes: bool = True
    ) -> tuple[torch.Tensor, str, str]:
        """Run forward-backward-optimizer steps using the given trainer class."""
        # Annotate after deepcopy: annotate_fn wrappers capture bound methods
        # that don't rebind correctly through copy.deepcopy.
        self.annotate_model(model)
        trainer = build_minimal_trainer(
            model,
            self.model_config,
            trainer_cls,
            compile_enable_passes=enable_passes,
            tokenizer=HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH),
        )
        global_valid_tokens = torch.tensor(
            BATCH_SIZE * SEQ_LEN, dtype=torch.float, device="cuda"
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        for _ in range(NUM_STEPS):
            optimizer.zero_grad()
            loss = trainer.forward_backward_step(
                input_dict={"input": self.inputs},
                labels=self.labels,
                global_valid_tokens=global_valid_tokens,
            )
            optimizer.step()

        return loss.detach().clone(), hash_model(model), hash_gradient(model)

    def _run_steps_with_precompile(
        self, model: nn.Module, *, enable_passes: bool = True
    ) -> tuple[torch.Tensor, str, str]:
        """Run steps using the precompile save/load path.

        Traces the model, saves the FX graph artifact to a temp dir,
        loads it back, then runs forward-backward-optimizer steps using
        the loaded artifact — identical to what happens during
        torchrun training with --compile.precompile_artifact_dir.
        """
        from torchtitan.experiments.graph_trainer.make_fx_tracer import (
            run_traced_train_step,
            trace_train_step,
        )
        from torchtitan.experiments.graph_trainer.passes import (
            apply_graph_passes,
            compile_time_passes,
            construct_default_graph_passes,
        )
        from torchtitan.experiments.graph_trainer.precompile import (
            precompile_fx_trace_load,
            precompile_fx_trace_save,
        )
        from torchtitan.experiments.graph_trainer.storage import DiskStorageAdapter
        from torchtitan.experiments.graph_trainer.trainer import make_fwd_bwd_step

        self.annotate_model(model)
        loss_fn = cross_entropy_loss
        fwd_bwd_fn = make_fwd_bwd_step(loss_fn)

        global_valid_tokens = torch.tensor(
            BATCH_SIZE * SEQ_LEN, dtype=torch.float, device="cuda"
        )
        extra_inputs: dict[str, torch.Tensor] = {}
        extra_kwargs: dict[str, torch.Tensor] = {}

        # Step 1: Trace the graph
        traced_result = trace_train_step(fwd_bwd_fn)(
            model,
            self.inputs,
            self.labels,
            global_valid_tokens,
            extra_inputs,
            extra_kwargs,
        )

        # Step 2: Apply compile-time passes (cleanup + regional_inductor)
        # before saving, so compiled Triton kernels are baked in
        if enable_passes:
            passes = compile_time_passes(traced_result)
            traced_result.gm = apply_graph_passes(
                traced_result.gm,
                traced_result.example_inputs,
                passes,
            )

        # Step 3: Save and load (serialize/deserialize roundtrip)
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorageAdapter(tmpdir)
            precompile_fx_trace_save(traced_result, storage)

            loaded_result = precompile_fx_trace_load(storage, expected_fingerprint="")

        # Step 4: Apply load-time passes (cudagraph)
        if enable_passes:
            passes = construct_default_graph_passes(loaded_result, precompiled=True)
            loaded_result.gm = apply_graph_passes(
                loaded_result.gm,
                loaded_result.example_inputs,
                passes,
            )

        # Step 4: Run training steps using the loaded artifact
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        for _ in range(NUM_STEPS):
            optimizer.zero_grad()
            outputs = run_traced_train_step(
                loaded_result,
                model,
                self.inputs,
                self.labels,
                global_valid_tokens,
                extra_inputs,
                extra_kwargs,
            )
            loss = outputs[0]
            grads = outputs[1:]
            params = [
                p
                for _, p in model.named_parameters(remove_duplicate=False)
                if p.requires_grad
            ]
            for param, grad in zip(params, grads):
                param.grad = grad
            optimizer.step()

        return loss.detach().clone(), hash_model(model), hash_gradient(model)

    def _assert_runs_match(
        self,
        run_a: tuple[torch.Tensor, str, str],
        run_b: tuple[torch.Tensor, str, str],
        msg_prefix: str = "",
    ) -> None:
        loss_a, model_hash_a, grad_hash_a = run_a
        loss_b, model_hash_b, grad_hash_b = run_b

        self.assertTrue(
            torch.equal(loss_a, loss_b),
            f"{msg_prefix}loss mismatch: {loss_a.item()} vs {loss_b.item()}",
        )
        self.assertEqual(model_hash_a, model_hash_b, f"{msg_prefix}model hash mismatch")
        self.assertEqual(
            grad_hash_a, grad_hash_b, f"{msg_prefix}gradient hash mismatch"
        )


class TestLlama3BitwiseDeterministic(BitwiseDeterministicBase):
    """Bitwise determinism tests for Llama3 debug model."""

    model_registry = staticmethod(llama3_model_registry)
    model_flavor = "debugmodel"
    annotate_model = staticmethod(annotate_llama)

    @unittest.skipUnless(
        has_cuda_capability(9, 0), "Numerics only match on H100 (sm_90+)"
    )
    def test_eager_self_deterministic(self):
        """Eager mode: results match hardcoded expected values.

        Run `EXPECTTEST_ACCEPT=1 pytest <this_file> ` to update the inline expected values.
        """
        loss, model_hash, grad_hash = self._run_steps(
            copy.deepcopy(self.model), Trainer
        )
        assert_expected_inline(str(loss.item()), """7.961757659912109""")
        assert_expected_inline(
            model_hash,
            """15134607def7232e128240d553c8ee7021a7edbc2ed44d86e927ba61e490b865""",
        )
        assert_expected_inline(
            grad_hash,
            """66bbbbc98b4c1635e42a133ac1fbd499a2b8633ca879f4121cf206708c21dbdf""",
        )

    def test_aot_fx_trace_vs_eager(self):
        """aot_fx_trace and eager produce bitwise identical losses and grads."""
        run_eager = self._run_steps(copy.deepcopy(self.model), Trainer)
        run_traced = self._run_steps(copy.deepcopy(self.model), GraphTrainer)

        self._assert_runs_match(run_eager, run_traced, "eager vs aot_fx_trace: ")

    def test_precompile_vs_trace(self):
        """Precompiled aot_fx_trace (save/load roundtrip) matches direct trace."""
        _set_deterministic()
        run_traced = self._run_steps(copy.deepcopy(self.model), GraphTrainer)
        _set_deterministic()
        run_precompile = self._run_steps_with_precompile(copy.deepcopy(self.model))

        self._assert_runs_match(run_traced, run_precompile, "trace vs precompile: ")


class TestDSv3BitwiseDeterministic(BitwiseDeterministicBase):
    """Bitwise determinism tests for DeepSeek-v3 debug model."""

    model_registry = staticmethod(dsv3_model_registry)
    model_flavor = "debugmodel"
    annotate_model = staticmethod(annotate_deepseekv3)

    @unittest.skipUnless(
        has_cuda_capability(9, 0), "Numerics only match on H100 (sm_90+)"
    )
    def test_eager_self_deterministic(self):
        """Eager mode: results match hardcoded expected values.

        Run `EXPECTTEST_ACCEPT=1 pytest <this_file> ` to update the inline expected values.
        """
        loss, model_hash, grad_hash = self._run_steps(
            copy.deepcopy(self.model), Trainer
        )
        assert_expected_inline(str(loss.item()), """7.4749956130981445""")
        assert_expected_inline(
            model_hash,
            """7db9791ff6b1c22f64eee52e68f61f0119352528eac8683bf75a899268968edc""",
        )
        assert_expected_inline(
            grad_hash,
            """30d87367fe7227032c71fe4fab7d5162bbc4b7311a4049711f2edd02442679f6""",
        )

    def test_aot_fx_trace_vs_eager(self):
        """aot_fx_trace and eager produce bitwise identical losses and grads."""
        run_eager = self._run_steps(copy.deepcopy(self.model), Trainer)
        run_traced = self._run_steps(copy.deepcopy(self.model), GraphTrainer)

        self._assert_runs_match(run_eager, run_traced, "eager vs aot_fx_trace: ")

    def test_precompile_vs_trace(self):
        """Precompiled aot_fx_trace (save/load roundtrip) matches direct trace."""
        _set_deterministic()
        run_traced = self._run_steps(copy.deepcopy(self.model), GraphTrainer)
        _set_deterministic()
        run_precompile = self._run_steps_with_precompile(copy.deepcopy(self.model))

        self._assert_runs_match(run_traced, run_precompile, "trace vs precompile: ")


class TestLlama3FlexAttnBitwiseDeterministic(BitwiseDeterministicBase):
    """Bitwise determinism tests for Llama3 with FlexAttention (debugmodel_flex_attn).

    aot_fx_trace compiles FlexAttention HOPs via regional_inductor into fused
    Triton kernels and produces bitwise identical results to eager.
    """

    model_registry = staticmethod(llama3_model_registry)
    model_flavor = "debugmodel_flex_attn"
    annotate_model = staticmethod(annotate_llama)

    @unittest.skipUnless(
        has_cuda_capability(9, 0), "Numerics only match on H100 (sm_90+)"
    )
    def test_eager_self_deterministic(self):
        """Eager results match hardcoded expected values.

        Run `EXPECTTEST_ACCEPT=1 pytest <this_file>` to update the inline expected values.
        """
        loss, model_hash, grad_hash = self._run_steps(
            copy.deepcopy(self.model), Trainer
        )
        assert_expected_inline(str(loss.item()), """7.961757183074951""")
        assert_expected_inline(
            model_hash,
            """714c6b36b72327f2f11da003a219b6ff84f83e785464133f729e4f82c1913232""",
        )
        assert_expected_inline(
            grad_hash,
            """2eb6e999ebe213e69f8e85ecabea46ab59be81f0981847c7c8e69765be0d6678""",
        )

    def test_aot_fx_trace_vs_eager(self):
        """aot_fx_trace with passes and eager produce bitwise identical results."""
        run_eager = self._run_steps(copy.deepcopy(self.model), Trainer)
        run_traced = self._run_steps(copy.deepcopy(self.model), GraphTrainer)
        self._assert_runs_match(run_eager, run_traced, "eager vs aot_fx_trace: ")


class TestDSv3FlexAttnBitwiseDeterministic(BitwiseDeterministicBase):
    """Bitwise determinism tests for DSv3 with FlexAttention (debugmodel_flex_attn).

    aot_fx_trace compiles FlexAttention HOPs via regional_inductor into fused
    Triton kernels and produces bitwise identical results to eager.
    """

    model_registry = staticmethod(dsv3_model_registry)
    model_flavor = "debugmodel_flex_attn"
    annotate_model = staticmethod(annotate_deepseekv3)

    @unittest.skipUnless(
        has_cuda_capability(9, 0), "Numerics only match on H100 (sm_90+)"
    )
    def test_eager_self_deterministic(self):
        """Eager results match hardcoded expected values.

        Run `EXPECTTEST_ACCEPT=1 pytest <this_file>` to update the inline expected values.
        """
        loss, model_hash, grad_hash = self._run_steps(
            copy.deepcopy(self.model), Trainer
        )
        assert_expected_inline(str(loss.item()), """7.4749956130981445""")
        assert_expected_inline(
            model_hash,
            """2dcc779af7bc5aeae2d39eff3898180a8156549b2f1582d77e8db237689e0c67""",
        )
        assert_expected_inline(
            grad_hash,
            """b3f5b911dea6c9d36f508b08300220d7f39f142a7c34a49b7ef2543abb2065dc""",
        )

    # TODO: OOMs during flex_attention compilation on A100 GPUs.
    # Revisit when GraphTrainer addresses peak memory during compilation.
    @unittest.skipUnless(
        has_cuda_capability(9, 0), "OOMs during flex_attention compilation on A100"
    )
    def test_aot_fx_trace_vs_eager(self):
        """aot_fx_trace with passes and eager produce bitwise identical results."""
        run_eager = self._run_steps(copy.deepcopy(self.model), Trainer)
        run_traced = self._run_steps(copy.deepcopy(self.model), GraphTrainer)
        self._assert_runs_match(run_eager, run_traced, "eager vs aot_fx_trace: ")


if __name__ == "__main__":
    unittest.main()
