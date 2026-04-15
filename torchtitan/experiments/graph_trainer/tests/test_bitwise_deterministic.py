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
import unittest
from collections.abc import Callable
from types import SimpleNamespace

import torch
import torch.nn as nn
from expecttest import assert_expected_inline
from tests.utils import hash_gradient, hash_model

from torchtitan.components.loss import cross_entropy_loss
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.distributed.utils import get_train_context
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
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
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


def _build_trainer(
    model: nn.Module,
    model_config,
    trainer_cls: type,
    *,
    enable_passes: bool = True,
) -> Trainer:
    """Build a minimal Trainer/GraphTrainer for single-GPU non-distributed testing.

    Uses object.__new__ to bypass __init__ because the full Trainer constructor
    requires a distributed environment, job config, and checkpoint manager that
    are unnecessary for single-GPU numerical verification. The attributes set
    below are the minimal set required by forward_backward_step().
    """
    trainer = object.__new__(trainer_cls)
    trainer.model_parts = [model]
    trainer.loss_fn = cross_entropy_loss
    trainer.parallel_dims = SimpleNamespace(pp_enabled=False, cp_enabled=False)
    trainer.train_context = get_train_context(False)
    trainer.model_config = model_config
    trainer.device = torch.device("cuda")
    trainer.tokenizer = HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH)

    if trainer_cls is GraphTrainer:
        trainer.config = SimpleNamespace(
            compile=SimpleNamespace(
                mode="aot_fx_trace",
                enable_passes=enable_passes,
            )
        )
        trainer._fwd_bwd_step_module = None
        trainer._traced_step = None

    return trainer


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class BitwiseDeterministicBase(unittest.TestCase):
    """Base class for bitwise determinism tests.

    Subclasses must set `model_registry` to the appropriate model registry function.
    """

    model_registry: Callable
    annotate_model: Callable
    model_flavor: str

    def setUp(self):
        if not hasattr(self, "model_registry"):
            self.skipTest("Base class")
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
        pass

    def _run_steps(
        self, model: nn.Module, trainer_cls: type, *, enable_passes: bool = True
    ) -> tuple[torch.Tensor, str, str]:
        """Run forward-backward-optimizer steps using the given trainer class."""
        # Annotate after deepcopy: annotate_fn wrappers capture bound methods
        # that don't rebind correctly through copy.deepcopy.
        self.annotate_model(model)
        trainer = _build_trainer(
            model, self.model_config, trainer_cls, enable_passes=enable_passes
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


# TODO: max_autotune=True causes multiple issues for FlexAttn bitwise tests:
# kernel config divergence between eager and traced paths, and triton shared
# memory OOMs for large head dims (DSv3). Investigate after stabilizing
# max_autotune with regional_inductor.
@unittest.skip("max_autotune breaks FlexAttn bitwise tests")
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


@unittest.skip("max_autotune breaks FlexAttn bitwise tests")
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
            """09bb71bafdd888cf46c5f5c7ccddb5441266f843f9a2255d1569121613035be9""",
        )
        assert_expected_inline(
            grad_hash,
            """8bb6e647c3edaa229cc65872086ccc5c4e1b7f1647bb01da4506ab777a64a0db""",
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
