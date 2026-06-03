# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bitwise deterministic guardrail for graph_trainer.

Tests that Trainer (eager) and GraphTrainer (aot_fx_trace) produce bitwise
identical losses and gradients on Llama3, DeepSeek-v3, and Qwen3 debug models.

Requires a CUDA GPU. Run with:
    pytest torchtitan/experiments/graph_trainer/tests/test_bitwise_deterministic.py -x
"""

import copy
import tempfile
import unittest
from collections.abc import Callable
from types import SimpleNamespace

import torch
import torch.nn as nn
from expecttest import assert_expected_inline
from tests.utils import hash_gradient, hash_model
from torch.nn.attention.flex_attention import flex_attention

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.experiments.graph_trainer.common_utils import (
    maybe_register_blockmask_pytree_node,
)
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
from torchtitan.experiments.graph_trainer.qwen3 import (
    model_registry as qwen3_model_registry,
)
from torchtitan.experiments.graph_trainer.qwen3.parallelize import annotate_qwen3
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
    torch.use_deterministic_algorithms(True)


_TOKENIZER_PATH = "./tests/assets/tokenizer"


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class BitwiseDeterministicBase(unittest.TestCase):
    """Base class for bitwise determinism tests.

    Subclasses must set `model_registry` to the appropriate model registry function.
    """

    model_registry: Callable
    annotate_model: Callable
    model_flavor: str
    attn_backend: str = "sdpa"

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
        model_spec = self.model_registry(
            self.model_flavor, attn_backend=self.attn_backend
        )
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
        self.positions = torch.arange(SEQ_LEN, device="cuda").repeat(BATCH_SIZE, 1)

    def tearDown(self):
        FlexAttention.inductor_configs = self._orig_inductor_configs
        FlexAttention._compiled_flex_attn = self._orig_compiled_flex_attn

    def _get_extra_kwargs(self, model: nn.Module) -> dict[str, object]:
        """Build extra_kwargs matching what post_dataloading_process produces.

        For FlexAttention models, this generates the BlockMask attention
        masks. For SDPA models, returns an empty dict.
        """
        from torchtitan.models.common.attention import FlexAttention as FlexAttnModule
        from torchtitan.models.common.decoder import Decoder

        if not isinstance(self.model_config, Decoder.Config):
            return {}
        layer = self.model_config.layers[0]
        inner_attention = getattr(layer.attention, "inner_attention", None)
        if not isinstance(inner_attention, FlexAttnModule.Config):
            return {}
        attention_masks = model.get_attention_masks(self.positions)
        return {"attention_masks": attention_masks}

    def _run_steps(
        self,
        model: nn.Module,
        trainer_cls: type,
        *,
        enable_passes: bool = True,
        numerics_changing_optim: bool = False,
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
            compile_numerics_changing_optim=numerics_changing_optim,
            tokenizer=HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH),
        )
        global_valid_tokens = torch.tensor(
            BATCH_SIZE * SEQ_LEN, dtype=torch.float, device="cuda"
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        for _ in range(NUM_STEPS):
            optimizer.zero_grad()
            loss = trainer.forward_backward_step(
                input_dict={"input": self.inputs, "positions": self.positions},
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
            minimal_fx_tracer,
            run_traced,
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
        loss_fn = CrossEntropyLoss.Config().build()
        fwd_bwd_fn = make_fwd_bwd_step(model, loss_fn)

        global_valid_tokens = torch.tensor(
            BATCH_SIZE * SEQ_LEN, dtype=torch.float, device="cuda"
        )
        extra_inputs: dict[str, torch.Tensor] = {}
        extra_kwargs: dict[str, object] = {
            "positions": self.positions,
            **self._get_extra_kwargs(model),
        }
        maybe_register_blockmask_pytree_node()

        # Step 1: Trace the graph
        traced_result = minimal_fx_tracer(fwd_bwd_fn, module=model)(
            self.inputs,
            self.labels,
            global_valid_tokens,
            extra_inputs,
            extra_kwargs,
        )

        # Step 2: Apply compile-time passes (cleanup + regional_inductor)
        # before saving, so compiled Triton kernels are baked in
        if enable_passes:
            config = SimpleNamespace(
                model_spec=SimpleNamespace(model=self.model_config),
                compile=SimpleNamespace(
                    memory_policy="default",
                    inductor_compilation="regional",
                    numerics_changing_optim=False,
                    cpu_offload_prefetch_n_layers=1,
                    cpu_offload_defer_n_layers=1,
                    cpu_offload_budget_gb=100.0,
                    enable_fsdp_ag_rs_overlap=False,
                ),
                parallelism=SimpleNamespace(
                    pipeline_parallel_degree=1,
                    fsdp_reshard_after_forward="default",
                    enable_async_tensor_parallel=False,
                ),
            )
            passes = compile_time_passes(traced_result, config)
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
            load_config = SimpleNamespace(
                model_spec=SimpleNamespace(model=self.model_config),
                compile=SimpleNamespace(
                    precompile_artifact_dir="precompiled",
                    inductor_compilation="regional",
                    disable_passes=[],
                ),
            )
            passes = construct_default_graph_passes(loaded_result, load_config)
            loaded_result.gm = apply_graph_passes(
                loaded_result.gm,
                loaded_result.example_inputs,
                passes,
            )

        # Step 4: Run training steps using the loaded artifact
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        for _ in range(NUM_STEPS):
            optimizer.zero_grad()
            outputs = run_traced(loaded_result, module=model)(
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
            """d8c4495bc41d103e3864433002d31be0823567938729396c44eb2f2782a47a23""",
        )
        assert_expected_inline(
            grad_hash,
            """926c46345abe29f427f072fb747375009cac66c4ab4be4b41d09661356089016""",
        )

    def test_aot_fx_trace_vs_eager(self):
        """aot_fx_trace and eager produce bitwise identical losses and grads."""
        run_eager = self._run_steps(copy.deepcopy(self.model), Trainer)
        run_traced = self._run_steps(copy.deepcopy(self.model), GraphTrainer)

        self._assert_runs_match(run_eager, run_traced, "eager vs aot_fx_trace: ")

    def test_precompile_vs_trace(self):
        """Precompiled aot_fx_trace (save/load roundtrip) matches direct trace."""
        run_traced = self._run_steps(copy.deepcopy(self.model), GraphTrainer)
        run_precompile = self._run_steps_with_precompile(copy.deepcopy(self.model))

        self._assert_runs_match(run_traced, run_precompile, "trace vs precompile: ")

    def test_numerics_changing_optim_run_to_run(self):
        """Two runs with numerics_changing_optim produce bitwise identical results."""
        run_a = self._run_steps(
            copy.deepcopy(self.model),
            GraphTrainer,
            numerics_changing_optim=True,
        )
        run_b = self._run_steps(
            copy.deepcopy(self.model),
            GraphTrainer,
            numerics_changing_optim=True,
        )

        self._assert_runs_match(run_a, run_b, "numerics_changing_optim run-to-run: ")


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
            """edaec1177d073cf99a24433a6381b23282bfbfe306c40cefcea5d4efaf14cd0a""",
        )
        assert_expected_inline(
            grad_hash,
            """ce80bf7a7186d63eb6231d684ecefe7a7846f1bc63c8fde794fefd462e9c2c5d""",
        )

    def test_aot_fx_trace_vs_eager(self):
        """aot_fx_trace and eager produce bitwise identical losses and grads."""
        run_eager = self._run_steps(copy.deepcopy(self.model), Trainer)
        run_traced = self._run_steps(copy.deepcopy(self.model), GraphTrainer)

        self._assert_runs_match(run_eager, run_traced, "eager vs aot_fx_trace: ")

    def test_precompile_vs_trace(self):
        """Precompiled aot_fx_trace (save/load roundtrip) matches direct trace."""
        run_traced = self._run_steps(copy.deepcopy(self.model), GraphTrainer)
        run_precompile = self._run_steps_with_precompile(copy.deepcopy(self.model))

        self._assert_runs_match(run_traced, run_precompile, "trace vs precompile: ")

    def test_numerics_changing_optim_run_to_run(self):
        """Two runs with numerics_changing_optim produce bitwise identical results."""
        run_a = self._run_steps(
            copy.deepcopy(self.model),
            GraphTrainer,
            numerics_changing_optim=True,
        )
        run_b = self._run_steps(
            copy.deepcopy(self.model),
            GraphTrainer,
            numerics_changing_optim=True,
        )

        self._assert_runs_match(run_a, run_b, "numerics_changing_optim run-to-run: ")


class TestLlama3FlexAttnBitwiseDeterministic(BitwiseDeterministicBase):
    """Bitwise determinism tests for Llama3 with FlexAttention (debugmodel_flex_attn).

    aot_fx_trace compiles FlexAttention HOPs via regional_inductor into fused
    Triton kernels and produces bitwise identical results to eager.
    """

    model_registry = staticmethod(llama3_model_registry)
    model_flavor = "debugmodel"
    attn_backend = "flex"
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
            """2cc38288f1641b058a56a1930af77dcb33c91fb12176cfdb59f436c9a2b3addd""",
        )
        assert_expected_inline(
            grad_hash,
            """9163c0f93a0a8dbf02208dae3ee0b427a97e3dd39e83b84847d2ed4b4e2bc495""",
        )

    def test_aot_fx_trace_vs_eager(self):
        """aot_fx_trace with passes and eager produce bitwise identical results."""
        run_eager = self._run_steps(copy.deepcopy(self.model), Trainer)
        run_traced = self._run_steps(copy.deepcopy(self.model), GraphTrainer)
        self._assert_runs_match(run_eager, run_traced, "eager vs aot_fx_trace: ")

    @unittest.skip("FlexAttention graphs contain unpicklable code objects")
    def test_precompile_vs_trace(self):
        """Precompiled aot_fx_trace (save/load roundtrip) matches direct trace."""
        run_traced = self._run_steps(copy.deepcopy(self.model), GraphTrainer)
        run_precompile = self._run_steps_with_precompile(copy.deepcopy(self.model))

        self._assert_runs_match(run_traced, run_precompile, "trace vs precompile: ")

    def test_numerics_changing_optim_run_to_run(self):
        """Two runs with numerics_changing_optim produce bitwise identical results."""
        run_a = self._run_steps(
            copy.deepcopy(self.model),
            GraphTrainer,
            numerics_changing_optim=True,
        )
        run_b = self._run_steps(
            copy.deepcopy(self.model),
            GraphTrainer,
            numerics_changing_optim=True,
        )

        self._assert_runs_match(run_a, run_b, "numerics_changing_optim run-to-run: ")


class TestDSv3FlexAttnBitwiseDeterministic(BitwiseDeterministicBase):
    """Bitwise determinism tests for DSv3 with FlexAttention (debugmodel_flex_attn).

    aot_fx_trace compiles FlexAttention HOPs via regional_inductor into fused
    Triton kernels and produces bitwise identical results to eager.
    """

    model_registry = staticmethod(dsv3_model_registry)
    model_flavor = "debugmodel"
    attn_backend = "flex"
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
            """d2670e9bf949d83c446bcce1ca468a23eda98c83c0cac83eafb15ddebde3c234""",
        )
        assert_expected_inline(
            grad_hash,
            """e90a28b41bdae0de5d1db20de40998b0508866efed561cce4373e595626e8e7a""",
        )

    # TODO: FlexAttention compilation exceeds resource limits on pre-Hopper GPUs.
    # Revisit when GraphTrainer addresses peak memory during compilation.
    @unittest.skipUnless(
        has_cuda_capability(9, 0),
        "flex_attention compilation exceeds resource limits on pre-Hopper GPUs",
    )
    def test_aot_fx_trace_vs_eager(self):
        """aot_fx_trace with passes and eager produce bitwise identical results."""
        run_eager = self._run_steps(copy.deepcopy(self.model), Trainer)
        run_traced = self._run_steps(copy.deepcopy(self.model), GraphTrainer)
        self._assert_runs_match(run_eager, run_traced, "eager vs aot_fx_trace: ")

    @unittest.skip("FlexAttention graphs contain unpicklable code objects")
    def test_precompile_vs_trace(self):
        """Precompiled aot_fx_trace (save/load roundtrip) matches direct trace."""
        run_traced = self._run_steps(copy.deepcopy(self.model), GraphTrainer)
        run_precompile = self._run_steps_with_precompile(copy.deepcopy(self.model))

        self._assert_runs_match(run_traced, run_precompile, "trace vs precompile: ")

    # TODO: FlexAttention compilation exceeds resource limits on pre-Hopper GPUs.
    @unittest.skipUnless(
        has_cuda_capability(9, 0),
        "flex_attention compilation exceeds resource limits on pre-Hopper GPUs",
    )
    def test_numerics_changing_optim_run_to_run(self):
        """Two runs with numerics_changing_optim produce bitwise identical results."""
        run_a = self._run_steps(
            copy.deepcopy(self.model),
            GraphTrainer,
            numerics_changing_optim=True,
        )
        run_b = self._run_steps(
            copy.deepcopy(self.model),
            GraphTrainer,
            numerics_changing_optim=True,
        )

        self._assert_runs_match(run_a, run_b, "numerics_changing_optim run-to-run: ")


class TestQwen3MoEBitwiseDeterministic(BitwiseDeterministicBase):
    """Bitwise determinism tests for Qwen3 MoE debug model."""

    model_registry = staticmethod(qwen3_model_registry)
    model_flavor = "debugmodel_moe"
    annotate_model = staticmethod(annotate_qwen3)

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
        assert_expected_inline(str(loss.item()), """7.297995567321777""")
        assert_expected_inline(
            model_hash,
            """c4f3d5d6a4dacffc82a0845ef620dcbdb053d9785ce64b8dd5b5e181f4fe2d1b""",
        )
        assert_expected_inline(
            grad_hash,
            """b24c4d0201f11a825bbf49269592968ffd53d5b52bc0486bcacafae91e90eee3""",
        )

    def test_aot_fx_trace_vs_eager(self):
        """aot_fx_trace and eager produce bitwise identical losses and grads."""
        run_eager = self._run_steps(copy.deepcopy(self.model), Trainer)
        run_traced = self._run_steps(copy.deepcopy(self.model), GraphTrainer)

        self._assert_runs_match(run_eager, run_traced, "eager vs aot_fx_trace: ")

    def test_precompile_vs_trace(self):
        """Precompiled aot_fx_trace (save/load roundtrip) matches direct trace."""
        run_traced = self._run_steps(copy.deepcopy(self.model), GraphTrainer)
        run_precompile = self._run_steps_with_precompile(copy.deepcopy(self.model))

        self._assert_runs_match(run_traced, run_precompile, "trace vs precompile: ")

    def test_numerics_changing_optim_run_to_run(self):
        """Two runs with numerics_changing_optim produce bitwise identical results."""
        run_a = self._run_steps(
            copy.deepcopy(self.model),
            GraphTrainer,
            numerics_changing_optim=True,
        )
        run_b = self._run_steps(
            copy.deepcopy(self.model),
            GraphTrainer,
            numerics_changing_optim=True,
        )

        self._assert_runs_match(run_a, run_b, "numerics_changing_optim run-to-run: ")


class TestQwen3MoEFlexAttnBitwiseDeterministic(BitwiseDeterministicBase):
    """Bitwise determinism tests for Qwen3 MoE with FlexAttention.

    aot_fx_trace compiles FlexAttention HOPs via regional_inductor into fused
    Triton kernels and produces bitwise identical results to eager.
    """

    model_registry = staticmethod(qwen3_model_registry)
    model_flavor = "debugmodel_moe"
    attn_backend = "flex"
    annotate_model = staticmethod(annotate_qwen3)

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
        assert_expected_inline(str(loss.item()), """7.297987461090088""")
        assert_expected_inline(
            model_hash,
            """85240276507f93d2dbc8b09d5dad5f86623bcd49423abe64fba787b74d7f6d81""",
        )
        assert_expected_inline(
            grad_hash,
            """74877b0fa386c66c3154e0371adb86c9255527f0d48fb623feb7ee70ee79d409""",
        )

    def test_aot_fx_trace_vs_eager(self):
        """aot_fx_trace with passes and eager produce bitwise identical results."""
        run_eager = self._run_steps(copy.deepcopy(self.model), Trainer)
        run_traced = self._run_steps(copy.deepcopy(self.model), GraphTrainer)
        self._assert_runs_match(run_eager, run_traced, "eager vs aot_fx_trace: ")

    @unittest.skip("FlexAttention graphs contain unpicklable code objects")
    def test_precompile_vs_trace(self):
        """Precompiled aot_fx_trace (save/load roundtrip) matches direct trace."""
        run_traced = self._run_steps(copy.deepcopy(self.model), GraphTrainer)
        run_precompile = self._run_steps_with_precompile(copy.deepcopy(self.model))

        self._assert_runs_match(run_traced, run_precompile, "trace vs precompile: ")

    def test_numerics_changing_optim_run_to_run(self):
        """Two runs with numerics_changing_optim produce bitwise identical results."""
        run_a = self._run_steps(
            copy.deepcopy(self.model),
            GraphTrainer,
            numerics_changing_optim=True,
        )
        run_b = self._run_steps(
            copy.deepcopy(self.model),
            GraphTrainer,
            numerics_changing_optim=True,
        )

        self._assert_runs_match(run_a, run_b, "numerics_changing_optim run-to-run: ")


if __name__ == "__main__":
    unittest.main()
