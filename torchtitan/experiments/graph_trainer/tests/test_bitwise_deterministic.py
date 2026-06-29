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

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import DebugConfig, ParallelismConfig, TrainingConfig
from torchtitan.experiments.graph_trainer.common_utils import (
    maybe_register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.configs import (
    EpOverlapConfig,
    GraphTrainerCompileConfig,
)
from torchtitan.experiments.graph_trainer.deepseek_v3 import (
    model_registry as dsv3_model_registry,
)
from torchtitan.experiments.graph_trainer.deepseek_v3.parallelize import (
    annotate_deepseekv3,
)
from torchtitan.experiments.graph_trainer.ep_eager_chunk import (
    maybe_apply_ep_overlap_eager_chunking,
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
    # The unsuffixed subclasses use SDPA (a test-only backend that exercises the
    # backend-agnostic graph machinery — precompile serialization, codegen,
    # determinism — without FlexAttention's unpicklable, non-tensor BlockMask).
    # The *FlexAttn subclasses override this to "flex".
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
        # Match Trainer.__init__: model configs consume runtime settings before
        # build. DSv3 uses the synced RoPE length to decide YaRN scaling.
        runtime_config = Trainer.Config(
            model_spec=model_spec,
            training=TrainingConfig(
                local_batch_size=BATCH_SIZE,
                seq_len=SEQ_LEN,
                steps=NUM_STEPS,
            ),
            parallelism=ParallelismConfig(),
            checkpoint=CheckpointManager.Config(initial_load_model_only=False),
            debug=DebugConfig(seed=SEED, deterministic=True),
        )
        self.model_config.update_from_config(config=runtime_config)
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
        compile_passes: list[str] | None = None,
        compile_ep_overlap_enabled: bool = False,
        compile_ep_overlap_chunk_dim: str = "batch",
        compile_ep_overlap_module_fqn: str = "layers.*",
        compile_ep_overlap_disable_early_grad_accumulation: bool = False,
        compile_inductor_compilation: str = "regional",
        compile_disable_passes: list[str] | None = None,
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
            compile_passes=compile_passes,
            compile_ep_overlap_enabled=compile_ep_overlap_enabled,
            compile_ep_overlap_chunk_dim=compile_ep_overlap_chunk_dim,
            compile_ep_overlap_module_fqn=compile_ep_overlap_module_fqn,
            compile_ep_overlap_disable_early_grad_accumulation=(
                compile_ep_overlap_disable_early_grad_accumulation
            ),
            compile_inductor_compilation=compile_inductor_compilation,
            compile_disable_passes=compile_disable_passes,
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
            extra_kwargs,
        )

        # Step 2: Apply compile-time passes (cleanup + regional_inductor)
        # before saving, so compiled Triton kernels are baked in
        if enable_passes:
            config = SimpleNamespace(
                model_spec=SimpleNamespace(model=self.model_config),
                compile=GraphTrainerCompileConfig(
                    enable=True,
                    mode="aot_fx_trace",
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
                compile=GraphTrainerCompileConfig(
                    enable=True,
                    mode="aot_fx_trace",
                    precompile_artifact_dir="precompiled",
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
        assert_expected_inline(str(loss.item()), """7.961757183074951""")
        assert_expected_inline(
            model_hash,
            """135abf5bcdaac7d1a0cbf1b8e0f2bf84ac788ce202f3ec4b23bc8b05d5ee2f1e""",
        )
        assert_expected_inline(
            grad_hash,
            """fe39cc3a984828640c2b9a95458f76cd82046f3147d9ad960845851731ceb4ad""",
        )

    def test_aot_fx_trace_vs_eager(self):
        """aot_fx_trace and eager produce bitwise identical losses and grads."""
        run_eager = self._run_steps(copy.deepcopy(self.model), Trainer)
        run_traced = self._run_steps(copy.deepcopy(self.model), GraphTrainer)

        self._assert_runs_match(run_eager, run_traced, "eager vs aot_fx_trace: ")

    def test_precompile_vs_trace(self):
        """Precompiled aot_fx_trace (save/load roundtrip) matches direct trace."""
        if self.attn_backend == "flex":
            # FlexAttention's BlockMask mask_mod closures are Python code objects
            # that pickle.dumps cannot serialize. The SDPA subclasses cover this
            # path (SDPA carries no such object).
            self.skipTest("FlexAttention graphs contain unpicklable code objects")
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
        assert_expected_inline(str(loss.item()), """7.474959373474121""")
        assert_expected_inline(
            model_hash,
            """99d414c98ec7de31e58ad41f3324a402dfa4f8fc4b3c57a406368af7360a2ec6""",
        )
        assert_expected_inline(
            grad_hash,
            """7f38de9544b31d1161c4191987663e2e8cc7be8b83b6226ff72a349a71aa1afa""",
        )

    def test_aot_fx_trace_vs_eager(self):
        """aot_fx_trace and eager produce bitwise identical losses and grads."""
        run_eager = self._run_steps(copy.deepcopy(self.model), Trainer)
        run_traced = self._run_steps(copy.deepcopy(self.model), GraphTrainer)

        self._assert_runs_match(run_eager, run_traced, "eager vs aot_fx_trace: ")

    def test_precompile_vs_trace(self):
        """Precompiled aot_fx_trace (save/load roundtrip) matches direct trace."""
        if self.attn_backend == "flex":
            # FlexAttention's BlockMask mask_mod closures are Python code objects
            # that pickle.dumps cannot serialize. The SDPA subclasses cover this
            # path (SDPA carries no such object).
            self.skipTest("FlexAttention graphs contain unpicklable code objects")
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
    """Bitwise determinism tests for Llama3 with FlexAttention (debugmodel).

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
        assert_expected_inline(str(loss.item()), """7.961757659912109""")
        assert_expected_inline(
            model_hash,
            """cfb3c5cba404ea7a6f9adae605277995d157cf3b624e5a1e3729a786996da350""",
        )
        assert_expected_inline(
            grad_hash,
            """b8c505488cf613d016ed55e0e4dbcfcd8a278cc9e27f3cf0154301cbefc7ca4a""",
        )

    def test_aot_fx_trace_vs_eager(self):
        """aot_fx_trace with passes and eager produce bitwise identical results."""
        run_eager = self._run_steps(copy.deepcopy(self.model), Trainer)
        run_traced = self._run_steps(copy.deepcopy(self.model), GraphTrainer)
        self._assert_runs_match(run_eager, run_traced, "eager vs aot_fx_trace: ")

    def test_precompile_vs_trace(self):
        """Precompiled aot_fx_trace (save/load roundtrip) matches direct trace."""
        if self.attn_backend == "flex":
            # FlexAttention's BlockMask mask_mod closures are Python code objects
            # that pickle.dumps cannot serialize. The SDPA subclasses cover this
            # path (SDPA carries no such object).
            self.skipTest("FlexAttention graphs contain unpicklable code objects")
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
    """Bitwise determinism tests for DSv3 with FlexAttention (debugmodel).

    aot_fx_trace compiles FlexAttention HOPs via regional_inductor into fused
    Triton kernels and produces bitwise identical results to eager.
    """

    model_registry = staticmethod(dsv3_model_registry)
    model_flavor = "debugmodel"
    attn_backend = "flex"
    annotate_model = staticmethod(annotate_deepseekv3)

    def _wrap_ep_chunk_eager_baseline(self, model: nn.Module, case: str) -> None:
        if case == "transformer_batch":
            mode, module_fqn = "batch", "layers.*"
        elif case == "moe_batch":
            mode, module_fqn = "batch", "layers.*.moe"
        elif case == "moe_seq":
            mode, module_fqn = "seq", "layers.*.moe"
        else:
            raise AssertionError(f"unknown EP chunk case {case}")
        maybe_apply_ep_overlap_eager_chunking(
            model,
            GraphTrainerCompileConfig(
                enable=True,
                ep_overlap=EpOverlapConfig(
                    enabled=True,
                    strategy="eager",
                    chunk_dim=mode,
                    module_fqn=module_fqn,
                ),
            ),
        )

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
        assert_expected_inline(str(loss.item()), """7.474959373474121""")
        assert_expected_inline(
            model_hash,
            """aff6268960d2b137a44756d322b72c0dc253ffbbc54077e4105fc3abed0fdd79""",
        )
        assert_expected_inline(
            grad_hash,
            """86f1656c38fa1e23d6b29955889bd5c4ad0c363f9755e2c4ea8dee142cacfcfe""",
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

    def test_precompile_vs_trace(self):
        """Precompiled aot_fx_trace (save/load roundtrip) matches direct trace."""
        if self.attn_backend == "flex":
            # FlexAttention's BlockMask mask_mod closures are Python code objects
            # that pickle.dumps cannot serialize. The SDPA subclasses cover this
            # path (SDPA carries no such object).
            self.skipTest("FlexAttention graphs contain unpicklable code objects")
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

    @unittest.skipUnless(
        has_cuda_capability(9, 0),
        "flex_attention compilation exceeds resource limits on pre-Hopper GPUs",
    )
    def test_ep_chunk_matches_eager_chunking_bitwise(self):
        """Fast single-GPU prerequisite for FlexAttention EP chunking numerics.

        This validates chunking logic, pass composability, and eager-chunked vs.
        graph-chunked numerics with and without the post-schedule concretization
        pass. It does not exercise real EP all-to-all communication,
        distributed sharding, or overlap behavior; the distributed DSV3 numerics
        tests provide that end-to-end coverage.
        """
        cases = [
            ("transformer_batch", "batch", "layers.*"),
            ("moe_batch", "batch", "layers.*.moe"),
            ("moe_seq", "seq", "layers.*.moe"),
        ]
        for case, mode, modules in cases:
            with self.subTest(case=case):
                eager_model = copy.deepcopy(self.model)
                self._wrap_ep_chunk_eager_baseline(eager_model, case)

                run_eager = self._run_steps(eager_model, Trainer)
                graph_model = copy.deepcopy(self.model)
                # The eager FlexAttention baseline compiles with concrete dims.
                # Reset Dynamo before tracing the graph-chunked production path,
                # which starts symbolic and then concretizes before Inductor.
                torch._dynamo.reset()
                run_traced = self._run_steps(
                    graph_model,
                    GraphTrainer,
                    compile_ep_overlap_enabled=True,
                    compile_ep_overlap_chunk_dim=mode,
                    compile_ep_overlap_module_fqn=modules,
                    compile_ep_overlap_disable_early_grad_accumulation=True,
                )

                self._assert_runs_match(
                    run_eager,
                    run_traced,
                    f"eager chunk vs ep_chunk {case}: ",
                )


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
        assert_expected_inline(str(loss.item()), """7.2979936599731445""")
        assert_expected_inline(
            model_hash,
            """dea001dde5db1b02f8b4c25fefc8ed517c390cacdfcea946d3829eb492e3c6dc""",
        )
        assert_expected_inline(
            grad_hash,
            """baf7140456134fe7256af1d8144bb6a579a731121a9843db177643b24eac9cd1""",
        )

    def test_aot_fx_trace_vs_eager(self):
        """aot_fx_trace and eager produce bitwise identical losses and grads."""
        run_eager = self._run_steps(copy.deepcopy(self.model), Trainer)
        run_traced = self._run_steps(copy.deepcopy(self.model), GraphTrainer)

        self._assert_runs_match(run_eager, run_traced, "eager vs aot_fx_trace: ")

    def test_precompile_vs_trace(self):
        """Precompiled aot_fx_trace (save/load roundtrip) matches direct trace."""
        if self.attn_backend == "flex":
            # FlexAttention's BlockMask mask_mod closures are Python code objects
            # that pickle.dumps cannot serialize. The SDPA subclasses cover this
            # path (SDPA carries no such object).
            self.skipTest("FlexAttention graphs contain unpicklable code objects")
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
        assert_expected_inline(str(loss.item()), """7.297994613647461""")
        assert_expected_inline(
            model_hash,
            """70ea7c9f5224ad2f8796270873f39d329ded42a9b3f437a14577d61c5d78e4c5""",
        )
        assert_expected_inline(
            grad_hash,
            """c336a44443add5c6af2b35de62e5f8584b2d0f8876390032ebceffc2a6608e03""",
        )

    def test_aot_fx_trace_vs_eager(self):
        """aot_fx_trace with passes and eager produce bitwise identical results."""
        run_eager = self._run_steps(copy.deepcopy(self.model), Trainer)
        run_traced = self._run_steps(copy.deepcopy(self.model), GraphTrainer)
        self._assert_runs_match(run_eager, run_traced, "eager vs aot_fx_trace: ")

    def test_precompile_vs_trace(self):
        """Precompiled aot_fx_trace (save/load roundtrip) matches direct trace."""
        if self.attn_backend == "flex":
            # FlexAttention's BlockMask mask_mod closures are Python code objects
            # that pickle.dumps cannot serialize. The SDPA subclasses cover this
            # path (SDPA carries no such object).
            self.skipTest("FlexAttention graphs contain unpicklable code objects")
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
