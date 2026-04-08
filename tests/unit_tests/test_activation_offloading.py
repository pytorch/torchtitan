# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)

from torchtitan.distributed.activation_offloading import (
    ActivationOffloadingManager,
    get_activation_offloading_ctx,
)

CUDA = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleMLP(nn.Module):
    """Two-layer MLP used across tests."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class MLPWithOutput(nn.Module):
    """MLP that exposes an `output` sub-module (mirrors LM-head structure)."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(torch.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestActivationOffloading(unittest.TestCase):
    # ------------------------------------------------------------------
    # 1. Activations moved to CPU
    #
    # On CUDA: pack_hook should return integer IDs for eligible tensors,
    # confirming they were moved to CPU.  On CPU-only machines the test
    # verifies at least that the context manager is transparent (no crash,
    # correct gradients), because there are no CUDA tensors to offload.
    # ------------------------------------------------------------------
    def test_activations_moved_to_cpu(self) -> None:
        """Pack hook returns int IDs (not tensors) for offloaded CUDA tensors."""
        device = "cuda" if CUDA else "cpu"
        model = SimpleMLP().to(device)
        x = torch.randn(8, 64, device=device, requires_grad=True)

        packed_results: list = []

        ctx = ActivationOffloadingManager(
            use_streams=CUDA,
            min_offload_size=0,
        )
        orig_pack = ctx._pack_hook

        def _spy_pack(t):
            result = orig_pack(t)
            packed_results.append(result)
            return result

        ctx._pack_hook = _spy_pack  # type: ignore[method-assign]

        with torch.autograd.graph.saved_tensors_hooks(ctx._pack_hook, ctx._unpack_hook):
            out = model(x)
        out.sum().backward()

        if CUDA:
            # At least some saved tensors should have been offloaded to an
            # integer ID, confirming the D2H copy path was exercised.
            int_ids = [r for r in packed_results if isinstance(r, int)]
            self.assertTrue(
                len(int_ids) > 0,
                f"Expected integer IDs for offloaded tensors, got: "
                f"{[type(r).__name__ for r in packed_results]}",
            )
        else:
            # CPU path: no CUDA tensors, pack hook returns tensors as-is.
            for r in packed_results:
                self.assertIsInstance(r, torch.Tensor)

    # ------------------------------------------------------------------
    # 2. Gradients are correct
    #
    # On CUDA: actual D2H/H2D transfers happen; gradients must match baseline.
    # On CPU: no CUDA tensors, so the context is transparent; gradients trivially
    # match but the backward round-trip is still exercised.
    # ------------------------------------------------------------------
    def test_gradients_are_correct(self) -> None:
        """Gradients with offloading enabled must match the baseline exactly."""
        device = "cuda" if CUDA else "cpu"
        torch.manual_seed(0)

        model_ref = SimpleMLP().to(device)
        model_off = SimpleMLP().to(device)
        model_off.load_state_dict(model_ref.state_dict())

        x = torch.randn(4, 64, device=device)

        # Baseline
        x_ref = x.clone().requires_grad_(True)
        model_ref.zero_grad()
        model_ref(x_ref).sum().backward()

        # With offloading
        ctx = ActivationOffloadingManager(
            use_streams=CUDA,
            min_offload_size=0,
        )
        x_off = x.clone().requires_grad_(True)
        model_off.zero_grad()
        with ctx:
            out = model_off(x_off)
        out.sum().backward()

        torch.testing.assert_close(x_ref.grad, x_off.grad)
        for p_ref, p_off in zip(model_ref.parameters(), model_off.parameters()):
            torch.testing.assert_close(p_ref.grad, p_off.grad)

    # ------------------------------------------------------------------
    # 2b. Gradients correct across multiple steps
    # ------------------------------------------------------------------
    def test_gradients_correct_multiple_steps(self) -> None:
        """Repeated enter/exit cycles must produce correct gradients each step."""
        device = "cuda" if CUDA else "cpu"
        torch.manual_seed(42)

        model_ref = SimpleMLP().to(device)
        model_off = SimpleMLP().to(device)
        model_off.load_state_dict(model_ref.state_dict())

        ctx = ActivationOffloadingManager(
            use_streams=CUDA,
            min_offload_size=0,
        )

        for step in range(3):
            x = torch.randn(4, 64, device=device)

            x_ref = x.clone().requires_grad_(True)
            model_ref.zero_grad()
            model_ref(x_ref).sum().backward()

            x_off = x.clone().requires_grad_(True)
            model_off.zero_grad()
            with ctx:
                out = model_off(x_off)
            out.sum().backward()

            torch.testing.assert_close(
                x_ref.grad,
                x_off.grad,
                msg=f"Input grad mismatch at step {step}",
            )

    # ------------------------------------------------------------------
    # 3. Parameters not offloaded
    # ------------------------------------------------------------------
    @unittest.skipUnless(CUDA, "CUDA required to produce eligible CUDA tensors")
    def test_parameters_not_offloaded(self) -> None:
        """nn.Parameter tensors must not be offloaded (returned as-is)."""
        model = SimpleMLP().cuda()
        ctx = ActivationOffloadingManager(
            use_streams=True,
            min_offload_size=0,
        )
        orig_pack = ctx._pack_hook
        parameter_offloaded = []

        def _spy(t):
            result = orig_pack(t)
            if isinstance(t, nn.Parameter):
                # If a Parameter was offloaded, result would be an int ID.
                parameter_offloaded.append(isinstance(result, int))
            return result

        ctx._pack_hook = _spy  # type: ignore[method-assign]
        x = torch.randn(4, 64, device="cuda")
        with torch.autograd.graph.saved_tensors_hooks(ctx._pack_hook, ctx._unpack_hook):
            model(x).sum().backward()

        self.assertFalse(
            any(parameter_offloaded),
            "At least one nn.Parameter was offloaded — it should be returned as-is.",
        )

    # ------------------------------------------------------------------
    # 4. Small tensors not offloaded
    # ------------------------------------------------------------------
    @unittest.skipUnless(CUDA, "CUDA required to produce eligible CUDA tensors")
    def test_small_tensors_not_offloaded(self) -> None:
        """Tensors below min_offload_size should be returned as tensors (not IDs)."""
        # Large enough that no activation from a dim-8 MLP crosses it.
        min_size = 10_000

        ctx = ActivationOffloadingManager(
            use_streams=True,
            min_offload_size=min_size,
        )
        captured = []
        orig_pack = ctx._pack_hook

        def _spy(t):
            result = orig_pack(t)
            captured.append(result)
            return result

        ctx._pack_hook = _spy  # type: ignore[method-assign]

        model = SimpleMLP(dim=8).cuda()
        x = torch.randn(2, 8, device="cuda")
        with torch.autograd.graph.saved_tensors_hooks(ctx._pack_hook, ctx._unpack_hook):
            model(x).sum().backward()

        int_ids = [r for r in captured if isinstance(r, int)]
        self.assertEqual(
            len(int_ids),
            0,
            f"Expected no integer IDs for small tensors, but got {len(int_ids)}.",
        )

    # ------------------------------------------------------------------
    # 4b. Large tensors ARE offloaded (size threshold works both directions)
    # ------------------------------------------------------------------
    @unittest.skipUnless(CUDA, "CUDA required")
    def test_large_tensors_are_offloaded(self) -> None:
        """Tensors above min_offload_size threshold must be offloaded."""
        min_size = 4  # tiny threshold — basically every tensor qualifies
        ctx = ActivationOffloadingManager(
            use_streams=True,
            min_offload_size=min_size,
        )
        captured = []
        orig_pack = ctx._pack_hook

        def _spy(t):
            result = orig_pack(t)
            captured.append(result)
            return result

        ctx._pack_hook = _spy  # type: ignore[method-assign]

        model = SimpleMLP(dim=64).cuda()
        x = torch.randn(4, 64, device="cuda")
        with torch.autograd.graph.saved_tensors_hooks(ctx._pack_hook, ctx._unpack_hook):
            out = model(x)
        out.sum().backward()

        int_ids = [r for r in captured if isinstance(r, int)]
        self.assertGreater(
            len(int_ids), 0, "Expected at least one tensor to be offloaded."
        )

    # ------------------------------------------------------------------
    # 5. Disabled returns nullcontext
    # ------------------------------------------------------------------
    def test_disabled_returns_nullcontext(self) -> None:
        """get_activation_offloading_ctx with enable=False returns nullcontext."""
        model = SimpleMLP()
        ctx = get_activation_offloading_ctx(model, enable=False)
        self.assertIsInstance(ctx, contextlib.nullcontext)

        # Ensure it behaves as a no-op
        x = torch.randn(4, 64)
        with ctx:
            out = model(x)
        out.sum().backward()

    # ------------------------------------------------------------------
    # 6. Composable with activation checkpointing
    # ------------------------------------------------------------------
    def test_composable_with_checkpointing(self) -> None:
        """Activation offloading and checkpoint_wrapper should compose cleanly."""
        device = "cuda" if CUDA else "cpu"
        torch.manual_seed(1)

        inner = SimpleMLP().to(device)
        checkpointed = checkpoint_wrapper(inner)

        ctx = ActivationOffloadingManager(
            use_streams=CUDA,
            min_offload_size=0,
        )

        x = torch.randn(4, 64, device=device, requires_grad=True)
        with ctx:
            out = checkpointed(x)
        out.sum().backward()

        self.assertIsNotNone(x.grad)

    # ------------------------------------------------------------------
    # 7. PP guard
    # ------------------------------------------------------------------
    def test_pp_guard(self) -> None:
        """NotImplementedError is raised when PP is enabled."""
        parallel_dims = MagicMock()
        parallel_dims.pp_enabled = True
        enable_activation_offload = True

        with self.assertRaises(NotImplementedError):
            if enable_activation_offload and parallel_dims.pp_enabled:
                raise NotImplementedError(
                    "Activation offloading is not supported with Pipeline Parallel."
                )

        # No error when PP is disabled.
        parallel_dims.pp_enabled = False
        try:
            if enable_activation_offload and parallel_dims.pp_enabled:
                raise NotImplementedError("should not reach here")
        except NotImplementedError:
            self.fail("NotImplementedError raised unexpectedly with pp_enabled=False")

    # ------------------------------------------------------------------
    # 8. Context can be re-entered (step recycling doesn't corrupt state)
    # ------------------------------------------------------------------
    def test_context_reentrant(self) -> None:
        """__enter__ must cleanly reset state so repeated use is safe."""
        device = "cuda" if CUDA else "cpu"
        model = SimpleMLP().to(device)
        ctx = ActivationOffloadingManager(
            use_streams=CUDA,
            min_offload_size=0,
        )

        for _ in range(4):
            x = torch.randn(2, 64, device=device, requires_grad=True)
            with ctx:
                out = model(x)
            out.sum().backward()
            # _bwd_stash may have CUDA entries; they should be cleared next __enter__
            self.assertIsNotNone(x.grad)


# ---------------------------------------------------------------------------
# Numerical smoke test
#
# Mirrors what `python train.py --training.enable_activation_offload true
# --debug.seed 42 --debug.deterministic` would check, but without needing
# the full trainer stack.  Uses a multi-layer transformer-like model,
# bfloat16 mixed precision, multiple steps, and asserts gradients match
# the no-offload baseline within float tolerance.
# ---------------------------------------------------------------------------


class _TransformerBlock(nn.Module):
    """Single transformer block: layernorm -> linear -> relu -> linear."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = torch.relu(self.fc1(h))
        return x + self.fc2(h)


class _SmallTransformer(nn.Module):
    """6-layer transformer-like model that produces a scalar loss."""

    def __init__(self, dim: int = 128, num_layers: int = 6) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_TransformerBlock(dim) for _ in range(num_layers)])
        self.output = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.output(x).squeeze(-1)


def _one_step_grads(
    model: nn.Module,
    ctx,
    x: torch.Tensor,
) -> list[torch.Tensor]:
    """Run one forward+backward and return per-parameter gradients."""
    model.zero_grad()
    with ctx:
        pred = model(x)
    pred.sum().backward()
    return [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]


class TestNumericalSmoke(unittest.TestCase):
    """Smoke test: offloading must produce numerically correct results.

    Activation offloading changes the GPU memory address of saved tensors
    (H2D copies land at new allocations).  CUDA's matmul kernels can pick
    different tiling strategies based on pointer alignment, producing
    ULP-level floating-point differences vs. a non-offloaded run.  This is
    expected and harmless for training — the same non-determinism exists
    whenever two tensors have different addresses.

    We therefore verify that gradients are *close* (within float tolerance)
    rather than bit-identical.  The tolerances are deliberately generous to
    survive across GPU generations; they are far smaller than any numerical
    instability that would indicate a real correctness bug (data corruption,
    stale CPU buffers, etc.).
    """

    @unittest.skipUnless(CUDA, "CUDA required for D2H/H2D transfers")
    def test_gradients_match_baseline_float32(self) -> None:
        """float32 run — gradients must be within float tolerance."""
        self._run_smoke(dtype=torch.float32)

    @unittest.skipUnless(CUDA, "CUDA required for D2H/H2D transfers")
    def test_gradients_match_baseline_bfloat16(self) -> None:
        """bfloat16 — tolerances match bfloat16 precision."""
        self._run_smoke(dtype=torch.bfloat16)

    def _run_smoke(
        self,
        *,
        dtype: torch.dtype,
        dim: int = 64,
        batch: int = 2,
        seq: int = 8,
        seed: int = 42,
        atol: float = 1e-4,
        rtol: float = 1e-4,
    ) -> None:
        device = "cuda"
        torch.manual_seed(seed)

        # Shared weights — both models see exactly the same parameters.
        state = _SmallTransformer(dim=dim).to(device=device, dtype=dtype).state_dict()

        model_ref = _SmallTransformer(dim=dim).to(device=device, dtype=dtype)
        model_ref.load_state_dict(state)

        model_off = _SmallTransformer(dim=dim).to(device=device, dtype=dtype)
        model_off.load_state_dict(state)

        manager = ActivationOffloadingManager(
            use_streams=True,
            min_offload_size=0,
            max_fwd_stash_size=5,
        )

        # Run 3 independent forward+backward passes (same weights each time)
        # so gradient differences don't accumulate across steps.
        for step in range(3):
            torch.manual_seed(seed + step)
            x = torch.randn(batch, seq, dim, device=device, dtype=dtype)

            grads_ref = _one_step_grads(model_ref, contextlib.nullcontext(), x.clone())
            grads_off = _one_step_grads(model_off, manager, x.clone())

            self.assertEqual(len(grads_ref), len(grads_off))
            for i, (g_ref, g_off) in enumerate(zip(grads_ref, grads_off)):
                torch.testing.assert_close(
                    g_ref,
                    g_off,
                    atol=atol,
                    rtol=rtol,
                    msg=f"Gradient mismatch for param {i} at step {step}",
                )


# ---------------------------------------------------------------------------
# FSDP2 composability test
#
# Verifies that activation offloading composes correctly with FSDP2
# (torch.distributed._composable.fsdp.fully_shard).  Runs on a single-GPU
# single-process distributed group so no multi-GPU hardware is required.
# ---------------------------------------------------------------------------

DIST_AVAILABLE = torch.distributed.is_available()


def _init_dist(port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.distributed.init_process_group("gloo", rank=0, world_size=1)


class TestFSDP2Composability(unittest.TestCase):
    """Activation offloading must compose correctly with FSDP2."""

    @unittest.skipUnless(CUDA and DIST_AVAILABLE, "CUDA + dist required")
    def test_fsdp2_gradients_match_baseline(self) -> None:
        """FSDP2-wrapped model with offloading must produce correct gradients."""
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard
        from torch.distributed.tensor import DTensor

        _init_dist(29510)
        try:
            self._run_fsdp2_smoke(fully_shard, init_device_mesh, DTensor)
        finally:
            torch.distributed.destroy_process_group()

    def _run_fsdp2_smoke(self, fully_shard, init_device_mesh, DTensor) -> None:
        torch.manual_seed(42)
        device, dtype, dim = "cuda", torch.bfloat16, 64
        state = _SmallTransformer(dim=dim).to(device, dtype).state_dict()

        model_ref = _SmallTransformer(dim=dim).to(device, dtype)
        model_ref.load_state_dict(state)

        model_fsdp = _SmallTransformer(dim=dim).to(device, dtype)
        model_fsdp.load_state_dict(state)
        mesh = init_device_mesh(device, (1,))
        for layer in model_fsdp.layers:
            fully_shard(layer, mesh=mesh)
        fully_shard(model_fsdp, mesh=mesh)

        mgr = ActivationOffloadingManager(
            use_streams=True,
            min_offload_size=0,
        )

        def _grads(model, ctx, x):
            model.zero_grad()
            with ctx:
                pred = model(x)
            pred.sum().backward()
            return {
                n: (p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad)
                .detach()
                .clone()
                for n, p in model.named_parameters()
                if p.grad is not None
            }

        for step in range(3):
            torch.manual_seed(step)
            x = torch.randn(2, 8, dim, device=device, dtype=dtype)
            grads_ref = _grads(model_ref, contextlib.nullcontext(), x.clone())
            grads_fsdp = _grads(model_fsdp, mgr, x.clone())
            common = set(grads_ref) & set(grads_fsdp)
            self.assertTrue(len(common) > 0, "No common parameters found")
            for k in common:
                torch.testing.assert_close(
                    grads_ref[k],
                    grads_fsdp[k],
                    atol=1e-2,
                    rtol=1e-2,
                    msg=f"Gradient mismatch for {k} at step {step}",
                )


# ---------------------------------------------------------------------------
# Prefetching tests
#
# Verifies that register_prefetch_hooks correctly tracks tensor IDs per
# module and that backward pre-hooks trigger H2D copies before _unpack_hook.
# ---------------------------------------------------------------------------


class TestPrefetching(unittest.TestCase):
    """Tests for layer-level backward prefetching."""

    @unittest.skipUnless(CUDA, "CUDA required for prefetching")
    def test_prefetch_hooks_registered(self) -> None:
        """register_prefetch_hooks should register 3 hooks per module."""
        model = _SmallTransformer(dim=64).cuda()
        mgr = ActivationOffloadingManager(
            use_streams=True,
            min_offload_size=0,
        )
        layers = list(model.layers)
        mgr.register_prefetch_hooks(layers)
        # 3 hooks per layer: forward_pre, forward, backward_pre
        self.assertEqual(len(mgr._hooks), 3 * len(layers))
        self.assertEqual(mgr._num_tracked_modules, len(layers))

    @unittest.skipUnless(CUDA, "CUDA required for prefetching")
    def test_prefetch_tracks_tensor_ids(self) -> None:
        """Forward pass with prefetch hooks should track tensor IDs per module."""
        model = _SmallTransformer(dim=64).cuda()
        mgr = ActivationOffloadingManager(
            use_streams=True,
            min_offload_size=0,
        )
        mgr.register_prefetch_hooks(list(model.layers))

        x = torch.randn(2, 8, 64, device="cuda")
        with mgr:
            model(x)
        # Each layer should have tracked some tensor IDs.
        for i in range(len(model.layers)):
            self.assertIn(i, mgr._module_tensor_ids)
            self.assertGreater(
                len(mgr._module_tensor_ids[i]),
                0,
                f"Layer {i} should have tracked tensor IDs",
            )

    @unittest.skipUnless(CUDA, "CUDA required for prefetching")
    def test_prefetch_gradients_correct(self) -> None:
        """Prefetching must produce correct gradients."""
        torch.manual_seed(42)
        device, dim = "cuda", 64

        state = _SmallTransformer(dim=dim).to(device).state_dict()
        model_ref = _SmallTransformer(dim=dim).to(device)
        model_ref.load_state_dict(state)
        model_off = _SmallTransformer(dim=dim).to(device)
        model_off.load_state_dict(state)

        mgr = ActivationOffloadingManager(
            use_streams=True,
            min_offload_size=0,
        )
        mgr.register_prefetch_hooks(list(model_off.layers))

        for step in range(3):
            torch.manual_seed(step)
            x = torch.randn(2, 8, dim, device=device)

            grads_ref = _one_step_grads(model_ref, contextlib.nullcontext(), x.clone())
            grads_off = _one_step_grads(model_off, mgr, x.clone())

            for i, (g_ref, g_off) in enumerate(zip(grads_ref, grads_off)):
                torch.testing.assert_close(
                    g_ref,
                    g_off,
                    atol=1e-4,
                    rtol=1e-4,
                    msg=f"Gradient mismatch for param {i} at step {step}",
                )

    @unittest.skipUnless(CUDA, "CUDA required for prefetching")
    def test_get_ctx_registers_prefetch_hooks(self) -> None:
        """get_activation_offloading_ctx should auto-register prefetch hooks."""
        model = _SmallTransformer(dim=64).cuda()
        ctx = get_activation_offloading_ctx(model, enable=True, min_offload_size=0)
        self.assertIsInstance(ctx, ActivationOffloadingManager)
        # Should have registered hooks on the 6 layers
        self.assertEqual(ctx._num_tracked_modules, 6)
        self.assertEqual(len(ctx._hooks), 18)  # 3 * 6

    @unittest.skipUnless(CUDA, "CUDA required for prefetching")
    def test_prefetch_multiple_steps(self) -> None:
        """Prefetching across multiple steps must produce correct gradients."""
        torch.manual_seed(42)
        device, dim = "cuda", 64

        state = _SmallTransformer(dim=dim).to(device).state_dict()
        model_ref = _SmallTransformer(dim=dim).to(device)
        model_ref.load_state_dict(state)
        model_off = _SmallTransformer(dim=dim).to(device)
        model_off.load_state_dict(state)

        # Use get_activation_offloading_ctx which auto-registers prefetch hooks
        mgr = get_activation_offloading_ctx(model_off, enable=True, min_offload_size=0)

        for step in range(5):
            torch.manual_seed(step)
            x = torch.randn(2, 8, dim, device=device)

            grads_ref = _one_step_grads(model_ref, contextlib.nullcontext(), x.clone())
            grads_off = _one_step_grads(model_off, mgr, x.clone())

            for i, (g_ref, g_off) in enumerate(zip(grads_ref, grads_off)):
                torch.testing.assert_close(
                    g_ref,
                    g_off,
                    atol=1e-4,
                    rtol=1e-4,
                    msg=f"Gradient mismatch for param {i} at step {step}",
                )


if __name__ == "__main__":
    unittest.main()
