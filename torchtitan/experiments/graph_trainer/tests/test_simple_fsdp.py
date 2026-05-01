# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn as nn

from torch.distributed._tensor import DTensor, Replicate
from torch.distributed.device_mesh import DeviceMesh
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

from torchtitan.config.configs import TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.common_utils import apply_simple_fsdp
from torchtitan.experiments.graph_trainer.make_fx_tracer import trace_train_step

_HAS_TORCHAO = importlib.util.find_spec("torchao") is not None


def _single_rank_parallel_dims() -> ParallelDims:
    return ParallelDims(
        dp_replicate=1,
        dp_shard=1,
        cp=1,
        tp=1,
        pp=1,
        ep=1,
        world_size=1,
    )


def _bf16_mixed_precision_training() -> TrainingConfig:
    return TrainingConfig(
        mixed_precision_param="bfloat16",
        mixed_precision_reduce="float32",
    )


def _replicated_dtensor(tensor: torch.Tensor, device_type: str) -> DTensor:
    mesh = DeviceMesh(device_type, [0], mesh_dim_names=("tp",))
    return DTensor.from_local(tensor, mesh, (Replicate(),))


def _count_op(gm: torch.fx.GraphModule, target) -> int:
    return sum(
        1
        for node in gm.graph.nodes
        if node.op == "call_function" and node.target is target
    )


class TestApplySimpleFSDPSingleRank(unittest.TestCase):
    """Verify simple_fsdp's MixedPrecisionPolicy actually casts params at NGPU=1."""

    def setUp(self):
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                init_method="tcp://localhost:12358",
                world_size=1,
                rank=0,
            )

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    @patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
    def test_param_cast_to_bf16_at_ngpu_1(self):
        """With ``mixed_precision_param=bfloat16``, the parametrized weight must
        yield bf16 — and a forward must run in bf16 — even when fsdp /
        dp_replicate / ep are all disabled. Without the unconditional
        simple_fsdp wrap, parameters silently stay in fp32 on a single GPU and
        any downstream bf16-only kernel (e.g. MXFP8) breaks.
        """
        model = nn.Linear(8, 8)
        self.assertEqual(model.weight.dtype, torch.float32)

        model = apply_simple_fsdp(
            model,
            parallel_dims=_single_rank_parallel_dims(),
            training=_bf16_mixed_precision_training(),
        )

        # Parametrization replaces ``weight`` access with a bf16 cast via
        # ``redistribute(forward_dtype=...)``.
        self.assertEqual(model.weight.dtype, torch.bfloat16)

        # Underlying storage stays in fp32 (the cast is applied per forward),
        # confirming this is true mixed precision rather than a one-shot
        # downcast that would lose master-weight precision.
        self.assertEqual(model._parameters["weight"].dtype, torch.float32)

        # End-to-end: forward against the parametrized weight produces bf16
        # activations.
        x = torch.randn(2, 8, dtype=torch.bfloat16)
        y = model(x)
        self.assertEqual(y.dtype, torch.bfloat16)


class _GroupedMMExperts(nn.Module):
    def __init__(self, *, num_experts: int, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(num_experts, hidden_dim, dim))
        self.w2 = nn.Parameter(torch.randn(num_experts, dim, hidden_dim))

    def forward(
        self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor
    ) -> torch.Tensor:
        w1 = self.w1.to_local() if isinstance(self.w1, DTensor) else self.w1
        w2 = self.w2.to_local() if isinstance(self.w2, DTensor) else self.w2
        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        h = torch._grouped_mm(
            x.bfloat16(),
            w1.bfloat16().transpose(-2, -1),
            offs=offsets,
        )
        return torch._grouped_mm(
            h,
            w2.bfloat16().transpose(-2, -1),
            offs=offsets,
        ).type_as(x)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(_HAS_TORCHAO, "torchao required")
class TestFSDPHookGraphCaptureSingleRank(unittest.TestCase):
    """Verify torchao FSDP-hook wrappers survive simple_fsdp graph capture."""

    def setUp(self):
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="tcp://localhost:12359",
                world_size=1,
                rank=0,
            )
        torch.manual_seed(0)

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def _require_fp8(self, op_name: str) -> None:
        if torch.cuda.get_device_capability() < (8, 9):
            self.skipTest(f"FP8 {op_name} requires SM89 or newer")

    def _require_mxfp8_grouped_mm(self) -> None:
        try:
            from torchao.prototype.moe_training.mxfp8_grouped_mm import (
                _SM100_KERNELS_AVAILABLE,
            )
        except ImportError:
            _SM100_KERNELS_AVAILABLE = False

        supported = (
            torch.cuda.get_device_capability() >= (10, 0) and _SM100_KERNELS_AVAILABLE
        )
        if not supported:
            self.skipTest("MXFP8 grouped GEMM requires SM100 torchao kernels")

    def _require_mxfp8_scaled_mm(self) -> None:
        if torch.cuda.get_device_capability() < (10, 0):
            self.skipTest("MXFP8 scaled GEMM requires SM100")

    def _grouped_mm_quantization_config(self, quantization: str):
        if quantization == "fp8":
            self._require_fp8("grouped GEMM")
            from torchao.prototype.moe_training.config import Float8TrainingOpConfig

            return Float8TrainingOpConfig()

        if quantization == "mxfp8":
            self._require_mxfp8_grouped_mm()
            from torchao.prototype.moe_training.config import (
                MXFP8TrainingOpConfig,
                MXFP8TrainingRecipe,
            )

            return MXFP8TrainingOpConfig.from_recipe(MXFP8TrainingRecipe.MXFP8_RCEIL)

        raise AssertionError(f"Unknown quantization type: {quantization}")

    def _apply_simple_fsdp_to_quantized_model(
        self,
        model: nn.Module,
        *,
        storage: str,
        dtensor_param_names: tuple[str, ...],
    ) -> nn.Module:
        if storage == "dtensor":
            for param_name in dtensor_param_names:
                param = getattr(model, param_name)
                setattr(
                    model,
                    param_name,
                    nn.Parameter(
                        _replicated_dtensor(param, "cuda"),
                        requires_grad=param.requires_grad,
                    ),
                )
        elif storage != "plain":
            raise AssertionError(f"Unknown storage type: {storage}")

        return apply_simple_fsdp(
            model,
            parallel_dims=_single_rank_parallel_dims(),
            training=_bf16_mixed_precision_training(),
        )

    def _build_quantized_scaled_mm_model(
        self,
        *,
        quantization: str,
        storage: str,
    ) -> nn.Module:
        linear_kwargs = dict(in_features=64, out_features=64, bias=False)
        if quantization == "fp8":
            self._require_fp8("scaled GEMM")

            from torchao.float8 import Float8LinearConfig as TorchAOFloat8LinearConfig

            from torchtitan.components.quantization import Float8Linear

            assert Float8Linear is not None
            model = Float8Linear(
                Float8Linear.Config(
                    **linear_kwargs,
                    _torchao_config=TorchAOFloat8LinearConfig.from_recipe_name(
                        "rowwise"
                    ),
                )
            ).cuda()
        elif quantization == "mxfp8":
            self._require_mxfp8_scaled_mm()

            from torchtitan.components.quantization import MXFP8Linear

            model = MXFP8Linear(
                MXFP8Linear.Config(
                    **linear_kwargs,
                    _recipe_name="mxfp8_rceil",
                )
            ).cuda()
        else:
            raise AssertionError(f"Unknown quantization type: {quantization}")

        return self._apply_simple_fsdp_to_quantized_model(
            model,
            storage=storage,
            dtensor_param_names=("weight",),
        )

    def _build_quantized_grouped_mm_model(
        self,
        *,
        quantization: str,
        storage: str,
    ) -> nn.Module:
        from torchao.quantization.quant_api import quantize_

        model = _GroupedMMExperts(num_experts=2, dim=64, hidden_dim=64).cuda()
        quantize_(
            model,
            config=self._grouped_mm_quantization_config(quantization),
            filter_fn=lambda mod, _fqn: isinstance(mod, _GroupedMMExperts),
        )

        return self._apply_simple_fsdp_to_quantized_model(
            model,
            storage=storage,
            dtensor_param_names=("w1", "w2"),
        )

    @parametrize("quantization", ("fp8", "mxfp8"))
    @parametrize("storage", ("plain", "dtensor"))
    def test_scaled_mm_in_graph(self, quantization: str, storage: str):
        """Graph capture keeps quantized linear weights on scaled GEMMs."""

        model = self._build_quantized_scaled_mm_model(
            quantization=quantization,
            storage=storage,
        )
        x = torch.randn(
            32,
            64,
            dtype=torch.bfloat16,
            device="cuda",
        )
        if storage == "dtensor":
            x = _replicated_dtensor(x, "cuda")

        def forward_fn(model, x):
            out = model(x)
            return out.to_local() if isinstance(out, DTensor) else out

        traced = trace_train_step(forward_fn)(model, x)

        num_scaled_mm = _count_op(traced.gm, torch.ops.aten._scaled_mm.default)
        num_mm = _count_op(traced.gm, torch.ops.aten.mm.default)

        self.assertGreaterEqual(
            num_scaled_mm,
            1,
            "Expected aten._scaled_mm in the captured graph. "
            "simple_fsdp likely bypassed the quantized linear compute path.",
        )
        self.assertEqual(
            num_mm,
            0,
            f"Expected no plain aten.mm in the captured graph. Found {num_mm}.",
        )

    @parametrize("quantization", ("fp8", "mxfp8"))
    @parametrize("storage", ("plain", "dtensor"))
    def test_scaled_grouped_mm_in_graph(self, quantization: str, storage: str):
        """Graph capture keeps quantized MoE weights on scaled grouped GEMMs."""

        model = self._build_quantized_grouped_mm_model(
            quantization=quantization,
            storage=storage,
        )
        x = torch.randn(
            32,
            64,
            dtype=torch.bfloat16,
            device="cuda",
        )
        num_tokens_per_expert = torch.tensor(
            [16, 16],
            dtype=torch.int64,
            device="cuda",
        )

        def forward_fn(model, x, num_tokens_per_expert):
            return model(x, num_tokens_per_expert)

        traced = trace_train_step(forward_fn)(model, x, num_tokens_per_expert)

        num_scaled_grouped_mm = _count_op(
            traced.gm, torch.ops.aten._scaled_grouped_mm.default
        )
        num_grouped_mm = _count_op(traced.gm, torch.ops.aten._grouped_mm.default)

        self.assertGreaterEqual(
            num_scaled_grouped_mm,
            2,
            "Expected aten._scaled_grouped_mm in the captured graph. "
            "simple_fsdp likely stripped the torchao training wrapper "
            "before grouped GEMM.",
        )
        self.assertEqual(
            num_grouped_mm,
            0,
            "Expected no plain aten._grouped_mm in the captured graph. "
            f"Found {num_grouped_mm}.",
        )


instantiate_parametrized_tests(TestFSDPHookGraphCaptureSingleRank)

if __name__ == "__main__":
    unittest.main()
