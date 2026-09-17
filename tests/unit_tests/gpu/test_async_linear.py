# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The async tensor-parallel linear backend, at the module and config level.

Three layers, cheapest first. If you are adding a new fused module, this file is
the template:

1. ``TestAsyncTensorParallelConfig`` -- no devices. Does the model transform
   select the async configs and preserve their shapes?
2. ``TestAsyncTensorParallelSharding`` -- a 2-rank gloo mesh. Do those contracts
   survive a real ``parallelize``? Still no CUDA: nothing here runs the fused ops.
3. Numerics for the underlying primitives live in ``test_distributed_linear.py`` (2
   GPUs), and an integration entry in ``tests/integration_tests/h100.py`` runs a
   real training step end to end; see ``dist_gemm`` there.

Both classes here run in CI. Note there is no GPU unit-test job, so anything
CUDA-guarded is developer-run only.
"""

import unittest
from unittest.mock import patch

import torch
from torch.distributed.device_mesh import init_device_mesh

from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.config.transform import (
    AsyncTensorParallelTransform,
    LinearLoRAHandler,
    LoRATransform,
    transform_model_config_,
)
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.common.async_linear import (
    AsyncColumnParallelLinear,
    AsyncRowParallelLinear,
)
from torchtitan.models.common.attention import QKVLinear
from torchtitan.models.common.decoder_sharding import (
    dense_sequence_parallel_placement,
    set_dense_ffn_sharding,
    set_gqa_attention_sharding,
)
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import (
    ColumnParallelLinear,
    Linear,
    RowParallelLinear,
)

DIM = 256
N_HEADS = 8


class TestAsyncTensorParallelConfig(unittest.TestCase):
    """Config-graph rewriting. No devices involved."""

    @staticmethod
    def _model_config():
        from torchtitan.models.llama3 import model_registry

        return model_registry("debugmodel").model

    def test_default_uses_sync_tp_projections(self):
        for layer in self._model_config().layers:
            self.assertIs(type(layer.attention.qkv_linear), QKVLinear.Config)
            self.assertIsInstance(
                layer.attention.qkv_linear.wqkv, ColumnParallelLinear.Config
            )
            self.assertIsInstance(layer.attention.wo, RowParallelLinear.Config)
            self.assertIsInstance(layer.feed_forward.w13, ColumnParallelLinear.Config)
            self.assertIsInstance(layer.feed_forward.w2, RowParallelLinear.Config)

    def test_transform_selects_async_linears(self):
        model = transform_model_config_(
            self._model_config(),
            [AsyncTensorParallelTransform()],
        )
        for layer in model.layers:
            self.assertIs(type(layer.attention.qkv_linear), QKVLinear.Config)
            self.assertIsInstance(
                layer.attention.qkv_linear.wqkv, AsyncColumnParallelLinear.Config
            )
            self.assertIsInstance(layer.attention.wo, AsyncRowParallelLinear.Config)
            self.assertIsInstance(
                layer.feed_forward.w13, AsyncColumnParallelLinear.Config
            )
            self.assertIsInstance(layer.feed_forward.w2, AsyncRowParallelLinear.Config)

    def test_stock_parameter_shapes_survive(self):
        """Fused modules keep the stock layouts, or checkpoints stop loading."""
        stock = self._model_config().layers[0].attention
        async_model = transform_model_config_(
            self._model_config(),
            [AsyncTensorParallelTransform()],
        )
        fused = async_model.layers[0].attention
        self.assertEqual(
            fused.qkv_linear.wqkv.in_features, stock.qkv_linear.wqkv.in_features
        )
        self.assertEqual(
            fused.qkv_linear.wqkv.out_features, stock.qkv_linear.wqkv.out_features
        )
        self.assertEqual(fused.wo.in_features, stock.wo.in_features)
        self.assertEqual(fused.wo.out_features, stock.wo.out_features)

    def test_sequence_parallel_disabled_is_rejected(self):
        """The fused GEMMs *are* the SP collectives, so SP off has nothing to fuse
        and wo would reduce-scatter where it must all-reduce."""
        model = transform_model_config_(
            self._model_config(),
            [AsyncTensorParallelTransform()],
        )
        attn = model.layers[0].attention
        with self.assertRaisesRegex(ValueError, "enable_sequence_parallel"):
            set_gqa_attention_sharding(attn, enable_sp=False)

    def test_sharding_setup_declares_common_communication_contracts(self):
        """Async attention and FFN implementations own their collectives."""
        stock_layer = self._model_config().layers[0]
        async_model = transform_model_config_(
            self._model_config(),
            [AsyncTensorParallelTransform()],
        )
        async_layer = async_model.layers[0]
        set_gqa_attention_sharding(stock_layer.attention, enable_sp=True)
        set_gqa_attention_sharding(async_layer.attention, enable_sp=True)
        set_dense_ffn_sharding(
            async_layer.feed_forward,
            attn_x_layout=dense_sequence_parallel_placement(),
            enable_sp=True,
        )

        self.assertIsNone(stock_layer.attention.sharding_config.in_dst_shardings)
        self.assertIsNone(stock_layer.attention.sharding_config.out_dst_shardings)
        self.assertIsNone(async_layer.attention.sharding_config.in_dst_shardings)
        self.assertIsNone(async_layer.attention.sharding_config.out_dst_shardings)

        self.assertIsNone(stock_layer.attention.qkv_linear.sharding_config)
        self.assertIsNone(async_layer.attention.qkv_linear.sharding_config)
        self.assertIsNotNone(
            async_layer.attention.qkv_linear.wqkv.sharding_config.in_src_shardings
        )
        self.assertIsNone(
            async_layer.attention.qkv_linear.wqkv.sharding_config.in_dst_shardings
        )
        self.assertIsNotNone(stock_layer.attention.wo.sharding_config.out_src_shardings)
        self.assertIsNone(stock_layer.attention.wo.sharding_config.out_dst_shardings)
        self.assertIsNotNone(async_layer.attention.wo.sharding_config.out_src_shardings)
        self.assertIsNone(async_layer.attention.wo.sharding_config.out_dst_shardings)
        self.assertIn(
            "weight", async_layer.attention.wo.sharding_config.state_shardings
        )

        self.assertIsNone(async_layer.feed_forward.sharding_config.in_dst_shardings)
        self.assertIsNotNone(async_layer.feed_forward.sharding_config.out_src_shardings)
        self.assertIsNone(async_layer.feed_forward.w13.sharding_config.in_dst_shardings)
        self.assertIsNotNone(
            async_layer.feed_forward.w2.sharding_config.out_src_shardings
        )
        self.assertIsNone(async_layer.feed_forward.w2.sharding_config.out_dst_shardings)

    def test_projection_boundaries_survive_lora_config_wrappers(self):
        """The transformed FFN boundary encloses LoRA projection work."""
        model = transform_model_config_(
            self._model_config(),
            [
                LoRATransform(handlers=(LinearLoRAHandler(),)),
            ],
        )
        layer = model.layers[0]
        set_gqa_attention_sharding(layer.attention, enable_sp=True)
        set_dense_ffn_sharding(
            layer.feed_forward,
            attn_x_layout=dense_sequence_parallel_placement(),
            enable_sp=True,
        )

        self.assertIsNone(layer.attention.sharding_config.in_dst_shardings)
        self.assertIsNone(layer.attention.qkv_linear.sharding_config)
        self.assertIsNone(
            layer.attention.qkv_linear.wqkv.sharding_config.in_dst_shardings
        )
        self.assertIsNone(layer.feed_forward.sharding_config.in_dst_shardings)
        self.assertIsNone(layer.feed_forward.w13.sharding_config.in_dst_shardings)

    def test_qkv_converter_is_rejected(self):
        """Async QKV does not support a converter-defined projection."""
        model = self._model_config()
        qkv = model.layers[0].attention.qkv_linear
        qkv.wqkv = LoRATransform(
            handlers=(LinearLoRAHandler(),),
            rank=2,
            alpha=4,
        ).transform(qkv.wqkv)
        with self.assertRaisesRegex(ValueError, "converted QKV projections"):
            transform_model_config_(model, [AsyncTensorParallelTransform()])


class TestAsyncTensorParallelSharding(DTensorTestBase):
    """The declared contracts, as they survive a real ``parallelize``.

    Contracts only -- nothing here runs the fused ops, so it needs no CUDA and
    does run in CI on a gloo mesh. Anything that actually calls symmetric memory
    belongs in a CUDA-guarded class.
    """

    @property
    def world_size(self) -> int:
        return 2

    def _parallel_dims(self) -> ParallelDims:
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=self.world_size,
            pp=1,
            ep=1,
            world_size=self.world_size,
        )
        with patch(
            "torchtitan.distributed.parallel_dims.device_type", self.device_type
        ):
            parallel_dims.build_mesh()
        return parallel_dims

    @with_comms
    def test_parallelize_keeps_async_collectives_in_projection_leaves(self):
        """Async linears remove the redundant synchronous redistributions."""
        from torchtitan.models.llama3.config_registry import llama3_debugmodel_dist_gemm

        parallel_dims = self._parallel_dims()
        attn_cfg = (
            llama3_debugmodel_dist_gemm(seq_len=2048)
            .model_spec.model.layers[0]
            .attention
        )
        set_gqa_attention_sharding(attn_cfg, enable_sp=True)
        attn = attn_cfg.build().to(self.device_type)
        attn.parallelize(parallel_dims)

        self.assertIsNone(attn._sharding_config.in_dst_shardings)
        self.assertIsNone(attn._sharding_config.out_dst_shardings)
        self.assertIsNone(attn.qkv_linear._sharding_config)
        self.assertIsNone(attn.qkv_linear.wqkv._sharding_config.in_dst_shardings)
        self.assertIsNotNone(attn.wo._sharding_config.out_src_shardings)
        self.assertIsNone(attn.wo._sharding_config.out_dst_shardings)
        self.assertIn("weight", attn.wo._sharding_config.state_shardings)


@unittest.skipUnless(
    torch.cuda.device_count() >= 2, "symmetric memory requires two CUDA devices"
)
class TestAsyncQKVNumerics(DTensorTestBase):
    """The async inner projection must preserve fused QKV behavior."""

    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_matches_standard_qkv_linear(self):
        from torchtitan.distributed.spmd_types import set_current_spmd_mesh

        R = self.world_size
        device = self.device_type
        dim, head_dim, num_heads, num_kv_heads = 64, 16, 8, 4
        num_tokens = 8 * R
        out_features = (num_heads + 2 * num_kv_heads) * head_dim

        stock_config = QKVLinear.Config(
            head_dim=head_dim,
            n_heads=num_heads,
            n_kv_heads=num_kv_heads,
            wqkv=Linear.Config(in_features=dim, out_features=out_features),
        )
        async_config = QKVLinear.Config(
            head_dim=head_dim,
            n_heads=num_heads,
            n_kv_heads=num_kv_heads,
            wqkv=AsyncColumnParallelLinear.Config(
                in_features=dim,
                out_features=out_features,
            ),
        )
        stock = stock_config.build().to(device=device, dtype=torch.bfloat16)
        async_qkv = async_config.build().to(device=device, dtype=torch.bfloat16)

        torch.manual_seed(0)
        with torch.no_grad():
            stock.wqkv.weight.copy_(torch.randn_like(stock.wqkv.weight))
            async_qkv.wqkv.weight = torch.nn.Parameter(
                stock.wqkv.weight.chunk(R, 0)[self.rank].contiguous()
            )

        x_TD = torch.randn(
            num_tokens,
            dim,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        expected = stock(x_TD)
        torch.stack([output.sum() for output in expected]).sum().backward()

        x_local_TD = x_TD.detach().chunk(R, 0)[self.rank].contiguous().requires_grad_()
        mesh = init_device_mesh(device, (R,), mesh_dim_names=("tp",))
        with set_current_spmd_mesh(mesh):
            actual = async_qkv(x_local_TD)
            torch.stack([output.sum() for output in actual]).sum().backward()

        for actual_projection, expected_projection in zip(actual, expected):
            torch.testing.assert_close(
                actual_projection,
                expected_projection.chunk(R, -2)[self.rank],
                atol=2e-2,
                rtol=2e-2,
            )
        torch.testing.assert_close(
            x_local_TD.grad,
            x_TD.grad.chunk(R, 0)[self.rank],
            atol=2e-2,
            rtol=2e-2,
        )
        torch.testing.assert_close(
            async_qkv.wqkv.weight.grad,
            stock.wqkv.weight.grad.chunk(R, 0)[self.rank],
            atol=2e-2,
            rtol=2e-2,
        )


@unittest.skipUnless(
    torch.cuda.device_count() >= 2, "symmetric memory requires two CUDA devices"
)
class TestAsyncFeedForwardNumerics(DTensorTestBase):
    """The async FFN projections must match the standard FFN under TP+SP.

    The test manually shards w13 colwise and w2 rowwise. DistGEMM must
    all-gather the sequence-sharded input before w13 and reduce-scatter w2's
    output. The standard forward can execute on these local shards, but it would
    produce an incorrect local partial result.
    """

    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_matches_standard_feed_forward(self):
        from torchtitan.distributed.spmd_types import set_current_spmd_mesh
        from torchtitan.models.common.config_utils import make_ffn_config

        R = self.world_size
        dev = self.device_type
        dim, hidden, num_tokens = 64, 128, 16 * R
        init = {"weight": torch.nn.init.zeros_}

        torch.manual_seed(0)
        standard = (
            make_ffn_config(
                dim=dim, hidden_dim=hidden, w1_param_init=init, w2w3_param_init=init
            )
            .build()
            .to(dev)
        )
        base_async_config = make_ffn_config(
            dim=dim,
            hidden_dim=hidden,
            w1_param_init=init,
            w2w3_param_init=init,
        )
        async_config = AsyncTensorParallelTransform().transform(base_async_config)
        dist_gemm = async_config.build().to(dev)

        with torch.no_grad():
            for m in (standard, dist_gemm):
                for w in (m.w13.weight, m.w2.weight):
                    torch.manual_seed(hash(tuple(w.shape)) % 2**31)
                    w.copy_(torch.randn_like(w) * 0.1)

        x = torch.randn(num_tokens, dim, device=dev)
        ref = standard(x)

        # Shard the dist-GEMM module's weights: w13 colwise, w2 rowwise.
        with torch.no_grad():
            dist_gemm.w13.weight = torch.nn.Parameter(
                standard.w13.weight.chunk(R, 0)[self.rank].contiguous()
            )
            dist_gemm.w2.weight = torch.nn.Parameter(
                standard.w2.weight.chunk(R, 1)[self.rank].contiguous()
            )

        # needs mesh_dim_names, and a "tp" axis for _tp_group_from_context
        mesh = init_device_mesh(self.device_type, (R,), mesh_dim_names=("tp",))
        with set_current_spmd_mesh(mesh):
            x_shard = x.chunk(R, 0)[self.rank].contiguous()
            out_shard = dist_gemm(x_shard)

        # DistGEMM returns this rank's sequence shard of the full result.
        torch.testing.assert_close(
            out_shard, ref.chunk(R, 0)[self.rank], atol=2e-3, rtol=2e-3
        )


@unittest.skipUnless(
    torch.cuda.device_count() >= 2, "symmetric memory requires two CUDA devices"
)
class TestAsyncFusedSwiGLUNumerics(DTensorTestBase):
    """The Triton-activation FFN with TP overlap must match native SwiGLU.

    Same communication contract as TestAsyncFeedForwardNumerics, with the
    Triton SiLU-and-multiply override composed on top. Lives here rather than in
    test_fused_swiglu.py, which is CPU-only by design.
    """

    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_matches_native_feed_forward(self):
        from torchtitan.distributed.spmd_types import set_current_spmd_mesh
        from torchtitan.models.common.config_utils import make_ffn_config
        from torchtitan.overrides.fused_swiglu import fused_swiglu

        R = self.world_size
        dev = self.device_type
        dim, hidden, num_tokens = 64, 128, 16 * R
        init = {"weight": torch.nn.init.zeros_}

        def make():
            return make_ffn_config(
                dim=dim,
                hidden_dim=hidden,
                w1_param_init=init,
                w2w3_param_init=init,
            )

        torch.manual_seed(0)
        native = make().build().to(dev)
        async_config = AsyncTensorParallelTransform().transform(make())
        async_config.activation_fn = fused_swiglu(async_config.activation_fn)
        fused = async_config.build().to(dev)
        self.assertIsInstance(fused, FeedForward)
        self.assertIsInstance(fused.w13, AsyncColumnParallelLinear)
        self.assertIsInstance(fused.w2, AsyncRowParallelLinear)

        with torch.no_grad():
            for w in (native.w13.weight, native.w2.weight):
                torch.manual_seed(hash(tuple(w.shape)) % 2**31)
                w.copy_(torch.randn_like(w) * 0.1)

        x = torch.randn(num_tokens, dim, device=dev)
        ref = native(x)

        # w13.weight is (2 * hidden/R, dim), with this rank's interleaved
        # colwise slice of both halves.
        with torch.no_grad():
            fused.w13.weight = torch.nn.Parameter(
                native.w13.weight.chunk(R, 0)[self.rank].contiguous()
            )
            fused.w2.weight = torch.nn.Parameter(
                native.w2.weight.chunk(R, 1)[self.rank].contiguous()
            )

        mesh = init_device_mesh(self.device_type, (R,), mesh_dim_names=("tp",))
        with set_current_spmd_mesh(mesh):
            out_shard = fused(x.chunk(R, 0)[self.rank].contiguous())

        torch.testing.assert_close(
            out_shard, ref.chunk(R, 0)[self.rank], atol=2e-3, rtol=2e-3
        )


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
