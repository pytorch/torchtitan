# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The async tensor-parallel linear backend, at the module and config level.

Three layers, cheapest first. If you are adding a new fused module, this file is
the template:

1. ``TestAsyncTensorParallelConfig`` -- no devices. Do synchronous and async
   projections share communication contracts, including after LoRA conversion?
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

import spmd_types as spmd
import torch
from spmd_types.checker import typecheck
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
    colwise_config,
    dense_sequence_parallel_placement,
    set_dense_ffn_sharding,
    set_gqa_attention_sharding,
)
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear

DIM = 256
N_HEADS = 8


class TestAsyncTensorParallelConfig(unittest.TestCase):
    """Config-graph rewriting. No devices involved."""

    @staticmethod
    def _model_config():
        from torchtitan.models.llama3 import model_registry

        return model_registry("debugmodel")

    def test_sharding_setup_declares_common_communication_contracts(self):
        """Async attention and FFN implementations own their collectives."""
        stock_layer = self._model_config().layers[0]
        async_model = transform_model_config_(
            self._model_config(),
            [AsyncTensorParallelTransform(enable_sequence_parallel=True)],
        )
        async_layer = async_model.layers[0]
        set_gqa_attention_sharding(stock_layer.attention, enable_sp=True)
        set_gqa_attention_sharding(async_layer.attention, enable_sp=True)
        for layer in (stock_layer, async_layer):
            set_dense_ffn_sharding(
                layer.feed_forward,
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
        self.assertTrue(stock_layer.feed_forward.w13.sharding_config.local_spmd)
        self.assertTrue(async_layer.feed_forward.w13.sharding_config.local_spmd)
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


class TestAsyncTensorParallelSharding(DTensorTestBase):
    """Projection contracts and synchronous boundary behavior.

    Nothing here runs the fused ops, so these tests need no CUDA and run in CI
    on a gloo mesh. Anything that calls symmetric memory belongs in a
    CUDA-guarded class.
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
            enable_sequence_parallel=True,
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
        attn_cfg = llama3_debugmodel_dist_gemm(seq_len=2048).model.layers[0].attention
        set_gqa_attention_sharding(attn_cfg, enable_sp=True)
        attn = attn_cfg.build().to(self.device_type)
        attn._parallelize(parallel_dims)

        self.assertIsNone(attn._sharding_config.in_dst_shardings)
        self.assertIsNone(attn._sharding_config.out_dst_shardings)
        self.assertIsNone(attn.qkv_linear._sharding_config)
        self.assertIsNone(attn.qkv_linear.wqkv._sharding_config.in_dst_shardings)
        self.assertIsNotNone(attn.wo._sharding_config.out_src_shardings)
        self.assertIsNone(attn.wo._sharding_config.out_dst_shardings)
        self.assertIn("weight", attn.wo._sharding_config.state_shardings)

    @with_comms
    def test_w13_tp_shards_the_matrix_row_dimension(self):
        from torchtitan.models.common.config_utils import make_ffn_config

        hidden_dim = 128
        init = {"weight": torch.nn.init.zeros_}
        ffn_config = make_ffn_config(
            dim=DIM,
            hidden_dim=hidden_dim,
            w1_param_init=init,
            w2w3_param_init=init,
        )
        set_dense_ffn_sharding(
            ffn_config,
            attn_x_layout=dense_sequence_parallel_placement(),
            enable_sp=True,
        )
        feed_forward = ffn_config.build().to(self.device_type)
        feed_forward._parallelize(self._parallel_dims())

        self.assertEqual(
            feed_forward.w13.weight.shape,
            (2, hidden_dim // self.world_size, DIM),
        )
        self.assertEqual(
            feed_forward.w2.weight.shape,
            (DIM, hidden_dim // self.world_size),
        )

    @with_comms
    def test_stacked_w13_typechecks(self):
        from torchtitan.distributed.spmd_types import (
            set_current_spmd_mesh,
            set_spmd_meshes,
        )
        from torchtitan.models.common.config_utils import make_ffn_config

        input_layout = dense_sequence_parallel_placement()
        init = {"weight": torch.nn.init.zeros_}
        ffn_config = make_ffn_config(
            dim=DIM,
            hidden_dim=128,
            w1_param_init=init,
            w2w3_param_init=init,
        )
        set_dense_ffn_sharding(
            ffn_config,
            attn_x_layout=input_layout,
            enable_sp=True,
        )
        feed_forward = ffn_config.build().to(self.device_type)
        parallel_dims = self._parallel_dims()
        feed_forward._parallelize(parallel_dims)

        x_local = torch.randn(8, DIM, device=self.device_type, requires_grad=True)
        mesh = parallel_dims.spmd_dense_mesh()
        set_spmd_meshes(
            dense_mesh=mesh,
            sparse_mesh=None,
            dense_sp_enabled=parallel_dims.sp_enabled,
        )
        with set_current_spmd_mesh(mesh), typecheck(local=False):
            spmd.assert_type(x_local, input_layout)
            output = feed_forward(x_local)
            output.sum().backward()

        self.assertEqual(output.shape, x_local.shape)

    @with_comms
    def test_gpt_oss_attention_reshapes_gathered_tokens(self):
        from torchtitan.distributed.spmd_types import (
            set_current_spmd_mesh,
            set_spmd_meshes,
        )
        from torchtitan.models.gpt_oss import model_registry
        from torchtitan.models.gpt_oss.sharding import set_gpt_oss_sharding_config

        class _IdentityRope(torch.nn.Module):
            def forward(self, q, k, positions):
                return q, k

        class _AttentionOutput(torch.nn.Module):
            def forward(self, q, k, v, *, out_transform=None, **kwargs):
                if out_transform is None:
                    return q
                lse = torch.zeros(q.shape[:2], device=q.device, dtype=q.dtype)
                return out_transform(q, lse)

        config = model_registry("debugmodel", seq_len=128, attn_backend="flex")
        set_gpt_oss_sharding_config(config, enable_sp=True, enable_ep=False)
        attention = config.layers[0].attention.build().to(self.device_type)
        attention.rope = _IdentityRope()
        attention.inner_attention = _AttentionOutput()

        parallel_dims = self._parallel_dims()
        attention._parallelize(parallel_dims)

        x_local = torch.randn(8, config.dim, device=self.device_type)
        mesh = parallel_dims.spmd_dense_mesh()
        set_spmd_meshes(
            dense_mesh=mesh,
            sparse_mesh=None,
            dense_sp_enabled=parallel_dims.sp_enabled,
        )
        with set_current_spmd_mesh(mesh):
            output = attention(x_local, None, None)

        self.assertEqual(output.shape, x_local.shape)


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
        input_layout = dense_sequence_parallel_placement()
        async_config.wqkv.sharding_config = colwise_config(input_layout=input_layout)
        stock = stock_config.build().to(device=device, dtype=torch.bfloat16)
        async_qkv = async_config.build().to(device=device, dtype=torch.bfloat16)

        torch.manual_seed(0)
        with torch.no_grad():
            stock.wqkv.weight.copy_(torch.randn_like(stock.wqkv.weight))
            async_qkv.wqkv.weight.copy_(stock.wqkv.weight)

        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=R,
            pp=1,
            ep=1,
            world_size=R,
            enable_sequence_parallel=True,
        )
        with patch("torchtitan.distributed.parallel_dims.device_type", device):
            parallel_dims.build_mesh()
        async_qkv._parallelize(parallel_dims)

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
        mesh = parallel_dims.spmd_dense_mesh()
        with set_current_spmd_mesh(mesh), typecheck(local=False):
            spmd.assert_type(x_local_TD, input_layout)
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

    DistGEMM must all-gather the sequence-sharded input before w13 and
    reduce-scatter w2's output. The forward runs with global SPMD typechecking
    to exercise the local boundary around the stacked w13 projection.
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
        async_config = AsyncTensorParallelTransform(
            enable_sequence_parallel=True
        ).transform(base_async_config)
        input_layout = dense_sequence_parallel_placement()
        set_dense_ffn_sharding(
            async_config,
            attn_x_layout=input_layout,
            enable_sp=True,
        )
        dist_gemm = async_config.build().to(dev)

        with torch.no_grad():
            for m in (standard, dist_gemm):
                for w in (m.w13.weight, m.w2.weight):
                    torch.manual_seed(hash(tuple(w.shape)) % 2**31)
                    w.copy_(torch.randn_like(w) * 0.1)

        x = torch.randn(num_tokens, dim, device=dev, requires_grad=True)
        ref = standard(x)
        ref.sum().backward()

        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=R,
            pp=1,
            ep=1,
            world_size=R,
            enable_sequence_parallel=True,
        )
        with patch("torchtitan.distributed.parallel_dims.device_type", dev):
            parallel_dims.build_mesh()
        dist_gemm._parallelize(parallel_dims)

        mesh = parallel_dims.spmd_dense_mesh()
        x_shard = x.detach().chunk(R, 0)[self.rank].contiguous().requires_grad_()
        with set_current_spmd_mesh(mesh), typecheck(local=False):
            spmd.assert_type(x_shard, input_layout)
            out_shard = dist_gemm(x_shard)
            out_shard.sum().backward()

        # DistGEMM returns this rank's sequence shard of the full result.
        torch.testing.assert_close(
            out_shard, ref.chunk(R, 0)[self.rank], atol=2e-3, rtol=2e-3
        )
        torch.testing.assert_close(
            x_shard.grad, x.grad.chunk(R, 0)[self.rank], atol=2e-3, rtol=2e-3
        )
        torch.testing.assert_close(
            dist_gemm.w13.weight.grad,
            standard.w13.weight.grad.chunk(R, 1)[self.rank],
            atol=2e-3,
            rtol=2e-3,
        )
        torch.testing.assert_close(
            dist_gemm.w2.weight.grad,
            standard.w2.weight.grad.chunk(R, 1)[self.rank],
            atol=2e-3,
            rtol=2e-3,
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
        async_config = AsyncTensorParallelTransform(
            enable_sequence_parallel=True
        ).transform(make())
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

        # w13.weight is (2, hidden/R, dim), with this rank's colwise slice of
        # both projections.
        with torch.no_grad():
            fused.w13.weight = torch.nn.Parameter(
                native.w13.weight.chunk(R, 1)[self.rank].contiguous()
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
