# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The dist-GEMM attention backend, at the module and config level.

Three layers, cheapest first. If you are adding a new fused module, this file is
the template:

1. ``TestDistGemmAttentionConfig`` -- no devices. Does ``tp_gemm_backend`` select the
   fused configs, does the sharding setup declare the right contracts for them,
   and does the runtime config reach them? Catches wiring mistakes in
   milliseconds.
2. ``TestDistGemmAttentionSharding`` -- a 2-rank gloo mesh. Do those contracts
   survive a real ``parallelize``? Still no CUDA: nothing here runs the fused ops.
3. Numerics for the underlying primitives live in ``test_distributed_linear.py`` (2
   GPUs), and an integration entry in ``tests/integration_tests/h100.py`` runs a
   real training step end to end; see ``dist_gemm`` there.

Both classes here run in CI. Note there is no GPU unit-test job, so anything
CUDA-guarded is developer-run only.
"""

import contextlib
import unittest
from unittest.mock import patch

import torch
from torch.distributed.device_mesh import init_device_mesh

from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.utils import get_spmd_backend, set_spmd_backend
from torchtitan.models.common.config_utils import make_gqa_config
from torchtitan.models.common.decoder_sharding import set_gqa_attention_sharding
from torchtitan.models.common.dist_gemm import (
    AllGatherFusedFeedForward,
    AllGatherFusedQKVLinear,
    RowParallelLinear,
)

DIM = 256
N_HEADS = 8


@contextlib.contextmanager
def use_spmd_backend(backend: str):
    """Temporarily select an SPMD backend without leaking test state."""
    previous_backend = get_spmd_backend()
    set_spmd_backend(backend)
    try:
        yield
    finally:
        set_spmd_backend(previous_backend)


class TestDistGemmAttentionConfig(unittest.TestCase):
    """Config-graph rewriting. No devices involved."""

    def test_tp_gemm_backend_selects_the_fused_configs(self):
        """model_registry(tp_gemm_backend="dist_gemm") swaps all three pieces."""
        from torchtitan.models.llama3 import model_registry

        spec = model_registry("debugmodel", tp_gemm_backend="dist_gemm")
        for layer in spec.model.layers:
            attn = layer.attention
            self.assertIsInstance(attn.qkv_linear, AllGatherFusedQKVLinear.Config)
            self.assertIsInstance(attn.wo, RowParallelLinear.Config)

    def test_default_tp_gemm_backend_is_untouched(self):
        """The default must stay stock, or every model silently changes."""
        from torchtitan.models.llama3 import model_registry

        for layer in model_registry("debugmodel").model.layers:
            self.assertNotIsInstance(layer.attention.wo, RowParallelLinear.Config)

    def test_stock_parameter_shapes_survive(self):
        """Fused modules keep the stock layouts, or checkpoints stop loading."""
        from torchtitan.models.llama3 import model_registry

        stock = model_registry("debugmodel").model.layers[0].attention
        fused = (
            model_registry("debugmodel", tp_gemm_backend="dist_gemm")
            .model.layers[0]
            .attention
        )
        self.assertEqual(
            fused.qkv_linear.wqkv.in_features, stock.qkv_linear.wqkv.in_features
        )
        self.assertEqual(
            fused.qkv_linear.wqkv.out_features, stock.qkv_linear.wqkv.out_features
        )
        self.assertEqual(fused.wo.in_features, stock.wo.in_features)
        self.assertEqual(fused.wo.out_features, stock.wo.out_features)

    def test_unfused_qkv_is_rejected(self):
        """The all-gather feeds one wqkv GEMM; separate wq/wk/wv has no schedule."""
        from torchtitan.models.common.attention import FlexAttention
        from torchtitan.models.common.rope import ComplexRoPE

        with self.assertRaisesRegex(ValueError, "requires fuse_qkv=True"):
            make_gqa_config(
                dim=DIM,
                n_heads=N_HEADS,
                wqkv_param_init={},
                wo_param_init={},
                inner_attention=FlexAttention.Config(),
                rope=ComplexRoPE.Config(dim=DIM // N_HEADS, max_seq_len=128),
                fuse_qkv=False,
                tp_gemm_backend="dist_gemm",
            )

    def test_dtensor_backend_is_rejected(self):
        """dist-GEMM is spmd_types-only; the DTensor backends are deprecated."""
        from torchtitan.models.llama3 import model_registry

        attn = model_registry("debugmodel", tp_gemm_backend="dist_gemm")
        attn = attn.model.layers[0].attention
        with use_spmd_backend("partial_dtensor"):
            with self.assertRaisesRegex(
                ValueError, "requires parallelism.spmd_backend"
            ):
                set_gqa_attention_sharding(attn, enable_sp=True)

    def test_sequence_parallel_disabled_is_rejected(self):
        """The fused GEMMs *are* the SP collectives, so SP off has nothing to fuse
        and wo would reduce-scatter where it must all-reduce."""
        from torchtitan.models.llama3 import model_registry

        attn = model_registry("debugmodel", tp_gemm_backend="dist_gemm")
        attn = attn.model.layers[0].attention
        with use_spmd_backend("spmd_types"):
            with self.assertRaisesRegex(ValueError, "enable_sequence_parallel"):
                set_gqa_attention_sharding(attn, enable_sp=False)

    def test_bias_on_w1_w3_is_rejected(self):
        """A bias must fail at config time, not silently fall back to stock.

        The fused all-gather takes no per-weight bias. Accepting the config and
        quietly running the unfused FFN would look exactly like the feature
        working.
        """
        from torchtitan.models.common.linear import Linear

        kw = {"in_features": DIM, "out_features": 4 * DIM}
        with self.assertRaisesRegex(ValueError, "does not support a bias"):
            AllGatherFusedFeedForward.Config(
                w1=Linear.Config(**kw, bias=True),
                w2=Linear.Config(in_features=4 * DIM, out_features=DIM),
                w3=Linear.Config(**kw),
            )

    def test_sharding_setup_declares_the_fused_contracts(self):
        """set_gqa_attention_sharding declares different contracts for dist-GEMM.

        Two differences from the stock block, both because the fused ops own the
        collectives themselves: the block declares no attention-boundary
        all-gather, and wo emits its final Shard(1) rather than a Partial for the
        framework to reduce-scatter. Declaring the stock Partial here would make
        the module fail its own out_src check at runtime.
        """
        from torchtitan.models.llama3 import model_registry

        stock = model_registry("debugmodel").model.layers[0].attention
        fused = (
            model_registry("debugmodel", tp_gemm_backend="dist_gemm")
            .model.layers[0]
            .attention
        )
        with use_spmd_backend("spmd_types"):
            set_gqa_attention_sharding(stock, enable_sp=True)
            set_gqa_attention_sharding(fused, enable_sp=True)

        self.assertIsNotNone(stock.sharding_config)
        self.assertIsNone(fused.sharding_config)

        self.assertIsNotNone(stock.wo.sharding_config.out_src_shardings)
        self.assertIsNone(fused.wo.sharding_config.out_src_shardings)
        self.assertIsNone(fused.wo.sharding_config.out_dst_shardings)
        # the weight sharding still has to be declared, or wo is never sharded
        self.assertIn("weight", fused.wo.sharding_config.state_shardings)


class TestDistGemmAttentionSharding(DTensorTestBase):
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
    def test_parallelize_preserves_the_fused_contracts(self):
        """Nothing downstream of the declaration undoes it.

        The config-level assertions live in
        ``test_sharding_setup_declares_the_fused_contracts``; this pins that they
        survive ``parallelize``, which is where an earlier revision had to patch
        them back because the sharding setup ran last and overwrote them.
        """
        from torchtitan.models.llama3.config_registry import llama3_debugmodel_dist_gemm

        parallel_dims = self._parallel_dims()
        attn_cfg = llama3_debugmodel_dist_gemm().model_spec.model.layers[0].attention
        with use_spmd_backend("spmd_types"):
            set_gqa_attention_sharding(attn_cfg, enable_sp=True)
            attn = attn_cfg.build().to(self.device_type)
            attn.parallelize(parallel_dims)

        self.assertIsNone(attn._sharding_config)
        self.assertIsNone(attn.wo._sharding_config.out_src_shardings)
        self.assertIsNone(attn.wo._sharding_config.out_dst_shardings)
        self.assertIn("weight", attn.wo._sharding_config.state_shardings)


@unittest.skipUnless(torch.cuda.is_available(), "symmetric memory requires CUDA")
class TestFusedFeedForwardNumerics(DTensorTestBase):
    """The fused FFN must match the stock one under TP+SP.

    Proves the fused path actually runs (a silent fallback would still match, so
    the weights are sharded per rank -- the stock module could not consume them)
    and that the sequence-major flatten/unflatten round-trips.
    """

    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_matches_stock_feed_forward(self):
        from torchtitan.distributed.spmd_types import set_current_spmd_mesh
        from torchtitan.models.common.config_utils import make_ffn_config

        R = self.world_size
        dev = self.device_type
        dim, hidden, bsz, seq = 64, 128, 2, 8 * R
        init = {"weight": torch.nn.init.zeros_}

        torch.manual_seed(0)
        stock = (
            make_ffn_config(
                dim=dim, hidden_dim=hidden, w1_param_init=init, w2w3_param_init=init
            )
            .build()
            .to(dev)
        )
        fused = (
            make_ffn_config(
                dim=dim,
                hidden_dim=hidden,
                w1_param_init=init,
                w2w3_param_init=init,
                tp_gemm_backend="dist_gemm",
            )
            .build()
            .to(dev)
        )

        with torch.no_grad():
            for m in (stock, fused):
                for w in (m.w1.weight, m.w2.weight, m.w3.weight):
                    torch.manual_seed(hash(tuple(w.shape)) % 2**31)
                    w.copy_(torch.randn_like(w) * 0.1)

        x = torch.randn(bsz, seq, dim, device=dev)
        ref = stock(x)

        # shard the fused module's weights: w1/w3 colwise, w2 rowwise
        with torch.no_grad():
            fused.w1.weight = torch.nn.Parameter(
                stock.w1.weight.chunk(R, 0)[self.rank].contiguous()
            )
            fused.w3.weight = torch.nn.Parameter(
                stock.w3.weight.chunk(R, 0)[self.rank].contiguous()
            )
            fused.w2.weight = torch.nn.Parameter(
                stock.w2.weight.chunk(R, 1)[self.rank].contiguous()
            )

        # needs mesh_dim_names, and a "tp" axis for _tp_group_from_context
        mesh = init_device_mesh(self.device_type, (R,), mesh_dim_names=("tp",))
        with use_spmd_backend("spmd_types"):
            with set_current_spmd_mesh(mesh):
                x_shard = x.chunk(R, 1)[self.rank].contiguous()
                out_shard = fused(x_shard)

        # fused returns this rank's sequence shard of the full-sequence result
        torch.testing.assert_close(
            out_shard, ref.chunk(R, 1)[self.rank], atol=2e-3, rtol=2e-3
        )


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
