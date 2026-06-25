# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the NVFP4 sequence-parallel block overrides.

Covers override targeting (parent attention/FFN blocks convert their child
Linears; the LM head stays stock), the ShardingConfig rewrites (parent keeps the
SP input; children gain NVFP4 buffer layouts and the rowwise out_dst=None
contract), the relocated parallelize()-time validation (local GEMM dim divisible
by 128; SP required under TP), the dual checkpoint contract (native state_dict
keeps the NVFP4 buffers; the HF export boundary strips them), and forward/backward
numerics (Blackwell).
"""

import unittest
from dataclasses import dataclass
from types import SimpleNamespace

import spmd_types as spmd
import torch
from torch.distributed.tensor import distribute_tensor, DTensor, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.config import apply_overrides, Configurable, OverrideConfig
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.common import Linear
from torchtitan.models.common.decoder_sharding import (
    colwise_config,
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    rowwise_config,
    set_dense_ffn_sharding,
)
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.llama3 import llama3_configs
from torchtitan.models.llama3.model import Llama3Model
from torchtitan.models.llama3.sharding import set_llama3_sharding_config
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
from torchtitan.overrides.nvfp4_linear import (
    _infer_tp_style,
    _is_seq_parallel,
    _nvfp4_colwise_sp,
    _nvfp4_rowwise_sp,
    _to_nvfp4_colwise,
    _to_nvfp4_rowwise,
    nvfp4_feed_forward,
    NVFP4Linear,
)

_MODULE = "torchtitan.overrides.nvfp4_linear"
_DIM = 512
_HIDDEN = 1024  # _DIM, _HIDDEN divisible by 128 * tp for tp <= 4


def _blackwell() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


def _blackwell_tp2() -> bool:
    return torch.cuda.device_count() >= 2 and all(
        torch.cuda.get_device_capability(i)[0] >= 10 for i in range(2)
    )


def _ffn_config(*, enable_sp: bool) -> FeedForward.Config:
    cfg = FeedForward.Config(
        w1=Linear.Config(in_features=_DIM, out_features=_HIDDEN),
        w2=Linear.Config(in_features=_HIDDEN, out_features=_DIM),
        w3=Linear.Config(in_features=_DIM, out_features=_HIDDEN),
    )
    attn_x_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I)
    )
    set_dense_ffn_sharding(cfg, attn_x_layout=attn_x_layout, enable_sp=enable_sp)
    return cfg


def _debugmodel_with_sharding():
    """Llama3 debugmodel config with sharding_config populated (as the trainer
    does in update_from_config before applying overrides)."""
    model_cfg = llama3_configs["debugmodel"](attn_backend="flex")
    set_llama3_sharding_config(model_cfg, enable_sp=True)
    return model_cfg


class _Root(Configurable):
    """Wrapper so the model config is a nested node and FQNs get a prefix."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        model: Llama3Model.Config

    def __init__(self, config: Config):
        self.config = config


class _GatedSharedExperts(FeedForward):
    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        gate: Linear.Config


class _MoeSlot(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        shared_experts: FeedForward.Config


class _LayerWithMoe(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        feed_forward: FeedForward.Config
        moe: _MoeSlot.Config


class _RootWithMoe(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        layers: list[_LayerWithMoe.Config]


class TestNVFP4Targeting(unittest.TestCase):
    def test_replaces_in_layer_linears_not_lm_head(self):
        """Parent block overrides convert child Linears; the LM head stays stock."""
        root = _Root.Config(model=_debugmodel_with_sharding())
        apply_overrides(OverrideConfig(imports=[_MODULE]), root)

        model = root.model
        layer0 = model.layers[0]
        for name in ("w1", "w2", "w3"):
            self.assertIsInstance(
                getattr(layer0.feed_forward, name), NVFP4Linear.Config, name
            )
        # debugmodel fuses QKV.
        self.assertIsInstance(layer0.attention.qkv_linear.wqkv, NVFP4Linear.Config)
        self.assertIsInstance(layer0.attention.wo, NVFP4Linear.Config)
        # The LM head stays stock (not a FeedForward / GQAttention child).
        self.assertIs(type(model.lm_head), Linear.Config)

    def test_skips_moe_shared_experts(self):
        """NVFP4 FFN override only targets decoder dense feed_forward blocks."""
        shared = _GatedSharedExperts.Config(
            w1=Linear.Config(in_features=_DIM, out_features=_HIDDEN),
            w2=Linear.Config(in_features=_HIDDEN, out_features=_DIM),
            w3=Linear.Config(in_features=_DIM, out_features=_HIDDEN),
            gate=Linear.Config(in_features=_DIM, out_features=1),
        )
        root = _RootWithMoe.Config(
            layers=[
                _LayerWithMoe.Config(
                    feed_forward=_ffn_config(enable_sp=True),
                    moe=_MoeSlot.Config(shared_experts=shared),
                )
            ]
        )

        apply_overrides(OverrideConfig(imports=[_MODULE]), root)

        layer = root.layers[0]
        self.assertIsInstance(layer.feed_forward.w1, NVFP4Linear.Config)
        self.assertIs(layer.moe.shared_experts, shared)
        self.assertIsInstance(layer.moe.shared_experts.gate, Linear.Config)


class TestNVFP4ShardingRemap(unittest.TestCase):
    def test_ffn_keeps_sequence_parallel_input(self):
        """Parent FFN must NOT bf16-gather x to Replicate before NVFP4 colwise."""
        cfg = _ffn_config(enable_sp=True)
        # Stock sharding gathers to Replicate (the bf16 all-gather we avoid).
        self.assertFalse(_is_seq_parallel(cfg.sharding_config.in_dst_shardings["x"]))

        out = nvfp4_feed_forward(cfg)
        sc = out.sharding_config
        self.assertTrue(_is_seq_parallel(sc.in_dst_shardings["x"]))
        self.assertIs(sc.in_dst_shardings["x"], sc.in_src_shardings["x"])

    def test_ffn_children_converted(self):
        out = nvfp4_feed_forward(_ffn_config(enable_sp=True))
        self.assertEqual(out.w1.tensor_parallel_style, "colwise")
        self.assertEqual(out.w3.tensor_parallel_style, "colwise")
        self.assertEqual(out.w2.tensor_parallel_style, "rowwise")
        for child in (out.w1, out.w2, out.w3):
            self.assertIsInstance(child, NVFP4Linear.Config)
            self.assertTrue(child.expects_sequence_parallel)

    def test_buffer_state_shardings_declared(self):
        """_distribute_states requires a placement entry for every direct buffer."""
        out = nvfp4_feed_forward(_ffn_config(enable_sp=True))
        for child in (out.w1, out.w2, out.w3):
            state = child.sharding_config.state_shardings
            self.assertIs(state["_sr_seed"], child.sr_seed_state_sharding)
            self.assertIs(
                state["_rht_sign_vector"], child.rht_sign_vector_state_sharding
            )
            self.assertIn("weight", state)  # inherited

    def test_buffer_state_shardings_come_from_nvfp4_config(self):
        """NVFP4Linear.Config owns runtime-buffer checkpoint placements."""
        sr_state = dense_param_placement(tp=spmd.I)
        rht_state = dense_param_placement(tp=spmd.R)
        src = NVFP4Linear.Config(
            in_features=_DIM,
            out_features=_HIDDEN,
            sr_seed_state_sharding=sr_state,
            rht_sign_vector_state_sharding=rht_state,
        )
        src.sharding_config = colwise_config()

        out = _to_nvfp4_colwise(src, sequence_parallel=True)
        state = out.sharding_config.state_shardings
        self.assertIs(out.sr_seed_state_sharding, sr_state)
        self.assertIs(out.rht_sign_vector_state_sharding, rht_state)
        self.assertIs(state["_sr_seed"], sr_state)
        self.assertIs(state["_rht_sign_vector"], rht_state)

    def test_non_sp_block_marks_children_non_sp(self):
        """When the block input is not SP, children carry expects_sequence_parallel
        =False so parallelize() can reject TP-without-SP."""
        out = nvfp4_feed_forward(_ffn_config(enable_sp=False))
        self.assertFalse(out.w1.expects_sequence_parallel)
        self.assertFalse(out.w2.expects_sequence_parallel)

    def test_rowwise_output_contract(self):
        """Rowwise reduce-scatters internally: out_src=SP, out_dst=None."""
        w2 = Linear.Config(in_features=_HIDDEN, out_features=_DIM)
        w2.sharding_config = rowwise_config(output_sp=True)
        nvfp4_w2 = _to_nvfp4_rowwise(w2, sequence_parallel=True)
        sc = nvfp4_w2.sharding_config
        self.assertIsNone(sc.out_dst_shardings)
        self.assertTrue(_is_seq_parallel(sc.out_src_shardings))

    def test_rowwise_non_sp_output_honors_contract(self):
        """Non-SP rowwise output: out_src is the reduce-scattered SP shard, but
        out_dst keeps the inherited (non-SP) contract so TorchTitan redistributes
        instead of the old hard-coded out_dst=None (which would emit the SP
        shard where the model declared Replicate)."""
        w2 = Linear.Config(in_features=_HIDDEN, out_features=_DIM)
        w2.sharding_config = rowwise_config(output_sp=False)
        inherited_out_dst = w2.sharding_config.out_dst_shardings
        self.assertFalse(_is_seq_parallel(inherited_out_dst))  # sanity: non-SP

        sc = _to_nvfp4_rowwise(w2, sequence_parallel=True).sharding_config
        self.assertTrue(_is_seq_parallel(sc.out_src_shardings))
        self.assertIs(sc.out_dst_shardings, inherited_out_dst)

    def test_colwise_output_contract_preserved(self):
        """Colwise keeps stock out_src (feature shard), out_dst stays None."""
        out = nvfp4_feed_forward(_ffn_config(enable_sp=True))
        sc = out.w1.sharding_config
        self.assertFalse(_is_seq_parallel(sc.out_src_shardings))
        self.assertIsNone(sc.out_dst_shardings)

    def test_infer_tp_style(self):
        self.assertEqual(_infer_tp_style(colwise_config()), "colwise")
        self.assertEqual(_infer_tp_style(rowwise_config(output_sp=True)), "rowwise")
        self.assertIsNone(_infer_tp_style(None))

    def test_missing_sharding_config_raises(self):
        """Applied off the Trainer path (before update_from_config populates
        sharding), the factory fails cleanly instead of an opaque AttributeError."""
        cfg = FeedForward.Config(
            w1=Linear.Config(in_features=_DIM, out_features=_HIDDEN),
            w2=Linear.Config(in_features=_HIDDEN, out_features=_DIM),
            w3=Linear.Config(in_features=_DIM, out_features=_HIDDEN),
        )
        self.assertIsNone(cfg.sharding_config)
        with self.assertRaisesRegex(ValueError, "sharding_config"):
            nvfp4_feed_forward(cfg)


def _stub_parallel_dims(
    *, tp_enabled: bool, spmd_backend: str = "default"
) -> SimpleNamespace:
    return SimpleNamespace(tp_enabled=tp_enabled, spmd_backend=spmd_backend)


class TestNVFP4Validate(unittest.TestCase):
    """The divisibility / SP gates relocated from the factory to parallelize()."""

    def _linear(self, in_f, out_f, *, style, tp, sp) -> NVFP4Linear:
        module = NVFP4Linear.Config(in_features=in_f, out_features=out_f).build()
        module.tensor_parallel_style = style
        module.world_size = tp
        module.expects_sequence_parallel = sp
        return module

    def test_divisible_dims_ok(self):
        m = self._linear(_DIM, _HIDDEN, style="colwise", tp=4, sp=True)
        m._validate(_stub_parallel_dims(tp_enabled=True))  # no raise

    def test_indivisible_local_dim_raises(self):
        # colwise: out // tp = 300 // 4 = 75, not a multiple of 128.
        m = self._linear(_DIM, 300, style="colwise", tp=4, sp=True)
        with self.assertRaisesRegex(ValueError, "divisible by 128"):
            m._validate(_stub_parallel_dims(tp_enabled=True))

    def test_uneven_colwise_tp_shard_raises(self):
        m = self._linear(_DIM, 513, style="colwise", tp=4, sp=True)
        with self.assertRaisesRegex(ValueError, "out_features divisible by TP"):
            m._validate(_stub_parallel_dims(tp_enabled=True))

    def test_uneven_rowwise_tp_shard_raises(self):
        m = self._linear(513, _DIM, style="rowwise", tp=4, sp=True)
        with self.assertRaisesRegex(ValueError, "in_features divisible by TP"):
            m._validate(_stub_parallel_dims(tp_enabled=True))

    def test_tp_without_sp_raises(self):
        m = self._linear(_DIM, _HIDDEN, style="colwise", tp=4, sp=False)
        with self.assertRaisesRegex(ValueError, "sequence parallelism"):
            m._validate(_stub_parallel_dims(tp_enabled=True))

    def test_spmd_types_tp_raises(self):
        m = self._linear(_DIM, _HIDDEN, style="colwise", tp=4, sp=True)
        with self.assertRaisesRegex(ValueError, "spmd_types"):
            m._validate(_stub_parallel_dims(tp_enabled=True, spmd_backend="spmd_types"))

    def test_non_tp_skips_sp_check(self):
        # tp disabled (single GPU / FSDP-only): SP irrelevant, dims checked at tp=1.
        m = self._linear(_DIM, _HIDDEN, style=None, tp=1, sp=False)
        m._validate(_stub_parallel_dims(tp_enabled=False))  # no raise


class TestNVFP4RuntimeShape(unittest.TestCase):
    def test_colwise_rejects_short_flattened_local_m(self):
        x = torch.empty(1, 64, _DIM, dtype=torch.bfloat16)
        w = torch.empty(_HIDDEN // 4, _DIM, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "flattened local M"):
            _nvfp4_colwise_sp(x, w, None, None, (), None, 4)

    def test_rowwise_rejects_unaligned_reduce_scatter_split(self):
        x = torch.empty(1, 512, _HIDDEN // 8, dtype=torch.bfloat16)
        w = torch.empty(_DIM, _HIDDEN // 8, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "reduce-scattered local M"):
            _nvfp4_rowwise_sp(x, w, None, None, (), None, 8)


class TestNVFP4ModuleLayout(unittest.TestCase):
    def test_built_module_params_and_buffers(self):
        """Built module exposes stock weight + the two NVFP4 buffers, nothing else."""
        module = NVFP4Linear.Config(in_features=_DIM, out_features=_HIDDEN).build()
        params = {name for name, _ in module.named_parameters()}
        self.assertEqual(params, {"weight"})
        module.init_states()
        buffers = dict(module.named_buffers())
        self.assertEqual(set(buffers), {"_sr_seed", "_rht_sign_vector"})
        self.assertEqual(buffers["_sr_seed"].dtype, torch.int64)
        self.assertEqual(tuple(buffers["_rht_sign_vector"].shape), (16,))


class TestNVFP4Checkpoint(unittest.TestCase):
    def _build(self) -> NVFP4Linear:
        module = NVFP4Linear.Config(in_features=_DIM, out_features=_HIDDEN).build()
        module.init_states()
        return module

    def test_native_state_dict_keeps_buffers(self):
        """Native NVFP4 checkpoints persist the RNG/RHT runtime buffers."""
        sd = self._build().state_dict()
        self.assertEqual(set(sd), {"weight", "_sr_seed", "_rht_sign_vector"})

    def test_native_roundtrip_preserves_buffers(self):
        src, dst = self._build(), self._build()
        dst.load_state_dict(src.state_dict())
        self.assertTrue(torch.equal(dst.weight, src.weight))
        self.assertTrue(torch.equal(dst._sr_seed, src._sr_seed))
        self.assertTrue(torch.equal(dst._rht_sign_vector, src._rht_sign_vector))

    def test_stock_state_dict_loads_before_init_states(self):
        stock = Linear.Config(in_features=_DIM, out_features=_HIDDEN).build()
        nvfp4 = NVFP4Linear.Config(in_features=_DIM, out_features=_HIDDEN).build()

        nvfp4.load_state_dict(stock.state_dict(), strict=False)

        self.assertIsNone(nvfp4._rht_sign_vector)
        self.assertIsNone(nvfp4._rht_sign_vector_tuple)
        nvfp4.init_states()
        self.assertIsNotNone(nvfp4._rht_sign_vector)
        self.assertIsNotNone(nvfp4._rht_sign_vector_tuple)

    def test_hf_export_strips_nvfp4_buffers(self):
        """The HF adapter export contains only stock keys -- no NVFP4 buffers."""
        config = _debugmodel_with_sharding()
        # Apply the FFN override factory directly, independent of the registry.
        for layer in config.layers:
            layer.feed_forward = nvfp4_feed_forward(layer.feed_forward)
        model = Llama3Model(config)
        model.init_states()
        self.assertIsInstance(
            model.get_submodule("layers.0.feed_forward.w1"), NVFP4Linear
        )

        sd = model.state_dict()
        self.assertTrue(any(k.endswith("feed_forward.w1._sr_seed") for k in sd))

        hf_sd = Llama3StateDictAdapter(config, hf_assets_path=None).to_hf(sd)
        self.assertIn("model.layers.0.mlp.gate_proj.weight", hf_sd)
        self.assertFalse(any("_sr_seed" in k for k in hf_sd))
        self.assertFalse(any("_rht_sign_vector" in k for k in hf_sd))


@unittest.skipUnless(_blackwell(), "NVFP4 Triton kernels require Blackwell (sm_100+)")
class TestNVFP4Numerics(unittest.TestCase):
    def _module(self) -> NVFP4Linear:
        module = (
            NVFP4Linear.Config(in_features=_DIM, out_features=_HIDDEN).build().cuda()
        )
        module.init_states(buffer_device=torch.device("cuda"))
        return module

    def test_forward_close_to_reference_and_backward_finite(self):
        module = self._module()
        # NVFP4 Triton GEMMs require M, K, N all divisible by 128.
        x = torch.randn(
            128, _DIM, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        out = module(x)
        ref = torch.nn.functional.linear(x, module.weight.to(x.dtype))
        # NVFP4 is a 4-bit format; use SQNR (the metric torchao's own NVFP4 tests
        # use) with the same >= 15 dB forward contract rather than a per-element tol.
        from torchao.quantization.utils import compute_error

        sqnr = compute_error(ref.float(), out.float())
        self.assertGreaterEqual(sqnr.item(), 15.0)

        out.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(module.weight.grad).all())

    def test_compiled_forward_runs(self):
        module = self._module()
        compiled = torch.compile(module)
        x = torch.randn(128, _DIM, device="cuda", dtype=torch.bfloat16)
        out = compiled(x)
        self.assertEqual(tuple(out.shape), (128, _HIDDEN))


@unittest.skipUnless(_blackwell_tp2(), "NVFP4 TP tests require two Blackwell GPUs")
class TestNVFP4TensorParallelNumerics(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    def _parallel_dims(self) -> ParallelDims:
        return ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=self.world_size,
            pp=1,
            ep=1,
            world_size=self.world_size,
        )

    def _parallelized_module(self, cfg: NVFP4Linear.Config) -> NVFP4Linear:
        module = cfg.build().to(self.device_type)
        parallel_dims = self._parallel_dims()
        module.parallelize(parallel_dims)
        torch.manual_seed(1234)
        module.init_states(buffer_device=torch.device(self.device_type))
        return module

    def _assert_shard(self, tensor: DTensor, dim: int) -> None:
        self.assertIsInstance(tensor, DTensor)
        self.assertEqual(len(tensor.placements), 1)
        self.assertIsInstance(tensor.placements[0], Shard)
        actual_dim = tensor.placements[0].dim
        if actual_dim < 0:
            actual_dim += tensor.ndim
        if dim < 0:
            dim += tensor.ndim
        self.assertEqual(actual_dim, dim)

    def _assert_sqnr(
        self, actual: torch.Tensor, expected: torch.Tensor, threshold: float
    ):
        from torchao.quantization.utils import compute_error

        sqnr = compute_error(expected.float(), actual.float())
        self.assertGreaterEqual(sqnr.item(), threshold)

    def _rank_scaled_input(
        self, shape: tuple[int, int, int], shard_dim: int
    ) -> torch.Tensor:
        gen = torch.Generator(device=self.device_type).manual_seed(42)
        x = torch.randn(
            shape, device=self.device_type, dtype=torch.bfloat16, generator=gen
        )
        scale = torch.ones(
            shape[shard_dim], device=self.device_type, dtype=torch.bfloat16
        )
        midpoint = shape[shard_dim] // 2
        scale[:midpoint] = 0.25
        scale[midpoint:] = 4.0
        view_shape = [1] * len(shape)
        view_shape[shard_dim] = shape[shard_dim]
        return x * scale.view(view_shape)

    def _weight(self, shape: tuple[int, int]) -> torch.Tensor:
        gen = torch.Generator(device=self.device_type).manual_seed(7)
        return torch.randn(
            shape, device=self.device_type, dtype=torch.bfloat16, generator=gen
        )

    def _check_against_reference(
        self,
        *,
        out: DTensor,
        x: DTensor,
        weight: DTensor,
        x_full: torch.Tensor,
        weight_full: torch.Tensor,
        out_shard_dim: int,
        x_shard_dim: int,
        weight_shard_dim: int,
    ) -> None:
        self._assert_shard(out, out_shard_dim)
        ref_out = torch.nn.functional.linear(x_full.float(), weight_full.float())
        self._assert_sqnr(out.full_tensor(), ref_out, 15.0)

        gen = torch.Generator(device=self.device_type).manual_seed(99 + self.rank)
        dy_local = torch.randn(
            out.to_local().shape,
            device=self.device_type,
            dtype=torch.bfloat16,
            generator=gen,
        )
        dy = DTensor.from_local(
            dy_local, out.device_mesh, out.placements, run_check=False
        )
        loss = (out.to_local().float() * dy_local.float()).sum()
        loss.backward()

        self._assert_shard(x.grad, x_shard_dim)
        self._assert_shard(weight.grad, weight_shard_dim)

        x_ref = x_full.float().detach().requires_grad_(True)
        weight_ref = weight_full.float().detach().requires_grad_(True)
        torch.nn.functional.linear(x_ref, weight_ref).backward(dy.full_tensor().float())
        self._assert_sqnr(x.grad.full_tensor(), x_ref.grad, 14.0)
        self._assert_sqnr(weight.grad.full_tensor(), weight_ref.grad, 14.0)

    @with_comms
    def test_tp_colwise_and_rowwise_math_and_placements(self):
        parallel_dims = self._parallel_dims()
        tp_mesh = parallel_dims.get_mesh("tp")
        B, L, D, H = 1, 256, 256, 512

        col_base = Linear.Config(in_features=D, out_features=H)
        col_base.sharding_config = colwise_config()
        col = self._parallelized_module(
            _to_nvfp4_colwise(col_base, sequence_parallel=True)
        )
        col_weight_full = self._weight((H, D))
        col.weight = torch.nn.Parameter(
            distribute_tensor(col_weight_full, tp_mesh, [Shard(0)])
        )
        col_x_full = self._rank_scaled_input((B, L, D), shard_dim=1)
        col_x = distribute_tensor(col_x_full, tp_mesh, [Shard(1)]).requires_grad_()
        col_out = col(col_x)
        self._check_against_reference(
            out=col_out,
            x=col_x,
            weight=col.weight,
            x_full=col_x_full,
            weight_full=col_weight_full,
            out_shard_dim=-1,
            x_shard_dim=1,
            weight_shard_dim=0,
        )

        row_base = Linear.Config(in_features=H, out_features=D)
        row_base.sharding_config = rowwise_config(output_sp=True)
        row = self._parallelized_module(
            _to_nvfp4_rowwise(row_base, sequence_parallel=True)
        )
        row_weight_full = self._weight((D, H))
        row.weight = torch.nn.Parameter(
            distribute_tensor(row_weight_full, tp_mesh, [Shard(1)])
        )
        row_x_full = self._rank_scaled_input((B, L, H), shard_dim=2)
        row_x = distribute_tensor(row_x_full, tp_mesh, [Shard(-1)]).requires_grad_()
        row_out = row(row_x)
        self._check_against_reference(
            out=row_out,
            x=row_x,
            weight=row.weight,
            x_full=row_x_full,
            weight_full=row_weight_full,
            out_shard_dim=1,
            x_shard_dim=-1,
            weight_shard_dim=1,
        )


if __name__ == "__main__":
    unittest.main()
