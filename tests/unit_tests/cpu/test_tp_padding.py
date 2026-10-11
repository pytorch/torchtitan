# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for TP head padding (``PaddedQKVLinear``).

Single-process tests cover the padded geometry, zero initialization, state-dict
hooks (native and Hugging Face), optimizer states and EMA copies. Multi-rank
tests run on real gloo meshes, because on a single-rank mesh a ``Shard`` and a
``Replicate`` placement are indistinguishable in the data. They check that:

* the hidden-padding tensors are sharded with uneven plain ``Shard`` placements
  and do not keep the gathered full tensor alive (``state_dict()`` results are
  cached by the checkpointer);
* restoring reshards to the live tensors' placements, because
  ``Optimizer.load_state_dict`` replaces the live state tensors instead of
  copying into them;
* weights, AdamW states and EMA copies saved under FSDP+TP with one padding load
  into FSDP+TP with another, via DCP.

Validation of the TP degree is tested in test_tp_kv_heads_validation.py.
"""

import tempfile
import unittest
from dataclasses import dataclass
from functools import partial
from unittest import mock

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir
from torchtitan.components.optim import AdamW, EMA, OptimizersContainer
from torchtitan.config.transform import convert_config_type
from torchtitan.models.common import compute_ffn_hidden_dim, Embedding, Linear, RMSNorm
from torchtitan.models.common.attention import GQAttention, InnerAttention
from torchtitan.models.common.attention.attention import _warn_tp_head_padding
from torchtitan.models.common.config_utils import make_ffn_config, make_gqa_config
from torchtitan.models.common.linear import ColumnParallelLinear, RowParallelLinear
from torchtitan.models.llama3 import Llama3Model
from torchtitan.models.llama3.model import Llama3TransformerBlock
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter

_DIM = 64
_HEAD_DIM = 8
_N_LAYERS = 2
_VOCAB_SIZE = 2048
_NUM_TOKENS = 16
_N_HEADS_UNPADDED = 6
_INIT = {"weight": partial(nn.init.trunc_normal_, std=0.02)}


class _CausalInnerAttention(InnerAttention):
    """Causal SDPA over ``(T, H, K)`` inputs, so the tests run on CPU."""

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__()

    def forward(self, q_THK, k_THK, v_THV, *, scale=None, enable_gqa=False, **kwargs):
        out_HTV = F.scaled_dot_product_attention(
            q_THK.transpose(0, 1),
            k_THK.transpose(0, 1),
            v_THV.transpose(0, 1),
            scale=scale,
            is_causal=True,
            enable_gqa=enable_gqa,
        )
        return out_HTV.transpose(0, 1)


def _attention_config(
    n_heads: int,
    n_kv_heads: int | None,
    *,
    head_dim: int = _HEAD_DIM,
    pad_heads_for_tp: bool = False,
) -> GQAttention.Config:
    return make_gqa_config(
        dim=_DIM,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        wqkv_param_init=_INIT,
        wo_param_init=_INIT,
        inner_attention=_CausalInnerAttention.Config(),
        rope=None,
        pad_heads_for_tp=pad_heads_for_tp,
    )


def _make_llama3_config(
    n_heads: int, n_kv_heads: int | None, *, pad_heads_for_tp: bool = False
) -> Llama3Model.Config:
    """Build a minimal Llama3Model.Config with the given head counts."""
    layers = [
        Llama3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(normalized_shape=_DIM),
            ffn_norm=RMSNorm.Config(normalized_shape=_DIM),
            attention=_attention_config(
                n_heads, n_kv_heads, pad_heads_for_tp=pad_heads_for_tp
            ),
            feed_forward=make_ffn_config(
                dim=_DIM,
                hidden_dim=compute_ffn_hidden_dim(_DIM, multiple_of=256),
                w13_param_init=_INIT,
                w2_param_init=_INIT,
            ),
        )
        for _ in range(_N_LAYERS)
    ]
    return Llama3Model.Config(
        max_context_length=4096,
        dim=_DIM,
        vocab_size=_VOCAB_SIZE,
        tok_embeddings=Embedding.Config(num_embeddings=_VOCAB_SIZE, embedding_dim=_DIM),
        norm=RMSNorm.Config(normalized_shape=_DIM),
        lm_head=Linear.Config(in_features=_DIM, out_features=_VOCAB_SIZE),
        layers=layers,
    )


def _mock_tp_degree(tp: int):
    return mock.patch(
        "torchtitan.distributed.spmd_types.spmd_mesh_size",
        side_effect=lambda axis: tp if axis == "tp" else 1,
    )


def _build_attention(config: GQAttention.Config, *, tp: int = 1) -> GQAttention:
    with _mock_tp_degree(tp):
        module = config.build()
    # Seed after construction: nn.Linear's constructor init consumes RNG in
    # proportion to the (padded) parameter size.
    torch.manual_seed(0)
    module.init_states()
    return module


def _init_state_dict(
    config: Llama3Model.Config, *, tp: int = 1
) -> dict[str, torch.Tensor]:
    with _mock_tp_degree(tp), torch.device("meta"):
        model = config.build()
    model.to_empty(device="cpu")
    torch.manual_seed(0)
    model.init_states()
    return model.state_dict()


class TestTPHeadPadding(unittest.TestCase):
    def setUp(self) -> None:
        _warn_tp_head_padding.cache_clear()

    def test_pads_kv_heads_and_keeps_group_size(self):
        config = _attention_config(n_heads=6, n_kv_heads=3, pad_heads_for_tp=True)
        with self.assertLogs(
            "torchtitan.models.common.attention", level="WARNING"
        ) as logs:
            attention = _build_attention(config, tp=4)

        self.assertIn("n_kv_heads 3 -> 4, n_heads 6 -> 8", logs.output[0])
        self.assertEqual(attention.qkv_linear.num_padded_kv_heads, 4)
        self.assertEqual(attention.qkv_linear.num_padded_q_heads, 8)
        self.assertEqual(attention.head_dim, _HEAD_DIM)
        self.assertEqual(
            attention.qkv_linear.wqkv.weight.shape, ((8 + 2 * 4) * _HEAD_DIM, _DIM)
        )
        self.assertEqual(attention.wo.weight.shape, (_DIM, 8 * _HEAD_DIM))
        # The config keeps the unpadded geometry.
        self.assertEqual(config.n_kv_heads, 3)
        self.assertEqual(config.n_heads, 6)

    def test_pads_mha(self):
        config = _attention_config(n_heads=6, n_kv_heads=None, pad_heads_for_tp=True)
        attention = _build_attention(config, tp=4)

        self.assertEqual(attention.qkv_linear.num_padded_kv_heads, 8)
        self.assertEqual(attention.qkv_linear.num_padded_q_heads, 8)

    def test_divisible_heads_are_not_padded(self):
        config = _attention_config(n_heads=8, n_kv_heads=4, pad_heads_for_tp=True)
        attention = _build_attention(config, tp=4)

        self.assertEqual(attention.qkv_linear.num_padded_kv_heads, 4)
        self.assertEqual(attention.wo.weight.shape, (_DIM, 8 * _HEAD_DIM))

    def test_rejects_implicit_weight_init(self):
        config = _attention_config(n_heads=6, n_kv_heads=3, pad_heads_for_tp=True)
        config.wo.param_init = None
        with self.assertRaisesRegex(ValueError, "explicit projection"):
            _build_attention(config, tp=4)

    def test_decoder_builds_padded_layers_and_warns_once(self):
        config = _make_llama3_config(n_heads=8, n_kv_heads=2, pad_heads_for_tp=True)
        with (
            _mock_tp_degree(4),
            torch.device("meta"),
            self.assertLogs(
                "torchtitan.models.common.attention", level="WARNING"
            ) as logs,
        ):
            model = config.build()

        self.assertEqual(len(logs.output), 1)
        self.assertIn("n_kv_heads 2 -> 4, n_heads 8 -> 16", logs.output[0])
        for layer in model.layers.values():
            self.assertEqual(layer.attention.qkv_linear.num_padded_kv_heads, 4)
            self.assertEqual(layer.attention.qkv_linear.num_padded_q_heads, 16)

    def test_state_dict_hides_padding(self):
        unpadded = _build_attention(_attention_config(n_heads=6, n_kv_heads=3))
        padded = _build_attention(
            _attention_config(n_heads=6, n_kv_heads=3, pad_heads_for_tp=True),
            tp=4,
        )

        state_dict = unpadded.state_dict()
        padded_state_dict = padded.state_dict()
        self.assertEqual(padded_state_dict.keys(), state_dict.keys())
        for key, value in state_dict.items():
            self.assertTrue(torch.equal(padded_state_dict[key], value), key)

    def test_load_unpadded_state_dict_zero_pads(self):
        unpadded = _build_attention(_attention_config(n_heads=6, n_kv_heads=3))
        padded = _build_attention(
            _attention_config(n_heads=6, n_kv_heads=3, pad_heads_for_tp=True),
            tp=4,
        )
        with torch.no_grad():
            for param in padded.parameters():
                param.fill_(1.0)

        padded.load_state_dict(unpadded.state_dict())

        num_rows = unpadded.qkv_linear.wqkv.weight.shape[0]
        num_columns = unpadded.wo.weight.shape[1]
        wqkv, wo = padded.qkv_linear.wqkv.weight, padded.wo.weight
        self.assertTrue(torch.equal(wqkv[:num_rows], unpadded.qkv_linear.wqkv.weight))
        self.assertEqual(wqkv[num_rows:].count_nonzero(), 0)
        self.assertTrue(torch.equal(wo[:, :num_columns], unpadded.wo.weight))
        self.assertEqual(wo[:, num_columns:].count_nonzero(), 0)

    def test_load_padded_state_dict(self):
        """Checkpoints saved with the padded shapes still load."""
        padded = _build_attention(
            _attention_config(n_heads=6, n_kv_heads=3, pad_heads_for_tp=True),
            tp=4,
        )
        raw_state_dict = {
            name: param.detach().clone() for name, param in padded.named_parameters()
        }
        with torch.no_grad():
            for param in padded.parameters():
                param.zero_()

        padded.load_state_dict(raw_state_dict)

        for name, param in padded.named_parameters():
            self.assertTrue(torch.equal(param, raw_state_dict[name]), name)

    def test_dtensor_state_dict_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            owns_process_group = not dist.is_initialized()
            if owns_process_group:
                dist.init_process_group(
                    "gloo",
                    init_method=f"file://{directory}/rendezvous",
                    rank=0,
                    world_size=1,
                )
            try:
                mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))
                unpadded = _build_attention(_attention_config(n_heads=6, n_kv_heads=3))
                padded = _build_attention(
                    _attention_config(n_heads=6, n_kv_heads=3, pad_heads_for_tp=True),
                    tp=4,
                )
                for linear, dim in ((padded.qkv_linear.wqkv, 0), (padded.wo, 1)):
                    linear.weight = nn.Parameter(
                        distribute_tensor(linear.weight.detach(), mesh, [Shard(dim)])
                    )

                # Clone: on a 1-rank mesh the hook output aliases the parameters.
                state_dict = {k: v.clone() for k, v in padded.state_dict().items()}
                for key, value in unpadded.state_dict().items():
                    saved = state_dict[key]
                    if isinstance(saved, DTensor):
                        saved = saved.full_tensor()
                    self.assertTrue(torch.equal(saved, value), key)

                with torch.no_grad():
                    for param in padded.parameters():
                        param.fill_(1.0)
                padded.load_state_dict(state_dict)

                num_rows = unpadded.qkv_linear.wqkv.weight.shape[0]
                wqkv = padded.qkv_linear.wqkv.weight.full_tensor()
                self.assertTrue(
                    torch.equal(wqkv[:num_rows], unpadded.qkv_linear.wqkv.weight)
                )
                self.assertEqual(wqkv[num_rows:].count_nonzero(), 0)
            finally:
                if owns_process_group:
                    dist.destroy_process_group()

    def test_hf_checkpoint_round_trip_strips_padding(self):
        config = _make_llama3_config(n_heads=8, n_kv_heads=2)
        padded_config = _make_llama3_config(
            n_heads=8, n_kv_heads=2, pad_heads_for_tp=True
        )
        state_dict = _init_state_dict(config)
        padded_state_dict = _init_state_dict(padded_config, tp=4)

        # The exported checkpoint matches the unpadded model exactly.
        hf_state_dict = Llama3StateDictAdapter(config, None).to_hf(state_dict)
        padded_adapter = Llama3StateDictAdapter(padded_config, None)
        padded_hf_state_dict = padded_adapter.to_hf(padded_state_dict)
        self.assertEqual(padded_hf_state_dict.keys(), hf_state_dict.keys())
        for key, value in hf_state_dict.items():
            self.assertTrue(torch.equal(padded_hf_state_dict[key], value), key)

        # Loading it restores the padded native state, padding included.
        loaded = padded_adapter.from_hf(padded_hf_state_dict)
        self.assertEqual(loaded.keys(), padded_state_dict.keys())
        for key, value in padded_state_dict.items():
            self.assertTrue(torch.equal(loaded[key], value), key)

    def test_padded_attention_matches_unpadded(self):
        unpadded = _build_attention(_attention_config(n_heads=6, n_kv_heads=3))
        padded = _build_attention(
            _attention_config(n_heads=6, n_kv_heads=3, pad_heads_for_tp=True),
            tp=4,
        )

        # Real heads draw the same initial values as the unpadded model, and
        # padded heads start at zero.
        num_real_qkv_rows = unpadded.qkv_linear.wqkv.weight.shape[0]
        num_real_wo_cols = unpadded.wo.weight.shape[1]
        padded_wqkv = padded.qkv_linear.wqkv.weight
        self.assertTrue(
            torch.equal(
                padded_wqkv[:num_real_qkv_rows], unpadded.qkv_linear.wqkv.weight
            )
        )
        self.assertEqual(padded_wqkv[num_real_qkv_rows:].count_nonzero(), 0)
        self.assertTrue(
            torch.equal(padded.wo.weight[:, :num_real_wo_cols], unpadded.wo.weight)
        )
        self.assertEqual(padded.wo.weight[:, num_real_wo_cols:].count_nonzero(), 0)

        torch.manual_seed(1)
        x_TD = torch.randn(_NUM_TOKENS, _DIM)
        grad_TD = torch.randn(_NUM_TOKENS, _DIM)
        unpadded(x_TD, None).backward(grad_TD)
        out_TD = padded(x_TD, None)
        out_TD.backward(grad_TD)

        torch.testing.assert_close(out_TD, unpadded(x_TD, None))
        # Padded heads receive exactly zero gradient, so they stay zero.
        padded_wqkv_grad = padded_wqkv.grad
        padded_wo_grad = padded.wo.weight.grad
        assert padded_wqkv_grad is not None and padded_wo_grad is not None
        self.assertEqual(padded_wqkv_grad[num_real_qkv_rows:].count_nonzero(), 0)
        self.assertEqual(padded_wo_grad[:, num_real_wo_cols:].count_nonzero(), 0)
        torch.testing.assert_close(
            padded_wqkv_grad[:num_real_qkv_rows], unpadded.qkv_linear.wqkv.weight.grad
        )
        torch.testing.assert_close(
            padded_wo_grad[:, :num_real_wo_cols], unpadded.wo.weight.grad
        )


def _padded_attention(tp: int, *, n_heads: int = 6, n_kv_heads: int = 3) -> GQAttention:
    config = _attention_config(n_heads, n_kv_heads, pad_heads_for_tp=True)
    return _build_attention(config, tp=tp)


def _trained_optimizer(attention: GQAttention) -> OptimizersContainer:
    """Take a few AdamW steps so the states are nonzero."""
    optimizer = OptimizersContainer.Config(
        optimizers=[AdamW.Config(pattern=r".*", lr=1e-2)]
    ).build(model_parts=[attention])
    torch.manual_seed(1)
    for _ in range(3):
        x_TD = torch.randn(_NUM_TOKENS, _DIM)
        optimizer.zero_grad()
        attention(x_TD, None).backward(torch.randn(_NUM_TOKENS, _DIM))
        optimizer.step()
    return optimizer


def _clone_state_dict(state_dict: dict) -> dict:
    return {
        k: v.clone() if isinstance(v, torch.Tensor) else v
        for k, v in state_dict.items()
    }


class TestTPHeadPaddingOptimizerState(unittest.TestCase):
    """Optimizer states hide TP head padding like the weights do."""

    def setUp(self) -> None:
        _warn_tp_head_padding.cache_clear()

    def test_state_dict_hides_padding(self):
        unpadded = _trained_optimizer(
            _build_attention(_attention_config(n_heads=6, n_kv_heads=3))
        ).state_dict()
        padded = _trained_optimizer(_padded_attention(tp=4)).state_dict()

        self.assertEqual(padded.keys(), unpadded.keys())
        for key, value in unpadded.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(padded[key], value, msg=key)
            else:
                self.assertEqual(padded[key], value, key)

    def test_load_zero_pads_states(self):
        saved = _clone_state_dict(
            _trained_optimizer(
                _build_attention(_attention_config(n_heads=6, n_kv_heads=3))
            ).state_dict()
        )
        padded_attention = _padded_attention(tp=4)
        optimizer = OptimizersContainer.Config(
            optimizers=[AdamW.Config(pattern=r".*", lr=1e-2)]
        ).build(model_parts=[padded_attention])

        optimizer.load_state_dict(saved)

        state = optimizer.optimizers[0].state[padded_attention.wo.weight]
        num_columns = _N_HEADS_UNPADDED * _HEAD_DIM
        self.assertEqual(state["exp_avg"].shape, padded_attention.wo.weight.shape)
        torch.testing.assert_close(
            state["exp_avg"][:, :num_columns], saved["state.wo.weight.exp_avg"]
        )
        self.assertEqual(state["exp_avg"][:, num_columns:].count_nonzero(), 0)
        self.assertEqual(state["exp_avg_sq"][:, num_columns:].count_nonzero(), 0)

    def test_load_padded_states(self):
        """States saved with the padded shapes still load."""
        trained = _trained_optimizer(_padded_attention(tp=4))
        raw = _clone_state_dict(trained.state_dict())
        optimizer = OptimizersContainer.Config(
            optimizers=[AdamW.Config(pattern=r".*", lr=1e-2)]
        ).build(model_parts=[_padded_attention(tp=4)])
        padded_wo = trained.model_parts[0].wo.weight
        raw["state.wo.weight.exp_avg"] = F.pad(
            raw["state.wo.weight.exp_avg"],
            (0, padded_wo.shape[1] - raw["state.wo.weight.exp_avg"].shape[1]),
        )

        optimizer.load_state_dict(raw)

        torch.testing.assert_close(
            optimizer.optimizers[0].state[optimizer.model_parts[0].wo.weight][
                "exp_avg"
            ],
            trained.optimizers[0].state[padded_wo]["exp_avg"],
        )

    def test_dcp_round_trip_across_tp_degrees(self):
        trained = _trained_optimizer(_padded_attention(tp=4))
        with tempfile.TemporaryDirectory() as directory:
            dcp.save(_clone_state_dict(trained.state_dict()), checkpoint_id=directory)
            # tp=2 pads 3 -> 4 KV heads like tp=4; tp=3 does not pad.
            for tp in (2, 3, 4):
                attention = _padded_attention(tp=tp)
                optimizer = OptimizersContainer.Config(
                    optimizers=[AdamW.Config(pattern=r".*", lr=1e-2)]
                ).build(model_parts=[attention])
                state_dict = optimizer.state_dict()
                dcp.load(state_dict, checkpoint_id=directory)
                optimizer.load_state_dict(state_dict)

                for key, value in trained.state_dict().items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(
                            optimizer.state_dict()[key], value, msg=f"tp={tp} {key}"
                        )

    def test_dtensor_states_hide_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            owns_process_group = not dist.is_initialized()
            if owns_process_group:
                dist.init_process_group(
                    "gloo",
                    init_method=f"file://{directory}/rendezvous",
                    rank=0,
                    world_size=1,
                )
            try:
                mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))
                attention = _padded_attention(tp=4)
                for linear, dim in ((attention.qkv_linear.wqkv, 0), (attention.wo, 1)):
                    linear.weight = nn.Parameter(
                        distribute_tensor(linear.weight.detach(), mesh, [Shard(dim)])
                    )
                optimizer = OptimizersContainer.Config(
                    optimizers=[AdamW.Config(pattern=r".*", lr=1e-2)]
                ).build(model_parts=[attention])

                state_dict = optimizer.state_dict()
                for key in ("state.wo.weight.exp_avg", "state.wo.weight.exp_avg_sq"):
                    self.assertIsInstance(state_dict[key], DTensor)
                    self.assertEqual(
                        state_dict[key].shape, (_DIM, _N_HEADS_UNPADDED * _HEAD_DIM)
                    )
                self.assertEqual(
                    state_dict["state.qkv_linear.wqkv.weight.exp_avg"].shape[0],
                    (_N_HEADS_UNPADDED + 2 * 3) * _HEAD_DIM,
                )
                # Loading restores the padded live states.
                optimizer.load_state_dict(_clone_state_dict(state_dict))
                live = optimizer.optimizers[0].state[attention.wo.weight]
                self.assertEqual(live["exp_avg"].shape, attention.wo.weight.shape)
            finally:
                if owns_process_group:
                    dist.destroy_process_group()


def _trained_ema(attention: GQAttention) -> EMA:
    """Track a few EMA firings while AdamW moves the weights."""
    optimizer = OptimizersContainer.Config(
        optimizers=[AdamW.Config(pattern=r".*", lr=1e-2)]
    ).build(model_parts=[attention])
    ema = EMA.Config(decays=[0.9]).build(model_parts=[attention])
    torch.manual_seed(1)
    for step in range(1, 4):
        optimizer.zero_grad()
        x_TD = torch.randn(_NUM_TOKENS, _DIM)
        attention(x_TD, None).backward(torch.randn(_NUM_TOKENS, _DIM))
        optimizer.step()
        ema.step(step)
    return ema


class TestTPHeadPaddingEMAState(unittest.TestCase):
    """EMA copies hide TP head padding like the weights do."""

    def setUp(self) -> None:
        _warn_tp_head_padding.cache_clear()

    def test_state_dict_hides_padding(self):
        unpadded = _trained_ema(
            _build_attention(_attention_config(n_heads=6, n_kv_heads=3))
        ).state_dict()
        padded = _trained_ema(_padded_attention(tp=4)).state_dict()

        self.assertEqual(padded.keys(), unpadded.keys())
        for key, value in unpadded.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(padded[key], value, msg=key)

    def test_dcp_round_trip_across_tp_degrees(self):
        trained = _trained_ema(_padded_attention(tp=4))
        key = "state.wo.weight.ema_params.decay_0p9"
        with tempfile.TemporaryDirectory() as directory:
            dcp.save(_clone_state_dict(trained.state_dict()), checkpoint_id=directory)
            # tp=2 pads 3 -> 4 KV heads like tp=4; tp=3 does not pad.
            for tp in (2, 3, 4):
                attention = _padded_attention(tp=tp)
                ema = EMA.Config(decays=[0.9]).build(model_parts=[attention])
                state_dict = ema.state_dict()
                dcp.load(state_dict, checkpoint_id=directory)
                ema.load_state_dict(state_dict)

                self.assertEqual(
                    state_dict[key].shape[1], _N_HEADS_UNPADDED * _HEAD_DIM
                )
                for name, value in trained.state_dict().items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(
                            ema.state_dict()[name], value, msg=f"tp={tp} {name}"
                        )
                live = ema.optimizers[0].state[attention.wo.weight]["ema_params"]
                self.assertEqual(live["decay_0p9"].shape, attention.wo.weight.shape)


class _CustomColumnLinear(ColumnParallelLinear):
    """Stands in for a quantized linear: a subclass with an extra config field."""

    @dataclass(kw_only=True, slots=True)
    class Config(ColumnParallelLinear.Config):
        marker: int = 0


class _CustomRowLinear(RowParallelLinear):
    @dataclass(kw_only=True, slots=True)
    class Config(RowParallelLinear.Config):
        marker: int = 0


class TestTPHeadPaddingModuleTypes(unittest.TestCase):
    def setUp(self) -> None:
        _warn_tp_head_padding.cache_clear()

    def test_linear_subclasses_keep_padding_hooks(self):
        """Config-level converters swap the projection classes before build."""
        config = _attention_config(n_heads=6, n_kv_heads=3, pad_heads_for_tp=True)
        config.qkv_linear.wqkv = convert_config_type(
            config.qkv_linear.wqkv, _CustomColumnLinear
        )
        config.wo = convert_config_type(config.wo, _CustomRowLinear)

        attention = _build_attention(config, tp=4)

        self.assertIsInstance(attention.qkv_linear.wqkv, _CustomColumnLinear)
        self.assertIsInstance(attention.wo, _CustomRowLinear)
        self.assertIn("weight", attention.qkv_linear.wqkv.padded_params)
        self.assertIn("weight", attention.wo.padded_params)
        unpadded = _build_attention(_attention_config(n_heads=6, n_kv_heads=3))
        self.assertEqual(attention.state_dict().keys(), unpadded.state_dict().keys())
        for key, value in unpadded.state_dict().items():
            self.assertEqual(attention.state_dict()[key].shape, value.shape, key)


def _assert_compact(test: DTensorTestBase, tensor: DTensor, name: str) -> None:
    """The local shard owns exactly its elements, not a gathered full tensor."""
    local = tensor.to_local()
    test.assertEqual(
        local.untyped_storage().nbytes(), local.numel() * local.element_size(), name
    )


class TestPaddedOptimizerStateDTensor(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @property
    def backend(self) -> str:
        return "gloo"

    @property
    def device_type(self) -> str:
        return "cpu"

    _N_HEADS = 6
    _N_KV_HEADS = 3
    _HEAD_DIM = 8

    def _sharded_attention(self) -> GQAttention:
        """Padded attention (tp=4: 3 -> 4 KV heads) sharded over 2 ranks."""
        attention = _build_attention(
            _attention_config(
                self._N_HEADS,
                self._N_KV_HEADS,
                head_dim=self._HEAD_DIM,
                pad_heads_for_tp=True,
            ),
            tp=4,
        )
        mesh = self.build_device_mesh()
        for linear, dim in ((attention.qkv_linear.wqkv, 0), (attention.wo, 1)):
            linear.weight = nn.Parameter(
                distribute_tensor(linear.weight.detach(), mesh, [Shard(dim)])
            )
        return attention

    def _check_hidden(self, state_dict: dict, suffixes: list[str]) -> None:
        num_real_rows = (self._N_HEADS + 2 * self._N_KV_HEADS) * self._HEAD_DIM
        num_real_columns = self._N_HEADS * self._HEAD_DIM
        for suffix in suffixes:
            wqkv = state_dict[f"state.qkv_linear.wqkv.weight.{suffix}"]
            wo = state_dict[f"state.wo.weight.{suffix}"]
            self.assertIsInstance(wqkv, DTensor)
            self.assertEqual(wqkv.shape, (num_real_rows, _DIM))
            self.assertEqual(wo.shape, (_DIM, num_real_columns))
            # Plain Shard placements can express the uneven unpadded shards.
            self.assertEqual(wqkv.placements, (Shard(0),))
            self.assertEqual(wo.placements, (Shard(1),))
            _assert_compact(self, wqkv, f"{suffix} wqkv")
            _assert_compact(self, wo, f"{suffix} wo")

    @with_comms
    def test_weights_hide_padding_without_retaining_full_tensor(self):
        attention = self._sharded_attention()
        state_dict = attention.state_dict()
        for name in ("qkv_linear.wqkv.weight", "wo.weight"):
            self.assertIsInstance(state_dict[name], DTensor)
            _assert_compact(self, state_dict[name], name)

    @with_comms
    def test_adamw_states_hide_padding_and_reshard_on_load(self):
        attention = self._sharded_attention()
        optimizer = OptimizersContainer.Config(
            optimizers=[AdamW.Config(pattern=r".*", lr=1e-2)]
        ).build(model_parts=[attention])

        state_dict = optimizer.state_dict()
        self._check_hidden(state_dict, ["exp_avg", "exp_avg_sq"])

        live_before = {
            name: tensor.placements
            for name, tensor in optimizer.optimizers[0]
            .state[attention.wo.weight]
            .items()
            if isinstance(tensor, DTensor)
        }
        optimizer.load_state_dict(dict(state_dict))
        live = optimizer.optimizers[0].state[attention.wo.weight]
        for name, placements in live_before.items():
            self.assertEqual(live[name].shape, attention.wo.weight.shape)
            self.assertEqual(live[name].placements, placements, name)
        self.assertEqual(
            live["exp_avg"]
            .full_tensor()[:, self._N_HEADS * self._HEAD_DIM :]
            .count_nonzero(),
            0,
        )

    @with_comms
    def test_ema_states_hide_padding_and_reshard_on_load(self):
        attention = self._sharded_attention()
        ema = EMA.Config(decays=[0.9]).build(model_parts=[attention])

        state_dict = ema.state_dict()
        self._check_hidden(state_dict, ["ema_params.decay_0p9"])
        # The EMA copy equals the weights it started from, minus the padding.
        torch.testing.assert_close(
            state_dict["state.wo.weight.ema_params.decay_0p9"].full_tensor(),
            attention.wo.weight.full_tensor()[:, : self._N_HEADS * self._HEAD_DIM],
        )

        ema.load_state_dict(dict(state_dict))
        live = ema.optimizers[0].state[attention.wo.weight]["ema_params"]["decay_0p9"]
        self.assertEqual(live.shape, attention.wo.weight.shape)
        self.assertEqual(live.placements, attention.wo.weight.placements)
        torch.testing.assert_close(
            live.full_tensor(), attention.wo.weight.full_tensor()
        )


class TestPaddedStateFSDPTP(DTensorTestBase):
    """FSDP+TP placements: DCP round trip of padded weights and states.

    Geometry (MHA, so ``n_kv_heads == n_heads``): 5 heads of size 2. The
    unpadded ``wqkv`` has 30 rows, which 4 ranks cannot split evenly, so the
    hidden-padding tensors are uneven nested ``Shard(0)`` while the live ones
    are ``(_StridedShard(0), Shard(0))``. tp=2 pads to 6 heads, tp=4 to 8, and
    tp=1 does not pad.
    """

    _N_HEADS = 5
    _HEAD_DIM = 2
    # tp -> (dp, tp) mesh shape over the 4 ranks.
    _MESHES = {1: (4, 1), 2: (2, 2), 4: (1, 4)}

    @property
    def world_size(self) -> int:
        return 4

    @property
    def backend(self) -> str:
        return "gloo"

    @property
    def device_type(self) -> str:
        return "cpu"

    def _real_shape(self, label: str) -> tuple[int, int]:
        if label == "wqkv":
            return (3 * self._N_HEADS * self._HEAD_DIM, _DIM)
        return (_DIM, self._N_HEADS * self._HEAD_DIM)

    def _build(self, tp: int) -> tuple[GQAttention, OptimizersContainer, EMA]:
        attention = _build_attention(
            _attention_config(
                self._N_HEADS,
                None,
                head_dim=self._HEAD_DIM,
                pad_heads_for_tp=True,
            ),
            tp=tp,
        )
        dp = 4 // tp
        mesh = init_device_mesh("cpu", self._MESHES[tp], mesh_dim_names=("dp", "tp"))
        # FSDP shards after TP, so a TP-sharded dim 0 gets _StridedShard on dp.
        fsdp = _StridedShard(0, split_factor=tp) if tp > 1 and dp > 1 else Shard(0)
        for linear, placements in (
            (attention.qkv_linear.wqkv, [fsdp, Shard(0)]),
            (attention.wo, [Shard(0), Shard(1)]),
        ):
            replicated = distribute_tensor(
                linear.weight.detach(), mesh, [Replicate(), Replicate()]
            )
            linear.weight = nn.Parameter(replicated.redistribute(mesh, placements))
        optimizer = OptimizersContainer.Config(
            optimizers=[AdamW.Config(pattern=r".*", lr=1e-2)]
        ).build(model_parts=[attention])
        optimizer.state_dict()  # materializes the AdamW states
        ema = EMA.Config(decays=[0.9]).build(model_parts=[attention])
        self._fill_states(optimizer, ema, attention)
        return attention, optimizer, ema

    def _live_states(
        self, optimizer: OptimizersContainer, ema: EMA, attention: GQAttention
    ) -> dict[str, DTensor]:
        states: dict[str, DTensor] = {}
        for label, param in (
            ("wqkv", attention.qkv_linear.wqkv.weight),
            ("wo", attention.wo.weight),
        ):
            for name, state in optimizer.optimizers[0].state[param].items():
                if name != "step":
                    states[f"adamw.{label}.{name}"] = state
            for name, state in ema.optimizers[0].state[param]["ema_params"].items():
                states[f"ema.{label}.{name}"] = state
        return states

    def _fill_states(
        self, optimizer: OptimizersContainer, ema: EMA, attention: GQAttention
    ) -> None:
        """Give the live states padding-independent values, with zero padding."""
        generator = torch.Generator().manual_seed(1)
        for key, state in self._live_states(optimizer, ema, attention).items():
            rows, columns = self._real_shape(key.split(".")[1])
            full = torch.zeros(state.shape)
            full[:rows, :columns] = torch.randn((rows, columns), generator=generator)
            mesh = state.device_mesh
            replicated = distribute_tensor(full, mesh, [Replicate()] * mesh.ndim)
            state.copy_(replicated.redistribute(mesh, state.placements))

    @with_comms
    @with_temp_dir
    def test_dcp_round_trip_across_tp_degrees(self):
        attention, optimizer, ema = self._build(tp=2)
        reference_states = {
            key: state.full_tensor()
            for key, state in self._live_states(optimizer, ema, attention).items()
        }
        reference_weights = {
            key: value.full_tensor() for key, value in attention.state_dict().items()
        }
        state_dict = {
            "model": attention.state_dict(),
            "optim": optimizer.state_dict(),
            "ema": ema.state_dict(),
        }
        # Hidden padding: unpadded global shapes, uneven plain Shard placements.
        for source in ("optim", "ema"):
            for key, value in state_dict[source].items():
                if isinstance(value, DTensor) and key.startswith("state."):
                    self.assertEqual(
                        value.shape, self._real_shape("wqkv" if "wqkv" in key else "wo")
                    )
                    self.assertFalse(
                        any(isinstance(p, _StridedShard) for p in value.placements), key
                    )
        dcp.save(state_dict, checkpoint_id=self.temp_dir)

        # tp=2 reloads with the same padding, tp=1 without any, tp=4 with more.
        for tp in (2, 1, 4):
            attention, optimizer, ema = self._build(tp)
            with torch.no_grad():
                for state in self._live_states(optimizer, ema, attention).values():
                    state.zero_()
            state_dict = {
                "model": attention.state_dict(),
                "optim": optimizer.state_dict(),
                "ema": ema.state_dict(),
            }
            dcp.load(state_dict, checkpoint_id=self.temp_dir)
            attention.load_state_dict(state_dict["model"])
            optimizer.load_state_dict(state_dict["optim"])
            ema.load_state_dict(state_dict["ema"])

            weights = {
                "wqkv": attention.qkv_linear.wqkv.weight,
                "wo": attention.wo.weight,
            }
            for key, value in reference_weights.items():
                label = "wqkv" if "wqkv" in key else "wo"
                param = weights[label]
                rows, columns = self._real_shape(label)
                full = param.full_tensor()
                torch.testing.assert_close(
                    full[:rows, :columns], value, msg=f"tp={tp} {key}"
                )
                self.assertEqual(full[rows:].count_nonzero(), 0, f"tp={tp} {key}")
                self.assertEqual(full[:, columns:].count_nonzero(), 0, f"tp={tp} {key}")
            for key, state in self._live_states(optimizer, ema, attention).items():
                label = key.split(".")[1]
                param = weights[label]
                self.assertEqual(state.shape, param.shape, f"tp={tp} {key}")
                self.assertEqual(state.placements, param.placements, f"tp={tp} {key}")
                rows, columns = self._real_shape(label)
                full = state.full_tensor()
                torch.testing.assert_close(
                    full[:rows, :columns],
                    reference_states[key][:rows, :columns],
                    msg=f"tp={tp} {key}",
                )
                self.assertEqual(full[rows:].count_nonzero(), 0, f"tp={tp} {key}")
                self.assertEqual(full[:, columns:].count_nonzero(), 0, f"tp={tp} {key}")


if __name__ == "__main__":
    unittest.main()
