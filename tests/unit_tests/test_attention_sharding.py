# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace

from torchtitan.distributed.context_parallel import ContextParallelMethod
from torchtitan.models.common.attention_sharding import (
    _validate_context_parallel,
    set_inner_attention_config,
)


def _attention_cfg(n_heads, n_kv_heads=None):
    return SimpleNamespace(
        inner_attention=SimpleNamespace(sharding_config=None),
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
    )


def _in_dst(attention_cfg, name):
    sharding_config = attention_cfg.inner_attention.sharding_config
    return sharding_config.in_dst_shardings[name].per_axis_spmd_types()


def _validate(
    attention_cfg,
    cp_method,
    *,
    tp,
    cp,
    spmd_backend="full_dtensor",
    load_balancer="headtail",
):
    _validate_context_parallel(
        attention_cfg,
        cp_method,
        tp=tp,
        cp=cp,
        spmd_backend=spmd_backend,
        load_balancer=load_balancer,
    )


class TestValidateContextParallel(unittest.TestCase):
    def test_ulysses_rejects_indivisible_heads(self):
        with self.assertRaises(ValueError):
            _validate(_attention_cfg(16, 8), "ulysses", tp=1, cp=3)

    def test_ulysses_accepts_divisible_heads(self):
        _validate(_attention_cfg(16, 8), "ulysses", tp=1, cp=4)

    def test_ulysses_divisor_is_tp_times_cp(self):
        # 2 heads divisible by tp=2 and cp=2 individually, but not by tp*cp=4.
        with self.assertRaises(ValueError):
            _validate(_attention_cfg(2, 2), "ulysses", tp=2, cp=2)

    def test_ulysses_checks_kv_heads(self):
        # 24 q heads divisible by 4; 6 kv heads not.
        with self.assertRaises(ValueError):
            _validate(_attention_cfg(24, 6), "ulysses", tp=1, cp=4)

    def test_ulysses_requires_spmd_backend(self):
        with self.assertRaises(ValueError):
            _validate(
                _attention_cfg(16, 8), "ulysses", tp=1, cp=4, spmd_backend="default"
            )

    def test_ulysses_accepts_spmd_types_backend(self):
        _validate(
            _attention_cfg(16, 8), "ulysses", tp=1, cp=4, spmd_backend="spmd_types"
        )

    def test_ulysses_warns_on_non_default_load_balancer(self):
        with self.assertLogs(level="WARNING") as captured:
            _validate(
                _attention_cfg(16, 8), "ulysses", tp=1, cp=4, load_balancer="ptrr"
            )
        self.assertTrue(any("ptrr" in msg for msg in captured.output))

    def test_ulysses_silent_on_default_load_balancer(self):
        with self.assertNoLogs(level="WARNING"):
            _validate(_attention_cfg(16, 8), "ulysses", tp=1, cp=4)

    def test_allgather_never_validates(self):
        _validate(
            _attention_cfg(16, 8), "allgather", tp=1, cp=3, spmd_backend="default"
        )

    def test_empty_never_validates(self):
        _validate(_attention_cfg(16, 8), "", tp=1, cp=3, spmd_backend="default")

    def test_mla_without_kv_heads_falls_back_to_n_heads(self):
        mla = SimpleNamespace(
            inner_attention=SimpleNamespace(sharding_config=None), n_heads=16
        )
        _validate(mla, "ulysses", tp=1, cp=4)
        with self.assertRaises(ValueError):
            _validate(mla, "ulysses", tp=1, cp=3)


class TestSetInnerAttentionConfig(unittest.TestCase):
    def test_each_method_sets_a_sharding_config(self):
        for cp_method in ("", "allgather", "ulysses"):
            attn = _attention_cfg(16, 8)
            set_inner_attention_config(attn, cp_method)
            self.assertIsNotNone(attn.inner_attention.sharding_config)

    def test_empty_maps_to_allgather(self):
        empty = _attention_cfg(16, 8)
        allgather = _attention_cfg(16, 8)
        set_inner_attention_config(empty, "")
        set_inner_attention_config(allgather, ContextParallelMethod.ALLGATHER)
        self.assertEqual(_in_dst(empty, "k_BLNH"), _in_dst(allgather, "k_BLNH"))

    def test_ulysses_differs_from_allgather(self):
        ulysses = _attention_cfg(16, 8)
        allgather = _attention_cfg(16, 8)
        set_inner_attention_config(ulysses, "ulysses")
        set_inner_attention_config(allgather, "allgather")
        self.assertNotEqual(_in_dst(ulysses, "q_BLNH"), _in_dst(allgather, "q_BLNH"))


if __name__ == "__main__":
    unittest.main()
