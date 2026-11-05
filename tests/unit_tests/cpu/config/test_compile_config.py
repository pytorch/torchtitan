# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from torchtitan.components.loss import compute_logprobs, cross_entropy_loss, mse_loss
from torchtitan.distributed.local_compile import local_compile, LocalCompileConfig
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.common.attention.attention import _compiled_create_block_mask
from torchtitan.models.common.moe.moe import _microbatch_load_balance_local_stats
from torchtitan.models.common.vision_encoder import compiled_create_block_mask


class CustomDeviceLocalCompileConfig(LocalCompileConfig):
    """Example configuration supplied by an out-of-tree backend."""

    @staticmethod
    def _compile(
        reference: Callable[..., Any],
        **compile_kwargs: Any,
    ) -> Callable[..., Any]:
        compile_kwargs.pop("options", None)
        compile_kwargs["backend"] = "custom_device_backend"
        return torch.compile(reference, **compile_kwargs)


@pytest.fixture(autouse=True)
def reset_local_compile():
    LocalCompileConfig().apply()
    yield
    LocalCompileConfig().apply()


def test_local_compile_config_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match=r"foo.*registered values"):
        LocalCompileConfig(regions=["foo"]).apply()


def test_local_compile_binds_once_from_existing_config(monkeypatch) -> None:
    compiled_calls = []

    @local_compile("test_local_compile", batch_invariant=False)
    def fn(value: int) -> tuple[str, int]:
        return "eager", value

    def fake_compile(reference, **kwargs):
        compiled_calls.append((reference, kwargs))

        def compiled(value: int) -> tuple[str, int]:
            return "compiled", value

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    LocalCompileConfig(regions=["test_local_compile"]).apply()

    assert fn(3) == ("compiled", 3)
    named_calls = [call for call in compiled_calls if call[0] is fn.__wrapped__]
    assert len(named_calls) == 1
    assert "backend" not in named_calls[0][1]
    assert named_calls[0][1]["fullgraph"] is True


def test_local_compile_forwards_compile_kwargs(monkeypatch) -> None:
    compiled_calls = []
    options = {"max_autotune": True}

    @local_compile(
        "test_compile_kwargs",
        batch_invariant=False,
        backend="eager",
        dynamic=True,
        fullgraph=True,
        options=options,
    )
    def fn(value: int) -> int:
        return value

    def fake_compile(reference, **kwargs):
        compiled_calls.append((reference, kwargs))
        return reference

    monkeypatch.setattr(torch, "compile", fake_compile)
    LocalCompileConfig(regions=["test_compile_kwargs"]).apply()

    assert fn(3) == 3
    named_calls = [call for call in compiled_calls if call[0] is fn.__wrapped__]
    assert named_calls[0][1] == {
        "backend": "eager",
        "fullgraph": True,
        "dynamic": True,
        "options": options,
    }


def test_local_compile_rejects_fullgraph_false() -> None:
    with pytest.raises(ValueError, match="fullgraph=True"):
        local_compile(
            "test_fullgraph",
            batch_invariant=False,
            fullgraph=False,
        )


def test_local_compile_rejects_non_batch_invariant_region(monkeypatch) -> None:
    @local_compile("test_non_batch_invariant", batch_invariant=False)
    def fn(value: int) -> int:
        return value

    monkeypatch.setattr(
        "torchtitan.distributed.local_compile.is_in_batch_invariant_mode",
        lambda: True,
    )
    with pytest.raises(
        ValueError, match=r"test_non_batch_invariant.*local_compile\.regions"
    ):
        LocalCompileConfig(regions=["test_non_batch_invariant"]).apply()


def test_loss_functions_use_local_compile(monkeypatch) -> None:
    compiled_names = []

    def fake_compile(reference, **kwargs):
        del kwargs
        compiled_names.append(reference.__name__)
        return reference

    monkeypatch.setattr(torch, "compile", fake_compile)
    LocalCompileConfig(regions=["loss"]).apply()

    assert [
        name
        for name in compiled_names
        if name
        in {
            cross_entropy_loss.__name__,
            mse_loss.__name__,
            compute_logprobs.__name__,
        }
    ] == [
        cross_entropy_loss.__name__,
        mse_loss.__name__,
        compute_logprobs.__name__,
    ]


def test_moe_aux_loss_uses_local_compile(monkeypatch) -> None:
    compiled = []

    def fake_compile(reference, **kwargs):
        compiled.append((reference, kwargs))
        return reference

    monkeypatch.setattr(torch, "compile", fake_compile)
    LocalCompileConfig(regions=["moe_aux_loss"]).apply()

    assert [
        call
        for call in compiled
        if call[0] is _microbatch_load_balance_local_stats.__wrapped__
    ] == [
        (
            _microbatch_load_balance_local_stats.__wrapped__,
            {"fullgraph": True},
        )
    ]


def test_custom_config_controls_all_torchtitan_compile_boundaries(
    monkeypatch,
) -> None:
    compiled_calls = []

    @local_compile("test_custom_backend", batch_invariant=False)
    def region() -> None:
        pass

    def fake_compile(reference, **kwargs):
        compiled_calls.append((reference, kwargs))
        return reference

    monkeypatch.setattr(torch, "compile", fake_compile)
    CustomDeviceLocalCompileConfig(regions=["test_custom_backend"]).apply()

    assert callable(FlexInnerAttention._compiled_flex_attn)
    assert callable(_compiled_create_block_mask)
    assert callable(compiled_create_block_mask)
    expected = {"backend": "custom_device_backend"}
    assert (region.__wrapped__, {**expected, "fullgraph": True}) in compiled_calls
    assert (
        flex_attention,
        {**expected, "fullgraph": True, "recompile_limit": 16},
    ) in compiled_calls
    assert (create_block_mask, expected) in compiled_calls
    assert sum(reference is create_block_mask for reference, _ in compiled_calls) == 2
