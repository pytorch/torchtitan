# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from torchtitan.components.loss import compute_logprobs, cross_entropy_loss, mse_loss
from torchtitan.distributed.local_compile import (
    apply_local_compile,
    compiled,
    local_compile,
    set_local_compile_backend,
)
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.common.attention.attention import _compiled_create_block_mask
from torchtitan.models.common.moe.moe import _microbatch_load_balance_local_stats
from torchtitan.models.common.vision_encoder import compiled_create_block_mask


@pytest.fixture(autouse=True)
def reset_local_compile():
    set_local_compile_backend("inductor")
    apply_local_compile([])
    yield
    set_local_compile_backend("inductor")
    apply_local_compile([])


def test_apply_local_compile_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match=r"foo.*registered values"):
        apply_local_compile(["foo"])


def test_local_compile_rebinds_active_region_when_backend_changes(monkeypatch) -> None:
    compiled_calls = []

    @local_compile("test_local_compile", batch_invariant=False)
    def fn(value: int) -> tuple[str, int]:
        return "eager", value

    def fake_compile(reference, **kwargs):
        compiled_calls.append((reference, kwargs))
        backend = kwargs["backend"]

        def compiled(value: int) -> tuple[str, int]:
            return backend, value

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    apply_local_compile(["test_local_compile"])

    assert fn(3) == ("inductor", 3)

    set_local_compile_backend("custom_device_backend")
    apply_local_compile(["test_local_compile"])

    assert fn(3) == ("custom_device_backend", 3)
    assert [kwargs for _, kwargs in compiled_calls] == [
        {"backend": "inductor", "fullgraph": True},
        {"backend": "custom_device_backend", "fullgraph": True},
    ]


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
    apply_local_compile(["test_compile_kwargs"])

    assert fn(3) == 3
    assert compiled_calls[0][1] == {
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
        ValueError, match=r"test_non_batch_invariant.*local_compile_regions"
    ):
        apply_local_compile(["test_non_batch_invariant"])


def test_loss_functions_use_local_compile(monkeypatch) -> None:
    compiled_names = []

    def fake_compile(reference, **kwargs):
        del kwargs
        compiled_names.append(reference.__name__)
        return reference

    monkeypatch.setattr(torch, "compile", fake_compile)
    apply_local_compile(["loss"])

    assert compiled_names == [
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
    apply_local_compile(["moe_aux_loss"])

    assert compiled == [
        (
            _microbatch_load_balance_local_stats.__wrapped__,
            {"backend": "inductor", "fullgraph": True},
        )
    ]


def test_compiled_helper_only_forwards_options_to_inductor(monkeypatch) -> None:
    compiled_calls = []
    options = {"max_autotune": False}

    def reference(value: int) -> int:
        return value

    def fake_compile(fn, **kwargs):
        compiled_calls.append((fn, kwargs))
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    helper = compiled(reference, inductor_options=options)
    set_local_compile_backend("custom_device_backend")

    assert compiled_calls == []
    assert helper(3) == 3
    set_local_compile_backend("inductor")
    assert helper(3) == 3
    assert compiled_calls == [
        (
            reference,
            {
                "backend": "custom_device_backend",
            },
        ),
        (
            reference,
            {
                "backend": "inductor",
                "options": options,
            },
        ),
    ]


def test_attention_helpers_do_not_forward_inductor_options_to_custom_backend(
    monkeypatch,
) -> None:
    compiled_calls = []

    def fake_compile(reference, **kwargs):
        if "options" in kwargs:
            raise TypeError("custom backend got unexpected keyword argument 'options'")
        compiled_calls.append((reference, kwargs))
        return lambda *args, **kwargs: None

    monkeypatch.setattr(torch, "compile", fake_compile)
    set_local_compile_backend("custom_device_backend")

    FlexInnerAttention._compiled_flex_attn()
    _compiled_create_block_mask()
    compiled_create_block_mask()

    assert compiled_calls == [
        (flex_attention, {"backend": "custom_device_backend"}),
        (create_block_mask, {"backend": "custom_device_backend"}),
        (create_block_mask, {"backend": "custom_device_backend"}),
    ]
