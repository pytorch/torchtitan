# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import Mock

import pytest
import torch

from torchtitan.tools.garbage_collector import GarbageCollector
from torchtitan.tools.utils import (
    activate_cuda_flash_attention,
    get_cuda_flash_attention_impl,
    get_local_device,
    validate_flash_attention_head_dims,
)


class _FakeDeviceModule:
    def __init__(self, num_devices: int):
        self.num_devices = num_devices

    def device_count(self) -> int:
        return self.num_devices


def test_gc_debug_collects_once_per_step(monkeypatch: pytest.MonkeyPatch) -> None:
    gc_collect = Mock()
    monkeypatch.setattr("torchtitan.tools.garbage_collector.gc.collect", gc_collect)
    garbage_collector = GarbageCollector.__new__(GarbageCollector)
    garbage_collector.debug = True

    for step in (1, 2):
        assert garbage_collector.run(step)
        gc_collect.assert_called_once_with(2)
        gc_collect.reset_mock()


def test_get_local_device_uses_local_rank_when_multiple_devices_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr("torchtitan.tools.utils.device_type", "cuda")
    monkeypatch.setattr("torchtitan.tools.utils.device_module", _FakeDeviceModule(8))

    assert get_local_device() == torch.device("cuda:3")


def test_get_local_device_uses_zero_when_one_device_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr("torchtitan.tools.utils.device_type", "cuda")
    monkeypatch.setattr("torchtitan.tools.utils.device_module", _FakeDeviceModule(1))

    assert get_local_device() == torch.device("cuda:0")


def test_get_local_device_rejects_out_of_range_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "8")
    monkeypatch.setattr("torchtitan.tools.utils.device_type", "cuda")
    monkeypatch.setattr("torchtitan.tools.utils.device_module", _FakeDeviceModule(8))

    with pytest.raises(ValueError, match="outside the visible cuda device count"):
        get_local_device()


@pytest.mark.parametrize(
    ("capability", "expected_impl"),
    [
        ((8, 0), None),
        ((9, 0), "FA3"),
        ((9, 1), "FA3"),
        ((10, 0), "FA4"),
        ((10, 3), "FA4"),
        ((11, 0), None),
    ],
)
def test_get_cuda_flash_attention_impl(monkeypatch, capability, expected_impl):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())

    assert get_cuda_flash_attention_impl() == expected_impl


def test_get_cuda_flash_attention_impl_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert get_cuda_flash_attention_impl() is None


def test_get_cuda_flash_attention_impl_on_rocm(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "7.0")

    assert get_cuda_flash_attention_impl() is None


@pytest.fixture
def cuda_flash_attention(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr("torch.nn.attention.current_flash_attention_impl", lambda: None)
    monkeypatch.setattr("torchtitan.observability.logging._logged", set())
    activate = Mock()
    monkeypatch.setattr("torch.nn.attention.activate_flash_attention_impl", activate)
    return activate


@pytest.mark.parametrize(
    ("capability", "installed", "expected"),
    [
        ((9, 0), set(), None),
        ((10, 0), set(), None),
        ((9, 0), {"flash_attn.cute.interface"}, None),
        ((10, 0), {"flash_attn_interface"}, None),
        ((9, 0), {"flash_attn_interface"}, "FA3"),
        ((10, 0), {"flash_attn.cute.interface"}, "FA4"),
    ],
)
def test_flash_attention_selection_respects_installed_wheels(
    monkeypatch, cuda_flash_attention, capability, installed, expected
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    monkeypatch.setattr(
        "importlib.util.find_spec", lambda name: object() if name in installed else None
    )
    assert get_cuda_flash_attention_impl() == expected


@pytest.mark.parametrize("missing", ["flash_attn", "flash_attn.cute"])
def test_flash_attention_missing_parent_package(
    monkeypatch, cuda_flash_attention, missing
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 0))
    monkeypatch.setattr(
        "importlib.util.find_spec",
        Mock(side_effect=ModuleNotFoundError(name=missing)),
    )
    assert get_cuda_flash_attention_impl() is None


@pytest.mark.parametrize(
    ("capability", "impl", "module"),
    [
        ((9, 0), "FA3", "flash_attn_interface"),
        ((10, 0), "FA4", "flash_attn.cute.interface"),
    ],
)
def test_flash_attention_activation(
    monkeypatch, cuda_flash_attention, capability, impl, module
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    find_spec = Mock(return_value=object())
    monkeypatch.setattr("importlib.util.find_spec", find_spec)
    activate_cuda_flash_attention()
    find_spec.assert_called_once_with(module)
    cuda_flash_attention.assert_called_once_with(impl)


def test_missing_flash_attention_warns_once(monkeypatch, cuda_flash_attention, caplog):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    activate_cuda_flash_attention()
    activate_cuda_flash_attention()
    cuda_flash_attention.assert_not_called()
    assert len(caplog.records) == 1
    assert "flash_attn_interface" in caplog.text
    assert "SM 9.0; using FA2" in caplog.text


@pytest.mark.parametrize("active", ["FA3", "custom"])
def test_flash_attention_preserves_active_backend(
    monkeypatch, cuda_flash_attention, active, caplog
):
    monkeypatch.setattr(
        "torch.nn.attention.current_flash_attention_impl", lambda: active
    )
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    activate_cuda_flash_attention()
    cuda_flash_attention.assert_not_called()
    assert not caplog.records


@pytest.mark.parametrize(
    "error",
    [
        ImportError("undefined symbol"),
        RuntimeError("missing kernels"),
        ValueError("unknown impl"),
    ],
)
def test_broken_flash_attention_activation_is_not_a_fallback(
    monkeypatch, cuda_flash_attention, error
):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    cuda_flash_attention.side_effect = error
    with pytest.raises(RuntimeError, match="FA3 activation failed on SM 9.0") as caught:
        activate_cuda_flash_attention()
    assert caught.value.__cause__ is error
    assert "flash_attn_interface" in str(caught.value)


def test_missing_backend_dependency_is_not_a_fallback(
    monkeypatch, cuda_flash_attention
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 0))
    error = ModuleNotFoundError(name="flash_attn_2_cuda")
    monkeypatch.setattr("importlib.util.find_spec", Mock(side_effect=error))
    with pytest.raises(
        RuntimeError, match="FA4 activation failed on SM 10.0"
    ) as caught:
        activate_cuda_flash_attention()
    assert caught.value.__cause__ is error
    cuda_flash_attention.assert_not_called()


@pytest.mark.parametrize("hip", [None, "7.0"])
def test_flash_attention_does_not_probe_non_cuda_backends(monkeypatch, hip):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: hip is not None)
    monkeypatch.setattr(torch.version, "hip", hip)
    capability = Mock(side_effect=AssertionError("must not query CUDA capability"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    assert get_cuda_flash_attention_impl() is None
    activate_cuda_flash_attention()
    validate_flash_attention_head_dims(192, v_head_dim=128)


@pytest.mark.parametrize("active", [None, "FA2"])
@pytest.mark.parametrize("capability", [(8, 0), (9, 0), (10, 0), (11, 0)])
def test_fa2_rejects_unequal_head_dims(
    monkeypatch, cuda_flash_attention, active, capability
):
    monkeypatch.setattr(
        "torch.nn.attention.current_flash_attention_impl", lambda: active
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    with pytest.raises(ValueError, match="Q/K=192, V=128") as caught:
        validate_flash_attention_head_dims(192, v_head_dim=128)
    message = str(caught.value)
    assert f"SM {capability[0]}.{capability[1]}" in message
    if capability[0] in (9, 10):
        assert "Install a compatible" in message
    else:
        assert "Install" not in message


@pytest.mark.parametrize("active", [None, "FA2", "FA3", "FA4"])
def test_supported_flash_attention_head_dims(monkeypatch, cuda_flash_attention, active):
    monkeypatch.setattr(
        "torch.nn.attention.current_flash_attention_impl", lambda: active
    )
    validate_flash_attention_head_dims(128, v_head_dim=128)
    if active in ("FA3", "FA4"):
        validate_flash_attention_head_dims(192, v_head_dim=128)
