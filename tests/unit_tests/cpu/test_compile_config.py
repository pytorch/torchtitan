# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest

from torchtitan.config import CompileConfig


def test_compile_config_default() -> None:
    config = CompileConfig()
    assert config.enable is False
    assert config.components == ["model", "loss"]


def test_compile_config_model_only() -> None:
    config = CompileConfig(enable=True, components=["model"])
    assert config.enable is True
    assert config.components == ["model"]


def test_compile_config_loss_only() -> None:
    config = CompileConfig(enable=True, components=["loss"])
    assert config.enable is True
    assert config.components == ["loss"]


def test_compile_config_empty_components() -> None:
    config = CompileConfig(components=[])
    assert config.components == []


def test_compile_config_rejects_unknown_component() -> None:
    with pytest.raises(ValueError, match=r"foo.*allowed values are.*loss.*model"):
        CompileConfig(components=["foo"])


def test_compile_config_rejects_unknown_component_when_disabled() -> None:
    with pytest.raises(ValueError, match=r"nope.*allowed values are.*loss.*model"):
        CompileConfig(enable=False, components=["nope"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enable_async_tensor_parallel": True},
        {
            "enable": True,
            "enable_async_tensor_parallel": True,
            "components": ["loss"],
        },
    ],
)
def test_compile_config_async_tp_requires_model_compile(kwargs: dict) -> None:
    with pytest.raises(
        ValueError,
        match="Async TP requires 'model' in --compile.components and --compile.enable",
    ):
        CompileConfig(**kwargs)


def _async_tp_trainer_config(*, tensor_parallel_degree: int):
    from torchtitan.models.llama3.config_registry import llama3_debugmodel

    config = llama3_debugmodel()
    config.compile.enable = True
    config.compile.components = ["model"]
    config.compile.enable_async_tensor_parallel = True
    config.parallelism.tensor_parallel_degree = tensor_parallel_degree
    return config


def test_async_tp_warns_without_tensor_parallelism() -> None:
    config = _async_tp_trainer_config(tensor_parallel_degree=1)
    with patch("torchtitan.trainer.logger.warning") as mock_warning:
        config.__post_init__()

    mock_warning.assert_called_once()
    message = mock_warning.call_args.args[0]
    assert "async" in message.lower()
    assert "--parallelism.tensor_parallel_degree > 1" in message


def test_async_tp_does_not_warn_with_tensor_parallelism() -> None:
    config = _async_tp_trainer_config(tensor_parallel_degree=2)
    with patch("torchtitan.trainer.logger.warning") as mock_warning:
        config.__post_init__()

    for call in mock_warning.call_args_list:
        message = call.args[0]
        assert "--parallelism.tensor_parallel_degree > 1" not in message
