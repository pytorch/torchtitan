# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Single entry point that registers both the TorchTitan model class and the
TorchTitan custom config parser with vLLM. Both registrations are wrapped in
dynamic classes that capture ``model_spec`` via closure — no global state.

Usage:
    from torchtitan.experiments.rl.models.vllm_registry import registry
    from torchtitan.experiments.rl.models.vllm_config_parser import (
        VLLM_MODEL_NAME, TORCHTITAN_CONFIG_FORMAT,
    )

    registry(model_spec, compile_config)
    # then construct EngineArgs(config_format=TORCHTITAN_CONFIG_FORMAT, ...)
"""

from __future__ import annotations

import dataclasses
from typing import Any

from torchtitan.config.configs import CompileConfig
from torchtitan.experiments.rl.models.vllm_config_parser import (
    add_custom_fields_to_config_dict,
    model_spec_to_hf_config_dict,
    TORCHTITAN_CONFIG_FORMAT,
    VLLM_MODEL_NAME,
)
from torchtitan.protocols.model_spec import ModelSpec


def registry(
    model_spec: ModelSpec,
    compile_config: CompileConfig,
) -> None:
    """Register the TorchTitan model class and the TorchTitan config parser with vLLM.

    Single entry point for vLLM integration. Must be called before creating
    a vLLM engine that uses a TorchTitan model. Registers two things:

      1. ``TorchTitanVLLMModelFromSpec`` (subclass of
         ``TorchTitanVLLMModelWrapper``) with vLLM's ``ModelRegistry`` under
         the name ``VLLM_MODEL_NAME``. This is what vLLM instantiates after
         it resolves the architecture from the parsed config.
      2. ``TorchTitanConfigParserForSpec`` (subclass of ``ConfigParserBase``)
         with vLLM's parser registry under ``TORCHTITAN_CONFIG_FORMAT``.
         This is what produces the ``PretrainedConfig`` from the torchtitan
         ``ModelSpec``.

    Args:
        model_spec: TorchTitan ModelSpec containing model config and components
        compile_config: Per-layer torch.compile config. When enabled, each
            TransformerBlock is compiled individually via ``apply_compile``
            during model construction.
    """
    from torchtitan.experiments.rl.models.vllm_wrapper import TorchTitanVLLMModelWrapper
    from transformers import PretrainedConfig
    from vllm.logger import init_logger
    from vllm.model_executor.models.registry import ModelRegistry
    from vllm.transformers_utils.config import register_config_parser
    from vllm.transformers_utils.config_parser_base import ConfigParserBase

    logger = init_logger(__name__)

    # Registration-time torchtitan-specific fields stamped onto the resulting
    # ``PretrainedConfig``. For per-engine runtime flags, callers should use
    # ``EngineArgs(hf_overrides={...})`` instead.
    #
    # CompileConfig is converted to a dict here (rather than stored as a
    # dataclass instance) so the resulting ``PretrainedConfig`` stays
    # JSON-serializable — vLLM's telemetry / caching paths sometimes call
    # ``hf_config.to_json()`` and a dataclass attribute would crash that.
    # Wrapper reconstructs via ``CompileConfig(**hf_config.compile_config)``.
    custom_hf_config_fields: dict[str, Any] = {
        "compile_config": dataclasses.asdict(compile_config),
    }

    # Dynamic model class capturing ModelSpec in the closure. The wrapper
    # reconstructs CompileConfig from hf_config (set by the ConfigParser
    # below), so we don't need to forward it through the constructor.
    class TorchTitanVLLMModelFromSpec(TorchTitanVLLMModelWrapper):
        def __init__(self, *, vllm_config, prefix=""):
            super().__init__(
                model_spec=model_spec,
                vllm_config=vllm_config,
                prefix=prefix,
            )

    TorchTitanVLLMModelFromSpec.__name__ = VLLM_MODEL_NAME
    TorchTitanVLLMModelFromSpec.__qualname__ = VLLM_MODEL_NAME
    ModelRegistry.register_model(VLLM_MODEL_NAME, TorchTitanVLLMModelFromSpec)

    # Dynamic config parser class capturing ModelSpec (and any registration-
    # time custom fields) in the closure.
    @register_config_parser(TORCHTITAN_CONFIG_FORMAT)
    class TorchTitanConfigParserForSpec(ConfigParserBase):
        def parse(
            self,
            model,
            trust_remote_code,
            revision=None,
            code_revision=None,
            **kwargs,
        ):
            config_dict = model_spec_to_hf_config_dict(model_spec)
            config_dict = add_custom_fields_to_config_dict(
                config_dict, **custom_hf_config_fields
            )
            return config_dict, PretrainedConfig.from_dict(config_dict)

    logger.info(
        f"Registered {VLLM_MODEL_NAME} + ConfigParser({TORCHTITAN_CONFIG_FORMAT!r}) "
        f"with vLLM (model={model_spec.name}, flavor={model_spec.flavor})"
    )
