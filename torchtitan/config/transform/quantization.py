# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantization model-config converters."""

import logging
from dataclasses import dataclass, field, fields
from importlib.util import find_spec
from typing import Literal

import torch

from torchtitan.models.common.attention import QKVLinear
from torchtitan.models.common.linear import (
    ColumnParallelLinear,
    GroupedLinear,
    Linear,
    RowParallelLinear,
    SharedExpertRowParallelLinear,
)
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.vision_encoder import InvariantRowParallelLinear
from torchtitan.quantization.mxfp8 import _mxfp8_linear_import_error, MXFP8Linear
from torchtitan.quantization.mxfp8.experts import _get_mxfp8_grouped_linear_cls
from torchtitan.quantization.nvfp4 import NVFP4Linear
from torchtitan.quantization.utils import get_quantized_linear, swap_token_dispatcher
from torchtitan.tools.utils import has_cuda_capability

from .converter import ModelConfigConverter


logger = logging.getLogger(__name__)

_QUANTIZABLE_LINEAR_CLASSES = (
    Linear,
    ColumnParallelLinear,
    RowParallelLinear,
    SharedExpertRowParallelLinear,
    InvariantRowParallelLinear,
)


def _validate_quantizable_linear(
    config: Linear.Config,
    fqn: str,
) -> None:
    owner = config._owner
    assert owner is not None
    if owner not in _QUANTIZABLE_LINEAR_CLASSES:
        supported = ", ".join(cls.__qualname__ for cls in _QUANTIZABLE_LINEAR_CLASSES)
        raise ValueError(
            f"Quantization does not support {owner.__qualname__} at {fqn!r}; "
            f"supported Linear classes are {supported}."
        )


class QuantizationConverter(ModelConfigConverter):
    """Base class for quantization converters.

    Subclasses define a nested Config and implement ``convert()``
    to transform the model config tree.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        pass


def _torchao_nightly_install_command() -> str:
    """Return the pip command that installs a torchao nightly for this torch.

    torchao nightlies live on download.pytorch.org rather than PyPI and are
    published per accelerator build, so the channel has to come from the torch
    actually installed. ``--upgrade`` matters as much as ``--pre``: an existing
    but older nightly is the common case, and without it pip leaves it alone.
    ``USE_CPP=0`` skips building the C++ extensions, which MXFP8 does not need.
    """
    if torch.version.cuda:
        channel = "cu" + torch.version.cuda.replace(".", "")
    elif torch.version.hip:
        channel = "rocm" + ".".join(torch.version.hip.split(".")[:2])
    else:
        channel = "cpu"
    return (
        "USE_CPP=0 python -m pip install --pre --upgrade torchao "
        f"--index-url https://download.pytorch.org/whl/nightly/{channel}"
    )


class MXFP8LinearConverter(QuantizationConverter):
    """Replace matching Linear.Config with MXFP8Linear.Config."""

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        fqns: list[str] = field(default_factory=list)
        """
        List of fully qualified names of modules to apply MXFP8 quantization to.
        Only Linear.Config entries whose FQN contains a match are converted.
        If empty, all Linear modules are converted.
        """
        linears_saving_inputs_for_backward_in_mxfp8: list[str] = field(
            default_factory=list
        )
        """FQN substrings selecting linears that save inputs in MXFP8 for backward.

        A linear can save either its BF16 input or a columnwise MXFP8 input for
        the backward pass.

        Without activation checkpointing, if the preceding operation already
        saves its BF16 output for backward, as flash attention does, that tensor
        is also available as this linear's input. Saving another MXFP8
        operands would increase memory usage, so this linear should save
        BF16. If no other operation retains the BF16 input, saving MXFP8 reduces
        activation memory and avoids columnwise quantization during backward.

        With full activation checkpointing, saved tensors from the original
        forward are discarded and reconstructed during backward. Today, a
        linear selected here produces its columnwise MXFP8 input in both the
        original forward and recomputation, even though the original result is
        discarded. An ideal checkpoint-aware policy could produce it only
        during recomputation, but distinguishing those executions would add
        complexity to the linear and its autograd contract. We intentionally
        apply the same policy to both.

        More granular ``torch.remat`` policies add further save-versus-recompute
        choices, so the optimal format depends on both model activation
        ownership and the activation-checkpointing policy. BF16 is therefore
        the conservative default, and users can opt selected modules into MXFP8
        with this list.
        """

        def __post_init__(self) -> None:
            if any(not fqn for fqn in self.linears_saving_inputs_for_backward_in_mxfp8):
                raise ValueError(
                    "MXFP8 linears_saving_inputs_for_backward_in_mxfp8 cannot "
                    "contain an empty FQN selector."
                )

    def __init__(self, config: Config):
        self.config = config

        if MXFP8Linear is None:
            raise ImportError(
                "MXFP8 linear layers need torchao's 32x32 swizzled cast "
                "kernels, which are not in any release up to v0.18.0, so a "
                "nightly is required:\n\n"
                f"    {_torchao_nightly_install_command()}\n"
            ) from _mxfp8_linear_import_error

        if not has_cuda_capability(10, 0):
            raise ValueError("MXFP8 is only supported on SM100 or later architectures")

    def convert(self, model_config):
        assert MXFP8Linear is not None
        fqns = self.config.fqns
        targets = [
            entry
            for entry in model_config.traverse(Linear.Config)
            if not fqns or any(target_fqn in entry[0] for target_fqn in fqns)
        ]

        block_size = MXFP8Linear.WEIGHT_BLOCK_SIZE
        for fqn, _config, parent, _attr in targets:
            if isinstance(parent, QKVLinear.Config) and parent.head_dim % block_size:
                raise ValueError(
                    "MXFP8 quantization of fused QKV requires head_dim divisible "
                    f"by {block_size} so weight scale blocks do not span Q, K, "
                    f"or V; got {fqn!r} with head_dim={parent.head_dim}."
                )

        selectors = self.config.linears_saving_inputs_for_backward_in_mxfp8
        target_fqns = [fqn for fqn, _config, _parent, _attr in targets]
        unmatched_fqn_selectors = {
            selector
            for selector in selectors
            if not any(selector in fqn for fqn in target_fqns)
        }
        if unmatched_fqn_selectors:
            raise ValueError(
                "MXFP8 linears_saving_inputs_for_backward_in_mxfp8 selectors "
                "did not match any converted Linear.Config: "
                f"{sorted(unmatched_fqn_selectors)}."
            )

        mxfp8_fqns = {
            fqn for fqn in target_fqns if any(selector in fqn for selector in selectors)
        }
        for fqn, config, parent, attr in targets:
            _validate_quantizable_linear(config, fqn)
            owner = config._owner
            assert owner is not None and issubclass(owner, Linear)
            config_cls = get_quantized_linear(MXFP8Linear, owner).Config
            new_config = config_cls(
                in_features=config.in_features,
                out_features=config.out_features,
                num_linears=config.num_linears,
                bias=config.bias,
                param_init=config.param_init,
                sharding_config=config.sharding_config,
                input_activation_format_for_backward=(
                    "mxfp8" if fqn in mxfp8_fqns else "bf16"
                ),
            )
            if parent is None:
                model_config = new_config
            elif isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)

        num_mxfp8 = len(mxfp8_fqns)
        num_bf16 = len(targets) - num_mxfp8
        logger.info(
            "Converted Linear layers to MXFP8Linear with saved input activation "
            f"formats: {num_bf16} bf16, {num_mxfp8} mxfp8"
        )
        logger.debug(f"Linears saving MXFP8 input activations: {sorted(mxfp8_fqns)}")
        return model_config


class MXFP8GroupedLinearConverter(QuantizationConverter):
    """Apply MXFP8 quantization to MoE expert grouped GEMMs."""

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        recipe_name: Literal["mxfp8_rceil"] = "mxfp8_rceil"
        """
        Quantization recipe name for grouped GEMMs. Options: ["mxfp8_rceil"]

        - mxfp8_rceil: MXFP8 dynamic quantization with RCEIL rounding mode
          when computing the e8m0 scale factors.
        """
        pad_multiple: int = 32
        """
        Pad per-expert token groups to this multiple for MXFP8 grouped GEMM alignment.
        The CuTeDSL quantization kernel on sm_100 requires multiples of 128.
        """

    def __init__(self, config: Config):
        self.config = config

        if find_spec("torchao") is None:
            raise ImportError(
                "torchao is not installed. Please install it to use MXFP8 MoE training."
            )

        if not has_cuda_capability(10, 0):
            raise ValueError("MXFP8 is only supported on SM100 or later architectures")

    def convert(self, model_config):
        routed_configs: dict[int, RoutedExperts.Config] = {}
        for _fqn, config, parent, attr in model_config.traverse(GroupedLinear.Config):
            if not isinstance(parent, RoutedExperts.Config):
                raise ValueError("GroupedLinear must be owned by RoutedExperts")
            routed_configs[id(parent)] = parent
            base_module_cls = type(config)._owner
            quantized_cls = _get_mxfp8_grouped_linear_cls(base_module_cls)
            config_cls = quantized_cls.Config  # type: ignore[attr-defined]
            new_config = config_cls(
                **{f.name: getattr(config, f.name) for f in fields(config)},
                recipe_name=self.config.recipe_name,
            )
            setattr(parent, attr, new_config)

        for routed_config in routed_configs.values():
            swap_token_dispatcher(routed_config, self.config.pad_multiple)

        logger.info(
            f"Converted GroupedLinear modules to use dynamic {self.config.recipe_name} "
            "quantization for grouped_mm"
        )
        return model_config


class NVFP4LinearConverter(QuantizationConverter):
    """Replace matching Linear.Config with NVFP4Linear.Config."""

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        fqns: list[str] = field(default_factory=list)
        """
        List of fully qualified names of modules to apply NVFP4 quantization to.
        Only Linear.Config entries whose FQN contains a match are converted.
        If empty, all Linear modules are converted -- pass explicit fqns to keep
        the LM head in bf16, which the mixed recipe leaves unquantized for stability.
        """

    def __init__(self, config: Config):
        self.config = config

        if NVFP4Linear is None:
            raise ImportError(
                "torchao is not installed or does not provide the NVFP4 training "
                "prototype. Install a torchao build with "
                "torchao.prototype.moe_training.nvfp4_training."
            )

        if not has_cuda_capability(10, 0):
            raise ValueError("NVFP4 is only supported on SM100 or later architectures")

    def convert(self, model_config):
        assert NVFP4Linear is not None
        fqns = self.config.fqns
        for fqn, config, parent, attr in model_config.traverse(Linear.Config):
            if not fqns or any(target_fqn in fqn for target_fqn in fqns):
                _validate_quantizable_linear(config, fqn)
                owner = config._owner
                assert owner is not None and issubclass(owner, Linear)
                config_cls = get_quantized_linear(NVFP4Linear, owner).Config
                new_config = config_cls(
                    in_features=config.in_features,
                    out_features=config.out_features,
                    num_linears=config.num_linears,
                    bias=config.bias,
                    param_init=config.param_init,
                    sharding_config=config.sharding_config,
                )
                if parent is None:
                    model_config = new_config
                elif isinstance(parent, list):
                    parent[attr] = new_config
                else:
                    setattr(parent, attr, new_config)

        logger.info("Converted Linear layers to NVFP4Linear")
        return model_config
