# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantization model-config converters."""

import logging
from dataclasses import dataclass, field, fields
from functools import partial
from importlib.util import find_spec
from typing import Literal

import torch
import torch._inductor.config

from torchtitan.models.common.linear import Linear, RouterGateLinear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.quantization.float8 import _get_float8_grouped_experts_cls, Float8Linear
from torchtitan.quantization.mxfp8 import _mxfp8_linear_import_error, MXFP8Linear
from torchtitan.quantization.mxfp8.experts import _get_mxfp8_grouped_experts_cls
from torchtitan.quantization.nvfp4 import NVFP4Linear
from torchtitan.quantization.utils import module_filter_fn, swap_token_dispatcher
from torchtitan.tools.utils import has_cuda_capability, has_rocm_capability


logger = logging.getLogger(__name__)


class QuantizationConverter(ModelConfigConverter):
    """Base class for quantization converters.

    Subclasses define a nested Config and implement ``convert()``
    to transform the model config tree.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        model_compile_enabled: bool = False
        """Whether torch.compile is enabled for the model."""


class Float8LinearConverter(QuantizationConverter):
    """Replace matching Linear.Config with Float8Linear.Config."""

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        recipe_name: Literal["rowwise", "rowwise_with_gw_hp"] = "rowwise"
        """Float8 recipe name."""

        filter_fqns: list[str] = field(default_factory=list)
        """
        List of fully qualified names of modules to skip applying float8 training to.
        nn.Linear modules with any dim size not divisible by 16 are always skipped
        due to hardware requirements.
        """

        emulate: bool = False
        """
        If True, emulation is used instead of hardware accelerated gemm.
        This is for test purpose only. Not compatible with torch.compile.
        """

    def __init__(self, config: Config):
        self.config = config

        if Float8Linear is None:
            raise ImportError(
                "torchao is not installed. Please install it to use float8 linear layers."
            )

        cfg = self.config
        filter_fqns = cfg.filter_fqns

        if (
            has_cuda_capability(8, 9)
            or has_rocm_capability(9, 4)
            or (cfg.emulate and not cfg.model_compile_enabled)
        ):
            pass
        else:
            raise ValueError(
                "Failed to swap to Float8Linear because float8 is only supported on "
                "NVIDIA SM89 or later, or AMD gfx942 (MI300) or later. "
                "To enable testing on older hardware, set `float8.emulate` to True in eager mode.",
            )

        try:
            from torchao.float8 import Float8LinearConfig as TorchAOFloat8LinearConfig
        except ImportError as e:
            raise ImportError(
                "torchao is not installed. Please install it to use float8 linear layers."
            ) from e

        if not hasattr(TorchAOFloat8LinearConfig, "from_recipe_name"):
            logger.warning(
                "Failed to use Float8 with recipe lookup because the torchao version "
                "is too old, please install torchao v0.9.0 or later and try again",
            )
            self.enabled = False
            return

        self.torchao_config = TorchAOFloat8LinearConfig.from_recipe_name(
            cfg.recipe_name
        )
        if cfg.emulate:
            self.torchao_config = TorchAOFloat8LinearConfig(emulate=True)
        logger.info(f"Float8 training active with recipe {cfg.recipe_name}")

        # short-term solution for https://github.com/pytorch/pytorch/issues/150859
        if cfg.recipe_name == "rowwise":
            torch._inductor.config.emulate_precision_casts = True
            logger.debug("Set torch._inductor.config.emulate_precision_casts to True")

        # Build filter function
        clean_fqns = [f for f in filter_fqns if f != "auto_filter_small_kn"]
        use_auto_filter = "auto_filter_small_kn" in filter_fqns
        if use_auto_filter:
            try:
                from torchao.float8 import _auto_filter_for_recipe

                logger.info(
                    "Using _auto_filter_for_recipe to avoid converting linear layers "
                    "with dims too small to benefit from float8 training. "
                    "See torchtitan/quantization/float8.md for more info."
                )
                self.filter_fn = _auto_filter_for_recipe(
                    cfg.recipe_name, filter_fqns=clean_fqns
                )
            except ImportError:
                logger.warning(
                    "Using default module_filter_fn for float8 model conversion. "
                    "To use _auto_filter_for_recipe, please install torchao nightly build."
                )
                self.filter_fn = partial(module_filter_fn, filter_fqns=clean_fqns)
        else:
            self.filter_fn = partial(module_filter_fn, filter_fqns=clean_fqns)

        self.enabled = True

    def convert(self, model_config):
        if not self.enabled:
            return model_config

        assert Float8Linear is not None
        for fqn, linear_config, parent, attr in model_config.traverse(Linear.Config):
            if self.filter_fn(linear_config, fqn):
                if isinstance(linear_config, RouterGateLinear.Config):
                    raise ValueError(
                        f"Float8 quantization does not support router gate {fqn!r}; "
                        "exclude it with filter_fqns."
                    )
                new_config = Float8Linear.Config(
                    in_features=linear_config.in_features,
                    out_features=linear_config.out_features,
                    bias=linear_config.bias,
                    param_init=linear_config.param_init,
                    _torchao_config=self.torchao_config,
                )
                if parent is None:
                    model_config = new_config
                elif isinstance(parent, list):
                    parent[attr] = new_config
                else:
                    setattr(parent, attr, new_config)

        logger.info("Swapped to Float8Linear layers")
        return model_config


class Float8GroupedExpertsConverter(QuantizationConverter):
    """Apply FP8 quantization to MoE expert grouped GEMMs."""

    # FP8: 16 byte alignment / 1 byte per elem = 16 elements.
    PAD_MULTIPLE = 16

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        pass

    def __init__(self, config: Config):
        self.config = config

        if find_spec("torchao") is None:
            raise ImportError(
                "torchao is not installed. Please install it to use float8 MoE training."
            )

        if not (has_cuda_capability(8, 9) or has_rocm_capability(9, 4)):
            raise ValueError(
                "Float8 MoE training only supported on NVIDIA SM89 or later, "
                "or AMD gfx942 (MI300) or later."
            )

        if not self.config.model_compile_enabled:
            logger.warning(
                "Compile is required for high performance float8 MoE training; "
                "enable it with --compile.enable"
            )

    def convert(self, model_config):
        for _fqn, config, parent, attr in model_config.traverse(GroupedExperts.Config):
            swap_token_dispatcher(parent, self.PAD_MULTIPLE)
            base_module_cls = type(config)._owner
            quantized_cls = _get_float8_grouped_experts_cls(base_module_cls)
            config_cls = quantized_cls.Config  # type: ignore[attr-defined]
            new_config = config_cls(
                **{f.name: getattr(config, f.name) for f in fields(config)},
            )
            if parent is None:
                model_config = new_config
            elif isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)

        logger.info(
            "Converted GroupedExperts to use dynamic float8 rowwise quantization "
            "with scaled grouped GEMMs"
        )
        return model_config


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

        quantized_router_fqns = [
            fqn
            for fqn, config, _parent, _attr in targets
            if isinstance(config, RouterGateLinear.Config)
        ]
        if quantized_router_fqns:
            raise ValueError(
                "MXFP8 quantization does not support router gates; exclude "
                f"{quantized_router_fqns} with fqns."
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
            new_config = MXFP8Linear.Config(
                in_features=config.in_features,
                out_features=config.out_features,
                bias=config.bias,
                param_init=config.param_init,
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


class MXFP8GroupedExpertsConverter(QuantizationConverter):
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

        if not self.config.model_compile_enabled:
            logger.warning(
                "torch.compile enablement is required for highest performance "
                "of MXFP8 dynamic quantization."
            )

    def convert(self, model_config):
        for _fqn, config, parent, attr in model_config.traverse(GroupedExperts.Config):
            # ``parent`` is the RoutedExperts.Config owning inner_experts + dispatcher.
            swap_token_dispatcher(parent, self.config.pad_multiple)
            base_module_cls = type(config)._owner
            quantized_cls = _get_mxfp8_grouped_experts_cls(base_module_cls)
            config_cls = quantized_cls.Config  # type: ignore[attr-defined]
            new_config = config_cls(
                **{f.name: getattr(config, f.name) for f in fields(config)},
                recipe_name=self.config.recipe_name,
            )
            if parent is None:
                model_config = new_config
            elif isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)

        logger.info(
            f"Converted GroupedExperts to use dynamic {self.config.recipe_name} "
            "quantization for grouped_mm ops"
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

        if not self.config.model_compile_enabled:
            logger.warning(
                "torch.compile enablement is required for highest performance "
                "of NVFP4 dynamic quantization."
            )

    def convert(self, model_config):
        assert NVFP4Linear is not None
        fqns = self.config.fqns
        for fqn, config, parent, attr in model_config.traverse(Linear.Config):
            if not fqns or any(target_fqn in fqn for target_fqn in fqns):
                if isinstance(config, RouterGateLinear.Config):
                    raise ValueError(
                        f"NVFP4 quantization does not support router gate {fqn!r}; "
                        "exclude it with fqns."
                    )
                new_config = NVFP4Linear.Config(
                    in_features=config.in_features,
                    out_features=config.out_features,
                    bias=config.bias,
                    param_init=config.param_init,
                )
                if parent is None:
                    model_config = new_config
                elif isinstance(parent, list):
                    parent[attr] = new_config
                else:
                    setattr(parent, attr, new_config)

        logger.info("Converted Linear layers to NVFP4Linear")
        return model_config
