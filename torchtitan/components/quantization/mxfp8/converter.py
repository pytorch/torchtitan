# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field, fields

from torchtitan.components.quantization import QuantizationConverter
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability

from ..utils import swap_token_dispatcher
from ._common import _MXFP8_SCALE_GROUP_ALIGNMENT, InputActivationFormatForBackward

_mxfp8_linear_import_error: ImportError | None = None

try:
    from .grouped_experts import _mxfp8_experts_cache, get_mxfp8_grouped_experts_cls
    from .linear import MXFP8Linear

except ImportError as import_error:
    MXFP8Linear = None
    get_mxfp8_grouped_experts_cls = None
    _mxfp8_experts_cache: dict[type, type] = {}
    _mxfp8_linear_import_error = import_error


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
                "TorchAO with the MXFP8 32x32 swizzled cast kernels is required "
                "for MXFP8 linear layers. Install TorchAO from source."
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
        input_activation_format_for_backward: InputActivationFormatForBackward = "bf16"
        """Format used to save routed-expert input activations for WGRAD.

        See ``MXFP8GroupedExperts.Config`` for the trade-off. BF16 is the
        conservative default because the routed input feeds more than one
        expert projection and stays alive regardless.
        """
        pad_multiple: int = _MXFP8_SCALE_GROUP_ALIGNMENT
        """
        Pad per-expert token groups to this multiple for MXFP8 grouped GEMM alignment.

        Two separate constraints apply, and the larger one wins. Columnwise
        WGRAD quantization scales 32 rows together, so a scale block must not
        span two experts. The blocked scale layout consumed by the grouped GEMM
        additionally starts each group on a 128-row block boundary, so a group
        whose size is not a multiple of 128 would misalign the scales against
        the quantized data.

        The default was 32 while this converter delegated to TorchAO, which
        padded internally to whatever its kernels needed, so the value never
        took effect -- the docstring already said 128 was required and both
        in-tree configs passed it explicitly. Now that TorchTitan drives the
        quantization, an unpadded group faults inside the scale rearrange
        instead, so the default matches the requirement and __post_init__
        rejects anything smaller.
        """

        def __post_init__(self) -> None:
            if self.pad_multiple % _MXFP8_SCALE_GROUP_ALIGNMENT:
                raise ValueError(
                    "MXFP8 grouped experts require pad_multiple to be a multiple "
                    f"of {_MXFP8_SCALE_GROUP_ALIGNMENT}; got {self.pad_multiple}."
                )

    def __init__(self, config: Config):
        self.config = config

        if get_mxfp8_grouped_experts_cls is None:
            raise ImportError(
                "TorchAO with the MXFP8 32x32 swizzled cast kernels is required "
                "for MXFP8 grouped experts. Install TorchAO from source."
            ) from _mxfp8_linear_import_error

        if not has_cuda_capability(10, 0):
            raise ValueError("MXFP8 is only supported on SM100 or later architectures")

        if not self.config.model_compile_enabled:
            logger.warning(
                "torch.compile enablement is required for highest performance "
                "of MXFP8 dynamic quantization."
            )

    def convert(self, model_config):
        assert get_mxfp8_grouped_experts_cls is not None
        for _fqn, config, parent, attr in model_config.traverse(GroupedExperts.Config):
            # ``parent`` is the RoutedExperts.Config owning inner_experts + dispatcher.
            swap_token_dispatcher(parent, self.config.pad_multiple)
            base_module_cls = type(config)._owner
            quantized_cls = get_mxfp8_grouped_experts_cls(base_module_cls)
            config_cls = quantized_cls.Config  # type: ignore[attr-defined]
            new_config = config_cls(
                **{f.name: getattr(config, f.name) for f in fields(config)},
                input_activation_format_for_backward=(
                    self.config.input_activation_format_for_backward
                ),
            )
            if parent is None:
                model_config = new_config
            elif isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)

        logger.info(
            "Converted GroupedExperts to MXFP8 grouped GEMMs with FSDP-managed "
            "32x32 weight quantization and saved input activation format "
            f"{self.config.input_activation_format_for_backward}"
        )
        return model_config
