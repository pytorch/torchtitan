# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-independent MX QAT selection and config transformation."""

from dataclasses import dataclass, field, replace

import torch

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.protocols.module import Module
from torchtitan.quantization.mx_qat.experts import (
    _get_mx_qat_grouped_experts_cls,
    activation_config,
    MXFakeQuantizeConfig,
    weight_config,
)
from torchtitan.quantization.mx_qat.linear import _get_mx_qat_linear_cls

from .base import convert_config_type, ModelConfigTransform


@dataclass(kw_only=True, slots=True)
class MXQATTransform(ModelConfigTransform):
    """Preserve BF16 masters and select QAT by exact module FQN.

    None selects every grouped-expert module; an empty tuple selects none.
    Dense projections are opt-in and use weight-only fake quantization.
    TorchAO configs, including kernel_preference, pass through unchanged.
    """

    grouped_expert_fqns: tuple[str, ...] | None = None
    linear_fqns: tuple[str, ...] = ()
    weight_fake_quant_config: "MXFakeQuantizeConfig" = field(
        default_factory=weight_config
    )
    activation_fake_quant_config: "MXFakeQuantizeConfig" = field(
        default_factory=activation_config
    )

    @classmethod
    def from_weight_fqns(cls, model: Module.Config, weight_fqns: set[str], **kwargs):
        """Translate adapter-resolved weights without assuming expert names.

        A grouped execution hook quantizes every expert matrix it consumes;
        reject partial selection instead of silently quantizing extra weights.
        Meta construction inspects registered shapes without allocating weights.
        """
        remaining = set(weight_fqns)
        groups, linears = [], []
        for fqn, config, _, _ in model.traverse(GroupedExperts.Config):
            with torch.device("meta"):
                module = config.build()
            weights = {
                f"{fqn}.{name}".lstrip(".")
                for name, parameter in module.named_parameters(recurse=False)
                if parameter.ndim == 3
            }
            if remaining & weights:
                if not weights <= remaining:
                    raise ValueError(
                        f"MX QAT requires all matrices in grouped module {fqn}"
                    )
                groups.append(fqn)
                remaining -= weights
        for fqn, _, _, _ in model.traverse(Linear.Config):
            key = f"{fqn}.weight".lstrip(".")
            if key in remaining:
                linears.append(fqn)
                remaining.remove(key)
        if remaining:
            raise ValueError(f"MX QAT cannot represent weights: {sorted(remaining)}")
        return cls(
            grouped_expert_fqns=tuple(groups), linear_fqns=tuple(linears), **kwargs
        )

    def transform(self, model: Module.Config) -> Module.Config:
        replacements = []
        missing = set()
        for config_type, targets, factory in (
            (
                GroupedExperts.Config,
                self.grouped_expert_fqns,
                _get_mx_qat_grouped_experts_cls,
            ),
            (Linear.Config, self.linear_fqns, _get_mx_qat_linear_cls),
        ):
            matched = set()
            for fqn, config, parent, attr in model.traverse(config_type):
                if targets is not None and fqn not in targets:
                    continue
                replacement = factory(type(config)._owner)
                new_config = convert_config_type(config, replacement)
                deltas = {"weight_fake_quant_config": self.weight_fake_quant_config}
                if config_type is GroupedExperts.Config:
                    deltas[
                        "activation_fake_quant_config"
                    ] = self.activation_fake_quant_config
                replacements.append((replace(new_config, **deltas), parent, attr))
                matched.add(fqn)
            missing.update(set(targets or ()) - matched)
        if missing:
            raise ValueError(f"MX QAT module FQNs did not match: {sorted(missing)}")
        for config, parent, attr in replacements:
            if parent is None:
                model = config
            elif isinstance(parent, list):
                parent[attr] = config
            else:
                setattr(parent, attr, config)
        return model


# Repeating QAT in one transform sequence is an error; applying it again
# to an existing tree is idempotent.
MXQATTransform.conflicts_with = (MXQATTransform,)
