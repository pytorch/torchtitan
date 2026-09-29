# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-independent MX QAT selection and config transformation."""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from torchtitan.models.common.linear import GroupedLinear, Linear
from torchtitan.protocols.module import Module
from torchtitan.quantization.mx_qat.experts import (
    _get_mx_qat_grouped_linear_cls,
    activation_config,
    MXFakeQuantizeConfig,
    weight_config,
)
from torchtitan.quantization.mx_qat.linear import _get_mx_qat_linear_cls

from .base import convert_config_type, ModelConfigTransform


@dataclass(kw_only=True, slots=True)
class MXQATTransform(ModelConfigTransform):
    """Preserve BF16 masters and select QAT by exact module FQN.

    None selects every grouped-linear module; an empty tuple selects none.
    Dense projections are opt-in and use weight-only fake quantization.
    TorchAO configs, including kernel_preference, pass through unchanged.
    """

    grouped_linear_fqns: tuple[str, ...] | None = None
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

        Each grouped linear owns one parameter, including its stacked projections.
        The checkpoint adapter validates that all HF aliases of a selected
        parameter use the same quantization policy.
        """
        remaining = set(weight_fqns)
        groups, linears = [], []
        for fqn, _, _, _ in model.traverse(GroupedLinear.Config):
            key = f"{fqn}.weight".lstrip(".")
            if key in remaining:
                groups.append(fqn)
                remaining.remove(key)
        for fqn, _, _, _ in model.traverse(Linear.Config):
            key = f"{fqn}.weight".lstrip(".")
            if key in remaining:
                linears.append(fqn)
                remaining.remove(key)
        if remaining:
            raise ValueError(f"MX QAT cannot represent weights: {sorted(remaining)}")
        return cls(
            grouped_linear_fqns=tuple(groups), linear_fqns=tuple(linears), **kwargs
        )

    def transform(self, model: Module.Config) -> Module.Config:
        replacements = []
        missing = set()
        handlers: tuple[
            tuple[
                type[Module.Config],
                tuple[str, ...] | None,
                Callable[[Any], type[Module]],
            ],
            ...,
        ] = (
            (
                GroupedLinear.Config,
                self.grouped_linear_fqns,
                _get_mx_qat_grouped_linear_cls,
            ),
            (Linear.Config, self.linear_fqns, _get_mx_qat_linear_cls),
        )
        for config_type, targets, factory in handlers:
            matched = set()
            for fqn, config, parent, attr in model.traverse(config_type):
                if targets is not None and fqn not in targets:
                    continue
                if (
                    config_type is GroupedLinear.Config
                    and self.activation_fake_quant_config.kernel_preference
                    != self.weight_fake_quant_config.kernel_preference
                ):
                    raise ValueError(
                        "MX QAT grouped linears require matching activation and weight "
                        "kernel_preference. Set both TorchAO configs to the same preference."
                    )
                owner = type(config)._owner
                assert owner is not None
                replacement = factory(owner)
                new_config = convert_config_type(config, replacement)
                deltas = {"weight_fake_quant_config": self.weight_fake_quant_config}
                if config_type is GroupedLinear.Config:
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
