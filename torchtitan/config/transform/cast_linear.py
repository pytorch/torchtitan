# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-config converter for fixed-dtype output projections."""

from dataclasses import dataclass, fields

from torchtitan.models.common.linear import CastLinear, Linear

from .converter import ModelConfigConverter

__all__ = ["LMHeadCastConverter"]


class LMHeadCastConverter(ModelConfigConverter):
    """Swap the decoder lm_head's ``Linear.Config`` to ``CastLinear.Config``.

    Walks the model config tree and replaces the ``lm_head`` node in place.
    Targets only the lm_head, so every other Linear stays a plain ``Linear``
    and LoRA transforms and quantization converters are unaffected.

    Note on trainer/inference bitwise parity: because the same ``model_config``
    backs both the trainer and the vLLM generator, the lm_head sees a matched
    cast chain on both sides. The inference weight, synced from the trainer,
    goes fp32 (trainer) -> bf16 (weight-sync) -> fp32 (lm_head cast); the
    trainer's own lm_head input follows the analogous fp32 (FSDP-sharded
    params) -> bf16 (all-gather) -> fp32 (lm_head cast). Both paths share the
    same lossy bf16 round-trip, so trainer/inference bitwise agreement is
    preserved rather than broken. TODO: investigate whether this
    fp32->bf16->fp32 cast pair can be removed (keep the weight in fp32 end to
    end) once that path is supported on both sides.
    """

    _TARGET = "lm_head"

    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        compute_dtype: str = "float32"
        """Forward matmul dtype for the lm_head (key into ``TORCH_DTYPE_MAP``)."""

    def __init__(self, config: Config):
        self.config = config

    def convert(self, model_config):
        found = False
        for fqn, linear_config, parent, attr in model_config.traverse(Linear.Config):
            if fqn.rsplit(".", 1)[-1] != self._TARGET:
                continue
            found = True
            shared_fields = {
                f.name: getattr(linear_config, f.name) for f in fields(linear_config)
            }
            new_config = CastLinear.Config(
                **shared_fields, compute_dtype=self.config.compute_dtype
            )
            if isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)
        if not found:
            raise ValueError(
                f"LMHeadCastConverter found no Linear named {self._TARGET!r} in the "
                "model config. The torchtitan decoder names its output projection "
                f"{self._TARGET!r} (see torchtitan/models/common/decoder.py)."
            )
        return model_config
