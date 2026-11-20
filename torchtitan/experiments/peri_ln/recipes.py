# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Peri-LN recipes using the standard Llama 3 Trainer and data pipeline."""

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.llama3.model import Llama3Model
from torchtitan.trainer import Trainer

from torchtitan_recipes.models.llama3 import llama3_8b as llama3_8b_base
from torchtitan_recipes.tests.models.llama3 import (
    llama3_debugmodel as llama3_debugmodel_base,
)

from .model import apply_peri_ln


def llama3_debugmodel(
    seq_len: int = 256,
    *,
    parallelism: ParallelismConfig | None = None,
    attn_backend: str = "flex",
) -> Trainer.Config:
    """Run ten deterministic steps on the existing C4 test fixture."""
    config = llama3_debugmodel_base(
        seq_len=seq_len, parallelism=parallelism, attn_backend=attn_backend
    )
    assert isinstance(config.model, Llama3Model.Config)
    config.model = apply_peri_ln(config.model)
    config.training.disable_cuda_graphs = True
    config.debug.seed = 42
    config.debug.deterministic = True
    return config


def llama3_8b(seq_len: int = 2048) -> Trainer.Config:
    """Use the standard 8B C4 recipe with Peri-LN blocks."""
    config = llama3_8b_base(seq_len=seq_len)
    assert isinstance(config.model, Llama3Model.Config)
    config.model = apply_peri_ln(config.model)
    return config
