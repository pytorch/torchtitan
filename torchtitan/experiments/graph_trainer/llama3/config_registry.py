# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.models.llama3.config_registry import (
    llama3_405b,
    llama3_70b,
    llama3_8b,
    llama3_debugmodel,
)

from . import model_registry


def graph_trainer_llama3_debugmodel() -> GraphTrainer.Config:
    config = to_graph_trainer_config(llama3_debugmodel(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_llama3_debugmodel_sdpa() -> GraphTrainer.Config:
    """Debug model on the test-only SDPA backend.

    Used by graph machinery tests (precompile artifact serialization, context
    parallel) that can't run on the default FlexAttention backend: its BlockMask
    is unpicklable (mask_mod code objects) and is not a tensor. SDPA exercises
    the same machinery without those obstacles. See
    ``build_decoder_config_for_backend``.
    """
    base = llama3_debugmodel()
    base.model_spec = model_registry("debugmodel", attn_backend="sdpa")
    config = to_graph_trainer_config(base, model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_llama3_debugmodel_sdpa_eager() -> GraphTrainer.Config:
    """SDPA debug model run eagerly (no graph tracing).

    Serves as the eager reference for the AutoParallel SDPA loss-compare test:
    with ``mode=None`` GraphTrainer.forward_backward_step delegates to the core
    (eager) Trainer path, so this is a plain eager FSDP+TP run of the same SDPA
    model the AutoParallel test traces. The default FlexAttention backend can't
    fill this role — flex + AutoParallel is unsupported (BlockMask flattening).
    """
    config = graph_trainer_llama3_debugmodel_sdpa()
    config.compile = GraphTrainerCompileConfig(enable=False, mode=None)
    return config


def graph_trainer_llama3_8b() -> GraphTrainer.Config:
    config = to_graph_trainer_config(llama3_8b(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_llama3_70b() -> GraphTrainer.Config:
    config = to_graph_trainer_config(llama3_70b(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_llama3_405b() -> GraphTrainer.Config:
    config = to_graph_trainer_config(llama3_405b(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config
