# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.distributed.context_parallel import HeadTailCPLoadBalancer
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.llama3 import model_registry as llama3_model_registry
from torchtitan.models.llama3.config_registry import (
    llama3_405b,
    llama3_70b,
    llama3_8b,
    llama3_debugmodel,
    llama3_debugmodel_dist_gemm,
    llama3_mxfp8_linear_converter_config,
)
from torchtitan.observability.sdc_replayer import SDCReplayer

from .model import GraphTrainerLlama3Model


def graph_trainer_llama3_debugmodel() -> GraphTrainer.Config:
    config = to_graph_trainer_config(
        llama3_debugmodel(), GraphTrainerLlama3Model.Config
    )
    config.compile = GraphTrainerCompileConfig()
    return config


def graph_trainer_llama3_debugmodel_sdc_replay() -> GraphTrainer.Config:
    config = graph_trainer_llama3_debugmodel()
    config.debug.deterministic = True
    config.debug.seed = 42
    config.training.disable_cuda_graphs = True
    config.training.steps = 2
    config.sdc_replayer = SDCReplayer.Config()
    return config


def graph_trainer_llama3_debugmodel_dist_gemm() -> GraphTrainer.Config:
    """Debug model with the attention and FFN TP collectives folded into the GEMMs.

    The point of running dist-GEMM here rather than only under the eager trainer:
    GraphTrainer traces the whole model, so this is what proves the fused autograd
    Functions survive tracing. Needs tensor_parallel_degree > 1 and CUDA.
    """
    config = to_graph_trainer_config(
        llama3_debugmodel_dist_gemm(),
        GraphTrainerLlama3Model.Config,
    )
    config.compile = GraphTrainerCompileConfig()
    return config


def graph_trainer_llama3_debugmodel_mxfp8() -> GraphTrainer.Config:
    base = llama3_debugmodel()
    base.model = llama3_model_registry(
        "debugmodel_mxfp8",
        seq_len=base.training.max_context_length,
        converters=[
            llama3_mxfp8_linear_converter_config(model_compile_enabled=True),
        ],
    )
    config = to_graph_trainer_config(base, GraphTrainerLlama3Model.Config)
    config.compile = GraphTrainerCompileConfig()
    return config


def graph_trainer_llama3_debugmodel_sdpa() -> GraphTrainer.Config:
    """Debug model on the test-only SDPA backend.

    Used by graph machinery tests (precompile artifact serialization, context
    parallel) that can't run on the default FlexInnerAttention backend: its BlockMask
    is unpicklable (mask_mod code objects) and is not a tensor. SDPA exercises
    the same machinery without those obstacles. See
    ``build_decoder_config_for_backend``.
    """
    base = llama3_debugmodel()
    from . import model_registry

    base.parallelism.context_parallel_load_balancer = HeadTailCPLoadBalancer.Config()
    base.model = model_registry(
        "debugmodel",
        seq_len=base.training.max_context_length,
        attn_backend="sdpa",
    )
    config = to_graph_trainer_config(base, GraphTrainerLlama3Model.Config)
    config.compile = GraphTrainerCompileConfig()
    return config


def graph_trainer_llama3_debugmodel_sdpa_cross_entropy_loss() -> GraphTrainer.Config:
    """SDPA debug model with standard cross-entropy loss."""
    config = graph_trainer_llama3_debugmodel_sdpa()
    config.loss = CrossEntropyLoss.Config(
        global_vocab_size=decoder_vocab_size(config.model),
    )
    return config


def graph_trainer_llama3_8b() -> GraphTrainer.Config:
    config = to_graph_trainer_config(
        llama3_8b(seq_len=8192), GraphTrainerLlama3Model.Config
    )
    config.compile = GraphTrainerCompileConfig()
    return config


def graph_trainer_llama3_8b_c4_test() -> GraphTrainer.Config:
    config = graph_trainer_llama3_8b()
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )
    return config


def graph_trainer_llama3_8b_mxfp8() -> GraphTrainer.Config:
    base = llama3_8b(seq_len=8192)
    # Swap dense Linear layers for MXFP8Linear before wrapping in the
    # graph_trainer config. graph_trainer always compiles the model, so the
    # MXFP8 converter's compile requirement is satisfied.
    base.model = llama3_model_registry(
        "8B",
        converters=[
            llama3_mxfp8_linear_converter_config(model_compile_enabled=True),
        ],
    )
    config = to_graph_trainer_config(base, GraphTrainerLlama3Model.Config)
    config.compile = GraphTrainerCompileConfig()
    return config


def graph_trainer_llama3_70b() -> GraphTrainer.Config:
    config = to_graph_trainer_config(
        llama3_70b(seq_len=8192), GraphTrainerLlama3Model.Config
    )
    config.compile = GraphTrainerCompileConfig()
    return config


def graph_trainer_llama3_405b() -> GraphTrainer.Config:
    config = to_graph_trainer_config(
        llama3_405b(seq_len=8192), GraphTrainerLlama3Model.Config
    )
    config.compile = GraphTrainerCompileConfig()
    return config
