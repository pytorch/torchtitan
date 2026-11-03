# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""b200 integration-test suite: run configurations and their test definitions.

Each configuration function is selected by module and name in the launched
training process, so it must stay importable from this module.
"""

from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.trainer import Trainer
from torchtitan_recipes.tests import _set_spmd_typechecking
from torchtitan_recipes.tests.models.deepseek_v3 import (
    deepseek_v3_debugmodel_dist_moe_bf16_fsdp2_ep2,
    deepseek_v3_debugmodel_dist_moe_mxfp8_fsdp2_ep2,
)
from torchtitan_recipes.tests.models.multimodal import (
    set_rank_conditional_image_presence,
)

from tests.integration_tests import IntegrationTestDefinition


def kimi_k3_debugmodel_mm() -> Trainer.Config:
    from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel

    config = kimi_k3_debugmodel(
        attn_backend="flex", parallelism=ParallelismConfig(expert_parallel_degree=2)
    )
    # DistMuon rejects TP-produced _StridedShard storage, so the TP coverage
    # keeps AdamW; kimi_k3_debugmodel_mm_muon covers the default optimizer.
    config.optim.optimizer = OptimizersContainer.Config(
        optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]
    )
    _set_spmd_typechecking(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.enable_sequence_parallel = True
    set_rank_conditional_image_presence(config)
    return config


def kimi_k3_debugmodel_mm_muon() -> Trainer.Config:
    """Per-head DistMuon with FSDP and EP."""
    from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel

    config = kimi_k3_debugmodel(
        attn_backend="flex", parallelism=ParallelismConfig(expert_parallel_degree=2)
    )
    _set_spmd_typechecking(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 2
    return config


def llama3_debugmodel_mxfp8_fsdp2() -> Trainer.Config:
    from torchtitan_recipes.tests.models.llama3 import llama3_debugmodel_mxfp8

    config = llama3_debugmodel_mxfp8()
    config.parallelism.data_parallel_shard_degree = 2
    return config


def llama3_debugmodel_nvfp4_fsdp2() -> Trainer.Config:
    from torchtitan_recipes.tests.models.llama3 import llama3_debugmodel_nvfp4

    config = llama3_debugmodel_nvfp4(seq_len=2048)
    config.parallelism.data_parallel_shard_degree = 2
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    return config


def kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4() -> Trainer.Config:
    from torchtitan.distributed.pipeline_parallel import (
        _generate_llm_fqn_per_model_part,
    )
    from torchtitan.models.kimi_k3.model import KimiK3Model

    from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel

    config = kimi_k3_debugmodel(parallelism=ParallelismConfig(expert_parallel_degree=2))
    # Type checking stays off under pipeline parallelism, as the other pipeline
    # recipes have it.
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.enable_sequence_parallel = True
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.parallelism.num_pp_microbatches = 4
    # Four stages per rank (the default is two), the shape where a hop can carry
    # no new block and a block has three later readers on its rank: core's split
    # for that many stages, with the model's end modules pinned.
    parallelism = config.parallelism
    split = _generate_llm_fqn_per_model_part(
        4 * parallelism.pipeline_parallel_degree,
        len(config.model.layers),
        parallelism.pipeline_parallel_first_stage_less_layers,
        parallelism.pipeline_parallel_last_stage_less_layers,
    )
    split[0][:0] = KimiK3Model.pipeline_first_stage_module_fqns
    split[-1].extend(KimiK3Model.pipeline_last_stage_module_fqns)
    parallelism.pipeline_parallel_module_fqns_per_model_part = split
    # DistMuon does not support tensor parallelism yet (#3353), so this cell
    # keeps AdamW the way kimi_k3_debugmodel_mm does.
    config.optim.optimizer = OptimizersContainer.Config(
        optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]
    )
    return config


def deepseek_v3_debugmodel_dist_moe_mxfp8_fsdp2_ep2_pp2() -> Trainer.Config:
    """Exercise eager PP schedule-derived Dist-MoE activation slots."""
    config = deepseek_v3_debugmodel_dist_moe_mxfp8_fsdp2_ep2()
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.parallelism.num_pp_microbatches = 4
    return config


def deepseek_v3_debugmodel_dist_moe_mxfp8_fsdp2_ep2_vmm() -> Trainer.Config:
    """Exercise host-backed VMM scratch with MXFP8 Dist-MoE."""
    from torchtitan.models.common.dist_moe.runtime import DistMoeRuntime

    config = deepseek_v3_debugmodel_dist_moe_mxfp8_fsdp2_ep2()
    runtime_config = config.dist_moe_runtime
    assert isinstance(runtime_config, DistMoeRuntime.Config)
    runtime_config.scratch_capacity_factor = 1.0
    runtime_config.vmm_capacity_factor = 4.0
    return config


def build_b200_tests_list() -> list[IntegrationTestDefinition]:
    """Build integration tests that require B200-class hardware."""
    return [
        IntegrationTestDefinition(
            configs=[kimi_k3_debugmodel_mm],
            test_descr="Kimi K3 multimodal SPMD-typed FSDP, TP and EP",
            test_name="kimi_k3_mm",
            ngpu=4,
        ),
        IntegrationTestDefinition(
            configs=[kimi_k3_debugmodel_mm_muon],
            test_descr="Kimi K3 multimodal per-head DistMuon FSDP+EP numerics",
            test_name="kimi_k3_mm_muon",
            ngpu=2,
            golden_numerics_path="tests/assets/losses/real_pg/kimi_k3_b200.txt",
            loss_compare_seed_config=kimi_k3_debugmodel_mm,
        ),
        IntegrationTestDefinition(
            configs=[kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4],
            test_descr="Kimi K3 FSDP, TP, EP and PP with 4 VPP stages",
            test_name="kimi_k3_fsdp2_tp2_ep2_pp2_vpp4",
            ngpu=8,
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_mxfp8_fsdp2],
            test_descr="MXFP8 linear with an FSDP-managed weight cache",
            test_name="mxfp8_linear_fsdp",
            ngpu=2,
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_nvfp4_fsdp2],
            test_descr="NVFP4 linear with an FSDP-managed weight cache",
            test_name="nvfp4_linear_fsdp",
            ngpu=2,
        ),
        IntegrationTestDefinition(
            configs=[
                deepseek_v3_debugmodel_dist_moe_bf16_fsdp2_ep2,
                deepseek_v3_debugmodel_dist_moe_mxfp8_fsdp2_ep2,
                deepseek_v3_debugmodel_dist_moe_mxfp8_fsdp2_ep2_vmm,
            ],
            test_descr="Eager BF16, MXFP8, and VMM Dist-MoE with FSDP and EP",
            test_name="dist_moe_eager_fsdp_ep_accelerator_graph",
            ngpu=2,
            use_real_pg=True,
        ),
        IntegrationTestDefinition(
            configs=[deepseek_v3_debugmodel_dist_moe_mxfp8_fsdp2_ep2_pp2],
            test_descr="Eager MXFP8 Dist-MoE PP with BF16 reduction",
            test_name="dist_moe_eager_fsdp_ep_pp_accelerator_graph",
            ngpu=4,
            use_real_pg=True,
        ),
    ]
