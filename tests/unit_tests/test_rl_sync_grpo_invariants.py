#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import types

import pytest
import torch


def _install_test_stubs() -> None:
    torchstore = types.ModuleType("torchstore")
    torchstore.LocalRankStrategy = type("LocalRankStrategy", (), {})
    torchstore.initialize = lambda *args, **kwargs: None
    torchstore.put_state_dict = lambda *args, **kwargs: None
    torchstore.get_state_dict = lambda *args, **kwargs: None
    sys.modules.setdefault("torchstore", torchstore)

    monarch = types.ModuleType("monarch")
    monarch_actor = types.ModuleType("monarch.actor")
    monarch_actor.Actor = type("Actor", (), {})
    monarch_actor.endpoint = lambda fn: fn
    monarch_actor.this_host = lambda: None
    monarch_spmd = types.ModuleType("monarch.spmd")

    async def _setup_torch_elastic_env_async(*args, **kwargs):
        return None

    monarch_spmd.setup_torch_elastic_env_async = _setup_torch_elastic_env_async
    monarch_rdma = types.ModuleType("monarch.rdma")
    monarch_rdma.is_rdma_available = lambda: False

    sys.modules.setdefault("monarch", monarch)
    sys.modules.setdefault("monarch.actor", monarch_actor)
    sys.modules.setdefault("monarch.spmd", monarch_spmd)
    sys.modules.setdefault("monarch.rdma", monarch_rdma)

    vllm = types.ModuleType("vllm")
    vllm.EngineArgs = type("EngineArgs", (), {"__init__": lambda self, **kwargs: None})
    vllm.LLMEngine = type("LLMEngine", (), {})
    vllm.SamplingParams = type(
        "SamplingParams", (), {"__init__": lambda self, **kwargs: None}
    )
    sys.modules.setdefault("vllm", vllm)

    vllm_config = types.ModuleType("vllm.config")
    vllm_config.AttentionConfig = type(
        "AttentionConfig", (), {"__init__": lambda self, **kwargs: None}
    )
    vllm_config.CompilationConfig = type(
        "CompilationConfig", (), {"__init__": lambda self, **kwargs: None}
    )
    vllm_config.VllmConfig = type("VllmConfig", (), {})
    vllm_config.get_current_vllm_config = lambda: types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(tensor_parallel_size=1),
        cache_config=None,
    )
    sys.modules.setdefault("vllm.config", vllm_config)

    vllm_logger = types.ModuleType("vllm.logger")
    import logging

    vllm_logger.init_logger = logging.getLogger
    sys.modules.setdefault("vllm.logger", vllm_logger)

    vllm_utils = types.ModuleType("vllm.utils")
    vllm_torch_utils = types.ModuleType("vllm.utils.torch_utils")
    vllm_torch_utils.weak_ref_tensor = lambda tensor: tensor
    vllm_utils.torch_utils = vllm_torch_utils
    sys.modules.setdefault("vllm.utils", vllm_utils)
    sys.modules.setdefault("vllm.utils.torch_utils", vllm_torch_utils)

    vllm_compilation = types.ModuleType("vllm.compilation")
    vllm_compilation_decorators = types.ModuleType("vllm.compilation.decorators")
    vllm_compilation_decorators.support_torch_compile = (
        lambda *args, **kwargs: (lambda cls: cls)
    )
    sys.modules.setdefault("vllm.compilation", vllm_compilation)
    sys.modules.setdefault(
        "vllm.compilation.decorators", vllm_compilation_decorators
    )

    vllm_model_executor = types.ModuleType("vllm.model_executor")
    vllm_model_executor_layers = types.ModuleType("vllm.model_executor.layers")
    vllm_model_executor_attention = types.ModuleType(
        "vllm.model_executor.layers.attention"
    )
    vllm_model_executor_attention.Attention = type(
        "Attention", (), {"__init__": lambda self, **kwargs: None}
    )
    vllm_batch_invariant = types.ModuleType(
        "vllm.model_executor.layers.batch_invariant"
    )
    vllm_batch_invariant.init_batch_invariance = lambda *args, **kwargs: None
    sys.modules.setdefault("vllm.model_executor", vllm_model_executor)
    sys.modules.setdefault("vllm.model_executor.layers", vllm_model_executor_layers)
    sys.modules.setdefault(
        "vllm.model_executor.layers.attention", vllm_model_executor_attention
    )
    sys.modules.setdefault(
        "vllm.model_executor.layers.batch_invariant", vllm_batch_invariant
    )

    vllm_registry = types.ModuleType("vllm.model_executor.models.registry")
    vllm_registry.ModelRegistry = type(
        "ModelRegistry", (), {"register_model": staticmethod(lambda *args, **kwargs: None)}
    )
    sys.modules.setdefault("vllm.model_executor.models.registry", vllm_registry)

    vllm_sampling_params = types.ModuleType("vllm.sampling_params")
    vllm_sampling_params.RequestOutputKind = types.SimpleNamespace(FINAL_ONLY="final")
    sys.modules.setdefault("vllm.sampling_params", vllm_sampling_params)

    vllm_v1_backend = types.ModuleType("vllm.v1.attention.backend")
    vllm_v1_backend.AttentionType = types.SimpleNamespace(
        ENCODER_ONLY="encoder_only",
        ENCODER="encoder",
    )
    sys.modules.setdefault("vllm.v1.attention.backend", vllm_v1_backend)

    vllm_v1_flash = types.ModuleType("vllm.v1.attention.backends.flash_attn")
    vllm_v1_flash.FlashAttentionBackend = type("FlashAttentionBackend", (), {})
    vllm_v1_flash.FlashAttentionImpl = type(
        "FlashAttentionImpl", (), {"__init__": lambda self, *args, **kwargs: None}
    )
    vllm_v1_flash.FlashAttentionMetadata = type("FlashAttentionMetadata", (), {})
    sys.modules.setdefault("vllm.v1.attention.backends.flash_attn", vllm_v1_flash)

    vllm_v1_registry = types.ModuleType("vllm.v1.attention.backends.registry")
    vllm_v1_registry.AttentionBackendEnum = types.SimpleNamespace(CUSTOM="custom")
    vllm_v1_registry.register_backend = lambda *args, **kwargs: (lambda cls: cls)
    sys.modules.setdefault("vllm.v1.attention.backends.registry", vllm_v1_registry)


_install_test_stubs()

from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.controller import _create_dedicated_meshes
from torchtitan.experiments.rl.sync_grpo_sum_digits.main import RLTrainer
from torchtitan.experiments.rl.types import Episode


def _episode(
    *,
    prompt_tokens: list[int],
    response_tokens: list[int],
    logprobs: list[float],
    group_id: str,
    policy_version: int,
    reward: float = 1.0,
    advantage: float = 0.5,
) -> Episode:
    return Episode(
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        logprobs=logprobs,
        group_id=group_id,
        text="sample",
        reward=reward,
        advantage=advantage,
        policy_version=policy_version,
    )


def test_shard_episodes_interleaves_by_dp_rank():
    trainer = object.__new__(RLTrainer)
    trainer.trainer_dp_degree = 2

    episodes = [
        _episode(
            prompt_tokens=[10, 11],
            response_tokens=[12, 13],
            logprobs=[-0.1, -0.2],
            group_id="g0",
            policy_version=7,
        ),
        _episode(
            prompt_tokens=[20],
            response_tokens=[21, 22, 23],
            logprobs=[-0.3, -0.4, -0.5],
            group_id="g0",
            policy_version=7,
        ),
        _episode(
            prompt_tokens=[30, 31, 32],
            response_tokens=[33],
            logprobs=[-0.6],
            group_id="g1",
            policy_version=7,
        ),
        _episode(
            prompt_tokens=[40, 41],
            response_tokens=[42, 43],
            logprobs=[-0.7, -0.8],
            group_id="g1",
            policy_version=7,
        ),
    ]

    shards = trainer._shard_episodes(episodes)

    assert len(shards) == 2
    assert [ep.prompt_tokens for ep in shards[0]] == [[10, 11], [30, 31, 32]]
    assert [ep.prompt_tokens for ep in shards[1]] == [[20], [40, 41]]


def test_collate_rank_episodes_preserves_rollout_policy_version_and_old_logprobs():
    episodes = [
        _episode(
            prompt_tokens=[10, 11],
            response_tokens=[12, 13],
            logprobs=[-0.1, -0.2],
            group_id="g0",
            policy_version=7,
        ),
        _episode(
            prompt_tokens=[30, 31, 32],
            response_tokens=[33],
            logprobs=[-0.6],
            group_id="g1",
            policy_version=7,
        ),
    ]

    batch = RLTrainer._collate_rank_episodes(
        episodes,
        policy_version=7,
        pad_token_id=99,
    )

    assert batch.policy_version == 7
    assert torch.equal(
        batch.old_logprobs,
        torch.tensor(
            [
                [0.0, 0.0, -0.1, -0.2],
                [0.0, 0.0, 0.0, -0.6],
            ],
            dtype=torch.float32,
        ),
    )


def test_shard_episodes_rejects_mixed_policy_versions():
    trainer = object.__new__(RLTrainer)
    trainer.trainer_dp_degree = 1

    episodes = [
        _episode(
            prompt_tokens=[1],
            response_tokens=[2],
            logprobs=[-0.1],
            group_id="g0",
            policy_version=3,
        ),
        _episode(
            prompt_tokens=[3],
            response_tokens=[4],
            logprobs=[-0.2],
            group_id="g0",
            policy_version=4,
        ),
    ]

    with pytest.raises(ValueError, match="policy_version"):
        trainer._shard_episodes(episodes)


def test_shard_then_collate_preserves_rollout_policy_version_and_old_logprobs():
    trainer = object.__new__(RLTrainer)
    trainer.trainer_dp_degree = 2

    episodes = [
        _episode(
            prompt_tokens=[10, 11],
            response_tokens=[12, 13],
            logprobs=[-0.1, -0.2],
            group_id="g0",
            policy_version=7,
        ),
        _episode(
            prompt_tokens=[20],
            response_tokens=[21, 22, 23],
            logprobs=[-0.3, -0.4, -0.5],
            group_id="g0",
            policy_version=7,
        ),
        _episode(
            prompt_tokens=[30, 31, 32],
            response_tokens=[33],
            logprobs=[-0.6],
            group_id="g1",
            policy_version=7,
        ),
        _episode(
            prompt_tokens=[40, 41],
            response_tokens=[42, 43],
            logprobs=[-0.7, -0.8],
            group_id="g1",
            policy_version=7,
        ),
    ]

    policy_version = trainer._validate_policy_versions(episodes)
    shards = trainer._shard_episodes(episodes)
    batches = [
        trainer._collate_rank_episodes(
            shard,
            policy_version=policy_version,
            pad_token_id=99,
        )
        for shard in shards
    ]

    assert len(batches) == 2
    assert batches[0].policy_version == 7
    assert batches[1].policy_version == 7
    assert torch.equal(
        batches[0].old_logprobs,
        torch.tensor(
            [
                [0.0, 0.0, -0.1, -0.2],
                [0.0, 0.0, 0.0, -0.6],
            ],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(
        batches[1].old_logprobs,
        torch.tensor(
            [
                [0.0, -0.3, -0.4, -0.5],
                [0.0, 0.0, -0.7, -0.8],
            ],
            dtype=torch.float32,
        ),
    )


def test_trainer_rejects_stale_policy_version():
    trainer = object.__new__(PolicyTrainer)
    trainer.policy_version = 5

    trainer._validate_batch_policy_version(5)

    with pytest.raises(ValueError, match="trainer expects policy_version=5"):
        trainer._validate_batch_policy_version(6)


class _FakeHostMesh:
    def __init__(self, name: str):
        self.name = name

    def slice(self, *, hosts):
        return _FakeHostMesh(f"{self.name}[{hosts.start}:{hosts.stop}]")

    def spawn_procs(self, per_host=None, bootstrap=None):
        return {
            "mesh": self.name,
            "per_host": per_host,
            "bootstrap": bootstrap,
        }


def test_dedicated_meshes_reuse_provisioner_for_shared_host_slice():
    root = _FakeHostMesh("root")
    meshes = _create_dedicated_meshes(
        root,
        gpus_per_node=8,
        gpu_requests=[4, 4, 0],
        node_assignments=[1, 0, 0],
    )

    first_bootstrap = meshes[0]["bootstrap"]
    second_bootstrap = meshes[1]["bootstrap"]
    assert first_bootstrap is not None
    assert second_bootstrap is not None

    original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        first_bootstrap()
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
        second_bootstrap()
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "4,5,6,7"
    finally:
        if original_cuda_visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices
