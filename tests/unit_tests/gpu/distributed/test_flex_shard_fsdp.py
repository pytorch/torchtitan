# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``parallelism.fsdp_backend=FlexShardConfig()`` against FSDP2, bit for bit."""

from itertools import product

import pytest
import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.config import PrecisionConfig
from torchtitan.config.configs import DebugConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.flex_shard_fsdp import FlexShardConfig
from torchtitan.distributed.fsdp import set_is_last_accumulation_step
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.llama3 import build_model_config as build_llama3_config
from torchtitan.models.qwen3 import build_model_config as build_qwen3_config

flex_shard = pytest.importorskip("flex_shard")

pytestmark = pytest.mark.multi_gpu

_SEQ_LEN = 128


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class TestFlexShardDecoder(DTensorTestBase):
    @property
    def world_size(self) -> int:
        # Sums of four values make the reduction order observable.
        return 4

    def _build(
        self,
        build_config,
        flavor: str,
        fsdp_backend: FlexShardConfig | None,
        reshard_after_forward: str,
        parallelism_context: ParallelismContext,
    ):
        parallelism = ParallelismConfig(
            data_parallel_shard_degree=self.world_size,
            fsdp_backend=fsdp_backend,
            fsdp_reshard_after_forward=reshard_after_forward,
        )
        model_config = build_config(
            flavor,
            parallelism=parallelism,
            seq_len=_SEQ_LEN,
            attn_backend="flex",
        )
        with parallelism_context.activate_spmd(), torch.device("meta"):
            model = model_config.build()
        model = model.parallelize(
            parallelism_context=parallelism_context,
            precision=PrecisionConfig(),
            parallelism=parallelism,
            local_compile=LocalCompileConfig(),
            activation_remat=None,
            dump_folder="",
        )
        dist_utils.set_determinism(
            parallelism_context,
            torch.device(self.device_type),
            DebugConfig(seed=0),
            distinct_seed_mesh_axes=[],
        )
        with parallelism_context.activate_spmd():
            model.to_empty(device=self.device_type)
            with torch.no_grad():
                model.init_weights(buffer_device=None)
        return model, model_config, parallelism, parallelism_context

    def _train(
        self, model, model_config, parallelism, parallelism_context, microbatches
    ):
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=True)
        # One packed sequence per microbatch, as the data loader produces.
        positions = torch.arange(_SEQ_LEN, device=self.device_type)
        history = []
        for step in range(3):
            losses = []
            for microbatch in range(microbatches):
                if microbatches > 1:
                    set_is_last_accumulation_step(
                        model,
                        microbatch == microbatches - 1,
                        fsdp_backend=parallelism.fsdp_backend,
                    )
                generator = torch.Generator().manual_seed(
                    1000 * step + 10 * microbatch + self.rank
                )
                tokens, labels = (
                    torch.randint(
                        model_config.vocab_size, (_SEQ_LEN,), generator=generator
                    ).to(self.device_type)
                    for _ in range(2)
                )
                inputs, labels, model_kwargs, _ = model.preprocess_inputs(
                    {"input": tokens, "labels": labels, "positions": positions},
                    parallelism_context=parallelism_context,
                    parallelism=parallelism,
                    # The trainer's count for MoE load balancing, without padding.
                    global_routing_token_counts=torch.tensor(
                        [_SEQ_LEN * self.world_size * microbatches],
                        device=self.device_type,
                    ),
                )
                with parallelism_context.activate_spmd():
                    logits = model(inputs, **model_kwargs)
                    loss = F.cross_entropy(logits.float(), labels)
                    with spmd.no_typecheck():
                        loss.backward()
                losses.append(loss.detach())
            grad_norm = dist_utils.clip_grad_norm_(
                list(model.parameters()), max_norm=1.0, foreach=True
            )
            grads = {
                fqn: _local(param.grad).clone()
                for fqn, param in model.named_parameters()
                if param.grad is not None
            }
            optim.step()
            optim.zero_grad()
            history.append((losses, grad_norm, grads, self._local_params(model)))
        return history

    def _local_params(self, model) -> dict[str, torch.Tensor]:
        return {
            fqn: _local(param).detach().clone()
            for fqn, param in model.named_parameters()
        }

    def _assert_equal_tensors(self, expected, actual, context: str) -> None:
        self.assertEqual(list(expected), list(actual), msg=context)
        for fqn, expected_tensor in expected.items():
            self.assertTrue(
                torch.equal(expected_tensor, actual[fqn]), msg=f"{context} {fqn}"
            )

    @with_comms
    def test_matches_fsdp2(self):
        # One context, as in training: both backends shard over its dp_shard mesh.
        parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=self.world_size,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=self.world_size,
            enable_sequence_parallel=False,
        )
        for (build_config, flavor), reshard_after_forward, microbatches in product(
            (
                (build_llama3_config, "debugmodel"),
                (build_qwen3_config, "debugmodel_moe"),
            ),
            ("default", "always"),
            (1, 2),
        ):
            context = (
                f"{flavor} "
                f"reshard_after_forward={reshard_after_forward} "
                f"microbatches={microbatches}"
            )
            fsdp2, *fsdp2_setup = self._build(
                build_config,
                flavor,
                None,
                reshard_after_forward,
                parallelism_context,
            )
            flex, *flex_setup = self._build(
                build_config,
                flavor,
                FlexShardConfig(),
                reshard_after_forward,
                parallelism_context,
            )
            for fqn, param in flex.named_parameters():
                self.assertTrue(flex_shard.is_flex_shard_param(param), msg=fqn)
            self._assert_equal_tensors(
                self._local_params(fsdp2), self._local_params(flex), f"{context} init"
            )
            expected_history = self._train(fsdp2, *fsdp2_setup, microbatches)
            actual_history = self._train(flex, *flex_setup, microbatches)
            for step, (expected, actual) in enumerate(
                zip(expected_history, actual_history, strict=True)
            ):
                step_context = f"{context} step {step}"
                for expected_loss, actual_loss in zip(
                    expected[0], actual[0], strict=True
                ):
                    self.assertTrue(
                        torch.equal(expected_loss, actual_loss),
                        msg=f"{step_context} loss",
                    )
                self.assertTrue(
                    torch.equal(expected[1], actual[1]), msg=f"{step_context} grad norm"
                )
                self._assert_equal_tensors(
                    expected[2], actual[2], f"{step_context} grad"
                )
                self._assert_equal_tensors(
                    expected[3], actual[3], f"{step_context} param"
                )
