# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``parallelism.fsdp_backend=FlexShardConfig()`` against FSDP2, bit for bit."""

import os
from itertools import product
from typing import Any, cast
from unittest.mock import patch

import pytest
import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.distributed.distributed_c10d as c10d
import torch.nn.functional as F
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torch.distributed.tensor import DTensor
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.config import PrecisionConfig
from torchtitan.config.configs import DebugConfig
from torchtitan.config.parallelism import (
    FlexShardConfig,
    FSDP2Config,
    ParallelismConfig,
)
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.flex_shard_fsdp import (
    flex_shard_bucket,
    flex_shard_reduce_meshes,
)
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.common.param_init import skip_param_init
from torchtitan.models.llama3 import build_model_config as build_llama3_config
from torchtitan.models.qwen3 import build_model_config as build_qwen3_config

flex_shard = pytest.importorskip("flex_shard")

pytestmark = pytest.mark.multi_gpu

_SEQ_LEN = 128


def _build_tied_llama3_config(flavor: str, **kwargs):
    """Llama 3 ``<flavor>_tied``: ``flavor`` with lm_head's weight shared with
    tok_embeddings, set up as the tied Llama 3 flavors (1B, 3B) do."""
    model_config = build_llama3_config(flavor.removesuffix("_tied"), **kwargs)
    model_config.enable_weight_tying = True
    model_config.tok_embeddings.param_init = {"weight": skip_param_init}
    return model_config


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
        fsdp_backend: FSDP2Config | FlexShardConfig,
        reshard_after_forward: str,
        parallelism_context: ParallelismContext,
    ):
        parallelism = ParallelismConfig(
            data_parallel_shard_degree=self.world_size,
            expert_parallel_degree=parallelism_context.ep,
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

    def _assert_initialized_as_one_device(
        self, model, model_config, parallelism, parallelism_context, context
    ) -> None:
        # One device's model, initialized from the same seed.
        del parallelism
        with parallelism_context.activate_spmd(), torch.device("meta"):
            reference = model_config.build()
        dist_utils.set_determinism(
            parallelism_context,
            torch.device(self.device_type),
            DebugConfig(seed=0),
            distinct_seed_mesh_axes=[],
        )
        with parallelism_context.activate_spmd():
            reference.to_empty(device=self.device_type)
            with torch.no_grad():
                reference.init_weights(buffer_device=None)
        full_params = dict(reference.named_parameters())
        for fqn, param in model.named_parameters():
            (placement,) = flex_shard.get_placements(param)
            expected = full_params[fqn]
            mesh = parallelism_context.get_mesh("dp_shard")
            if parallelism_context.ep > 1 and ".routed_experts." in fqn:
                # EP splits the experts, and edp_shard shards each EP rank's part.
                ep_mesh = parallelism_context.get_mesh("ep")
                expected = expected.chunk(ep_mesh.size())[ep_mesh.get_local_rank()]
                mesh = parallelism_context.get_mesh("edp_shard")
            expected = expected.chunk(mesh.size(), placement.dim)
            self.assertTrue(
                torch.equal(param, expected[mesh.get_local_rank()]),
                msg=f"{context} init {fqn}",
            )

    def _train(
        self,
        model,
        model_config,
        parallelism,
        parallelism_context,
        microbatches,
        chunked_loss,
        *,
        optim=None,
    ):
        if optim is None:
            optim = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=True)
        if chunked_loss:
            # As the training engine sets it up.
            loss_fn = ChunkedLossWrapper.Config().build()
            loss_fn.set_lm_head(
                model.lm_head,
                flex_shard_bucket=(
                    flex_shard_bucket(model, "lm_head")
                    if isinstance(parallelism.fsdp_backend, FlexShardConfig)
                    else None
                ),
            )
            model._skip_lm_head = True
        # One packed sequence per microbatch, as the data loader produces.
        positions = torch.arange(_SEQ_LEN, device=self.device_type)
        history = []
        for step in range(3):
            losses = []
            for microbatch in range(microbatches):
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
                    output = model(inputs, **model_kwargs)
                    if chunked_loss:
                        loss, _ = loss_fn(
                            output,
                            labels,
                            torch.tensor(_SEQ_LEN, device=self.device_type),
                        )
                    else:
                        loss = F.cross_entropy(output.float(), labels)
                    with spmd.no_typecheck():
                        loss.backward()
                losses.append(loss.detach())
            flex_shard_enabled = isinstance(parallelism.fsdp_backend, FlexShardConfig)
            grad_norm = dist_utils.clip_grad_norm_(
                list(model.parameters()),
                max_norm=1.0,
                foreach=True,
                flex_shard_enabled=flex_shard_enabled,
                flex_shard_reduce_meshes=(
                    flex_shard_reduce_meshes(parallelism_context)
                    if flex_shard_enabled
                    else None
                ),
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

    def _check_matches_fsdp2(
        self, *, ep: int, cases, chunked_loss: bool = False
    ) -> None:
        # One context, as in training: both backends shard over its meshes.
        parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=self.world_size,
            cp=1,
            tp=1,
            pp=1,
            ep=ep,
            world_size=self.world_size,
            enable_sequence_parallel=False,
        )
        for (build_config, flavor), reshard_after_forward, microbatches in cases:
            context = (
                f"{flavor} "
                f"reshard_after_forward={reshard_after_forward} "
                f"microbatches={microbatches} chunked_loss={chunked_loss}"
            )
            fsdp2, *fsdp2_setup = self._build(
                build_config,
                flavor,
                FSDP2Config(),
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
            self._assert_initialized_as_one_device(flex, *flex_setup, context)
            # Each of FSDP2's DTensor shards draws its own random values, so
            # start FlexShard from FSDP2's parameters.
            flex_params = dict(flex.named_parameters())
            with torch.no_grad():
                for fqn, param in fsdp2.named_parameters():
                    flex_params[fqn].copy_(_local(param))
            histories, num_reduce_scatters = [], []
            for model, setup in ((fsdp2, fsdp2_setup), (flex, flex_setup)):
                # FSDP2 calls dist.reduce_scatter_single, and FlexShard
                # dist.reduce_scatter_tensor, which calls c10d's.
                with patch.object(
                    c10d, "reduce_scatter_single", wraps=c10d.reduce_scatter_single
                ) as reduce_scatter, patch.object(
                    dist, "reduce_scatter_single", reduce_scatter
                ):
                    histories.append(
                        self._train(model, *setup, microbatches, chunked_loss)
                    )
                num_reduce_scatters.append(reduce_scatter.call_count)
            # Gradients are grouped into reduce-scatters as FSDP2 groups them,
            # e.g. ChunkedLossWrapper's lm_head grads of all chunks go into one.
            # Values can't tell when the sums are exact. The first check makes
            # sure the patches see the reduce-scatters.
            self.assertGreater(num_reduce_scatters[0], 0, msg=context)
            self.assertEqual(
                num_reduce_scatters[0], num_reduce_scatters[1], msg=context
            )
            expected_history, actual_history = histories
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

    @with_comms
    def test_matches_fsdp2(self):
        # Two microbatches cover one, and the reshard policy doesn't depend on
        # the model.
        self._check_matches_fsdp2(
            ep=1,
            cases=[
                ((build_llama3_config, "debugmodel"), "default", 2),
                ((build_qwen3_config, "debugmodel_moe"), "always", 2),
                ((_build_tied_llama3_config, "debugmodel_tied"), "default", 2),
            ],
        )

    @with_comms
    def test_matches_fsdp2_with_expert_parallelism(self):
        # Each MoE block's routed experts shard over edp_shard (dp_shard / ep
        # ranks) within each EP rank's experts, and the rest over dp_shard. The
        # loss is ChunkedLossWrapper's, as in the Qwen3 MoE recipes: it runs
        # lm_head on its own on each chunk, with a backward each. lm_head's
        # bucket keeps its grads unsharded across the chunks, as FSDP2's group
        # does, and reduce-scatters them at the last chunk.
        self._check_matches_fsdp2(
            ep=2,
            cases=product(
                [(build_qwen3_config, "debugmodel_moe")], ("default", "always"), (1, 2)
            ),
            chunked_loss=True,
        )

    @with_comms
    def test_matches_fsdp2_with_tied_weights_and_chunked_loss(self):
        # With tied weights, lm_head's bucket is also tok_embeddings', and
        # flex_shard_bucket finds it by the shared weight's second name.
        self._check_matches_fsdp2(
            ep=1,
            cases=[((_build_tied_llama3_config, "debugmodel_tied"), "default", 2)],
            chunked_loss=True,
        )

    @with_comms
    @with_temp_dir
    def test_checkpoints_match_fsdp2(self):
        # FlexShard loads FSDP2's checkpoint of the model and its AdamW states,
        # and saves it again chunk for chunk and bit for bit, also as a bf16
        # export. With ep=2, the routed experts are split over ep, then over
        # edp_shard.
        parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=self.world_size,
            cp=1,
            tp=1,
            pp=1,
            ep=2,
            world_size=self.world_size,
            enable_sequence_parallel=False,
        )

        def build(fsdp_backend, folder, **config):
            model, *setup = self._build(
                build_qwen3_config,
                "debugmodel_moe",
                fsdp_backend,
                "default",
                parallelism_context,
            )
            optimizers = OptimizersContainer(
                OptimizersContainer.Config(
                    optimizers=[AdamW.Config(pattern=".*", lr=1e-3)]
                ),
                model_parts=[model],
            )
            checkpointer = CheckpointManager(
                CheckpointManager.Config(
                    folder=folder,
                    interval=1,
                    keep_latest_k=0,
                    export_dtype="bfloat16",
                    **config,
                ),
                model_parts=[model],
                optimizers=optimizers,
                lr_schedulers=cast(Any, None),
                ema=None,
                extra_states={},
                sd_adapter=None,
                base_folder=self.temp_dir,
            )
            return model, setup, optimizers, checkpointer

        model, setup, optimizers, checkpointer = build(FSDP2Config(), "fsdp2")
        self._train(model, *setup, 1, False, optim=optimizers)
        checkpointer.save(3)
        checkpointer.save(3, last_step=True)
        checkpointer.close()

        # FlexShard's own initialization differs from FSDP2's, so only a load
        # gives it FSDP2's values.
        flex, _, flex_optimizers, checkpointer = build(
            FlexShardConfig(),
            "flex",
            initial_load_path=os.path.join(self.temp_dir, "fsdp2", "step-3"),
            initial_load_model_only=False,
        )
        self.assertTrue(checkpointer.load())
        # Saving again gives the same checkpoint even if ranks swapped their
        # shards, so check that each rank loaded FSDP2's.
        self._assert_equal_tensors(
            self._local_params(model), self._local_params(flex), "loaded param"
        )
        self._assert_equal_tensors(
            self._optimizer_states(model, optimizers),
            self._optimizer_states(flex, flex_optimizers),
            "loaded AdamW state",
        )
        checkpointer.save(3)
        checkpointer.save(3, last_step=True)
        checkpointer.close()
        for checkpoint in ("step-3", os.path.join("exports", "step-3")):
            self._assert_equal_checkpoints(
                os.path.join(self.temp_dir, "fsdp2", checkpoint),
                os.path.join(self.temp_dir, "flex", checkpoint),
            )

    def _optimizer_states(self, model, optimizers) -> dict[str, torch.Tensor]:
        states = {p: s for o in optimizers.optimizers for p, s in o.state.items()}
        return {
            f"{fqn} {name}": _local(value)
            for fqn, param in model.named_parameters()
            for name, value in states[param].items()
        }

    def _assert_equal_checkpoints(self, expected_dir: str, actual_dir: str) -> None:
        """Assert that two DCP checkpoints hold the same keys, the same chunks
        of full tensors of the same shapes and dtypes, and the same values.

        Rank 0 compares and shares the outcome, so a mismatch fails every rank
        at once instead of leaving the others in a collective.
        """
        error: list[Exception | None] = [None]
        if self.rank == 0:
            key = "keys"
            try:
                metadata = [
                    dcp.FileSystemReader(path).read_metadata().state_dict_metadata
                    for path in (expected_dir, actual_dir)
                ]
                self.assertEqual(sorted(metadata[0]), sorted(metadata[1]))
                for key, expected in metadata[0].items():
                    actual = metadata[1][key]
                    self.assertIs(type(actual), type(expected))
                    if isinstance(expected, TensorStorageMetadata):
                        self.assertEqual(
                            (actual.size, actual.properties.dtype, _chunks(actual)),
                            (
                                expected.size,
                                expected.properties.dtype,
                                _chunks(expected),
                            ),
                        )
                states = []
                for path in (expected_dir, actual_dir):
                    dcp_to_torch_save(path, f"{path}.pt")
                    states.append(torch.load(f"{path}.pt", weights_only=False))
                for key, expected in states[0].items():
                    if isinstance(expected, torch.Tensor):
                        self.assertTrue(torch.equal(expected, states[1][key]))
                    else:
                        self.assertEqual(expected, states[1][key])
            except Exception as e:
                error = [AssertionError(f"{actual_dir} vs {expected_dir}, {key}: {e}")]
        dist.broadcast_object_list(error)
        if error[0] is not None:
            raise error[0]


def _chunks(
    metadata: TensorStorageMetadata,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """A checkpointed tensor's non-empty chunks as (offsets, sizes) pairs.

    FSDP2 also saves an empty chunk for each rank without data.
    """
    return sorted(
        (tuple(chunk.offsets), tuple(chunk.sizes))
        for chunk in metadata.chunks
        if all(chunk.sizes)
    )
