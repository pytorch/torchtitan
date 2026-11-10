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
from torchtitan.config.parallelism import FSDP2Config, ParallelismConfig
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
        steps=range(3),
    ):
        if optim is None:
            optim = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=True)
        if chunked_loss:
            # As the training engine sets it up.
            loss_fn = ChunkedLossWrapper.Config().build()
            loss_fn.set_lm_head(model.lm_head, fsdp_root=model)
            model._skip_lm_head = True
        # One packed sequence per microbatch, as the data loader produces.
        positions = torch.arange(_SEQ_LEN, device=self.device_type)
        history = []
        for step in steps:
            losses = []
            for microbatch in range(microbatches):
                if microbatches > 1:
                    set_is_last_accumulation_step(model, microbatch == microbatches - 1)
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

    def _assert_equal_histories(
        self, expected_history, actual_history, context, *, first_step=0
    ):
        for step, (expected, actual) in enumerate(
            zip(expected_history, actual_history, strict=True), start=first_step
        ):
            step_context = f"{context} step {step}"
            for expected_loss, actual_loss in zip(expected[0], actual[0], strict=True):
                self.assertTrue(
                    torch.equal(expected_loss, actual_loss), msg=f"{step_context} loss"
                )
            self.assertTrue(
                torch.equal(expected[1], actual[1]), msg=f"{step_context} grad norm"
            )
            self._assert_equal_tensors(expected[2], actual[2], f"{step_context} grad")
            self._assert_equal_tensors(expected[3], actual[3], f"{step_context} param")

    def _parallelism_context(self, ep: int) -> ParallelismContext:
        # One context per test, as in training: both backends shard over its
        # meshes.
        return ParallelismContext(
            dp_replicate=1,
            dp_shard=self.world_size,
            cp=1,
            tp=1,
            pp=1,
            ep=ep,
            world_size=self.world_size,
            enable_sequence_parallel=False,
        )

    def _check_matches_fsdp2(
        self, *, ep: int, cases, chunked_loss: bool = False
    ) -> None:
        parallelism_context = self._parallelism_context(ep)
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
            self._assert_equal_tensors(
                self._local_params(fsdp2), self._local_params(flex), f"{context} init"
            )
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
            # e.g. ChunkedLossWrapper's head grads kept from earlier
            # microbatches go into one. Values can't tell when sums are exact.
            # The first check makes sure the patches see the reduce-scatters.
            self.assertGreater(num_reduce_scatters[0], 0, msg=context)
            self.assertEqual(
                num_reduce_scatters[0], num_reduce_scatters[1], msg=context
            )
            self._assert_equal_histories(*histories, context)

    @with_comms
    def test_matches_fsdp2(self):
        self._check_matches_fsdp2(
            ep=1,
            cases=product(
                (
                    (build_llama3_config, "debugmodel"),
                    (build_qwen3_config, "debugmodel_moe"),
                ),
                ("default", "always"),
                (1, 2),
            ),
        )

    @with_comms
    def test_matches_fsdp2_with_expert_parallelism(self):
        # Each MoE block's routed experts shard over edp_shard (half of
        # dp_shard), and the rest of the block over dp_shard. The loss is
        # ChunkedLossWrapper's, as in the Qwen3 MoE recipes: it runs lm_head on
        # its own on each chunk, with a backward each. Its bucket keeps its
        # grads unsharded across the chunks, as FSDP2's group does, and
        # reduce-scatters them at the last chunk, or with grads kept from
        # earlier microbatches, in the decoder backward.
        self._check_matches_fsdp2(
            ep=2,
            cases=product(
                [(build_qwen3_config, "debugmodel_moe")], ("default", "always"), (1, 2)
            ),
            chunked_loss=True,
        )

    def _check_checkpoints_across_backends(self, build_config, flavor, *, ep: int):
        parallelism_context = self._parallelism_context(ep)

        def build(fsdp_backend):
            model, *setup = self._build(
                build_config, flavor, fsdp_backend, "default", parallelism_context
            )
            optimizers = OptimizersContainer(
                OptimizersContainer.Config(
                    optimizers=[AdamW.Config(pattern=".*", lr=1e-3)]
                ),
                model_parts=[model],
            )
            # As the training engine does. Qwen3 MoE's hooks reset its
            # load-balancing loss's per-step state, which isn't checkpointed.
            type(model)._register_optimizer_hooks(
                optimizers, [model], parallelism_context
            )

            def train(steps):
                return self._train(
                    model, *setup, 1, False, optim=optimizers, steps=steps
                )

            return model, optimizers, train

        def checkpointer(model, optimizers, folder, **config):
            return CheckpointManager(
                CheckpointManager.Config(
                    folder=os.path.join(self.temp_dir, folder),
                    interval=1,
                    keep_latest_k=0,
                    **config,
                ),
                model_parts=[model],
                optimizers=optimizers,
                lr_schedulers=cast(Any, None),
                ema=None,
                extra_states={},
                sd_adapter=type(model).state_dict_adapter_cls(model.config, None),
            )

        def path(folder):
            return os.path.join(self.temp_dir, folder)

        # FSDP2's uninterrupted run, which saves after steps 3 and 6, and
        # exports in bf16 and as a Hugging Face checkpoint after step 3.
        model, optimizers, train = build(FSDP2Config())
        reference = train(range(3))
        fsdp2 = checkpointer(model, optimizers, "fsdp2")
        fsdp2.save(3)
        export = checkpointer(model, optimizers, "fsdp2", export_dtype="bfloat16")
        export.save(3, last_step=True)
        export.close()
        export = checkpointer(model, optimizers, "fsdp2_hf", last_save_in_hf=True)
        export.save(3, last_step=True)
        export.close()
        reference += train(range(3, 6))
        fsdp2.save(6)
        fsdp2.close()

        # FlexShard's checkpoints after the same 3 steps are FSDP2's, saved in
        # sync and threaded async mode, and exported in bf16.
        model, optimizers, train = build(FlexShardConfig())
        train(range(3))
        for folder, config in (
            ("flex", {}),
            ("flex_async", {"async_mode": "async"}),
            ("flex", {"export_dtype": "bfloat16"}),
        ):
            flex = checkpointer(model, optimizers, folder, **config)
            flex.save(3, last_step="export_dtype" in config)
            flex.close()
        self._assert_equal_checkpoints(path("fsdp2/step-3"), path("flex/step-3"))
        self._assert_equal_checkpoints(path("fsdp2/step-3"), path("flex_async/step-3"))
        self._assert_equal_checkpoints(
            path("fsdp2/exports/step-3"), path("flex/exports/step-3")
        )

        # FlexShard loads the Hugging Face export into FSDP2's step-3
        # parameters, and its next step's losses, grad norm and grads, which
        # don't depend on the optimizer state, are FSDP2's.
        model, optimizers, train = build(FlexShardConfig())
        hf_load = checkpointer(
            model,
            optimizers,
            "flex_hf",
            initial_load_path=path("fsdp2_hf/exports/step-3"),
            initial_load_in_hf=True,
        )
        self.assertTrue(hf_load.load())
        hf_load.close()
        self._assert_equal_tensors(
            reference[2][3], self._local_params(model), "HF load"
        )
        ((losses, grad_norm, grads, _),) = train(range(3, 4))
        for expected_loss, loss in zip(reference[3][0], losses, strict=True):
            self.assertTrue(torch.equal(expected_loss, loss), msg="HF load loss")
        self.assertTrue(torch.equal(reference[3][1], grad_norm), msg="HF load norm")
        self._assert_equal_tensors(reference[3][2], grads, "HF load grad")

        # Each backend resumes from the other's step-3 checkpoint, and its next
        # 3 steps are FSDP2's uninterrupted ones.
        for fsdp_backend, source, target in (
            (FlexShardConfig(), "fsdp2", "flex"),
            (FSDP2Config(), "flex", "fsdp2"),
        ):
            model, optimizers, train = build(fsdp_backend)
            resumed = checkpointer(
                model,
                optimizers,
                f"{source}_to_{target}",
                initial_load_path=path(f"{source}/step-3"),
                initial_load_model_only=False,
            )
            self.assertTrue(resumed.load())
            self._assert_equal_histories(
                reference[3:],
                train(range(3, 6)),
                f"{source} to {target}",
                first_step=3,
            )
            # The load replaced the state dicts' tensors, whose replacements
            # declare their layouts too.
            resumed.save(6)
            resumed.close()
            self._assert_equal_checkpoints(
                path("fsdp2/step-6"), path(f"{source}_to_{target}/step-6")
            )

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

    @with_comms
    @with_temp_dir
    def test_checkpoints_across_backends(self):
        # FlexShard's checkpoints hold the same chunks of the same full tensors
        # as FSDP2's, so either backend resumes from the other's checkpoints,
        # bit for bit. Llama 3 shards its stacked linears on dim 1, which the
        # Hugging Face conversion splits.
        self._check_checkpoints_across_backends(build_llama3_config, "debugmodel", ep=1)

    @with_comms
    @with_temp_dir
    def test_checkpoints_across_backends_with_expert_parallelism(self):
        # The routed experts' chunks cover every ep rank's experts.
        self._check_checkpoints_across_backends(
            build_qwen3_config, "debugmodel_moe", ep=2
        )


def _chunks(metadata: TensorStorageMetadata) -> list[tuple[tuple[int, ...], ...]]:
    """A checkpointed tensor's non-empty chunks as (offsets, sizes) pairs.

    FSDP2 also saves an empty chunk for each rank without data.
    """
    return sorted(
        (tuple(chunk.offsets), tuple(chunk.sizes))
        for chunk in metadata.chunks
        if all(chunk.sizes)
    )
