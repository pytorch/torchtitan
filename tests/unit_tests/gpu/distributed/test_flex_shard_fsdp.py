# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``parallelism.fsdp_backend=FlexShardConfig()`` against FSDP2, bit for bit."""

import os
import re
from functools import partial
from itertools import count, product
from typing import Any, cast, Literal
from unittest.mock import patch

import pytest
import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.distributed.distributed_c10d as c10d
from safetensors.torch import save_file
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torch.distributed.tensor import DTensor, Shard
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import AdamW, DistMuon, EMA, OptimizersContainer
from torchtitan.config import PrecisionConfig
from torchtitan.config.configs import DebugConfig
from torchtitan.config.parallelism import (
    FlexShardConfig,
    FSDP2Config,
    ParallelismConfig,
)
from torchtitan.config.transform import ContextParallelTransform
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.accelerator_graph import (
    accelerator_graph_teardown,
    NUM_ACCELERATOR_GRAPH_WARMUP_STEPS,
    wrap_fwd_bwd_with_accelerator_graph,
    wrap_with_accelerator_graph,
)
from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
    Owned,
)
from torchtitan.distributed.flex_shard_fsdp import (
    flex_shard_bucket,
    flex_shard_reduce_meshes,
    load_hf_into_local_shards,
)
from torchtitan.distributed.fsdp import post_validate_fsdp_for_accelerator_graph
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.common.attention.cp_attention import (
    AllGatherKVFlexInnerAttention,
)
from torchtitan.models.common.param_init import skip_param_init
from torchtitan.models.deepseek_v3 import build_model_config as build_deepseek_v3_config
from torchtitan.models.deepseek_v3.mtp import MTPCrossEntropyLoss
from torchtitan.models.llama3 import build_model_config as build_llama3_config
from torchtitan.models.qwen3 import build_model_config as build_qwen3_config

flex_shard = pytest.importorskip("flex_shard")

pytestmark = pytest.mark.multi_gpu

# Tokens per cp rank: CP's flex attention needs a 128-token block on each.
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


@instantiate_parametrized_tests
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
            data_parallel_shard_degree=parallelism_context.dp_shard,
            tensor_parallel_degree=parallelism_context.tp,
            enable_sequence_parallel=parallelism_context.enable_sequence_parallel,
            context_parallel_degree=parallelism_context.cp,
            expert_parallel_degree=parallelism_context.ep,
            fsdp_backend=fsdp_backend,
            fsdp_reshard_after_forward=reshard_after_forward,
        )
        model_config = build_config(
            flavor,
            parallelism=parallelism,
            seq_len=_SEQ_LEN * parallelism_context.cp,
            attn_backend="flex",
        )
        if parallelism_context.cp_enabled:
            # As the CP recipes convert it.
            model_config = ContextParallelTransform(
                inner_attention_map={FlexInnerAttention: AllGatherKVFlexInnerAttention}
            ).transform(model_config)
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
        # Ranks are laid out (dp_shard, cp, tp), and the buckets shard over
        # dp_shard and cp flattened.
        tp = parallelism_context.tp
        tp_rank, bucket_rank = self.rank % tp, self.rank // tp
        mismatched = []
        for fqn, param in model.named_parameters():
            (placement,) = flex_shard.get_placements(param)
            expected = full_params[fqn]
            if parallelism_context.ep > 1 and ".routed_experts." in fqn:
                # EP splits the experts, and edp_shard shards each EP rank's part.
                ep_mesh = parallelism_context.get_mesh("ep")
                expected = expected.chunk(ep_mesh.size())[ep_mesh.get_local_rank()]
                edp_shard = parallelism_context.get_mesh("edp_shard")
                expected = expected.chunk(edp_shard.size(), placement.dim)
                expected = expected[edp_shard.get_local_rank()]
            else:
                # TP splits a parameter in the dim where its tp rank's shape
                # differs, and the bucket shards that shard.
                for dim, (full, local) in enumerate(
                    zip(expected.shape, flex_shard.get_global_shape(param))
                ):
                    if full != local:
                        expected = expected.chunk(tp, dim)[tp_rank]
                expected = expected.chunk(self.world_size // tp, placement.dim)
                expected = expected[bucket_rank]
            if not torch.equal(param, expected):
                mismatched.append(fqn)
        # A shard can match on some ranks only, e.g. rank 0's when each rank
        # initializes its own shard. Fail on all ranks, not only on the others,
        # which would leave the rest waiting in training's collectives.
        mismatched_by_rank: list[Any] = [None] * self.world_size
        dist.all_gather_object(mismatched_by_rank, mismatched)
        self.assertFalse(
            any(mismatched_by_rank),
            msg=f"{context} init, mismatches by rank: {mismatched_by_rank}",
        )

    def _train(
        self,
        model,
        model_config,
        parallelism,
        parallelism_context,
        microbatches,
        chunked_loss,
        accelerator_graph: Literal["step", "microbatch_group"] | None = None,
        *,
        optim=None,
    ):
        if optim is None:
            optim = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=True)
        # With MTP layers, one prediction, one loss and one token count per
        # objective.
        num_objectives = 1 + len(getattr(model, "mtp_layers", None) or ())
        loss_cls = CrossEntropyLoss if num_objectives == 1 else MTPCrossEntropyLoss
        # Vocab-parallel under TP, as in the recipes.
        loss_config = loss_cls.Config(global_vocab_size=model_config.vocab_size)
        if chunked_loss:
            # As the training engine sets it up.
            loss_fn = ChunkedLossWrapper.Config(loss_fn=loss_config).build()
            loss_fn.set_lm_head(
                model.lm_head,
                flex_shard_bucket=(
                    flex_shard_bucket(model, "lm_head")
                    if isinstance(parallelism.fsdp_backend, FlexShardConfig)
                    else None
                ),
            )
            model._skip_lm_head = True
        else:
            loss_fn = loss_config.build()
        # Made outside the graphs: a host-to-device copy can't be captured.
        global_loss_token_counts = torch.tensor(
            _SEQ_LEN if num_objectives == 1 else [_SEQ_LEN] * num_objectives,
            device=self.device_type,
        )

        def forward_backward(prepared_microbatches) -> torch.Tensor:
            losses = []
            for inputs, labels, model_kwargs in prepared_microbatches:
                with parallelism_context.activate_spmd():
                    output = model(inputs, **model_kwargs)
                    loss, _ = loss_fn(output, labels, global_loss_token_counts)
                    with spmd.no_typecheck():
                        loss.backward()
                losses.append(loss.detach())
            return torch.stack(losses)

        # As the training engine wraps it: eager warmup calls, one capture,
        # then replays. "step" captures a step's microbatches. With
        # "microbatch_group", each microbatch is a group, and its graph is
        # replayed for each, accumulating into the existing sharded grads.
        run_forward_backward = forward_backward
        run_validation = None
        if accelerator_graph == "step":
            run_forward_backward = wrap_fwd_bwd_with_accelerator_graph(
                forward_backward,
                parameters=model.parameters(),
                num_warmup_iterations=NUM_ACCELERATOR_GRAPH_WARMUP_STEPS,
            )

            # As the validator with training graphs: after each step, a forward
            # without grad, eager once, then captured in the training graphs'
            # pool before the training capture, and replayed.
            @torch.no_grad()
            def validate(inputs, labels, model_kwargs) -> torch.Tensor:
                with parallelism_context.activate_spmd():
                    output = model(inputs, **model_kwargs)
                    loss, _ = loss_fn(output, labels, global_loss_token_counts)
                    post_validate_fsdp_for_accelerator_graph(
                        [model],
                        flex_shard_enabled=isinstance(
                            parallelism.fsdp_backend, FlexShardConfig
                        ),
                    )
                return loss

            run_validation = wrap_with_accelerator_graph(
                validate, num_warmup_iterations=1
            )
        elif accelerator_graph == "microbatch_group":
            run_group = wrap_with_accelerator_graph(
                forward_backward,
                num_warmup_iterations=NUM_ACCELERATOR_GRAPH_WARMUP_STEPS,
            )

            def run_forward_backward(prepared_microbatches) -> torch.Tensor:
                # A replay overwrites the previous replay's outputs.
                return torch.cat(
                    [
                        run_group([prepared]).clone()
                        for prepared in prepared_microbatches
                    ]
                )

        # One packed sequence per microbatch, as the data loader produces. CP
        # shards it, and the tp and cp ranks of a data-parallel rank share it.
        seq_len = _SEQ_LEN * parallelism_context.cp
        positions = torch.arange(seq_len, device=self.device_type)
        dp_rank = self.rank // (parallelism_context.cp * parallelism_context.tp)
        history = []
        # With graphs: the warmup calls, the capture and at least one replay.
        num_steps = (
            3 if accelerator_graph is None else NUM_ACCELERATOR_GRAPH_WARMUP_STEPS + 2
        )
        for step in range(num_steps):
            prepared_microbatches = []
            for microbatch in range(microbatches):
                generator = torch.Generator().manual_seed(
                    1000 * step + 10 * microbatch + dp_rank
                )
                tokens, labels = (
                    torch.randint(
                        model_config.vocab_size, (seq_len,), generator=generator
                    ).to(self.device_type)
                    for _ in range(2)
                )
                # In the SPMD context, as the trainer does, so that CP shards
                # the inputs.
                with parallelism_context.activate_spmd():
                    inputs, labels, model_kwargs, _ = model.preprocess_inputs(
                        {"input": tokens, "labels": labels, "positions": positions},
                        parallelism_context=parallelism_context,
                        parallelism=parallelism,
                        # The trainer's count for MoE load balancing, without
                        # padding.
                        global_routing_token_counts=torch.tensor(
                            [seq_len * parallelism_context.dp_shard * microbatches]
                            * num_objectives,
                            device=self.device_type,
                        ),
                    )
                prepared_microbatches.append((inputs, labels, model_kwargs))
            losses = list(run_forward_backward(prepared_microbatches).clone())
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
            # As the training engine: the per-group graph accumulates into
            # the grads that existed at capture.
            optim.zero_grad(set_to_none=accelerator_graph != "microbatch_group")
            if run_validation is not None:
                model.eval()
                losses.append(run_validation(*prepared_microbatches[0]).clone())
                model.train()
            history.append((losses, grad_norm, grads, self._local_params(model)))
        if accelerator_graph is not None:
            accelerator_graph_teardown()
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
        self,
        *,
        ep: int,
        cases,
        tp: int = 1,
        cp: int = 1,
        sequence_parallel: bool = False,
        accelerator_graph: Literal["step", "microbatch_group"] | None = None,
        dist_muon: bool = False,
    ) -> None:
        # One context, as in training: both backends shard over its meshes.
        parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=self.world_size // (tp * cp),
            cp=cp,
            tp=tp,
            pp=1,
            ep=ep,
            world_size=self.world_size,
            enable_sequence_parallel=sequence_parallel,
        )
        for (
            (build_config, flavor),
            reshard_after_forward,
            microbatches,
            chunked_loss,
        ) in cases:
            context = (
                f"{flavor} "
                f"reshard_after_forward={reshard_after_forward} "
                f"microbatches={microbatches} chunked_loss={chunked_loss} "
                f"accelerator_graph={accelerator_graph} dist_muon={dist_muon}"
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
                        self._train(
                            model,
                            *setup,
                            microbatches,
                            chunked_loss,
                            accelerator_graph,
                            optim=_dist_muon_optimizers(model, ep=ep)
                            if dist_muon
                            else None,
                        )
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
            self._assert_equal_histories(*histories, context)

    def _assert_equal_histories(
        self, expected_history, actual_history, context: str
    ) -> None:
        for step, (expected, actual) in enumerate(
            zip(expected_history, actual_history, strict=True)
        ):
            step_context = f"{context} step {step}"
            for expected_loss, actual_loss in zip(expected[0], actual[0], strict=True):
                self.assertTrue(
                    torch.equal(expected_loss, actual_loss),
                    msg=f"{step_context} loss",
                )
            self.assertTrue(
                torch.equal(expected[1], actual[1]), msg=f"{step_context} grad norm"
            )
            self._assert_equal_tensors(expected[2], actual[2], f"{step_context} grad")
            self._assert_equal_tensors(expected[3], actual[3], f"{step_context} param")

    @with_comms
    def test_matches_fsdp2(self):
        # Two microbatches cover one, and the reshard policy doesn't depend on
        # the model.
        self._check_matches_fsdp2(
            ep=1,
            cases=[
                ((build_llama3_config, "debugmodel"), "default", 2, False),
                ((build_qwen3_config, "debugmodel_moe"), "always", 2, False),
                ((_build_tied_llama3_config, "debugmodel_tied"), "default", 2, False),
            ],
        )

    @with_comms
    def test_matches_fsdp2_with_accelerator_graphs(self):
        # As the recipes train by default: in accelerator graphs, with
        # ChunkedLossWrapper, whose per-chunk backwards run inside the graph,
        # and with an eval graph in the same pool. EP stays 1: graphs under EP
        # need HybridEP, and FlexShard under EP supports only the all-to-all
        # dispatcher.
        self._check_matches_fsdp2(
            ep=1,
            cases=product(
                (
                    (build_llama3_config, "debugmodel"),
                    (build_qwen3_config, "debugmodel_moe"),
                ),
                ("default",),
                (1, 2),
                (True,),
            ),
            accelerator_graph="step",
        )

    @with_comms
    def test_matches_fsdp2_with_accelerator_graph_per_microbatch_group(self):
        # training.accelerator_graph_per_microbatch_group: each replay
        # reduce-scatters its microbatch's grads and adds them to the sharded
        # grads in place.
        self._check_matches_fsdp2(
            ep=1,
            cases=product(
                (
                    (build_llama3_config, "debugmodel"),
                    (build_qwen3_config, "debugmodel_moe"),
                ),
                ("default",),
                (2,),
                (True,),
            ),
            accelerator_graph="microbatch_group",
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
                [(build_qwen3_config, "debugmodel_moe")],
                ("default", "always"),
                (1, 2),
                (True,),
            ),
        )

    @with_comms
    def test_matches_fsdp2_with_tied_weights_and_chunked_loss(self):
        # With tied weights, lm_head's bucket is also tok_embeddings', and
        # flex_shard_bucket finds it by the shared weight's second name.
        self._check_matches_fsdp2(
            ep=1,
            cases=[
                ((_build_tied_llama3_config, "debugmodel_tied"), "default", 2, True)
            ],
        )

    @with_comms
    def test_matches_fsdp2_for_deepseek_v3(self):
        # DeepSeek V3 (without MTP layers) shards through Decoder._apply_fsdp,
        # as the other decoders do. Its debug model adds MLA attention, a dense
        # first block before the MoE blocks, and shared experts next to the
        # routed ones. The loss is ChunkedLossWrapper's, as in its recipes.
        self._check_matches_fsdp2(
            ep=2,
            cases=product(
                [(build_deepseek_v3_config, "debugmodel")],
                ("default",),
                (1, 2),
                (True,),
            ),
        )

    def _check_deepseek_v3_mtp_matches_fsdp2(self, *, ep: int) -> None:
        # One MTP layer, as in deepseek_v3_debugmodel_mtp. The embedding and
        # lm_head run once per objective, so their buckets sum two calls' grads
        # in each backward. The loss is MTPCrossEntropyLoss on full logits, and
        # inside ChunkedLossWrapper, as in the recipe, which runs lm_head on its
        # own on the chunks of both outputs.
        self._check_matches_fsdp2(
            ep=ep,
            cases=product(
                [(partial(build_deepseek_v3_config, num_mtp_layers=1), "debugmodel")],
                ("default",),
                (1, 2),
                (False, True),
            ),
        )

    @with_comms
    def test_matches_fsdp2_for_deepseek_v3_mtp(self):
        self._check_deepseek_v3_mtp_matches_fsdp2(ep=1)

    @with_comms
    def test_matches_fsdp2_for_deepseek_v3_mtp_with_expert_parallelism(self):
        self._check_deepseek_v3_mtp_matches_fsdp2(ep=2)

    @with_comms
    def test_matches_fsdp2_with_tensor_parallelism(self):
        # TP splits a parameter in one dim, which the bucket shards within each
        # tp rank's shard when it is the bucket's dim too. Under sequence
        # parallelism, the norm weights are typed R on tp, as are Qwen3's QK
        # norms and router: each tp rank's grad is partial, and both backends
        # all-reduce it over tp before the reduce-scatter. The grad norm reduces
        # over dp_shard, then tp. Qwen3 MoE needs ep >= tp, and the experts'
        # norm reduces over edp_shard, then ep.
        self._check_matches_fsdp2(
            ep=2,
            tp=2,
            sequence_parallel=True,
            cases=product(
                (
                    (build_llama3_config, "debugmodel"),
                    (build_qwen3_config, "debugmodel_moe"),
                ),
                ("default",),
                (1, 2),
                (True,),
            ),
        )

    @with_comms
    def test_matches_fsdp2_with_context_parallelism(self):
        # Each cp rank's grads cover its own tokens. FSDP2 sums them in its
        # reduce-scatter over dp_shard and cp, flattened into one axis, and so
        # do FlexShard's buckets. Qwen3's routed experts shard over it too.
        self._check_matches_fsdp2(
            ep=1,
            cp=2,
            cases=product(
                (
                    (build_llama3_config, "debugmodel"),
                    (build_qwen3_config, "debugmodel_moe"),
                ),
                ("default",),
                (1, 2),
                (False,),
            ),
        )

    @with_comms
    def test_dist_muon_matches_fsdp2(self):
        # DistMuon plans from FlexShard's storage layouts and steps its plain
        # local shards, with the collectives and updates it runs for FSDP2's
        # DTensors.
        self._check_matches_fsdp2(
            ep=1,
            cases=product(
                (
                    (build_llama3_config, "debugmodel"),
                    (build_qwen3_config, "debugmodel_moe"),
                ),
                ("default",),
                (2,),
                (False,),
            ),
            dist_muon=True,
        )

    @with_comms
    def test_dist_muon_matches_fsdp2_with_expert_parallelism(self):
        # DistMuon computes the routed experts where they're stored, on the
        # (edp_shard, ep) mesh, as for FSDP2's DTensors.
        self._check_matches_fsdp2(
            ep=2,
            cases=[((build_qwen3_config, "debugmodel_moe"), "default", 2, True)],
            dist_muon=True,
        )

    @with_comms
    @with_temp_dir
    @parametrize("tp", [1, 2])
    def test_checkpoints_match_fsdp2(self, tp: int):
        # With tp=2, TP splits the dense parameters in dim 0 or 1, and the
        # buckets shard each tp rank's part. Under sequence parallelism, the
        # norm weights are replicated over tp, so both backends save each of
        # their chunks once.
        self._check_checkpoints_match_fsdp2(tp=tp)

    def _check_checkpoints_match_fsdp2(
        self, *, tp: int, dist_muon: bool = False
    ) -> None:
        # FlexShard loads FSDP2's checkpoint of the model and its optimizer
        # states, and saves it again chunk for chunk and bit for bit, also as a
        # bf16 export. With ep=2, the routed experts are split over ep, then
        # over edp_shard.
        parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=self.world_size // tp,
            cp=1,
            tp=tp,
            pp=1,
            ep=2,
            world_size=self.world_size,
            enable_sequence_parallel=tp > 1,
        )

        def build(fsdp_backend, folder, **config):
            model, *setup = self._build(
                build_qwen3_config,
                "debugmodel_moe",
                fsdp_backend,
                "default",
                parallelism_context,
            )
            optimizers = (
                _dist_muon_optimizers(model, ep=2)
                if dist_muon
                else OptimizersContainer(
                    OptimizersContainer.Config(
                        optimizers=[AdamW.Config(pattern=".*", lr=1e-3)]
                    ),
                    model_parts=[model],
                )
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
            "loaded optimizer state",
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

    @with_comms
    @with_temp_dir
    def test_dist_muon_checkpoints_match_fsdp2(self):
        # DistMuon's momentum is a plain local shard with its parameter's
        # layout, so FlexShard loads and saves FSDP2's DistMuon checkpoints,
        # the routed experts' momentum split over ep, then over edp_shard.
        self._check_checkpoints_match_fsdp2(tp=1, dist_muon=True)

    @with_comms
    @with_temp_dir
    def test_ema_checkpoints_match_fsdp2(self):
        # FlexShard's EMA copies are plain local shards, which its checkpoints
        # place in the full tensors as FSDP2's do. Trained from FSDP2's initial
        # weights, it saves FSDP2's checkpoint, EMA included. It also loads
        # FSDP2's checkpoint into an EMA in GPU memory and into one offloaded
        # to pinned CPU memory, and saves it again. With ep=2, the routed
        # experts are split over ep, then over edp_shard.
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

        def build(
            fsdp_backend, folder, *, initial_params=None, offload_to_cpu=False, **config
        ):
            model, *setup = self._build(
                build_qwen3_config,
                "debugmodel_moe",
                fsdp_backend,
                "default",
                parallelism_context,
            )
            if initial_params is not None:
                with torch.no_grad():
                    for fqn, param in model.named_parameters():
                        param.copy_(initial_params[fqn])
            optimizers = OptimizersContainer(
                OptimizersContainer.Config(
                    optimizers=[AdamW.Config(pattern=".*", lr=1e-3)]
                ),
                model_parts=[model],
            )
            ema = EMA.Config(decays=[0.9], offload_to_cpu=offload_to_cpu).build(
                model_parts=[model]
            )
            # As Optim.step updates it, after each optimizer step.
            steps = count(1)
            optimizers.register_step_post_hook(lambda *_: ema.step(next(steps)))
            checkpointer = CheckpointManager(
                CheckpointManager.Config(
                    folder=folder, interval=1, keep_latest_k=0, **config
                ),
                model_parts=[model],
                optimizers=optimizers,
                lr_schedulers=cast(Any, None),
                ema=ema,
                extra_states={},
                sd_adapter=None,
                base_folder=self.temp_dir,
            )
            return model, setup, optimizers, checkpointer

        model, setup, optimizers, checkpointer = build(FSDP2Config(), "fsdp2")
        initial_params = self._local_params(model)
        self._train(model, *setup, 1, False, optim=optimizers)
        checkpointer.save(3)
        checkpointer.close()
        expected = os.path.join(self.temp_dir, "fsdp2", "step-3")

        model, setup, optimizers, checkpointer = build(
            FlexShardConfig(), "flex", initial_params=initial_params
        )
        self._train(model, *setup, 1, False, optim=optimizers)
        checkpointer.save(3)
        checkpointer.close()
        self._assert_equal_checkpoints(
            expected, os.path.join(self.temp_dir, "flex", "step-3")
        )

        for offload_to_cpu in (False, True):
            folder = f"flex_loaded_offload_to_cpu_{offload_to_cpu}"
            *_, checkpointer = build(
                FlexShardConfig(),
                folder,
                offload_to_cpu=offload_to_cpu,
                initial_load_path=expected,
                initial_load_model_only=False,
            )
            self.assertTrue(checkpointer.load())
            checkpointer.save(3)
            checkpointer.close()
            self._assert_equal_checkpoints(
                expected, os.path.join(self.temp_dir, folder, "step-3")
            )

    def _check_hf_loads_match_fsdp2(
        self, build_config, flavor: str, *, ep: int, tp: int = 1
    ):
        # FSDP2 trains three steps and exports a Hugging Face checkpoint. A
        # fresh FSDP2 model and a fresh FlexShard model, each from its own
        # initialization, load it: FSDP2 gets the exported parameters,
        # FlexShard's shards are their slices, and both train alike from them.
        parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=self.world_size // tp,
            cp=1,
            tp=tp,
            pp=1,
            ep=ep,
            world_size=self.world_size,
            enable_sequence_parallel=tp > 1,
        )

        def build(fsdp_backend, folder, **config):
            model, *setup = self._build(
                build_config, flavor, fsdp_backend, "default", parallelism_context
            )
            checkpointer = CheckpointManager(
                CheckpointManager.Config(folder=folder, **config),
                model_parts=[model],
                optimizers=cast(Any, None),
                lr_schedulers=cast(Any, None),
                ema=None,
                extra_states={},
                sd_adapter=type(model).state_dict_adapter_cls(model.config, None),
                base_folder=self.temp_dir,
            )
            return model, setup, checkpointer

        model, setup, export = build(FSDP2Config(), "fsdp2", last_save_in_hf=True)
        self._train(model, *setup, 1, False)
        export.save(3, last_step=True)
        export.close()
        exported = {
            fqn: cast(DTensor, param).full_tensor()
            for fqn, param in model.named_parameters()
        }

        loaded = []
        for fsdp_backend, folder in (
            (FSDP2Config(), "fsdp2"),
            (FlexShardConfig(), "flex"),
        ):
            model, setup, checkpointer = build(
                fsdp_backend,
                folder,
                initial_load_path=os.path.join(
                    self.temp_dir, "fsdp2", "exports", "step-3"
                ),
                initial_load_in_hf=True,
            )
            # FlexShard loads each parameter alone, so a conversion that needs
            # another parameter fails here, as it would across the groups of a
            # larger model.
            with patch(
                "torchtitan.distributed.flex_shard_fsdp._HF_LOAD_GROUP_BYTES", 1
            ):
                self.assertTrue(checkpointer.load())
            checkpointer.close()
            loaded.append((model, setup))
        (fsdp2, _), (flex, _) = loaded
        for fqn, param in fsdp2.named_parameters():
            self.assertTrue(
                torch.equal(cast(DTensor, param).full_tensor(), exported[fqn]),
                msg=f"FSDP2 {fqn}",
            )
        layouts = flex_shard.get_flex_shard_global_layouts(flex)
        mismatches = [
            fqn
            for fqn, param in flex.named_parameters()
            if not all(
                torch.equal(
                    param[_box(local_offset, size)],
                    exported[fqn][_box(global_offset, size)],
                )
                for global_offset, local_offset, size in zip(
                    layouts[fqn].global_offsets,
                    layouts[fqn].local_offsets,
                    layouts[fqn].local_sizes,
                    strict=True,
                )
            )
        ]
        # Fail on every rank together. Otherwise a rank whose shards match,
        # e.g. a tp rank whose columns start at 0, would wait in the training's
        # collectives for one that failed.
        all_mismatches: list[Any] = [None] * self.world_size
        dist.all_gather_object(all_mismatches, mismatches)
        self.assertFalse(
            any(all_mismatches), msg=f"FlexShard shards, by rank: {all_mismatches}"
        )
        self._assert_equal_histories(
            *(self._train(model, *setup, 1, False) for model, setup in loaded),
            f"{flavor} after the Hugging Face load",
        )

    @with_comms
    @with_temp_dir
    @parametrize("tp", [1, 2])
    def test_hf_loads_match_fsdp2(self, tp: int):
        # Llama 3 splits the packed QKV and the stacked w13, and permutes the
        # query and key rows for rotary embeddings. TP splits wo and w2 in
        # their last dim, so each shard holds columns of its rows of o_proj and
        # down_proj, and the other weights in a leading dim. The buckets then
        # shard each tp rank's part.
        self._check_hf_loads_match_fsdp2(build_llama3_config, "debugmodel", ep=1, tp=tp)

    @with_comms
    @with_temp_dir
    def test_hf_loads_match_fsdp2_with_tied_weights(self):
        # to_hf gives no tensor for lm_head's weight, which tok_embeddings'
        # Hugging Face tensor loads.
        self._check_hf_loads_match_fsdp2(
            _build_tied_llama3_config, "debugmodel_tied", ep=1
        )

    @with_comms
    @with_temp_dir
    def test_hf_loads_match_fsdp2_with_expert_parallelism(self):
        # Qwen3 splits the routed experts into one tensor per expert, which
        # ep=2 splits over EP ranks, then edp_shard.
        self._check_hf_loads_match_fsdp2(build_qwen3_config, "debugmodel_moe", ep=2)

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


@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize("group_bytes", [1 << 30, 1])
def test_hf_load_into_uneven_shards(tmp_path, group_bytes: int, tp: int) -> None:
    # Each simulated rank of (tp, dp_shard=5) holds the box torch.chunk gives it
    # of each parameter: its chunk of the bucket's dim (1 for w13, else 0) and,
    # with tp > 1, of the last dim, as tensor parallelism splits row-wise
    # parallel weights. Shards are uneven and end inside attention heads,
    # unlike the parity tests'. Each rank loads its shards without a process
    # group, with all parameters in one group or each alone.
    dp_shard = 5
    model_config = build_llama3_config(
        "debugmodel",
        parallelism=ParallelismConfig(),
        seq_len=_SEQ_LEN,
        attn_backend="flex",
    )
    with torch.device("meta"):
        model = model_config.build()
    sd_adapter = type(model).state_dict_adapter_cls(model_config, None)
    generator = torch.Generator().manual_seed(0)
    full = {
        key: torch.randn(value.shape, generator=generator)
        for key, value in model.state_dict().items()
    }
    hf_state_dict = sd_adapter.to_hf(
        {key: value.clone() for key, value in full.items()}
    )
    save_file(
        {key: value.contiguous() for key, value in hf_state_dict.items()},
        str(tmp_path / "model.safetensors"),
    )
    for tp_rank, dp_rank in product(range(tp), range(dp_shard)):
        state_dict, boxes = {}, {}
        for key, value in full.items():
            splits = {1 if key.endswith("w13.weight") else 0: (dp_shard, dp_rank)}
            if value.ndim > 1:
                splits[value.ndim - 1] = (tp, tp_rank)
            offsets, sizes = [], []
            for dim, length in enumerate(value.shape):
                num_chunks, rank = splits.get(dim, (1, 0))
                chunk = -(-length // num_chunks)
                offsets.append(min(rank * chunk, length))
                sizes.append(min(chunk, length - offsets[-1]))
            state_dict[key] = torch.zeros(sizes)
            flex_shard.set_global_layout(
                state_dict[key],
                flex_shard.GlobalLayout(
                    global_shape=tuple(value.shape),
                    global_offsets=(tuple(offsets),),
                    local_offsets=((0,) * value.ndim,),
                    local_sizes=(tuple(sizes),),
                ),
            )
            boxes[key] = _box(offsets, sizes)
        with patch(
            "torchtitan.distributed.flex_shard_fsdp._HF_LOAD_GROUP_BYTES",
            group_bytes,
        ):
            load_hf_into_local_shards(
                sd_adapter,
                state_dict,
                sd_adapter.get_hf_storage_reader(str(tmp_path)),
            )
        for key, local in state_dict.items():
            assert torch.equal(local, full[key][boxes[key]]), (tp_rank, dp_rank, key)


def _dist_muon_optimizers(model, *, ep: int) -> OptimizersContainer:
    """DistMuon for the transformer blocks' matrices, and AdamW for the other
    parameters, as the DistMuon recipes split them.

    The compute layouts take each of DistMuon's paths: the fused QKV weight per
    head, in place where its row shards hold whole heads; the other matrices
    each gathered whole on an owner; a stacked linear's matrices, spread over
    the ranks from their row shards; and the routed experts where they're
    stored.
    """

    def compute_layout(**shardings_by_mesh_axis) -> ComputeLayout:
        return ComputeLayout(
            shardings_by_mesh_axis=shardings_by_mesh_axis,
            compute_dtype=torch.bfloat16,
        )

    per_expert = (
        compute_layout(dp_shard=Shard(0))
        if ep == 1
        else ComputeLayout(
            shardings_by_mesh_axis={"edp_shard": Shard(0), "ep": Shard(0)},
            # EP splits the experts first, then edp_shard each ep rank's.
            shard_order_by_tensor_dim={0: ("ep", "edp_shard")},
            compute_dtype=torch.bfloat16,
        )
    )
    head_dim = model.layers["0"].attention.head_dim
    layouts = {
        "attention.qkv_linear.wqkv.weight": compute_layout(
            dp_shard=BlockShard(dim=0, block_sizes=(head_dim,))
        ),
        "attention.wo.weight": compute_layout(dp_shard=Owned()),
        "feed_forward.w13.weight": compute_layout(dp_shard=Shard(0)),
        "feed_forward.w2.weight": compute_layout(dp_shard=Owned()),
        "feed_forward.router.gate.weight": compute_layout(dp_shard=Owned()),
        "feed_forward.routed_experts.w13.weight": per_expert,
        "feed_forward.routed_experts.w2.weight": per_expert,
    }
    compute_sharding_by_fqn = {}
    bucket_configs = []
    for layer_id, block in model.layers.items():
        layer_layouts = {
            f"layers.{layer_id}.{name}": layouts[name]
            for name, _ in block.named_parameters()
            if name in layouts
        }
        compute_sharding_by_fqn.update(layer_layouts)
        bucket_configs.append(BucketConfig(patterns=tuple(layer_layouts)))
    return OptimizersContainer(
        OptimizersContainer.Config(
            optimizers=[
                DistMuon.Config(
                    pattern=f"^(?:{'|'.join(map(re.escape, compute_sharding_by_fqn))})$",
                    compute_sharding_by_fqn=compute_sharding_by_fqn,
                    bucket_configs=bucket_configs,
                ),
                AdamW.Config(pattern=".*", lr=1e-3),
            ]
        ),
        model_parts=[model],
    )


def _box(offsets, sizes) -> tuple[slice, ...]:
    return tuple(
        slice(offset, offset + size)
        for offset, size in zip(offsets, sizes, strict=True)
    )


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
