# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import unittest

from unittest.mock import patch

import pytest
import torch
import torch_remat as remat
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_utils import DeterministicGuard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.activation_remat import (
    ActivationRematConfig,
    DefaultActivationRematConfig,
    FullActivationRematConfig,
)
from torchtitan.distributed.spmd_types import set_current_spmd_mesh, set_spmd_meshes
from torchtitan.models.base.module import Module, ModuleDict
from torchtitan.models.common.activation import (
    BinaryActivationFn,
    ClampedSwiGLU,
    SwiGLU,
)
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.moe.token_dispatcher import (
    AllToAllTokenDispatcher,
    DeepEPTokenDispatcher,
    HybridEPTokenDispatcher,
)
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.gpt_oss.moe import GptOssGroupedLinear

deep_ep = pytest.importorskip("deep_ep")
# DeepEP < 2.5 named the unified buffer ``ElasticBuffer``.
_EP_BUFFER = getattr(deep_ep, "EPBuffer", None) or getattr(
    deep_ep, "ElasticBuffer", None
)

pytestmark = pytest.mark.multi_gpu

# DeepEP's combine needs the hidden size to be a multiple of 256.
_MODEL_DIM = 256
_NUM_TOKENS = 64
_TOP_K = 2


class _EPBlock(Module):
    def __init__(
        self,
        num_experts: int,
        dispatcher_config,
        *,
        grouped_linear_cls: type[GroupedLinear],
        activation_fn: BinaryActivationFn.Config,
        output_postprocess: Module.Config | None,
    ):
        super().__init__()
        # RoutedExperts.forward runs after EP has selected this rank's local
        # expert-weight shard. This test bypasses parallelization, so construct
        # that local view directly while the dispatcher retains the global E.
        num_local_experts = num_experts // torch.distributed.get_world_size()
        routed_experts = RoutedExperts.__new__(RoutedExperts)
        Module.__init__(routed_experts)
        routed_experts.w13 = grouped_linear_cls.Config(
            group_size=num_local_experts,
            in_features=_MODEL_DIM,
            out_features=_MODEL_DIM,
            num_linears=2,
        ).build()
        routed_experts.w2 = grouped_linear_cls.Config(
            group_size=num_local_experts,
            in_features=_MODEL_DIM,
            out_features=_MODEL_DIM,
        ).build()
        routed_experts.activation_fn = activation_fn.build()
        routed_experts.output_postprocess = (
            output_postprocess.build() if output_postprocess is not None else None
        )
        routed_experts.token_dispatcher = dispatcher_config.build()
        self.gate = torch.nn.Linear(_MODEL_DIM, num_experts, bias=False)
        with torch.no_grad():
            for parameter in routed_experts.parameters():
                parameter.normal_(std=0.05)
        self.routed_experts = routed_experts
        self.num_experts = num_experts

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        num_tokens = x_TD.shape[0]
        token_ids_T1 = torch.arange(num_tokens, device=x_TD.device).unsqueeze(-1)
        offsets_1K = torch.arange(_TOP_K, device=x_TD.device).unsqueeze(0)
        expert_ids_TK = (token_ids_T1 + offsets_1K) % self.num_experts
        # Scores depend on the gate, so the router gets a gradient too.
        scores_TK = torch.sigmoid(self.gate(x_TD).float()).gather(1, expert_ids_TK)
        routing_map_TE = torch.zeros(
            num_tokens, self.num_experts, dtype=torch.bool, device=x_TD.device
        ).scatter_(1, expert_ids_TK, True)
        out_TD = self.routed_experts(
            x_TD,
            scores_TK,
            expert_ids_TK,
            routing_map_TE,
        )
        # The loss is a bare consumer of the routed-expert output.
        remat.recompute_needs_tensor(out_TD)
        return out_TD.float().square().sum()


class _PostW2(_EPBlock):
    """Reference DeepEP experts with router scores applied after W2."""

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        num_tokens = x_TD.shape[0]
        token_ids_T1 = torch.arange(num_tokens, device=x_TD.device).unsqueeze(-1)
        offsets_1K = torch.arange(_TOP_K, device=x_TD.device).unsqueeze(0)
        expert_ids_TK = (token_ids_T1 + offsets_1K) % self.num_experts
        # Scores depend on the gate, so the router gets a gradient too.
        scores_TK = torch.sigmoid(self.gate(x_TD).float()).gather(1, expert_ids_TK)
        routing_map_TE = torch.zeros(
            num_tokens, self.num_experts, dtype=torch.bool, device=x_TD.device
        ).scatter_(1, expert_ids_TK, True)
        experts = self.routed_experts
        routed_RD, counts_E, metadata = experts.token_dispatcher.dispatch(
            x_TD, scores_TK, expert_ids_TK, routing_map_TE
        )
        offsets_E = counts_E.cumsum(dim=0, dtype=torch.int32)
        gate_up_R2F = experts.w13(routed_RD.bfloat16(), offsets_E)
        hidden_RF = experts.activation_fn(gate_up_R2F, offsets=offsets_E)
        routed_output_RD = experts.w2(hidden_RF, offsets_E).type_as(routed_RD)
        if experts.output_postprocess is not None:
            routed_output_RD = experts.output_postprocess(routed_output_RD)
        routed_output_RD = (
            routed_output_RD * metadata.scores_R.reshape(-1, 1).to(torch.float32)
        ).to(routed_output_RD.dtype)
        out_TD = experts.token_dispatcher.combine(
            routed_output_RD,
            metadata,
            x_TD,
        )
        remat.recompute_needs_tensor(out_TD)
        return out_TD.float().square().sum()


class _Model(Module):
    def __init__(self, block: Module):
        super().__init__()
        self.layers = ModuleDict({"0": block})

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return self.layers["0"](x_TD)


def _run_forward_backward(
    model: Module,
    x_TD: torch.Tensor,
) -> list[torch.Tensor]:
    model.zero_grad(set_to_none=True)
    input_TD = x_TD.detach().clone().requires_grad_(True)
    loss = model(input_TD)
    loss.backward()
    assert input_TD.grad is not None
    grads = [loss.detach().clone(), input_TD.grad.detach().clone()]
    grads += [parameter.grad.detach().clone() for parameter in model.parameters()]
    return grads


class _EPActivationCheckpointing(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def _check_ep_activation_checkpointing(
        self,
        ac_configs: tuple,
        *,
        num_dispatch_replays: int,
        num_combine_replays: int,
        dispatcher_config,
        expert_configs: tuple[
            tuple[type[GroupedLinear], BinaryActivationFn.Config, Module.Config | None],
            ...,
        ],
        buffer_type,
        dispatch_method: str,
        combine_method: str,
        exact: bool,
        capture: bool = False,
    ) -> None:
        mesh = init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("ep",),
        )
        set_spmd_meshes(
            dense_mesh=mesh,
            sparse_mesh=mesh,
            dense_sp_enabled=False,
        )
        num_experts = 4 * self.world_size
        for activation_remat, (
            grouped_linear_cls,
            activation_fn,
            output_postprocess,
        ) in itertools.product(ac_configs, expert_configs):
            # Spawned ranks run the test method without setUp, so set determinism
            # here, matching set_determinism in torchtitan/distributed/utils.py.
            # Filling empty outputs must stay off: it races with comm streams.
            with (
                DeterministicGuard(True, fill_uninitialized_memory=False),
                self.subTest(
                    ac=type(activation_remat).__qualname__,
                    linear=grouped_linear_cls.__name__,
                    postprocess=output_postprocess is not None,
                ),
                torch.autograd.set_multithreading_enabled(False),
                set_current_spmd_mesh(mesh),
            ):
                torch.manual_seed(42)
                baseline = _Model(
                    _EPBlock(
                        num_experts,
                        dispatcher_config,
                        grouped_linear_cls=grouped_linear_cls,
                        activation_fn=activation_fn,
                        output_postprocess=output_postprocess,
                    )
                ).to(self.device_type, torch.bfloat16)
                ac_model = _Model(
                    _EPBlock(
                        num_experts,
                        dispatcher_config,
                        grouped_linear_cls=grouped_linear_cls,
                        activation_fn=activation_fn,
                        output_postprocess=output_postprocess,
                    )
                ).to(self.device_type, torch.bfloat16)
                ac_model.load_state_dict(baseline.state_dict())
                # DeepEP: compare with router scores applied after W2. HybridEP:
                # compare with AllToAll, which has no capacity rows past the last
                # expert offset that could reach the postprocess weight grad.
                is_deepep = isinstance(dispatcher_config, DeepEPTokenDispatcher.Config)
                reference = None
                if is_deepep or isinstance(
                    dispatcher_config, HybridEPTokenDispatcher.Config
                ):
                    reference_block = _PostW2 if is_deepep else _EPBlock
                    reference = _Model(
                        reference_block(
                            num_experts,
                            (
                                dispatcher_config
                                if is_deepep
                                else AllToAllTokenDispatcher.Config(
                                    num_experts=num_experts, top_k=_TOP_K
                                )
                            ),
                            grouped_linear_cls=grouped_linear_cls,
                            activation_fn=activation_fn,
                            output_postprocess=output_postprocess,
                        )
                    ).to(self.device_type, torch.bfloat16)
                    reference.load_state_dict(baseline.state_dict())
                activation_remat.apply(ac_model)
                # As in training, the buffer is created after AC is applied. All
                # models share it, so the baseline also runs deterministic when
                # the AC model replays.
                baseline.layers["0"].routed_experts.token_dispatcher.init_buffer()
                ac_model.layers["0"].routed_experts.token_dispatcher.init_buffer()
                if reference is not None:
                    reference.layers["0"].routed_experts.token_dispatcher.init_buffer()

                num_calls = {"dispatch": 0, "combine": 0}
                original = {
                    "dispatch": getattr(buffer_type, dispatch_method),
                    "combine": getattr(buffer_type, combine_method),
                }

                def counted(name):
                    def call(*args, **kwargs):
                        num_calls[name] += 1
                        return original[name](*args, **kwargs)

                    return call

                torch.manual_seed(self.rank)
                x_TD = torch.randn(
                    _NUM_TOKENS,
                    _MODEL_DIM,
                    device=self.device_type,
                    dtype=torch.bfloat16,
                )
                if capture:
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            _run_forward_backward(ac_model, x_TD)
                    torch.cuda.current_stream().wait_stream(stream)
                with (
                    patch.object(
                        buffer_type,
                        dispatch_method,
                        autospec=True,
                        side_effect=counted("dispatch"),
                    ),
                    patch.object(
                        buffer_type,
                        combine_method,
                        autospec=True,
                        side_effect=counted("combine"),
                    ),
                ):
                    expected = _run_forward_backward(baseline, x_TD)
                    baseline_calls = dict(num_calls)
                    num_calls.update(dispatch=0, combine=0)
                    if capture:
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            actual = _run_forward_backward(ac_model, x_TD)
                        graph.replay()
                    else:
                        actual = _run_forward_backward(ac_model, x_TD)

                if reference is not None:
                    reference_values = _run_forward_backward(reference, x_TD)
                    # Normwise within BF16 rounding: a wrong postprocess weight grad
                    # has a relative error near 1.
                    bf16_rtol = 2 * torch.finfo(torch.bfloat16).eps
                    for value, reference_value in zip(expected, reference_values):
                        if is_deepep and output_postprocess is not None and exact:
                            torch.testing.assert_close(
                                value, reference_value, rtol=0, atol=0
                            )
                            continue
                        error_norm = torch.linalg.vector_norm(
                            value.float() - reference_value.float()
                        )
                        reference_norm = torch.linalg.vector_norm(
                            reference_value.float()
                        )
                        self.assertLessEqual(
                            error_norm.item(), bf16_rtol * reference_norm.item()
                        )

                # Forward dispatch and combine, plus their backward passes.
                self.assertEqual(baseline_calls, {"dispatch": 2, "combine": 2})
                self.assertEqual(
                    num_calls,
                    {
                        "dispatch": 2 + num_dispatch_replays,
                        "combine": 2 + num_combine_replays,
                    },
                    msg=f"unexpected EP communication replays: {num_calls}",
                )
                if exact:
                    for actual_grad, expected_grad in zip(actual, expected):
                        torch.testing.assert_close(
                            actual_grad, expected_grad, rtol=0, atol=0
                        )
                else:
                    # Not bitwise: the receive order can differ between the two
                    # runs, which changes the summation order of the expert
                    # weight grads.
                    for actual_grad, expected_grad in zip(actual, expected):
                        torch.testing.assert_close(actual_grad, expected_grad)


@unittest.skipUnless(
    torch.cuda.device_count() >= 2 and _EP_BUFFER is not None,
    "requires two CUDA devices and DeepEP v2",
)
class TestDeepEPActivationCheckpointing(_EPActivationCheckpointing):
    def _check_deepep_activation_checkpointing(self, ac_configs, *, num_replays):
        self._check_ep_activation_checkpointing(
            ac_configs,
            num_dispatch_replays=num_replays,
            num_combine_replays=0,
            dispatcher_config=DeepEPTokenDispatcher.Config(
                num_experts=4 * self.world_size,
                top_k=_TOP_K,
                hidden_dim=_MODEL_DIM,
                num_max_tokens_per_rank=_NUM_TOKENS,
            ),
            expert_configs=(
                (GroupedLinear, SwiGLU.Config(), None),
                (GptOssGroupedLinear, ClampedSwiGLU.Config(), None),
                (
                    GroupedLinear,
                    SwiGLU.Config(),
                    RMSNorm.Config(normalized_shape=_MODEL_DIM),
                ),
                (
                    GptOssGroupedLinear,
                    ClampedSwiGLU.Config(),
                    RMSNorm.Config(normalized_shape=_MODEL_DIM),
                ),
            ),
            buffer_type=_EP_BUFFER,
            dispatch_method="dispatch",
            combine_method="combine",
            exact=bool(num_replays),
        )

    @with_comms
    def test_dispatch_and_combine_are_saved_not_replayed(self):
        self._check_deepep_activation_checkpointing(
            (
                DefaultActivationRematConfig(),
                ActivationRematConfig(save_regions=["*ep_communication"]),
            ),
            num_replays=0,
        )

    @with_comms
    def test_region_ac_replay_is_exact(self):
        self._check_deepep_activation_checkpointing(
            (FullActivationRematConfig(), ActivationRematConfig(save_regions=[])),
            num_replays=1,
        )


@unittest.skipUnless(
    torch.cuda.device_count() >= 2 and hasattr(deep_ep, "HybridEPBuffer"),
    "requires two CUDA devices and HybridEP",
)
class TestHybridEPActivationCheckpointing(_EPActivationCheckpointing):
    def _check_hybridep_activation_checkpointing(
        self, ac_configs, *, num_replays, capture=False
    ):
        for capacity_factor in (1.0,) if capture else (None, 1.0):
            with self.subTest(capacity_factor=capacity_factor):
                self._check_ep_activation_checkpointing(
                    ac_configs,
                    num_dispatch_replays=num_replays,
                    num_combine_replays=num_replays,
                    dispatcher_config=HybridEPTokenDispatcher.Config(
                        num_experts=4 * self.world_size,
                        top_k=_TOP_K,
                        hidden_dim=_MODEL_DIM,
                        num_max_tokens_per_rank=_NUM_TOKENS,
                        non_blocking_capacity_factor=capacity_factor,
                    ),
                    expert_configs=(
                        (GroupedLinear, SwiGLU.Config(), None),
                        (
                            GroupedLinear,
                            SwiGLU.Config(),
                            RMSNorm.Config(normalized_shape=_MODEL_DIM),
                        ),
                    ),
                    buffer_type=deep_ep.HybridEPBuffer,
                    dispatch_method="dispatch_with_permute",
                    combine_method="combine_with_unpermute",
                    exact=True,
                    capture=capture,
                )

    @with_comms
    def test_dispatch_and_combine_are_saved_not_replayed(self):
        # Copying saved tensors must not release the CPU ID owning the handle.
        with remat.saved_tensors_hooks(
            lambda tensor: tensor.detach().clone(), lambda tensor: tensor
        ):
            self._check_hybridep_activation_checkpointing(
                (
                    DefaultActivationRematConfig(),
                    ActivationRematConfig(save_regions=["*ep_communication"]),
                ),
                num_replays=0,
            )

    @with_comms
    def test_region_ac_replay_is_exact(self):
        self._check_hybridep_activation_checkpointing(
            (FullActivationRematConfig(), ActivationRematConfig(save_regions=[])),
            num_replays=1,
        )

    @with_comms
    def test_saved_communication_is_graph_capturable(self):
        self._check_hybridep_activation_checkpointing(
            (DefaultActivationRematConfig(),),
            num_replays=0,
            capture=True,
        )
