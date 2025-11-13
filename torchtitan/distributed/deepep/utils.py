###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
from typing import Optional

import torch
from deep_ep import Config

# from .fused_indices_converter import fused_indices_to_multihot

# from .deepep import fused_dispatch, fused_combine
from .fused_a2a import fused_combine, fused_dispatch, set_deepep_num_sms
from .fused_indices_converter import fused_indices_to_multihot


def permute(
    tokens,
    routing_map,
    probs: Optional[torch.Tensor] = None,
):
    num_tokens, hidden = tokens.shape
    num_experts = routing_map.shape[1]
    permuted_probs = None
    # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
    routing_map = routing_map.bool().T.contiguous()

    # Create a dense expert-to-token mapping from the sparse token-to-expert mapping
    token_indices = (
        torch.arange(num_tokens, device=routing_map.device)
        .unsqueeze(0)
        .expand(num_experts, -1)
    )
    sorted_indices = token_indices.masked_select(routing_map)

    if probs is not None:
        permuted_probs = probs.T.contiguous().masked_select(routing_map)

    # use the mapping to permute the tokens
    permuted_input = tokens.index_select(0, sorted_indices)

    return permuted_input, permuted_probs, sorted_indices


def unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
):
    _, hidden = restore_shape
    input_dtype = permuted_tokens.dtype
    # Create an output tensor filled with zeros
    output_tokens = torch.zeros(
        restore_shape, dtype=permuted_tokens.dtype, device=permuted_tokens.device
    )
    # Scatter add the permuted_input back to the original positions
    output_tokens.scatter_add_(
        0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted_tokens
    )
    return output_tokens.to(dtype=input_dtype)


class PrimusTurboDeepepManager:

    _supported_backend_type = ["deepep", "mori"]
    cuda_dtoh_stream = None

    def __init__(
        self,
        router_topk: int,
        permute_fusion: bool = False,
        capacity_factor: Optional[float] = None,
        num_experts: Optional[int] = None,
        router_dtype: Optional[str] = None,
        backend_type: str = "deepep",
        deep_num_cus: int = 32,
        use_cuda_num_token_per_expert: bool = False,
        sync_free_moe: bool = False,
        num_worst_tokens: int = 0,
        dispatch_tuned_config: Optional[tuple] = None,
        combine_tuned_config: Optional[tuple] = None,
    ):
        self.router_topk = router_topk
        self.capacity_factor = capacity_factor
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        # self.num_local_experts = num_local_experts
        self.router_dtype = router_dtype
        self.num_worst_tokens = num_worst_tokens

        # Metadata
        self.token_indices: Optional[torch.Tensor] = None
        self.token_probs: Optional[torch.Tensor] = None
        # Handle used for combine operation

        self.handle = None

        if backend_type not in self._supported_backend_type:
            raise ValueError(f"only support {self._supported_backend_type}")

        self.backend_type = backend_type
        self.deep_num_cus = deep_num_cus
        self.use_cuda_num_token_per_expert = use_cuda_num_token_per_expert
        self.sync_free_moe = sync_free_moe

        def _get_deepep_config(config: tuple) -> Config:
            """
            Create a DeepEP Config from a tuned config tuple.

            The Config takes 5 parameters: Config(num_sms, param1, param2, param3, param4)

            Args:
                config: A tuple of 4 integers (param1, param2, param3, param4) that will be
                       unpacked after num_sms to create the full Config.

            Example tuned configs from DeepEP library (based on world size):

            Dispatch configs:
              2 ranks:  (24, 256, 6, 128)
              4 ranks:  (6, 256, 6, 128)
              8 ranks:  (6, 256, 6, 128)
              16 ranks: (36, 288, 20, 128)
              32 ranks: (32, 288, 8, 128)

            Combine configs:
              2 ranks:  (10, 256, 6, 128)
              4 ranks:  (9, 256, 6, 128)
              8 ranks:  (4, 256, 6, 128)
              16 ranks: (4, 288, 12, 128)
              32 ranks: (1, 288, 8, 128)

            Returns:
                Config object initialized with num_sms and the unpacked config tuple.
            """
            return Config(deep_num_cus, *config)

        if dispatch_tuned_config is not None:
            self.dispatch_config = _get_deepep_config(dispatch_tuned_config)
        else:
            self.dispatch_config = None

        if combine_tuned_config is not None:
            self.combine_config = _get_deepep_config(combine_tuned_config)
        else:
            self.combine_config = None

        if not self.sync_free_moe:
            if PrimusTurboDeepepManager.cuda_dtoh_stream is None:
                PrimusTurboDeepepManager.cuda_dtoh_stream = torch.cuda.Stream()

    @classmethod
    def maybe_cpu_sync(cls):
        if cls.cuda_dtoh_stream is not None:
            cls.cuda_dtoh_stream.synchronize()

    def setup_metadata(self, top_scores: torch.Tensor, selected_indices: torch.Tensor):
        # [bs * seqlen * topk]
        self.token_probs = top_scores.view(-1, self.router_topk)
        self.token_indices = selected_indices.view(-1, self.router_topk)
        # Mask the indices of dropped tokens with -1
        if self.capacity_factor is not None:
            mask = self.token_probs == 0
            self.token_indices = self.token_indices.masked_fill(mask, -1)

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        group: torch.distributed.ProcessGroup,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> torch.Tensor:
        # DeepEP only supports float32 probs
        self.num_local_experts = self.num_experts // torch.distributed.get_world_size(
            group
        )
        if self.token_probs.dtype != torch.float32:
            if self.token_probs.dtype in [torch.bfloat16, torch.float16]:
                print(
                    "DeepEP only supports float32 probs, please set --moe-router-dtype=fp32"
                )

            self.token_probs = self.token_probs.float()  # downcast or upcast

        (
            hidden_states,
            dispatched_indices,
            dispatched_probs,
            num_tokens_per_expert,
            handle,
        ) = fused_dispatch(
            hidden_states,
            self.token_indices,
            self.token_probs,
            self.num_experts,
            group,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
            use_cuda_num_token_per_expert=self.use_cuda_num_token_per_expert,
            num_worst_tokens=self.num_worst_tokens,
        )

        # use_cuda_num_token_per_expert not support on internode deepep for now!
        if not isinstance(num_tokens_per_expert, torch.Tensor):
            num_tokens_per_expert = torch.tensor(num_tokens_per_expert)

        self.handle = handle
        self.tokens_per_expert = num_tokens_per_expert
        self.dispatched_indices = dispatched_indices
        self.dispatched_probs = dispatched_probs
        self.num_recv_tokens = None

        if self.sync_free_moe:
            num_tokens = hidden_states.size(0)
            self.num_recv_tokens = torch.tensor(
                [self.router_topk * num_tokens], device="cpu", pin_memory=True
            )
        else:
            # Use async try to overlap cpu overhead.
            num_recv_tokens = torch.sum(self.tokens_per_expert)
            if num_recv_tokens.device.type != "cpu":
                self.cuda_dtoh_stream.wait_stream(torch.cuda.current_stream())
                num_recv_tokens.record_stream(self.cuda_dtoh_stream)
                with self.cuda_dtoh_stream:
                    self.num_recv_tokens = torch.empty_like(
                        num_recv_tokens,
                        dtype=num_recv_tokens.dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                    self.num_recv_tokens.copy_(num_recv_tokens, non_blocking=True)
            else:
                self.num_recv_tokens = num_recv_tokens
        return hidden_states

    def combine(
        self,
        hidden_states: torch.Tensor,
        group: torch.distributed.ProcessGroup,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> torch.Tensor:
        hidden_states, event = fused_combine(
            hidden_states,
            group,
            self.handle,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        # Release the handle after combine operation
        self.handle = None
        return hidden_states

    def get_restored_hidden_states_by_experts(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = unpermute(
            hidden_states,
            self.reversed_mapping_for_combine,
            restore_shape=self.hidden_shape_before_permute,
        )
        return hidden_states

    def get_permuted_hidden_states_by_experts(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if True:
            (
                self.dispatched_routing_map,
                self.dispatched_probs,
            ) = fused_indices_to_multihot(
                self.dispatched_indices, self.dispatched_probs, self.num_local_experts
            )
        else:
            raise RuntimeError("Please enable permute_fusion")
            (
                self.dispatched_routing_map,
                self.dispatched_probs,
            ) = self._indices_to_multihot(
                self.dispatched_indices, self.dispatched_probs
            )

        self.hidden_shape_before_permute = hidden_states.shape
        assert (
            self.dispatched_probs.dtype == torch.float32
        ), "DeepEP only supports float32 probs"

        hidden_states, permuted_probs, self.reversed_mapping_for_combine = permute(
            hidden_states,
            self.dispatched_routing_map,
            probs=self.dispatched_probs,
        )
        if self.router_dtype == "fp64":
            permuted_probs = permuted_probs.to(torch.float64)

        return hidden_states, permuted_probs


class PrimusTurboFlexTokenDispatcher:
    """
    PrimusTurbo token dispatcher using DeepEP or MORI.
    """

    turbo_deepep_backend: str = "deepep"
    turbo_sync_free_moe: bool = False
    turbo_deepep_num_worst_tokens: int = 0
    use_turbo_grouped_mlp: bool = False

    # default
    # turbo_deepep_num_cus: int = 32
    # turbo_deepep_dispatch_tuned_config: Optional[tuple] = None
    # turbo_deepep_combine_tuned_config: Optional[tuple] = None

    # === EP=4 Config (COMMENTED OUT - optimized for 4 GPUs) ===
    # Optimal configs from comprehensive num_sms grid tuning for B200 single-node EP=4
    # source: scripts/deepep/torchtitan_deepep_tune/summary/num_sms_tuning_summary.json (timestamp: 2025-11-13T16:22:33)
    # Tuned for: Qwen3-30B-A3B (128 experts, topk=8, hidden=2048, 4 GPUs, B200 ~600 SMs)
    # Comprehensive tuning tested num_sms: [8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 128]
    # turbo_deepep_num_cus: int = 128  # 56.3% faster dispatch, 61.8% faster combine vs num_sms=24
    # turbo_deepep_dispatch_tuned_config: Optional[tuple] = (32, 1024, 8, 128)  # 496.28 GB/s, 83.0% improvement vs worst
    # turbo_deepep_combine_tuned_config: Optional[tuple] = (16, 256, 8, 128)    # 476.36 GB/s, 84.0% improvement vs worst

    # === EP=8 Config (ACTIVE - optimized for 8 GPUs) ===
    # Optimal configs from comprehensive num_sms grid tuning for B200 single-node EP=8
    # source: scripts/deepep/torchtitan_deepep_tune/summary/num_sms_tuning_summary.json (timestamp: 2025-11-13T17:06:53)
    # Tuned for: Qwen3-30B-A3B (128 experts, topk=8, hidden=2048, 8 GPUs, B200 ~600 SMs)
    # Comprehensive tuning tested num_sms: [8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 128]
    # Key differences from EP=4: Smaller dispatch chunk (18 vs 32) and buffer (512 vs 1024) due to 8-way communication
    turbo_deepep_num_cus: int = (
        128  # 41.2% faster dispatch, 48.3% faster combine vs num_sms=24
    )
    turbo_deepep_dispatch_tuned_config: Optional[tuple] = (
        18,
        512,
        8,
        128,
    )  # 516.71 GB/s, 76.8% improvement vs worst
    turbo_deepep_combine_tuned_config: Optional[tuple] = (
        16,
        256,
        8,
        128,
    )  # 485.75 GB/s, 79.2% improvement vs worst

    def __init__(
        self,
        moe_router_topk: int,
        num_moe_experts: int,
    ):
        """
        Initialize the token dispatcher.
        """
        self.shared_experts = None

        self.tp_size = 1

        set_deepep_num_sms(self.turbo_deepep_num_cus)

        self._comm_manager = PrimusTurboDeepepManager(
            router_topk=self.tp_size * moe_router_topk,
            permute_fusion=False,
            capacity_factor=None,
            num_experts=self.tp_size * num_moe_experts,
            backend_type=self.turbo_deepep_backend,
            # use_cuda_num_token_per_expert=self.use_turbo_grouped_mlp,
            # NOTE: if return cuda token_per_expert, turbo groupgemm will cause cpu sync
            use_cuda_num_token_per_expert=False,
            sync_free_moe=False,
            num_worst_tokens=0,
            dispatch_tuned_config=self.turbo_deepep_dispatch_tuned_config,
            combine_tuned_config=self.turbo_deepep_combine_tuned_config,
        )

    def dispatch_preprocess(
        self, top_scores: torch.Tensor, selected_indices: torch.Tensor
    ):
        self._comm_manager.setup_metadata(top_scores, selected_indices)

    def token_dispatch(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor = None,
        group: torch.distributed.ProcessGroup = None,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ):
        return (
            self._comm_manager.dispatch(
                hidden_states, group, async_finish, allocate_on_comm_stream
            ),
            self._comm_manager.dispatched_probs,
        )

    def dispatch_postprocess(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        (
            global_input_tokens,
            permuted_probs,
        ) = self._comm_manager.get_permuted_hidden_states_by_experts(hidden_states)
        tokens_per_expert = self._comm_manager.tokens_per_expert
        return global_input_tokens, tokens_per_expert, permuted_probs

    def combine_preprocess(self, hidden_states: torch.Tensor):
        hidden_states = self._comm_manager.get_restored_hidden_states_by_experts(
            hidden_states
        )
        return hidden_states

    def token_combine(
        self,
        hidden_states: torch.Tensor,
        group: torch.distributed.ProcessGroup = None,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ):
        return self._comm_manager.combine(
            hidden_states, group, async_finish, allocate_on_comm_stream
        )

    def combine_postprocess(self, hidden_states: torch.Tensor):
        return hidden_states.view(self.hidden_shape)
