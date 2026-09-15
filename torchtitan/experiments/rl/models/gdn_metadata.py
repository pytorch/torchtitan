# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fixed-address request metadata for captured Attention Gym GDN execution."""

from dataclasses import dataclass

import torch


@dataclass
class GDNGraphMetadata:
    """One shared metadata buffer set per vLLM attention group.

    One extra null slot covers the padded token tail. Each token bucket uses a
    deterministic prefix view; staging may change values but never addresses.
    """

    query_start_loc: torch.Tensor
    state_indices: torch.Tensor
    has_initial_state: torch.Tensor

    @classmethod
    def allocate(cls, num_reqs: int, *, device: torch.device) -> "GDNGraphMetadata":
        return cls(
            query_start_loc=torch.zeros(num_reqs + 2, device=device, dtype=torch.int32),
            state_indices=torch.zeros(num_reqs + 1, device=device, dtype=torch.int32),
            has_initial_state=torch.zeros(
                num_reqs + 1, device=device, dtype=torch.bool
            ),
        )

    def for_capacity(self, token_capacity: int) -> "GDNGraphMetadata":
        num_reqs = min(self.state_indices.numel() - 1, token_capacity)
        return GDNGraphMetadata(
            self.query_start_loc[: num_reqs + 2],
            self.state_indices[: num_reqs + 1],
            self.has_initial_state[: num_reqs + 1],
        )

    def clear(self, *, token_capacity: int) -> None:
        """Prepare an inert dummy batch outside CUDA graph capture."""
        self.query_start_loc[0].zero_()
        self.query_start_loc[1:].fill_(token_capacity)
        self.state_indices.zero_()
        self.has_initial_state.zero_()

    def update(
        self,
        query_start_loc: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_state: torch.Tensor | None,
        *,
        num_reqs: int,
        num_tokens: int,
        token_capacity: int,
    ) -> None:
        """Stage a real batch before replay without changing any buffer address."""
        if not 0 <= num_reqs < self.state_indices.numel():
            raise ValueError("GDN request count exceeds captured metadata capacity")
        if not 0 <= num_tokens <= token_capacity:
            raise ValueError("GDN token count exceeds captured token capacity")
        self.query_start_loc[: num_reqs + 1].copy_(query_start_loc[: num_reqs + 1])
        self.query_start_loc[num_reqs + 1 :].fill_(token_capacity)
        self.state_indices[:num_reqs].copy_(state_indices[:num_reqs])
        self.state_indices[num_reqs:].zero_()
        if has_initial_state is None:
            self.has_initial_state[:num_reqs].fill_(True)
        else:
            self.has_initial_state[:num_reqs].copy_(has_initial_state[:num_reqs])
        self.has_initial_state[num_reqs:].zero_()
