# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fixed-capacity NCCL EP BF16 transport with explicit derivatives."""

import os
import threading
import weakref
from dataclasses import dataclass

import torch
from torch.distributed._functional_collectives import wait_tensor

# TODO: These private TokenSwitch ops come from
# https://github.com/pytorch/pytorch/pull/204973 (BF16) and
# https://github.com/pytorch/pytorch/pull/204974 (the quantized dispatch, whose
# side channel carries source-row identity bytes next to BF16 rows). Switch to
# public APIs once torch.distributed._token_switch exposes them.
from torch.distributed._token_switch import (
    _combine_async_op,
    _combine_backward_async_op,
    _dispatch_backward_async_op,
    _dispatch_quantized_async_op,
    _NCCLEPObject,
    TokenSwitchNCCL,
)

from torchtitan.distributed.nccl_ep_capture import routing_capture_plan

# Shape suffixes: T = source tokens, D = hidden features,
# R = received expert rows, e = local experts, K = top-k experts,
# U = aligned routing metadata bytes, P = source assignment positions,
# X = arbitrary row payload features.


_buffers = {}


class _Buffer:
    def __init__(self, key):
        (
            group,
            num_experts,
            max_tokens,
            capacity,
            hidden_dim,
            top_k,
            ht_flow,
        ) = key
        self.key = key
        self.users = 0
        self.switch = TokenSwitchNCCL(
            group,
            num_experts,
            max_tokens,
            capacity,
            hidden_dim * 2 + 16 * (top_k + 1),
            ht_flow=ht_flow,
        )


def acquire_buffer(
    group,
    num_experts,
    max_tokens,
    capacity,
    hidden_dim,
    top_k,
    *,
    ht_flow="push_pull",
):
    key = (
        group,
        num_experts,
        max_tokens,
        capacity,
        hidden_dim,
        top_k,
        ht_flow,
    )
    if key not in _buffers:
        _buffers[key] = _Buffer(key)
    buffer = _buffers[key]
    buffer.users += 1
    return buffer


def release_buffer(buffer):
    assert buffer.users > 0
    buffer.users -= 1
    if buffer.users == 0:
        del _buffers[buffer.key]
        buffer.switch.close()


class RoutingSlots:
    """Retain independent native mappings through backward and graph teardown."""

    def __init__(self, switch, ids, num_local_experts, *, group, alignment=32):
        self.switch = switch
        self.group = group
        self.num_local_experts = num_local_experts
        self.alignment = alignment
        self.slots = []
        self.generations = []
        self.lock = threading.RLock()
        self._create(ids, "expert_major")

    def _create(self, ids, layout, per_expert_token_counts=None):
        # Collective: creating a handle also runs the routing update.
        routing = self.switch.create_routing(
            ids,
            per_expert_token_counts,
            layout=layout,
            alignment=self.alignment,
        )
        self.slots.append([routing, False, False])
        self.generations.append(0)
        return len(self.slots) - 1

    def acquire(self, ids, layout, per_expert_token_counts=None):
        """Lease a routing slot for ``ids``.

        Returns ``(index, routing, updated)``. ``updated`` is True when the slot
        was created for ``ids``; creation already ran the collective routing
        update, so the caller must not update again. Every call therefore runs
        exactly one collective update on each rank, whatever the local slot
        state.
        """
        with self.lock:
            capturing = ids.is_cuda and torch.cuda.is_current_stream_capturing()
            plan = routing_capture_plan.get()
            if capturing and plan is None:
                raise RuntimeError(
                    "NCCL EP CUDA capture requires a routing capture plan owner; "
                    "initialize the dispatcher before constructing CUDA graph wrappers."
                )
            updated = False
            for index, (routing, leased, _) in enumerate(self.slots):
                if (
                    not leased
                    and routing.topk_idx.shape == ids.shape
                    and routing.layout == layout
                ):
                    break
            else:
                if capturing:
                    raise RuntimeError(
                        "Warm up NCCL EP routing demand before CUDA capture."
                    )
                index = self._create(ids, layout, per_expert_token_counts)
                routing = self.slots[index][0]
                updated = True
            self.slots[index][1] = True
            self.slots[index][2] = capturing
            self.generations[index] += 1
            if plan is not None:
                if capturing:
                    plan.captured.append((self, index, self.generations[index]))
                else:
                    plan.demands[(self, ids.shape, layout)] += 1
            return index, routing, updated

    def release(self, index, generation=None):
        with self.lock:
            if generation is not None and self.generations[index] != generation:
                return
            if not self.slots[index][2]:
                self.slots[index][1] = False


class _RoutingLease:
    def __init__(self, pool, index):
        self.pool = pool
        self.index = index
        weakref.finalize(self, pool.release, index, pool.generations[index])


def _leased_native(routing, slots, index):
    """Return the versioned native routing, holding its slot while it lives."""
    # TODO: Routing exposes only the raw handle publicly; the versioned
    # _NCCLEPObject is needed so stale-routing use raises. Replace with a
    # public accessor in torch.distributed._token_switch
    # (https://github.com/pytorch/pytorch/pull/204973).
    native = routing._opaque_handle
    native.lease = _RoutingLease(slots, index)
    return native


@torch.library.custom_op(
    "nccl_ep::prepare_deferred_routing",
    mutates_args=(),
    schema=(
        "(Tensor ids, torch.distributed._token_switch._NCCLEPObject pool, "
        "int num_local_experts) -> "
        "torch.distributed._token_switch._NCCLEPObject"
    ),
)
def _prepare_routing(ids, pool, num_local_experts):
    slots = pool.value
    index, routing, updated = slots.acquire(ids.contiguous(), "expert_major")
    try:
        if not updated:
            routing = slots.switch.update_routing(routing, ids.contiguous())
        return _leased_native(routing, slots, index)
    except BaseException:
        slots.release(index)
        raise


@torch.library.custom_op(
    "nccl_ep::prepare_scanned_routing",
    mutates_args=(),
    schema=(
        "(Tensor ids, torch.distributed._token_switch._NCCLEPObject pool, "
        "int num_local_experts) -> "
        "(torch.distributed._token_switch._NCCLEPObject, Tensor)"
    ),
)
def _prepare_scanned_routing(ids, pool, num_local_experts):
    slots = pool.value
    counts_e = ids.new_empty((num_local_experts,), dtype=torch.int32)
    index, routing, updated = slots.acquire(
        ids.contiguous(), "expert_major", per_expert_token_counts=counts_e
    )
    try:
        if not updated:
            routing = slots.switch.update_routing(
                routing, ids.contiguous(), per_expert_token_counts=counts_e
            )
        return _leased_native(routing, slots, index), counts_e
    except BaseException:
        slots.release(index)
        raise


@_prepare_scanned_routing.register_fake
def _prepare_scanned_routing_fake(ids, pool, num_local_experts):
    return _NCCLEPObject(), ids.new_empty((num_local_experts,), dtype=torch.int32)


_prepare_scanned_routing.register_effect(torch.library.EffectType.ORDERED)


@_prepare_routing.register_fake
def _prepare_routing_fake(ids, pool, num_local_experts):
    return _NCCLEPObject()


_prepare_routing.register_effect(torch.library.EffectType.ORDERED)


@dataclass(frozen=True, kw_only=True)
class DispatchMetadata:
    native: _NCCLEPObject
    scores_R: torch.Tensor  # noqa: N815
    # None when native already returns canonical (source-ordered) rows.
    order_R: torch.Tensor | None  # noqa: N815
    inverse_R: torch.Tensor | None  # noqa: N815
    valid_R: torch.Tensor  # noqa: N815
    counts_e: torch.Tensor  # noqa: N815
    alignment: int


def source_identity_bytes(
    x_TD: torch.Tensor, ep_rank: int, max_source_tokens: int
) -> torch.Tensor:
    """Encode source rows in an aligned suffix; zero identifies native padding."""
    source_T = torch.arange(x_TD.shape[0], device=x_TD.device, dtype=torch.int64)
    source_T = source_T + ep_rank * max_source_tokens + 1
    return row_identity_bytes(source_T)


def row_identity_bytes(rows_R):
    shifts_U = torch.arange(8, device=rows_R.device, dtype=torch.int64) * 8
    identity_RU = ((rows_R[:, None] >> shifts_U) & 255).to(torch.uint8)
    return torch.cat((identity_RU, torch.zeros_like(identity_RU)), dim=1)


def source_rows(metadata_RU):
    shifts_U = torch.arange(8, device=metadata_RU.device, dtype=torch.int64) * 8
    return (metadata_RU[:, -16:-8].to(torch.int64) << shifts_U).sum(-1)


def canonical_expert_order(
    identity_RU: torch.Tensor,
    counts_e: torch.Tensor,
    max_source_tokens: int,
    *,
    native_source_order: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    """Order unique top-k assignments by source within each native expert bucket.

    Returns ``(order_R, inverse_R, valid_R)``; the permutation is ``None`` when
    native rows are already in canonical order.
    """
    source_R = source_rows(identity_RU)
    valid_R = source_R != 0
    rows_R = torch.arange(identity_RU.shape[0], device=identity_RU.device)
    expert_R = torch.bucketize(rows_R, counts_e.cumsum(0), right=True)
    keys_R = expert_R * (max_source_tokens + 2) + torch.where(
        valid_R, source_R, max_source_tokens + 1
    )
    # Include the unused-capacity expert bucket and its padding sentinel.
    max_key = (counts_e.numel() + 1) * (max_source_tokens + 2) - 1
    if max_key <= torch.iinfo(torch.int32).max:
        keys_R = keys_R.to(torch.int32)
    if native_source_order:
        torch._assert_async(
            (keys_R[1:] >= keys_R[:-1]).all(),
            "Native scan must return source-ordered expert-major rows.",
        )
        return None, None, valid_R
    order_R = keys_R.argsort(stable=True)
    # Integer addition avoids plain scatter's deterministic sorted fallback.
    inverse_R = torch.zeros_like(order_R)
    inverse_R.scatter_add_(0, order_R, rows_R)
    return order_R, inverse_R, valid_R[order_R]


class _CanonicalRowPermutation(torch.autograd.Function):
    """Use the inverse of a complete row permutation for its derivative."""

    @staticmethod
    def forward(ctx, values_RX, order_R, inverse_R):  # pyrefly: ignore[bad-override]
        ctx.save_for_backward(inverse_R)
        return values_RX[order_R]

    @staticmethod
    def backward(ctx, grad_output_RX):  # pyrefly: ignore[bad-override]
        (inverse_R,) = ctx.saved_tensors
        grad_values_RX = grad_output_RX[inverse_R]
        # Match accumulation into positive zero, including negative zero inputs.
        return (
            grad_values_RX
            - torch.copysign(torch.zeros_like(grad_values_RX), grad_values_RX),
            None,
            None,
        )


class _Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx,
        x_TD,
        scores_TK,
        ids_TK,
        pool,
        capacity,
        ep_rank,
        max_source_tokens,
        num_local_experts,
    ):
        native, counts_e = _prepare_scanned_routing(ids_TK, pool, num_local_experts)
        identity_TU = source_identity_bytes(x_TD, ep_rank, max_source_tokens)
        # The quantized dispatch forwards BF16 rows unchanged alongside opaque
        # side-channel bytes, here each row's source identity.
        received_RD, scores_R, _, identity_RU = _dispatch_quantized_async_op(
            native,
            x_TD,
            identity_TU,
            scores_TK.contiguous(),
            capacity,
            "expert_major",
            initialize_tokens=False,
        )
        counts_e = wait_tensor(counts_e)
        identity_RU = wait_tensor(identity_RU)
        received_RD = wait_tensor(received_RD)
        ctx.native = native
        ctx.num_tokens, ctx.top_k = scores_TK.shape
        ctx.mark_non_differentiable(counts_e, identity_RU)
        return (
            received_RD,
            wait_tensor(scores_R),
            counts_e,
            native,
            identity_RU,
        )

    @staticmethod
    def backward(  # pyrefly: ignore[bad-override]
        ctx,
        grad_tokens_RD,
        grad_scores_R,
        grad_counts,
        grad_native,
        grad_identity,
    ):
        tokens_TD, scores_TK = _combine_backward_async_op(
            ctx.native,
            grad_tokens_RD.contiguous(),
            grad_scores_R.contiguous(),
            ctx.num_tokens,
            ctx.top_k,
        )
        tokens_TD, scores_TK = wait_tensor(tokens_TD), wait_tensor(scores_TK)
        return tokens_TD, scores_TK, None, None, None, None, None, None


def _zero_bucket_padding_(rows_RD, valid_R, counts_e, alignment, order_R=None):
    """Zero the uninitialized alignment rows inside each GEMM group.

    Canonical order puts the at most ``alignment - 1`` padding rows
    last in each bucket, so only those last rows of each bucket are visited.
    Visited real rows are written back unchanged, so duplicate indices all
    write the same value. With ``order_R``, ``rows_RD`` is in native order and
    canonical row ``i`` is native row ``order_R[i]``. Rows past the last bucket
    stay uninitialized: grouped GEMMs skip them and native combine gathers only
    mapped rows.
    """
    ends_e = counts_e.cumsum(0)
    offsets_J = torch.arange(1, alignment, device=ends_e.device)
    rows_N = (ends_e[:, None] - offsets_J).clamp_min(0).flatten()
    keep_N1 = valid_R[rows_N, None]
    if order_R is not None:
        rows_N = order_R[rows_N]
    rows_RD[rows_N] = torch.where(keep_N1, rows_RD[rows_N], 0)


def dispatch(
    x_TD,
    scores_TK,
    ids_TK,
    pool,
    capacity,
    ep_rank,
    max_source_tokens,
    ep_size,
    num_local_experts,
    *,
    alignment,
    ht_flow="push_pull",
):
    sorted_ids_TK = ids_TK.sort(-1).values
    torch._assert_async(
        (
            (sorted_ids_TK[:, 1:] != sorted_ids_TK[:, :-1]) | (sorted_ids_TK[:, 1:] < 0)
        ).all(),
        "NCCL EP requires unique top-k experts per source token.",
    )
    (received_RD, scores_R, counts_e, native, identity_RU,) = _Dispatch.apply(
        x_TD,
        scores_TK,
        ids_TK,
        pool,
        capacity,
        ep_rank,
        max_source_tokens,
        num_local_experts,
    )
    order_R, inverse_R, valid_R = canonical_expert_order(
        identity_RU,
        counts_e,
        max_source_tokens * ep_size,
        # Native returns source-ordered rows only on the AllGather + scan
        # metadata path with pull_push; the default count exchange does not.
        native_source_order=(
            os.environ.get("NCCL_EP_HT_AG_SCAN_MODE") == "1" and ht_flow == "pull_push"
        ),
    )
    if order_R is not None:
        received_RD = _CanonicalRowPermutation.apply(received_RD, order_R, inverse_R)
        scores_R = _CanonicalRowPermutation.apply(scores_R, order_R, inverse_R)
    received_RD = torch.where(valid_R[:, None], received_RD, 0)
    scores_R = torch.where(valid_R, scores_R, 0)
    metadata = DispatchMetadata(
        native=native,
        scores_R=scores_R,
        order_R=order_R,
        inverse_R=inverse_R,
        valid_R=valid_R,
        counts_e=counts_e,
        alignment=alignment,
    )
    return received_RD, counts_e, metadata


class _Combine(torch.autograd.Function):
    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx,
        x_RD,
        native,
        num_tokens,
        valid_R,
        counts_e,
        alignment,
        order_R,
    ):
        ctx.native = native
        ctx.alignment = alignment
        ctx.save_for_backward(valid_R, counts_e, order_R)
        ctx.capacity = x_RD.shape[0]
        output_TD = _combine_async_op(
            native, x_RD.contiguous(), num_tokens, ctx.capacity
        )
        return wait_tensor(output_TD)

    @staticmethod
    def backward(ctx, grad_tokens_TD):  # pyrefly: ignore[bad-override]
        # Each expert contribution receives the same source-token derivative.
        output_RD = wait_tensor(
            _dispatch_backward_async_op(
                ctx.native,
                grad_tokens_TD.contiguous(),
                ctx.capacity,
                initialize_tokens=False,
            )
        )
        # The derivative feeds W2's backward directly, whose weight
        # gradient reads every row of each group, so padding rows must
        # not hold uninitialized (possibly NaN) values.
        valid_R, counts_e, order_R = ctx.saved_tensors
        _zero_bucket_padding_(output_RD, valid_R, counts_e, ctx.alignment, order_R)
        return output_RD, None, None, None, None, None, None


def combine(routed_output_RD, metadata, num_tokens):
    """Sum each source token's expert rows.

    The expert already scaled its W2 input by ``scores_R``, which is zero on
    invalid rows, so combine is a plain sum; native combine reads only mapped
    rows.
    """
    if metadata.order_R is not None:
        routed_output_RD = _CanonicalRowPermutation.apply(
            routed_output_RD, metadata.inverse_R, metadata.order_R
        )
    return _Combine.apply(
        routed_output_RD,
        metadata.native,
        num_tokens,
        metadata.valid_R,
        metadata.counts_e,
        metadata.alignment,
        metadata.order_R,
    )
