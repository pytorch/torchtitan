# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantile Balancing for the Kimi K3 MoE router (tech report sec 2.3.3).

The router bias enters Top-k selection only, so it regulates dispatch without
touching the mixture weights or the router's gradients. Quantile Balancing
solves for that bias each step instead of stepping it by a fixed amount.
"""

from __future__ import annotations

import torch
from torch.distributed.tensor import DTensor


def topk_with_cutoff(
    scores_TE: torch.Tensor,
    bias_E: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-(k+1) routing: ``(expert_ids_TK, cutoff_T)``.

    The cutoff is the ``(k+1)``-th biased score, the threshold an expert must
    exceed to enter that token's Top-k.
    """
    E = scores_TE.size(-1)
    if top_k + 1 > E:
        raise ValueError(
            f"Quantile Balancing routes with Top-(k+1), so top_k+1="
            f"{top_k + 1} must not exceed num_experts={E}"
        )
    vals, ids = torch.topk(scores_TE + bias_E, top_k + 1, dim=-1)
    return ids[..., :top_k], vals[..., top_k]


def quantile_balance_bias(
    scores_TE: torch.Tensor,
    cutoff_T: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Exact QB bias (Eq. 14), the reference form for small batches and tests.

    Returns an ``(E,)`` zero-mean bias for the next step.
    """
    n = scores_TE.size(-1)
    margins_TE = (scores_TE - cutoff_T.unsqueeze(-1)).float()
    # "lower" keeps the result on an actual margin, so the count lands on the
    # target rather than between two order statistics.
    b_hat = -torch.quantile(margins_TE, 1.0 - top_k / n, dim=0, interpolation="lower")
    return b_hat - b_hat.mean()


def margin_histogram(
    scores_TE: torch.Tensor,
    cutoff_T: torch.Tensor,
    *,
    num_bins: int = 512,
    lo: float = -1.0,
    hi: float = 1.0,
) -> torch.Tensor:
    """Per-expert histogram of the margins ``s_{:,j} - alpha``, ``(E, num_bins)``.

    Counts are additive across ranks and accumulation steps, so one all-reduce
    reconstructs the whole-batch distribution; margins outside ``[lo, hi]`` are
    clamped into the end bins.
    """
    E = scores_TE.size(-1)
    margins_TE = (scores_TE - cutoff_T.unsqueeze(-1)).float()
    edges = torch.linspace(lo, hi, num_bins + 1, device=margins_TE.device)
    idx = torch.bucketize(margins_TE.clamp(lo, hi), edges[1:-1])
    counts = torch.zeros(E, num_bins, dtype=torch.long, device=margins_TE.device)
    idx_ET = idx.t().contiguous()
    counts.scatter_add_(1, idx_ET, torch.ones_like(idx_ET))
    return counts


def quantile_balance_bias_histogram(
    counts_EB: torch.Tensor,
    top_k: int,
    *,
    lo: float = -1.0,
    hi: float = 1.0,
) -> torch.Tensor:
    """QB bias from pooled margin histograms, the form used at scale.

    ``counts_EB`` is the per-rank histograms already summed with one
    all-reduce; the exact quantile would need every margin in the global batch.
    Returns an ``(E,)`` zero-mean bias for the next step; ``num_bins`` sets the
    residual imbalance the estimator leaves.
    """
    E, num_bins = counts_EB.shape
    target = 1.0 - top_k / E
    edges = torch.linspace(lo, hi, num_bins + 1, device=counts_EB.device)
    total = counts_EB.sum(dim=1, keepdim=True).clamp(min=1)
    cdf = counts_EB.cumsum(dim=1).float() / total.float()
    b_hat = -_interp_quantile(cdf, target, edges, lo, hi, num_bins)
    return b_hat - b_hat.mean()


def _interp_quantile(
    cdf: torch.Tensor,
    target: float,
    edges: torch.Tensor,
    lo: float,
    hi: float,
    num_bins: int,
) -> torch.Tensor:
    """Quantile value where ``cdf`` crosses ``target``, interpolated in-bin.

    Snapping to the crossing bin's left edge instead would restrict the bias to
    a lattice and stop the iteration at a lattice point rather than the target.
    """
    bin_idx = (cdf < target).sum(dim=1).clamp(max=num_bins - 1)
    cdf_at = cdf.gather(1, bin_idx.unsqueeze(1)).squeeze(1)
    below = (bin_idx - 1).clamp(min=0)
    cdf_below = torch.where(
        bin_idx > 0,
        cdf.gather(1, below.unsqueeze(1)).squeeze(1),
        torch.zeros_like(cdf_at),
    )
    span = (cdf_at - cdf_below).clamp(min=1e-12)
    frac = ((target - cdf_below) / span).clamp(0.0, 1.0)
    width = (hi - lo) / num_bins
    return edges[bin_idx] + frac * width


def expert_loads(
    scores_TE: torch.Tensor,
    bias_E: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """``(E,)`` token count each expert would receive under ``bias_E``.

    Re-routes with the new bias, so it measures the trajectory rather than the
    per-step solve.
    """
    ids, _ = topk_with_cutoff(scores_TE, bias_E, top_k)
    return torch.bincount(ids.reshape(-1), minlength=scores_TE.size(-1))


# ----- Runtime integration ------------------------------------------------ #
# A forward hook on each router recomputes Top-(k+1) for the cutoff and adds to
# the layer's histogram; the step pools them over the loss mesh and overwrites
# expert_bias_E, which makes this the only writer of the bias.


class QuantileBalancer:
    """Drives Quantile Balancing over a training run.

    One ``(E, num_bins)`` int32 histogram per MoE layer: 169 MiB at K3's full
    size (896 experts, 512 bins, 92 layers), traded against resolution.
    """

    def __init__(
        self,
        model_parts,
        *,
        num_bins: int = 512,
        lo: float = -1.0,
        hi: float = 1.0,
        loss_group=None,
    ) -> None:
        self.num_bins = num_bins
        self.lo = lo
        self.hi = hi
        self.loss_group = loss_group
        self._handles: list = []
        # Layer identity is the MoE module itself; dict preserves insertion
        # order so the all-reduce stacks histograms in a stable layer order.
        self._counts: dict[int, torch.Tensor] = {}
        self._moes: dict[int, torch.nn.Module] = {}
        self._top_k: dict[int, int] = {}

        for moe in self._iter_moes(model_parts):
            if getattr(moe, "expert_bias_E", None) is None:
                raise ValueError(
                    "Quantile Balancing needs the expert_bias_E buffer, which "
                    "only exists when load_balance_coeff is set on the MoE"
                )
            key = id(moe)
            self._moes[key] = moe
            self._top_k[key] = moe.router.top_k
            # Pre-allocated, not lazily in the hook: a first-call branch changes
            # the op sequence between a forward and its checkpoint recompute.
            bias_E = moe.expert_bias_E
            assert isinstance(bias_E, torch.Tensor)
            self._counts[key] = torch.zeros(
                bias_E.numel(),
                self.num_bins,
                dtype=torch.int32,
                device=bias_E.device,
            )
            self._handles.append(moe.router.register_forward_hook(self._make_hook(key)))

    @staticmethod
    def _iter_moes(model_parts):
        from torchtitan.models.common.moe import MoE

        for part in model_parts:
            for m in part.modules():
                if isinstance(m, MoE):
                    yield m

    def _make_hook(self, key: int):
        def hook(router, args, output):
            # Router returns (topk_scores_BLK, topk_expert_ids_BLK, scores_BLE).
            scores_BLE = output[2]
            bias_E = self._moes[key].expert_bias_E
            assert isinstance(bias_E, torch.Tensor)
            with torch.no_grad():
                # Under TP the gate's output is a DTensor and expert_bias_E is
                # not; the scores are Replicate there, so to_local is exact.
                if isinstance(scores_BLE, DTensor):
                    scores_BLE = scores_BLE.to_local()
                scores_TE = scores_BLE.detach().reshape(-1, scores_BLE.size(-1))
                bias = bias_E.to_local() if isinstance(bias_E, DTensor) else bias_E
                _, cutoff_T = topk_with_cutoff(
                    scores_TE, bias.detach(), self._top_k[key]
                )
                counts = margin_histogram(
                    scores_TE,
                    cutoff_T,
                    num_bins=self.num_bins,
                    lo=self.lo,
                    hi=self.hi,
                ).to(torch.int32)
                self._counts[key].add_(counts)
            # Python state, not a tensor op, so the checkpoint op sequence is
            # unchanged; a recompute scales every count alike, which the solve ignores.
            self._armed = True

        return hook

    @torch.no_grad()
    def step(self) -> None:
        """Solve for and install each layer's bias. Call once per optimizer step."""
        if not getattr(self, "_armed", False):
            return  # no forward ran since the last step (e.g. step 0 resume)
        import torch.distributed as dist

        if self.loss_group is not None and dist.is_initialized():
            # Every layer's histogram has the same shape, so they stack into one
            # exact integer SUM rather than a blocking all-reduce per layer. The
            # loss mesh excludes tp, where the router's scores are Replicate and a
            # second sum would scale the histogram by the tp degree.
            keys = list(self._counts)
            stacked = torch.stack([self._counts[k] for k in keys])
            dist.all_reduce(stacked, group=self.loss_group, op=dist.ReduceOp.SUM)
            for i, key in enumerate(keys):
                self._counts[key].copy_(stacked[i])

        for key, counts in self._counts.items():
            bias = quantile_balance_bias_histogram(
                counts.to(torch.int64), self._top_k[key], lo=self.lo, hi=self.hi
            )
            target = self._moes[key].expert_bias_E
            assert isinstance(target, torch.Tensor)
            if isinstance(target, DTensor):
                target.to_local().copy_(bias.to(target.dtype))
            else:
                target.copy_(bias.to(target.dtype))
        # Zero in place rather than clear: the hook must find the same
        # buffer on every pass to stay branch-free under recompute.
        for counts in self._counts.values():
            counts.zero_()
        # Core's hook is not registered under this registration, so its per-step
        # reset of the MoE's own token counter falls to this step.
        for moe in self._moes.values():
            counter = getattr(moe, "tokens_per_expert_E", None)
            if isinstance(counter, torch.Tensor):
                counter.zero_()
        self._armed = False

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def register_quantile_balancing(
    optimizers, model_parts, parallel_dims, *, num_bins: int = 512
) -> QuantileBalancer:
    """``post_optimizer_build_fn`` that replaces the sign rule with QB.

    The slot holds one function, so registering this one leaves core's
    ``register_moe_load_balancing_hook`` unregistered.
    """
    loss_mesh = parallel_dims.get_optional_mesh("loss")
    balancer = QuantileBalancer(
        model_parts,
        num_bins=num_bins,
        loss_group=None if loss_mesh is None else loss_mesh.get_group(),
    )
    optimizers.register_step_pre_hook(lambda *a, **kw: balancer.step())
    return balancer
