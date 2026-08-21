# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantile Balancing for the Kimi K3 MoE router (tech report sec 2.3.3).

    Auxiliary-loss-free routing adds a per-expert bias to the router score used for
    Top-k selection only, so it regulates dispatch without touching the mixture
    weights or the router's gradients. Quantile Balancing solves for that bias
    instead of stepping it, which removes the step-size trade-off.

    See ``phase13_k3like_48b_posttrain/QUANTILE_BALANCING.md``.
    """

from __future__ import annotations

import torch


def topk_with_cutoff(
    scores_TE: torch.Tensor,
    bias_E: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-(k+1) routing: the k taken routes, plus the cutoff.

    Args:
        scores_TE: ``(T, E)`` raw router scores ``s = Sigmoid(W_r x)``.
        bias_E: ``(E,)`` current expert bias (selection only).
        top_k: ``k``.

    Returns:
        ``(expert_ids_TK, cutoff_T)``. The cutoff is the ``(k+1)``-th biased
        score, i.e. the threshold an expert must exceed to enter that token's
        Top-k; taking it from Top-(k+1) routing avoids a separate token-side
        quantile.
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
    """Exact QB bias (Eq. 14). Reference form for small batches and tests.

    Args:
        scores_TE: ``(T, E)`` raw router scores.
        cutoff_T: ``(T,)`` cutoffs from :func:`topk_with_cutoff`.
        top_k: ``k``.

    Returns:
        ``(E,)`` zero-mean bias, to be used on the NEXT step. Exact up to ties
        at the threshold -- see the module docstring on the atom at margin 0.
    """
    n = scores_TE.size(-1)
    margins_TE = (scores_TE - cutoff_T.unsqueeze(-1)).float()
    # Per expert, over tokens. "lower" interpolation keeps the result on an
    # actual margin value, which is what makes the count land on the target
    # exactly rather than between two order statistics.
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
    """Per-expert histogram of the margins ``s_{:,j} - alpha``.

    Counts are ADDITIVE across ranks and accumulation steps, which is what
    lets one all-reduce reconstruct the whole-batch distribution.

    Returns:
        ``(E, num_bins)`` int64 counts. Margins outside ``[lo, hi]`` are
        clamped into the end bins, so no token is dropped.
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
    """QB bias read from pooled margin histograms -- the method used at scale.

    Args:
        counts_EB: ``(E, num_bins)`` pooled counts. Sum the per-rank
            histograms with a single all-reduce before calling this.
        top_k: ``k``.

    Returns:
        ``(E,)`` zero-mean bias for the next step.

    Accuracy, measured on a deliberately skewed n=16 / k=2 / m=4096 router by
    iterating the update to its fixed point and reading the resulting load
    coefficient of variation (see :func:`expert_loads`), from cv 0.607:

        exact quantile      -> 0.053 after 60 updates, still descending
        histogram,  256     -> 0.160     histogram, 2048  -> 0.104
        histogram,  512     -> 0.147     histogram, 8192  -> 0.092

    The estimator trades residual imbalance for being computable at all: the
    exact quantile needs every margin in the global batch, millions of values
    per expert per step at K3's scale. The plateau is resolution-limited, so
    ``num_bins`` is the knob.

    Two findings about that plateau, both measured, neither obvious:

    * Interpolating inside the crossing bin is essential, not a refinement.
      Snapping to the bin's left edge restricts the bias to a lattice, making
      the update map piecewise constant; the iteration then locks at cv 0.232
      and never moves again. See :func:`_interp_quantile`.
    * The margin distribution has an ATOM at exactly 0, because ``alpha_i`` is
      itself one of the scores: ``s_ij - alpha_i`` is exactly 0 whenever expert
      j is token i's (k+1)-th, which for the most over-subscribed expert was 419
      of 4096 tokens. Handling that atom explicitly (counting it separately and
      placing the quantile at 0 when the target falls inside it) made the
      plateau WORSE, 0.154 vs 0.147, so it is not the limiting factor and the
      machinery is not carried. What DID help was dropping those boundary
      tokens from the estimate entirely -- excluded from both the bins and the
      total, the plateau fell to 0.118 at 512 bins. That is a deviation from
      Eq. 14 as written, so it is recorded here rather than adopted: this module
      implements the published rule, and a departure from it belongs in an
      ablation with training evidence behind it.
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

    Interpolating rather than snapping to the crossing bin's left edge matters:
    snapping restricts the bias to a lattice of bin edges, making the update map
    piecewise constant, so the per-step iteration terminates at a lattice fixed
    point instead of converging. Measured on the skewed n=16 setup, snapping
    locked at load cv 0.232 forever.
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

    The quantity QB drives toward the target load ``q = m*k/n``. Note it
    RE-ROUTES with the new bias, so it also moves every cutoff; it is the
    trajectory measure, not a check of the per-step quantile solve.
    """
    ids, _ = topk_with_cutoff(scores_TE, bias_E, top_k)
    return torch.bincount(ids.reshape(-1), minlength=scores_TE.size(-1))


# ----- Runtime integration ------------------------------------------------ #
#
# The pieces above are pure functions. Wiring them to a training run needs
# three things the sign rule does not: the per-token cutoff alpha, margins
# pooled over the WHOLE global batch, and a bias that is SOLVED rather than
# accumulated.
#
# Where each comes from:
#   * alpha -- the router returns raw ``scores_BLE`` as its third output, so a
#     forward hook can recompute Top-(k+1) and take the (k+1)-th biased score.
#     One extra topk on (B, L, E), negligible beside the expert GEMMs, and it
#     needs no change to core's router.
#   * global-batch pooling -- histogram counts are additive, so accumulating
#     across gradient-accumulation micro-batches is just addition, and one
#     all-reduce over the loss mesh covers the sharded token axes.
#   * solved bias -- the update OVERWRITES ``expert_bias_E`` instead of adding
#     to it. Core's optimizer hook still applies its sign-rule delta first;
#     overwriting makes that delta irrelevant rather than fighting it, and
#     keeping core's hook registered is what keeps the buffer allocated and the
#     per-expert token counts zeroed each step.


class QuantileBalancer:
    """Drives Quantile Balancing over a training run.

    Usage as a ``post_optimizer_build_fn``::

        post_optimizer_build_fn=register_quantile_balancing

    Memory: one ``(E, num_bins)`` int32 histogram per MoE layer, e.g. 896
    experts x 512 bins x 4 B x 92 layers ~= 169 MiB at K3's full size. Reduce
    ``num_bins`` to trade quantile resolution for that.
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
            with torch.no_grad():
                scores_TE = scores_BLE.detach().reshape(-1, scores_BLE.size(-1))
                bias = bias_E.to_local() if hasattr(bias_E, "to_local") else bias_E
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
                prev = self._counts.get(key)
                self._counts[key] = counts if prev is None else prev + counts

        return hook

    @torch.no_grad()
    def step(self) -> None:
        """Solve for and install each layer's bias. Call once per optimizer step."""
        if not self._counts:
            return  # no forward ran since the last step (e.g. step 0 resume)
        import torch.distributed as dist

        if self.loss_group is not None and dist.is_initialized():
            # Counts are additive, so one SUM all-reduce reconstructs the
            # whole-batch margin distribution regardless of how tokens are
            # sharded across dp/cp.
            #
            # One collective for every layer, not one per layer. Reducing inside
            # the loop below issues a blocking all-reduce per MoE layer, and they
            # serialise: 92 of them per optimizer step at K3's depth, each paying
            # full latency for num_bins int32 values. Every histogram has the
            # same shape (num_bins is one attribute, not per layer) and the same
            # dtype, so they stack. Integer SUM is exact, so this cannot move the
            # numbers -- only the number of round trips.
            keys = list(self._counts)
            stacked = torch.stack([self._counts[k] for k in keys])
            dist.all_reduce(stacked, group=self.loss_group, op=dist.ReduceOp.SUM)
            for i, key in enumerate(keys):
                self._counts[key] = stacked[i]

        for key, counts in self._counts.items():
            bias = quantile_balance_bias_histogram(
                counts.to(torch.int64), self._top_k[key], lo=self.lo, hi=self.hi
            )
            target = self._moes[key].expert_bias_E
            if hasattr(target, "to_local"):
                target.to_local().copy_(bias.to(target.dtype))
            else:
                target.copy_(bias.to(target.dtype))
        self._counts.clear()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def register_quantile_balancing(
    optimizers, model_parts, parallel_dims, *, num_bins: int = 512
) -> QuantileBalancer:
    """``post_optimizer_build_fn`` that replaces the sign rule with QB.

    Registered AFTER core's ``register_moe_load_balancing_hook`` equivalent, so
    the solved bias is the last write each step.
    """
    loss_mesh = parallel_dims.get_optional_mesh("loss")
    balancer = QuantileBalancer(
        model_parts,
        num_bins=num_bins,
        loss_group=None if loss_mesh is None else loss_mesh.get_group(),
    )
    optimizers.register_step_pre_hook(lambda *a, **kw: balancer.step())
    return balancer
