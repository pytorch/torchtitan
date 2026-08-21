# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""(Per-Head) Muon optimizer for the Kimi K3 experiment.

Muon (Jordan et al., 2024): momentum SGD whose update direction is
orthogonalized via a Newton-Schulz iteration -- for 2-D weight
matrices, replace the raw momentum G with ~ (G G^T)^-1/2 G, an
approximate orthogonal factor. K3 uses a Per-Head Muon variant; the
"per-head" part orthogonalizes each attention head's projection block
independently (heads share no orthogonality).

Scope (honest): the BASE Muon algorithm is published and implemented
faithfully here. The exact K3 Per-Head variant (which projections,
head grouping, Nesterov details) reconciles at 7.27; this provides a
correct, testable base + a per-head reshape hook. Non-2-D params
(embeddings, norms, biases, KDA vectors) fall back to AdamW, as in the
reference Muon recipe.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim.adamw import adamw as _torch_adamw
from torch.optim.optimizer import Optimizer

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.tools.logging import logger


def _newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate orthogonalization of a 2-D matrix via Newton-Schulz.

    Quintic iteration (Jordan's coefficients). Operates in bf16 for
    speed; returns same shape as G.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(Optimizer):
    """Muon for 2-D matrices; AdamW fallback for everything else.

    Args:
        lr: learning rate for the Muon (matrix) group.
        momentum: heavy-ball momentum.
        nesterov: use Nesterov momentum.
        ns_steps: Newton-Schulz iterations.
        per_head: if set, a param whose ``_muon_heads`` attribute is an
            int H reshapes to (H, out/H, in) and orthogonalizes each
            head block independently (Per-Head Muon).
        adamw_lr / adamw_betas / adamw_eps / weight_decay: fallback
            AdamW hyperparameters for non-2-D params.
    """

    def __init__(
        self,
        params,
        lr: float = 2e-2,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        per_head: bool = True,
        adamw_lr: float = 3e-4,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            per_head=per_head,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def _warn_if_per_head_is_inert(self) -> None:
        """Per-head Muon needs ``_muon_heads`` tags; without any it is just
        Muon. That degeneration is invisible in the loss, so say so once."""
        if getattr(self, "_per_head_checked", False):
            return
        self._per_head_checked = True
        for group in self.param_groups:
            if not group.get("per_head") or not group.get("use_muon", True):
                continue
            if any(getattr(p, "_muon_heads", None) for p in group["params"]):
                continue
            logger.warning(
                "Muon(per_head=True) but no parameter in this group carries "
                "_muon_heads, so every update falls back to full-matrix "
                "orthogonalization. Call tag_per_head_muon(model) before "
                "building the optimizer."
            )

    @torch.no_grad()
    def step(self, closure=None):
        self._warn_if_per_head_is_inert()
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                # Muon applies to 2-D weight matrices; else AdamW.
                if g.ndim == 2 and min(g.shape) > 1:
                    self._muon_update(p, g, group)
                else:
                    self._adamw_update(p, g, group)
        return loss

    def _muon_update(self, p, g, group):
        st = self.state[p]
        if "momentum_buffer" not in st:
            st["momentum_buffer"] = torch.zeros_like(g)
        buf = st["momentum_buffer"]
        buf.mul_(group["momentum"]).add_(g)
        d = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf

        heads = getattr(p, "_muon_heads", None)
        if group["per_head"] and heads and d.size(0) % heads == 0:
            # Orthogonalize each head's row-block independently.
            hd = d.view(heads, d.size(0) // heads, d.size(1))
            o = torch.stack(
                [_newton_schulz(hd[i], group["ns_steps"]) for i in range(heads)]
            ).view_as(d)
        else:
            o = _newton_schulz(d, group["ns_steps"])
        # scale by sqrt(max(1, rows/cols)) per the Muon recipe
        scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
        p.add_(o, alpha=-group["lr"] * scale)

    def _adamw_update(self, p, g, group):
        """AdamW for the params Muon does not orthogonalize, via torch's own kernel.

        This was a hand-rolled clone of torch.optim.adamw. The math was identical --
        checked before replacing it, not after: over 5 float32 steps the parameters
        came out BIT-identical and exp_avg_sq identical. The one difference is
        exp_avg, which drifts to ~1e-7 because torch fuses the first-moment update as
        a lerp where the clone did mul_ then add_. So this is a reuse change with a
        declared float32 last-bit difference in momentum state, not a bit-exact
        refactor, and a long run will not reproduce the clone's trajectory exactly.

        ``step`` is kept as a tensor because that is what the functional API takes.
        """
        st = self.state[p]
        if "exp_avg" not in st:
            st["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
            st["exp_avg"] = torch.zeros_like(g)
            st["exp_avg_sq"] = torch.zeros_like(g)
        b1, b2 = group["adamw_betas"]
        _torch_adamw(
            [p],
            [g],
            [st["exp_avg"]],
            [st["exp_avg_sq"]],
            [],
            [st["step"]],
            foreach=False,
            capturable=False,
            differentiable=False,
            fused=False,
            grad_scale=None,
            found_inf=None,
            has_complex=False,
            amsgrad=False,
            beta1=b1,
            beta2=b2,
            lr=group["adamw_lr"],
            weight_decay=group["weight_decay"],
            eps=group["adamw_eps"],
            maximize=False,
        )


# Report sec 2.5 scopes the per-head refinement to the Q, K and V projections:
# "instead of applying Newton-Schulz orthogonalization to the full Q, K, and V
# projection matrices, we partition their momentum matrices along the head
# dimension and orthogonalize each head's block separately." o_proj is excluded
# deliberately -- it is the head axis on its INPUT side, so a row partition
# would not correspond to heads at all.
_PER_HEAD_MLA = ("q_proj", "q_b_proj", "kv_b_proj")
_PER_HEAD_KDA = ("q_proj", "k_proj", "v_proj")


def tag_per_head_muon(model: nn.Module) -> int:
    """Mark every Q/K/V projection with its head count. Returns the count.

    Per-Head Muon is driven by a ``_muon_heads`` attribute on the parameter,
    which nothing set outside the tests -- so a real run silently degenerated to
    plain full-matrix Muon. Call this before building the optimizer.

    The head count is read from the owning attention module rather than guessed
    from shapes, and a projection whose output width is not a multiple of its
    head count is left untagged instead of partitioned wrongly.
    """
    from torchtitan.models.kimi_k3.model import KimiDeltaAttention, KimiMLAAttention

    tagged = 0
    for module in model.modules():
        if isinstance(module, KimiMLAAttention):
            names, heads = _PER_HEAD_MLA, module.num_heads
        elif isinstance(module, KimiDeltaAttention):
            names, heads = _PER_HEAD_KDA, module.num_heads
        else:
            continue
        for name in names:
            proj = getattr(module, name, None)
            if proj is None:
                continue
            weight = getattr(proj, "weight", None)
            if weight is None or weight.dim() != 2:
                continue
            if weight.size(0) % heads != 0:
                # e.g. a fused projection whose rows do not tile by head. Better
                # to run full-matrix Muon on it than to partition into blocks
                # that are not heads.
                continue
            # kv_b_proj's per-head block holds that head's K_nope rows AND its V
            # rows; "partition along the head dimension" keeps them together,
            # which is what a fused KV matrix makes them.
            weight._muon_heads = heads
            tagged += 1
    return tagged


# ----- Wiring Muon into torchtitan's optimizer container ------------------ #


class KimiOptimizersContainer(OptimizersContainer):
    """``OptimizersContainer`` that also knows about Muon.

    Core's ``_resolve_optimizer_cls`` hardcodes ``{Adam, AdamW}`` and raises
    ``NotImplementedError`` for anything else, and CLAUDE.md rules out editing
    core to accommodate an experiment. Subclassing keeps the addition local: the
    Config's ``_owner`` machinery builds this class, so a flavor pointing at
    ``KimiOptimizersContainer.Config`` gets Muon resolution and nothing else
    changes.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        """Needed even though it adds no fields.

        Configurable sets ``_owner`` per Config CLASS. Inheriting the parent's
        Config verbatim means ``_owner`` still points at OptimizersContainer, so
        ``build()`` returns core's container and Muon resolution never happens --
        the smoke failed with "Optimizer Muon not added" for exactly that reason.
        """

    @staticmethod
    def _resolve_optimizer_cls(name: str) -> type:
        if name == "Muon":
            return Muon
        return OptimizersContainer._resolve_optimizer_cls(name)


# Report sec 2.5: Muon for the matrix parameters, with the per-head refinement on
# the attention projections. Everything that is not a 2-D weight matrix -- norms,
# biases, the 1-D KDA parameters, embeddings and the LM head -- stays on AdamW,
# which is the standard Muon recipe rather than something specific to K3.
# Parameters Muon skips. Split by dimensionality because weight decay applies
# to one subset and not the other: decaying a 1-D parameter shrinks a gain or an
# offset toward zero, which is a change in the function rather than the
# capacity control decay is meant to be. The 2-D and 3-D entries keep decay.
_MUON_EXCLUDE_1D_PATTERNS = (
    r".*norm.*",  # RMSNorm gains
    r".*\.bias$",
    r".*A_log$",  # KDA decay rates
    r".*dt_bias$",
)
_MUON_EXCLUDE_DECAY_PATTERNS = (
    r".*embed_tokens.*",
    r".*lm_head.*",
    # AttnRes pseudo-queries are [1, D]: 2-D by ndim, so they stay here, but the
    # step function treats them as vectors (see step()'s min(shape) > 1 test).
    # Whether they should also be decay-exempt is a separate numerics question.
    r".*_res_proj\.weight$",
    r".*conv1d.*",  # short conv weights are 3-D
)
_MUON_EXCLUDE_PATTERNS = _MUON_EXCLUDE_1D_PATTERNS + _MUON_EXCLUDE_DECAY_PATTERNS


def default_muon(
    lr: float = 2e-2,
    *,
    adamw_lr: float = 3e-4,
    momentum: float = 0.95,
    ns_steps: int = 5,
) -> "OptimizersContainer.Config":
    """Muon on the matrix parameters, AdamW on everything else.

    The two learning rates are deliberately different: Muon's update is
    orthogonalized, so its scale is decoupled from the gradient magnitude and it
    wants a much larger lr than AdamW on the same model. Passing one lr for both
    is the usual way to make Muon look bad.
    """
    from torchtitan.components.optimizer import ParamGroupConfig

    adamw_kwargs = {"lr": adamw_lr, "betas": (0.9, 0.95), "eps": 1e-8}
    return KimiOptimizersContainer.Config(
        param_groups=[
            # AdamW first, and its no-decay half before its decaying half: the
            # container assigns each parameter to the FIRST matching pattern, so
            # narrower sets have to precede wider ones, and both have to precede
            # the catch-all Muon group.
            ParamGroupConfig(
                pattern="|".join(_MUON_EXCLUDE_1D_PATTERNS),
                optimizer_name="AdamW",
                optimizer_kwargs={**adamw_kwargs, "weight_decay": 0.0},
            ),
            ParamGroupConfig(
                pattern="|".join(_MUON_EXCLUDE_DECAY_PATTERNS),
                optimizer_name="AdamW",
                optimizer_kwargs={**adamw_kwargs, "weight_decay": 0.1},
            ),
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="Muon",
                optimizer_kwargs={
                    "lr": lr,
                    "momentum": momentum,
                    "ns_steps": ns_steps,
                    "per_head": True,
                },
            ),
        ],
        implementation="for-loop",
    )
