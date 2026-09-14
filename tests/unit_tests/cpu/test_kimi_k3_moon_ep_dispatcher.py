# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for the MoonEP unit.

MoonEP itself needs NVLink hardware and its package; nothing here touches
either. The wiring is driven end to end through ``moonep_fake``: two ranks as
threads, the fake Buffer's collectives as barriers, a test-chosen duplication
map so prefetch slots and slot-grad reduction are exercised. The reference is
the dense per-token computation with every expert's fp32 weights.
"""

import threading

import pytest
import torch
import torch.nn as nn

from torchtitan.models.common.activation import SiTUGLU
from torchtitan.models.kimi_k3 import model_registry
from torchtitan.models.kimi_k3.moon_ep_dispatcher import (
    _import_moonep,
    MoonEPTokenDispatcher,
)
from torchtitan.models.kimi_k3.moon_ep_experts import MoonEPGroupedExperts

from tests.unit_tests.cpu.kimi_k3_moonep_fake import FakeMoonEPWorld, grouped_mm_loop


def _find_dispatcher(model):
    for _, mod in model.named_modules():
        if hasattr(mod, "token_dispatcher"):
            return mod.token_dispatcher
    raise AssertionError("no routed experts found")


def test_moonep_spec_selects_dispatcher_and_experts_sized_by_latent():
    spec = model_registry("debugmodel", moe_comm_backend="moonep")
    model = spec.model.build()
    dispatcher = _find_dispatcher(model)
    assert isinstance(dispatcher, MoonEPTokenDispatcher)
    layer = next(m for _, m in model.named_modules() if hasattr(m, "routed_down"))
    # The routed experts consume routed_down's output, so the buffer width is
    # the latent width, not the model width.
    assert dispatcher.hidden_dim == layer.routed_down.weight.shape[0]
    assert isinstance(layer.routed_experts.inner_experts, MoonEPGroupedExperts)


def test_moonep_ep1_falls_back_to_local_dispatch():
    """With no EP mesh the local fallback runs and moonep is never imported."""
    spec = model_registry("debugmodel", moe_comm_backend="moonep")
    model = spec.model.build()
    dispatcher = _find_dispatcher(model)
    dispatcher.wire_meshes(ep_mesh=None)
    num_tokens, num_experts, top_k = 8, dispatcher.num_experts, dispatcher.top_k
    x_TD = torch.randn(num_tokens, dispatcher.hidden_dim)
    scores_TK, ids_TK = torch.rand(num_tokens, num_experts).topk(top_k, dim=-1)
    counts_E = torch.zeros(num_experts, dtype=torch.long).scatter_add_(
        0, ids_TK.reshape(-1), torch.ones(num_tokens * top_k, dtype=torch.long)
    )
    routed_RD, counts_e, metadata = dispatcher.dispatch(
        x_TD, scores_TK, ids_TK, counts_E
    )
    assert counts_e.sum().item() == num_tokens * top_k
    assert dispatcher.combine(routed_RD, metadata, x_TD).shape == x_TD.shape


def test_moonep_import_guard_names_the_package():
    try:
        import moonep  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("moonep installed; the guard has nothing to raise")
    with pytest.raises(ImportError, match="MoonshotAI/MoonEP"):
        _import_moonep()


# --- the unit, end to end, against a dense reference ---------------------- #

R, E, K, S, D, F = 2, 8, 2, 16, 8, 16
# Rank 0 gets a copy of expert 5 (home rank 1); rank 1 a copy of expert 1.
DUP = {(0, 0): 5, (1, 0): 1}


def _reference(x_all, weights_all, ids_all, w1, w2, w3, beta, linear_beta):
    """Every token through its K experts, weighted, in fp32 -- what MoonEP
    must reproduce up to bf16 rounding."""
    act = SiTUGLU.Config(beta=beta, linear_beta=linear_beta).build()
    out = torch.zeros_like(x_all)
    for t in range(x_all.shape[0]):
        for k in range(K):
            e = int(ids_all[t, k])
            g = x_all[t] @ w1[e].T
            u = x_all[t] @ w3[e].T
            h = act(g, u)
            out[t] += weights_all[t, k] * (h @ w2[e].T)
    return out


def _run_rank(rank, world, params, inputs, results):
    torch.manual_seed(0)
    experts = MoonEPGroupedExperts(
        MoonEPGroupedExperts.Config(
            dim=D,
            hidden_dim=F,
            num_experts=E,
            activation_fn=SiTUGLU.Config(beta=4.0, linear_beta=25.0),
        )
    )
    lo, hi = rank * (E // R), (rank + 1) * (E // R)
    experts.w1_EFD = nn.Parameter(params["w1"][lo:hi].clone())
    experts.w2_EDF = nn.Parameter(params["w2"][lo:hi].clone())
    experts.w3_EFD = nn.Parameter(params["w3"][lo:hi].clone())
    dispatcher = MoonEPTokenDispatcher(
        MoonEPTokenDispatcher.Config(
            num_experts=E, top_k=K, hidden_dim=D, num_max_tokens_per_rank=S
        )
    )
    dispatcher._buffer_factory = lambda **kw: world.buffer_for(rank)
    mesh = world.mesh_for(rank)
    dispatcher.wire_meshes(ep_mesh=mesh)
    experts.attach(dispatcher, world.backend_for(rank), mesh)

    x, weights, ids, counts = inputs[rank]
    x = x.clone().requires_grad_(True)
    routed, rows, metadata = dispatcher.dispatch(x, weights, ids, counts)
    assert rows.numel() == E + E // R, "counts span the E + B table rows"
    assert rows[E:].sum().item() > 0, "the duplication map put tokens in a slot"
    expert_out = experts(routed, rows)
    out = dispatcher.combine(expert_out, metadata, x)
    out.float().sum().backward()
    results[rank] = (
        out.detach().float(),
        x.grad.clone(),
        experts.w1_EFD.grad.clone(),
        experts.w2_EDF.grad.clone(),
        experts.w3_EFD.grad.clone(),
    )


def test_moonep_unit_matches_dense_reference_with_duplicated_experts(monkeypatch):
    # The expert GEMMs run on CPU here; torch._grouped_mm is CUDA only.
    monkeypatch.setattr(
        "torchtitan.models.kimi_k3.moon_ep_experts._grouped_mm", grouped_mm_loop
    )
    torch.manual_seed(1)
    params = {
        "w1": torch.randn(E, F, D) * 0.2,
        "w2": torch.randn(E, D, F) * 0.2,
        "w3": torch.randn(E, F, D) * 0.2,
    }
    inputs = {}
    for r in range(R):
        x = torch.randn(S, D)
        weights, ids = torch.rand(S, E).topk(K, dim=-1)
        counts = torch.zeros(E, dtype=torch.long).scatter_add_(
            0, ids.reshape(-1), torch.ones(S * K, dtype=torch.long)
        )
        inputs[r] = (x, weights, ids.to(torch.int64), counts)
    world = FakeMoonEPWorld(
        num_ranks=R,
        num_experts=E,
        top_k=K,
        tokens_per_rank=S,
        hidden_dim=D,
        num_prefetch_slots=E // R,
        dup_map=DUP,
    )
    results = {}
    threads = [
        threading.Thread(target=_run_rank, args=(r, world, params, inputs, results))
        for r in range(R)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert len(results) == R, "a rank died or deadlocked in a fake collective"

    # Reference on the concatenated batch, fp32 params, weights applied.
    x_all = torch.cat([inputs[r][0] for r in range(R)])
    w_all = torch.cat([inputs[r][1] for r in range(R)])
    ids_all = torch.cat([inputs[r][2] for r in range(R)])
    x_ref = x_all.clone().requires_grad_(True)
    p_ref = {n: params[n].clone().requires_grad_(True) for n in params}
    ref = _reference(
        x_ref, w_all, ids_all, p_ref["w1"], p_ref["w2"], p_ref["w3"], 4.0, 25.0
    )
    ref.sum().backward()

    out = torch.cat([results[r][0] for r in range(R)])
    torch.testing.assert_close(out, ref.detach(), atol=5e-2, rtol=5e-2)
    grad_x = torch.cat([results[r][1] for r in range(R)])
    torch.testing.assert_close(grad_x, x_ref.grad, atol=5e-2, rtol=5e-2)
    for name, idx in (("w1", 2), ("w2", 3), ("w3", 4)):
        got = torch.cat([results[r][idx] for r in range(R)])
        # Includes the rows that other ranks computed in their prefetch slots
        # and reduced back home.
        torch.testing.assert_close(got, p_ref[name].grad, atol=5e-2, rtol=5e-2)
