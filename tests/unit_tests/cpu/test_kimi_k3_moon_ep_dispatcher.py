# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for the MoonEP wiring: what the spec selects, the import guard, and the mesh precondition.

Nothing here touches MoonEP itself, which needs its package and NVLink
multicast. The comparison against a dense reference runs against the real
package in ``tests/unit_tests/gpu/test_kimi_k3_moon_ep.py``.
"""

from types import SimpleNamespace

import pytest
import torch

from torchtitan.models.kimi_k3 import model_registry
from torchtitan.models.kimi_k3.moon_ep_dispatcher import (
    _import_moonep,
    MoonEPTokenDispatcher,
)
from torchtitan.models.kimi_k3.moon_ep_experts import (
    check_moonep_mesh,
    MoonEPGroupedExperts,
)


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


def _dims(*, dp_shard, cp=1, tp=1, ep, dp_replicate=False):
    # core keeps the efsdp axis whenever ep > 1 and sizes it dp_shard * cp * tp // ep.
    efsdp = SimpleNamespace(size=lambda: dp_shard * cp * tp // ep)
    return SimpleNamespace(
        dp_replicate_enabled=dp_replicate,
        dp_shard=dp_shard,
        cp=cp,
        tp=tp,
        ep=ep,
        get_optional_mesh=lambda _name, include_singleton_axes=False: efsdp,
    )


def test_moonep_mesh_requires_efsdp_of_one():
    check_moonep_mesh(_dims(dp_shard=2, ep=2))
    check_moonep_mesh(_dims(dp_shard=1, cp=2, ep=2))
    for dims in (
        _dims(dp_shard=2, cp=2, ep=2),
        _dims(dp_shard=2, tp=2, ep=2),
        _dims(dp_shard=4, ep=2),
    ):
        with pytest.raises(NotImplementedError, match="efsdp == 1"):
            check_moonep_mesh(dims)
    with pytest.raises(NotImplementedError, match="dp_replicate"):
        check_moonep_mesh(_dims(dp_shard=2, ep=2, dp_replicate=True))
