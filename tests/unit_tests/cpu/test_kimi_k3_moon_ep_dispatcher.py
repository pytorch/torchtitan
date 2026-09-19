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

from unittest.mock import patch

import pytest
import torch
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.parallel_dims import ParallelDims
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


class TestMoonEPMeshPrecondition(DTensorTestBase):
    """The mesh check against real ParallelDims meshes."""

    @property
    def world_size(self) -> int:
        return 4

    def _dims(self, **kwargs) -> ParallelDims:
        dims = ParallelDims(pp=1, world_size=self.world_size, **kwargs)
        dims.build_mesh()
        return dims

    @with_comms
    def test_moonep_mesh_requires_efsdp_of_one(self):
        with patch(
            "torchtitan.distributed.parallel_dims.device_type", self.device_type
        ):
            check_moonep_mesh(self._dims(dp_replicate=1, dp_shard=4, cp=1, tp=1, ep=4))
            with self.assertRaisesRegex(NotImplementedError, "efsdp == 1"):
                check_moonep_mesh(
                    self._dims(dp_replicate=1, dp_shard=4, cp=1, tp=1, ep=2)
                )
            with self.assertRaisesRegex(NotImplementedError, "dp_replicate"):
                check_moonep_mesh(
                    self._dims(dp_replicate=2, dp_shard=2, cp=1, tp=1, ep=2)
                )
