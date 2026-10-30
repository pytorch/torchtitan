# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The full-scale Kimi K3 recipe selects Dist-MoE before graph validation."""

import pytest

from torchtitan_recipes.models import kimi_k3


def test_full_recipe_selects_dist_moe_before_graph_validation(monkeypatch) -> None:
    """Graph support validates the transformed backend, not the stock dispatcher."""
    pytest.importorskip(
        "dist_moe", reason="Full MXFP8 recipes require the optional Dist-MoE package"
    )
    monkeypatch.setattr(
        "torchtitan.distributed.accelerator_graph.accelerator_graphs_supported",
        lambda: True,
    )
    kimi_k3.kimi_k3_dist_moe_mxfp8_gb200_512()
