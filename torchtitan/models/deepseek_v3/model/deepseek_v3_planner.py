# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
DeepSeek V3 Load Planner for DCP that handles grouped expert tensors.

This planner validates that grouped expert tensors can be formed from individual experts
in the checkpoint before creating read items.
"""

import re
from typing import Any, List, Optional

from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner, _create_read_items
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.checkpoint.planner import LoadPlan
from torchtitan.models.deepseek_v3.model.metadata import (
    DeepSeekV3Metadata,
)

class DeepSeekV3LoadPlanner(DefaultLoadPlanner):
    """Load planner for DeepSeek V3 that handles grouped expert tensor validation."""

    def __init__(self):
        """Initialize the DeepSeek V3 load planner."""
        super().__init__()
        self.valid_grouped_experts = set()

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Optional[DeepSeekV3Metadata] = None,
        is_coordinator: bool = False,
    ) -> None:
        super().set_up_planner(state_dict, metadata.sd_metadata, is_coordinator)
        # Build cache of valid grouped expert FQNs once during setup
        self.metadata = metadata.sd_metadata
        self.io_metadata = metadata.io_metadata
        self.valid_grouped_experts = self._build_valid_grouped_experts()

    def _build_valid_grouped_experts(self) -> set:
        """Build cache of valid grouped expert FQNs from checkpoint metadata."""
        # Group individual experts by (layer, weight_type)
        experts_by_group = {}
        # Match only weight tensors, explicitly exclude scale tensors
        expert_pattern = r'model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(\w+)\.weight$'
        hf_to_tt_weight_map = {'gate_proj': 'w1', 'down_proj': 'w2', 'up_proj': 'w3'}

        # Count total expert entries
        total_expert_entries = 0

        for idx in self.io_metadata.storage_data.keys():
            match = re.match(expert_pattern, idx.fqn)
            if match:
                total_expert_entries += 1
                layer_idx, expert_idx, hf_weight_type = match.groups()
                tt_weight_type = hf_to_tt_weight_map.get(hf_weight_type)

                if tt_weight_type:
                    group_key = (layer_idx, tt_weight_type)
                    if group_key not in experts_by_group:
                        experts_by_group[group_key] = []
                    experts_by_group[group_key].append(int(expert_idx))

        # If no expert entries found, the checkpoint might not have individual experts
        # This could mean experts are already grouped or use a different naming pattern
        if total_expert_entries == 0:
            return set()

        # Determine which grouped expert FQNs are valid
        # We just need to have at least one expert for each weight type in each layer
        valid_fqns = set()

        if len(experts_by_group) == 0:
            return set()

        for (layer_idx, tt_weight_type), expert_indices in experts_by_group.items():
            expert_indices = sorted(expert_indices)

            # As long as we have at least one expert, we can create a grouped tensor
            if len(expert_indices) > 0:
                grouped_fqn = f"layers.{layer_idx}.moe.experts.{tt_weight_type}"
                valid_fqns.add(grouped_fqn)

        return valid_fqns

    def create_local_plan(self) -> LoadPlan:
        """Create a local load plan starting from the model's state_dict."""
        requests = []

        # Process each tensor in the model's state_dict
        for fqn, obj in self.state_dict.items():
            if self._is_grouped_expert_tensor(fqn) and fqn not in self.valid_grouped_experts:
                raise RuntimeError(f"Grouped expert tensor {fqn} cannot be loaded from checkpoint")

            # Create read items for all tensors (both regular and grouped)
            self._validate_and_create_read_items(fqn, obj, self.metadata, requests)

        return LoadPlan(requests)

    def _validate_and_create_read_items(self, fqn: str, obj: Any, metadata: Any, requests: List) -> None:
        """Validate tensor and add read items to requests."""
        if fqn not in metadata.state_dict_metadata:
            raise RuntimeError(f"Missing key in checkpoint metadata: {fqn}")

        md = metadata.state_dict_metadata[fqn]

        # Create read items (handle DTensor submesh)
        if isinstance(obj, DTensor):
            if obj.device_mesh.get_coordinate() is not None:
                requests += _create_read_items(fqn, md, obj)
        else:
            requests += _create_read_items(fqn, md, obj)

    def _is_grouped_expert_tensor(self, fqn: str) -> bool:
        """Check if this FQN represents a grouped expert tensor."""
        # Match grouped expert tensors but exclude shared expert tensors
        return 'moe.experts' in fqn

