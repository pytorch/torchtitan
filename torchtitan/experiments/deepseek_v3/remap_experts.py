import csv
from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn

# import model class we want to re-map experts for
from model import MoE, MoEGate


def remap_experts_from_csv(
    model: nn.Module,
    csv_file_path: str,
    layer_col: str = "layer",
    order_col: str = "expert_order",
):
    """
    Remap experts in MoE layers based on ordering specified in a CSV file.

    Args:
        model: nn.Module model instance
        csv_file_path: Path to CSV file containing expert remapping instructions
        layer_col: Column name containing layer identifiers (e.g., 'layer1', 'layer2')
        order_col: Column name containing comma-separated expert order (e.g., '3,1,0,2')

    CSV Format Example:
        layer,expert_order
        layer_5,"3,1,0,2"
        layer_8,"2,0,3,1"
        layer_11,"1,3,2,0"

    The expert order should be a comma-separated list of integers representing the new order of experts.
    For example, "3,1,0,2" means expert 3 should be first, expert 1 second, etc.
    """


def apply_routing_remap_to_moe():
    """
    Remap experts in MoE class based on ordering specified in a CSV file.

    """

    # Stash original forward
    if not hasattr(MoE, "_original_forward"):
        MoE._original_forward = MoE.forward

    def forward_with_remapping(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape

        # Get expert indices and weights from gate
        topk_index, topk_weight = self.gate(hidden_states)

        if hasattr(self, "expert_routing_remap"):
            original_shape = topk_index.shape
            topk_index_flat = topk_index.flatten()

            # apply the remapping
            remapped_idx = self.expert_routing_remap[topk_index_flat]
            topk_index = remapped_idx.view(original_shape)

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        # which comms
        if self.shuffle_method == "symm_mem":
            y = self.moe_on_device(hidden_states, topk_index, topk_weight)
        else:  # "torch_all_to_all"
            y = self.moe_forward(hidden_states, topk_index, topk_weight)

        y = y.view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    # Replace the forward method
    MoE.forward = forward_with_remapping


def apply_expert_remapping_with_routing(
    model: nn.Module,
    remap_file: str,
    layer_col: str = "layer",
    order_col: str = "expert_order",
):
    """
    Remap experts in MoE class based on ordering specified in a CSV file.

    Args:
        model: the model to apply the remapping to
        remap_file: the path to the CSV file containing the remapping

    """
    # apply forward remapping
    apply_routing_remap_to_moe()

    # load remapping from CSV
    remap_experts_from_csv(model, remap_file, layer_col, order_col)

    print(f"Expert remapping applied successfully with routing updates! {remap_file}")
