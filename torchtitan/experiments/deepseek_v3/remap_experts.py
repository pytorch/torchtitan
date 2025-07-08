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

    layer_remapping = {}
    try:
        with open(csv_file_path, "r") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                layer = row[layer_col].strip()
                order = row[order_col].strip().strip('"')
                # Parse the order string into a list of integers
                order = [int(x.strip()) for x in order.split(",")]
                layer_remapping[layer_name] = expert_order
    except Exception as e:
        raise ValueError(f"Error reading CSV file {csv_file_path}: {e}")

    print(f"Loaded remapping for {len(layer_remapping)} layers from {csv_file_path}")

    # Apply remapping to each specified layer
    # Apply remapping to each specified layer
    for layer_name, new_order in layer_remapping.items():
        # Extract layer index from layer name (e.g., "layer_5" -> 5)
        try:
            if layer_name.startswith("layer_"):
                layer_idx = int(layer_name.split("_")[1])
            elif layer_name.startswith("layer"):
                layer_idx = int(layer_name.replace("layer", ""))
            else:
                layer_idx = int(layer_name)
        except ValueError:
            print(f"Warning: Could not parse layer index from '{layer_name}', skipping")
            continue

        # Get the layer
        layer_key = str(layer_idx)

        if layer_key not in model.model.layers:
            print(f"Warning: Layer {layer_idx} not found in model, skipping")
            continue

        layer = model.model.layers[layer_key]

        # Check if this layer has MoE
        if not isinstance(layer.mlp, MoE):
            print(f"Warning: Layer {layer_idx} does not have MoE, skipping")
            continue

        print(f"Remapping experts in layer {layer_idx} with order {new_order}")
        _remap_layer_experts(layer.mlp, new_order)


def _remap_layer_experts(moe_layer: "MoE", new_order: List[int]):
    """
    Remap experts within a single MoE layer.
    """
    n_routed_experts = moe_layer.config.n_routed_experts

    # Validate the new order
    if len(new_order) != n_routed_experts:
        raise ValueError(
            f"New order {new_order} does not match the number of routed experts {n_routed_experts}"
        )

    if set(new_order) != set(range(n_routed_experts)):
        raise ValueError(
            f"New order {new_order} does not contain all experts from 0 to {n_routed_experts-1}"
        )

    # create the routing remap tensor
    # effectively, gate output of route to expert i is now new_order[i]
    routing_remap = torch.tensor(
        new_order, dtype=torch.long, device=next(moe_layer.parameters()).device
    )

    # Store the remapping in the MoE layer for use during forward pass
    moe_layer.register_buffer("expert_routing_remap", routing_remap, persistent=True)

    # Also remap the gate weights to maintain the same routing behavior
    # The gate.weight has shape [n_routed_experts, hidden_size]
    if hasattr(moe_layer.gate, "weight"):
        old_gate_weights = moe_layer.gate.weight.data.clone()
        # Remap gate weights: new_weight[i] = old_weight[old_expert_that_should_be_at_position_i]
        # We need the inverse mapping: which old expert should be at each new position
        inverse_order = [0] * len(new_order)
        for new_pos, old_expert in enumerate(new_order):
            inverse_order[old_expert] = new_pos

        for new_pos, old_pos in enumerate(inverse_order):
            moe_layer.gate.weight.data[new_pos] = old_gate_weights[old_pos]

    # If using topk_method "noaux_tc", also remap the correction bias
    if hasattr(moe_layer.gate, "e_score_correction_bias"):
        old_bias = moe_layer.gate.e_score_correction_bias.data.clone()
        inverse_order = [0] * len(new_order)
        for new_pos, old_expert in enumerate(new_order):
            inverse_order[old_expert] = new_pos

        for new_pos, old_pos in enumerate(inverse_order):
            moe_layer.gate.e_score_correction_bias.data[new_pos] = old_bias[old_pos]

    # For expert parallelism, we need to handle local expert remapping
    # but ultimately this should be at load time, not runtime...runtime for now as experimental
    # Get all expert IDs that should be present on this rank
    local_expert_ids = set(int(k) for k in moe_layer.experts.keys())

    if local_expert_ids:
        # Store the old experts
        old_experts = OrderedDict()
        for expert_id, expert in moe_layer.experts.items():
            old_experts[expert_id] = expert

        # Clear the experts dict
        moe_layer.experts.clear()

        # Reassign experts based on new ordering
        # The expert indices in the ModuleDict should reflect the new global ordering
        for old_global_id in local_expert_ids:
            old_expert = old_experts[str(old_global_id)]

            # Find where this expert should go in the new ordering
            try:
                new_global_id = new_order.index(old_global_id)
            except ValueError:
                print(
                    f"Warning: Expert {old_global_id} not found in new_order, keeping at same position"
                )
                new_global_id = old_global_id

            # Check if this expert should still be on this rank after remapping
            new_ep_rank = new_global_id // moe_layer.experts_per_rank

            if new_ep_rank == moe_layer.ep_rank:
                # This expert should stay on this rank
                moe_layer.experts[str(new_global_id)] = old_expert
                print(
                    f"    Moved expert {old_global_id} -> {new_global_id} (staying on rank {moe_layer.ep_rank})"
                )
            else:
                print(
                    f"    Warning: Expert {old_global_id} -> {new_global_id} should move to rank {new_ep_rank} (currently on {moe_layer.ep_rank})"
                )
                # This should not be needed, we want to load at runtime via SPMD the sorted experts for final impl...
                # For now, keep the expert at its original position with a warning
                moe_layer.experts[str(old_global_id)] = old_expert

    print(f"Applied expert remapping {new_order} to layer")
    print(f"Routing remap tensor: {routing_remap}")


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


def create_expert_remapping_csv(
    model: nn.Module,
    output_path: str,
    layer_indices: Optional[List[int]] = None,
):
    """
    Create a template CSV file for expert remapping.

    Args:
        model: DeepseekForCausalLM model instance
        output_path: Path where to save the CSV file
        layer_indices: List of layer indices to include. If None, includes all MoE layers.
    """

    moe_layers = []

    # Find all MoE layers
    for layer_idx, layer in model.model.layers.items():
        if isinstance(layer.mlp, MoE):
            layer_num = int(layer_idx)
            if layer_indices is None or layer_num in layer_indices:
                n_experts = layer.mlp.config.n_routed_experts
                # Create default ordering (0, 1, 2, ...)
                default_order = ",".join(map(str, range(n_experts)))
                moe_layers.append(
                    {
                        "layer": f"layer_{layer_num}",
                        "expert_order": default_order,
                        "num_experts": n_experts,
                    }
                )

    # Write CSV file
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["layer", "expert_order", "num_experts"]
        )
        writer.writeheader()
        writer.writerows(moe_layers)

    print(f"Created template CSV with {len(moe_layers)} MoE layers at {output_path}")
    print("Edit the 'expert_order' column to specify your desired expert ordering.")


def test_routing_remapping(
    model: nn.module, layer_idx: int, batch_size: int = 2, seq_len: int = 10
):
    """
    Test the routing remapping by running a forward pass and checking expert assignments.

    Args:
        model: model instance
        layer_idx: Layer index to test
        batch_size: Batch size for test input
        seq_len: Sequence length for test input
    """

    layer_key = str(layer_idx)
    if layer_key not in model.model.layers:
        print(f"Layer {layer_idx} not found in model")
        return

    layer = model.model.layers[layer_key]
    if not isinstance(layer.mlp, MoE):
        print(f"Layer {layer_idx} is not an MoE layer")
        return

    moe_layer = layer.mlp
    device = next(moe_layer.parameters()).device

    # Create random input
    hidden_states = torch.randn(
        batch_size,
        seq_len,
        moe_layer.config.hidden_size,
        device=device,
        dtype=next(moe_layer.parameters()).dtype,
    )

    print(f"Testing routing remapping for layer {layer_idx}")
    print(f"Input shape: {hidden_states.shape}")

    # Get gate outputs directly
    topk_idx, topk_weight = moe_layer.gate(hidden_states)
    print(f"Original gate outputs (first few): {topk_idx.flatten()[:10]}")

    if hasattr(moe_layer, "expert_routing_remap"):
        # Apply remapping manually to see the effect
        remapped_idx = moe_layer.expert_routing_remap[topk_idx.flatten()]
        print(f"Remapped indices (first few): {remapped_idx[:10]}")
        print(f"Routing remapping: {moe_layer.expert_routing_remap.cpu().tolist()}")
    else:
        print("No routing remapping found")

    # Test full forward pass
    try:
        output = moe_layer(hidden_states)
        print(f"Forward pass successful. Output shape: {output.shape}")
    except Exception as e:
        print(f"Error in forward pass: {e}")
