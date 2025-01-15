import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class ExpertTokenTracker:
    """Tracks the flow of tokens-> experts through the model layers and updates the token-expert map accordingly."""

    """def __init__(self, num_experts: int, num_tokens: int, device: torch.device) -> None:
        self.num_experts = num_experts
        self.num_tokens = num_tokens
        self.device = device
        self.token_expert_map: torch.Tensor = torch.zeros(
            (num_experts, num_tokens), dtype=torch.int32, device=device
        )
        self.token_expert_map.fill_(-1)
    """

    def __init__(
        self, num_layers: int, local_rank: int = 0, base_filename: str = "token_paths"
    ):
        self.num_layers = num_layers
        self.base_filename = base_filename
        self.reset_tracking()
        self.local_rank = local_rank
        self.layer0_selected_tokens: torch.Tensor = torch.empty(0)
        self.layer1_selected_tokens: torch.Tensor = torch.empty(0)
        self.layer2_selected_tokens: torch.Tensor = torch.empty(0)
        self.layer3_selected_tokens: torch.Tensor = torch.empty(0)

    def reset_tracking(self):
        """Reset all tracking data."""
        # Dictionary to store token paths
        # Key: token_id, Value: list of expert assignments
        self.token_paths: Dict[int, List[int]] = {}
        self.current_layer = 0

    def record_assignments(self, selected_token_indices: torch.Tensor):
        """
                Record expert assignments for tokens in the current layer.
                format:
                selected tokens: tensor([[1, 2, 0, 4, 3, 7, 5, 6],
        [rank0]:        [0, 3, 6, 1, 5, 7, 2, 4],
        [rank0]:        [3, 2, 5, 4, 6, 7, 0, 1],
        [rank0]:        [4, 6, 3, 5, 7, 2, 1, 0]], device='cuda:0') torch.Size([4, 8])

        """
        if self.current_layer >= self.num_layers:
            raise ValueError(
                f"Attempting to record assignments for layer {self.current_layer} "
                f"but model only has {self.num_layers} layers"
            )

        print(
            f"record assignments, layer {self.current_layer}: selected_token_indices: {selected_token_indices}"
        )
        num_elems = selected_token_indices.numel()
        shape = selected_token_indices.shape
        print(
            f"record_assignments for rank {self.local_rank=}, num_elems: {num_elems}, shape: {shape}"
        )
        if self.current_layer == 0:
            self.layer0_selected_tokens = selected_token_indices
        elif self.current_layer == 1:
            self.layer1_selected_tokens = selected_token_indices
        elif self.current_layer == 2:
            self.layer2_selected_tokens = selected_token_indices
        elif self.current_layer == 3:
            self.layer3_selected_tokens = selected_token_indices
        else:
            raise ValueError(
                f"Attempting to record assignments for layer {self.current_layer} "
                f"but model only has {self.num_layers} layers"
            )
        # Convert tensors to CPU and numpy for processing
        # token_ids = token_ids.detach().cpu()
        # expert_assignments = expert_assignments.detach().cpu()

        # Record assignments for each token
        # for token_id, expert_id in zip(token_ids.numpy(), expert_assignments.numpy()):
        #    if token_id not in self.token_paths:
        #        self.token_paths[token_id] = [0] * self.num_layers
        #    self.token_paths[token_id][self.current_layer] = int(expert_id)

        self.current_layer += 1
        if self.current_layer == self.num_layers:
            self.current_layer = 0
            # process the token paths
            path1, path2, path3, path_summary = self.create_routing_traces()
            print(f"rank {self.local_rank=}, path_summary: {path_summary}")
            print(f"rank {self.local_rank=}, path1: {path1}")
            # print(f"rank {self.local_rank=}, path2: {path2}")
            # print(f"rank {self.local_rank=}, path3: {path3}")
            # reset
            self.layer0_selected_tokens = torch.empty(0)
            self.layer1_selected_tokens = torch.empty(0)
            self.layer2_selected_tokens = torch.empty(0)
            self.layer3_selected_tokens = torch.empty(0)

    def get_timestamped_filename(self, custom_filename: Optional[str] = None) -> str:
        """
        Generate a timestamped filename for the CSV output.

        Args:
            custom_filename: Optional custom base filename to override the default

        Returns:
            Timestamped filename string
        """
        base = custom_filename if custom_filename else self.base_filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        return f"{base}_{self.local_rank}_{timestamp}.csv"

    def save_token_paths(self, output_file: Optional[str] = None):
        """
        Save token paths to a CSV file with timestamp.

        Args:
            output_file: Optional custom output path. If not provided,
                        uses base_filename with timestamp
        """
        if output_file is None:
            output_file = self.get_timestamped_filename()

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(
                ["token_id"] + [f"layer_{i}" for i in range(self.num_layers)]
            )

            # Write token paths sorted by token ID
            for token_id in sorted(self.token_paths.keys()):
                writer.writerow([token_id] + self.token_paths[token_id])

    def create_routing_traces(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Create three routing trace tensors and a summary of complete paths:
        1. Primary routing tensor: First expert assignment for each token
        2. Duplicate routing tensor: Second expert assignment for each token
        3. Triplicate routing tensor: Third expert assignment for each token
        4. Summary dict of complete paths (tokens routed through both layers)

        Args:
            layer0_assignments: Tensor of shape [num_experts, tokens_per_expert]
            layer1_assignments: Tensor of shape [num_experts, tokens_per_expert]

        Returns:
            Tuple containing:
            - Primary routing trace: Tensor of shape [num_tokens, num_layers]
            - Duplicate routing trace: Tensor of shape [num_tokens, num_layers]
            - Triplicate routing trace: Tensor of shape [num_tokens, num_layers]
            - Path summary: Dict with complete path statistics
        """
        num_tokens = 128
        num_layers = 4

        layer0_assignments = self.layer0_selected_tokens
        layer1_assignments = self.layer1_selected_tokens
        layer2_assignments = self.layer2_selected_tokens
        layer3_assignments = self.layer3_selected_tokens

        device = layer0_assignments.device

        # Initialize routing trace tensors with -1
        primary_trace = torch.full(
            (num_tokens, num_layers), -1, dtype=torch.long, device=device
        )
        duplicate_trace = torch.full(
            (num_tokens, num_layers), -1, dtype=torch.long, device=device
        )
        triplicate_trace = torch.full(
            (num_tokens, num_layers), -1, dtype=torch.long, device=device
        )

        # Track complete paths
        complete_paths = {
            "primary": [],  # tokens with complete single path
            "duplicate": [],  # tokens with complete double path
            "triplicate": [],  # tokens with complete triple path
        }

        # Fill routing traces for each layer
        for token_id in range(num_tokens):
            # Layer 0
            experts_layer0 = torch.where(layer0_assignments == token_id)[0]
            if len(experts_layer0) > 0:
                primary_trace[token_id, 0] = experts_layer0[0]
                if len(experts_layer0) > 1:
                    duplicate_trace[token_id, 0] = experts_layer0[1]
                    if len(experts_layer0) > 2:
                        triplicate_trace[token_id, 0] = experts_layer0[2]

            # Layer 1
            experts_layer1 = torch.where(layer1_assignments == token_id)[0]
            if len(experts_layer1) > 0:
                primary_trace[token_id, 1] = experts_layer1[0]
                if len(experts_layer1) > 1:
                    duplicate_trace[token_id, 1] = experts_layer1[1]
                    if len(experts_layer1) > 2:
                        triplicate_trace[token_id, 1] = experts_layer1[2]

            # Layer 2
            experts_layer2 = torch.where(layer2_assignments == token_id)[0]
            if len(experts_layer2) > 0:
                primary_trace[token_id, 1] = experts_layer2[0]
                if len(experts_layer2) > 1:
                    duplicate_trace[token_id, 1] = experts_layer2[1]
                    if len(experts_layer2) > 2:
                        triplicate_trace[token_id, 1] = experts_layer2[2]

            # Layer 3
            experts_layer3 = torch.where(layer3_assignments == token_id)[0]
            if len(experts_layer3) > 0:
                primary_trace[token_id, 1] = experts_layer3[0]
                if len(experts_layer3) > 1:
                    duplicate_trace[token_id, 1] = experts_layer3[1]
                    if len(experts_layer3) > 2:
                        triplicate_trace[token_id, 1] = experts_layer3[2]

            # Check for complete paths (token routed through both layers)
            has_primary = (primary_trace[token_id] != -1).all().item()
            has_duplicate = (duplicate_trace[token_id] != -1).all().item()
            has_triplicate = (triplicate_trace[token_id] != -1).all().item()

            if has_triplicate:
                complete_paths["triplicate"].append(token_id)
            if has_duplicate:
                complete_paths["duplicate"].append(token_id)
            if has_primary:
                complete_paths["primary"].append(token_id)

        # Create summary dictionary
        path_summary = {
            "complete_paths": complete_paths,
            "stats": {
                "total_tokens": num_tokens,
                "tokens_with_primary_path": len(complete_paths["primary"]),
                "tokens_with_duplicate_path": len(complete_paths["duplicate"]),
                "tokens_with_triplicate_path": len(complete_paths["triplicate"]),
            },
        }

        return primary_trace, duplicate_trace, triplicate_trace, path_summary
