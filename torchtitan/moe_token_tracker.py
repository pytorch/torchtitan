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
        self,
        num_layers: int,
        local_rank: int = 0,
        num_tokens: int = 128,
        base_filename: str = "token_paths",
    ):
        self.num_layers = num_layers
        self.base_filename = base_filename
        self.reset_tracking()
        self.local_rank = local_rank
        self.num_tokens = num_tokens
        # routing assignment for each layer
        self.layer_assignments = {}

    def reset_tracking(self):
        """Reset all tracking data."""
        # Dictionary to store token paths
        # Key: token_id, Value: list of expert assignments
        self.token_paths: Dict[int, List[int]] = {}
        self.current_layer = 0
        # Clear stored layer assignments
        self.layer_assignments = {i: torch.empty(0) for i in range(self.num_layers)}

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

        # Store assignments for current layer
        self.layer_assignments[self.current_layer] = selected_token_indices

        self.current_layer += 1
        if self.current_layer == self.num_layers:
            self.current_layer = 0
            # Process the token paths
            path1, path2, path3, path_summary = self.create_routing_traces()
            print(f"\nrank {self.local_rank=}, path_summary: {path_summary}")
            print(f"rank {self.local_rank=}, path1: {path1}\n")
            # Reset layer assignments
            self.reset_tracking()

    def create_routing_traces(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Create routing trace tensors for each layer and return a summary of complete paths."""
        num_tokens = self.num_tokens
        device = next(
            tensor for tensor in self.layer_assignments.values() if tensor.numel() > 0
        ).device

        # Initialize routing trace tensors with -1
        primary_trace = torch.full(
            (num_tokens, self.num_layers), -1, dtype=torch.long, device=device
        )
        duplicate_trace = torch.full(
            (num_tokens, self.num_layers), -1, dtype=torch.long, device=device
        )
        triplicate_trace = torch.full(
            (num_tokens, self.num_layers), -1, dtype=torch.long, device=device
        )

        complete_paths = {"primary": [], "duplicate": [], "triplicate": []}

        # Fill routing traces for each token
        for token_id in range(num_tokens):
            # Process each layer
            for layer in range(self.num_layers):
                layer_assignments = self.layer_assignments[layer]
                if layer_assignments.numel() > 0:  # Check if layer has assignments
                    experts = torch.where(layer_assignments == token_id)[0]

                    if len(experts) > 0:
                        primary_trace[token_id, layer] = experts[0]
                        if len(experts) > 1:
                            duplicate_trace[token_id, layer] = experts[1]
                            if len(experts) > 2:
                                triplicate_trace[token_id, layer] = experts[2]

            # Check for complete paths through all layers
            has_primary = (primary_trace[token_id] != -1).all().item()
            has_duplicate = (duplicate_trace[token_id] != -1).all().item()
            has_triplicate = (triplicate_trace[token_id] != -1).all().item()

            if has_triplicate:
                complete_paths["triplicate"].append(token_id)
            if has_duplicate:
                complete_paths["duplicate"].append(token_id)
            if has_primary:
                complete_paths["primary"].append(token_id)

        path_summary = {
            "complete_paths": complete_paths,
            "stats": {
                "total_tokens": num_tokens,
                "tokens_with_primary_path": len(complete_paths["primary"]),
                "tokens_with_duplicate_path": len(complete_paths["duplicate"]),
                "tokens_with_triplicate_path": len(complete_paths["triplicate"]),
                "num_layers": self.num_layers,
            },
        }

        return primary_trace, duplicate_trace, triplicate_trace, path_summary

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
