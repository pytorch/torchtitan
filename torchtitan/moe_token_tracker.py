import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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

    def __init__(self, num_layers: int, base_filename: str = "token_paths"):
        self.num_layers = num_layers
        self.base_filename = base_filename
        self.reset_tracking()

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
        print(f"record_assignments, current_layer: {self.current_layer}")
        print(f"record assignments: selected_token_indices: {selected_token_indices}")
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
        return f"{base}_{timestamp}.csv"

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
