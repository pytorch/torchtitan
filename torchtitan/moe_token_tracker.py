import csv
import glob
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

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
        self.save_path = None
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
        # reset on first iter
        if self.current_layer == 0:
            self.reset_tracking()

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

    def save_routing_traces(
        self,
        primary_trace: torch.Tensor,
        duplicate_trace: torch.Tensor,
        triplicate_trace: torch.Tensor,
        path_summary: Dict,
        output_dir: Optional[str] = None,
    ):
        """
        Save routing traces and summary to files.

        Args:
            primary_trace: Primary routing tensor
            duplicate_trace: Duplicate routing tensor
            triplicate_trace: Triplicate routing tensor
            path_summary: Dictionary containing routing statistics
            output_dir: Optional output directory (defaults to current directory)
        """
        if output_dir is None:
            output_dir = "routing_traces"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        out_dir = Path(output_dir) / timestamp / f"rank_{self.local_rank}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving routing traces to {out_dir}")
        self.save_path = Path(output_dir) / timestamp

        # Save tensors as numpy arrays
        np.save(out_dir / "primary_trace.npy", primary_trace.cpu().numpy())
        np.save(out_dir / "duplicate_trace.npy", duplicate_trace.cpu().numpy())
        np.save(out_dir / "triplicate_trace.npy", triplicate_trace.cpu().numpy())

        # Save path summary as JSON

        with open(out_dir / "path_summary.json", "w") as f:
            # Convert any tensor values to Python types
            clean_summary = json.loads(
                json.dumps(
                    path_summary,
                    default=lambda x: x.tolist() if isinstance(x, torch.Tensor) else x,
                )
            )
            json.dump(clean_summary, f, indent=2)

    # @staticmethod
    def load_and_combine_traces(
        self, trace_dir: str, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Load and combine routing traces from multiple ranks.

        Args:
            trace_dir: Directory containing routing trace subdirectories
            device: Target device for loaded tensors

        Returns:
            Combined primary, duplicate, and triplicate traces, and combined summary
        """
        # Find all rank directories
        trace_dir = self.save_path
        rank_dirs = sorted(Path(trace_dir).glob("rank_*"))
        print(f"rank_dirs: {rank_dirs}")

        if not rank_dirs:
            raise ValueError(f"No rank directories found in {trace_dir}")

        # Initialize combined tensors and statistics
        combined_primary = None
        combined_duplicate = None
        combined_triplicate = None
        combined_summary = {
            "complete_paths": {
                "primary": set(),
                "duplicate": set(),
                "triplicate": set(),
            },
            "stats": {
                "total_tokens": 0,
                "tokens_with_primary_path": 0,
                "tokens_with_duplicate_path": 0,
                "tokens_with_triplicate_path": 0,
                "num_layers": None,
            },
        }

        # Load and combine traces from each rank
        for rank_dir in rank_dirs:
            # Load trace tensors
            primary = torch.from_numpy(np.load(rank_dir / "primary_trace.npy")).to(
                device
            )
            duplicate = torch.from_numpy(np.load(rank_dir / "duplicate_trace.npy")).to(
                device
            )
            triplicate = torch.from_numpy(
                np.load(rank_dir / "triplicate_trace.npy")
            ).to(device)

            # Load summary
            with open(rank_dir / "path_summary.json", "r") as f:
                summary = json.load(f)

            # Initialize combined tensors if needed
            if combined_primary is None:
                combined_primary = primary
                combined_duplicate = duplicate
                combined_triplicate = triplicate
                combined_summary["stats"]["num_layers"] = summary["stats"]["num_layers"]
            else:
                # Combine traces by taking non-negative values
                combined_primary = torch.where(primary != -1, primary, combined_primary)
                combined_duplicate = torch.where(
                    duplicate != -1, duplicate, combined_duplicate
                )
                combined_triplicate = torch.where(
                    triplicate != -1, triplicate, combined_triplicate
                )

            # Update complete paths sets
            for path_type in ["primary", "duplicate", "triplicate"]:
                combined_summary["complete_paths"][path_type].update(
                    summary["complete_paths"][path_type]
                )

        # Convert sets to sorted lists
        for path_type in ["primary", "duplicate", "triplicate"]:
            combined_summary["complete_paths"][path_type] = sorted(
                combined_summary["complete_paths"][path_type]
            )

        # Update final statistics
        combined_summary["stats"].update(
            {
                "tokens_with_primary_path": len(
                    combined_summary["complete_paths"]["primary"]
                ),
                "tokens_with_duplicate_path": len(
                    combined_summary["complete_paths"]["duplicate"]
                ),
                "tokens_with_triplicate_path": len(
                    combined_summary["complete_paths"]["triplicate"]
                ),
                "total_tokens": primary.shape[0],
            }
        )

        return (
            combined_primary,
            combined_duplicate,
            combined_triplicate,
            combined_summary,
        )

    def process_and_save_all(self):
        """Process current assignments and save all outputs."""
        # Create routing traces
        primary_trace, duplicate_trace, triplicate_trace, path_summary = (
            self.create_routing_traces()
        )

        # Save token paths to CSV
        self.save_token_paths()

        # Save routing traces and summary
        self.save_routing_traces(
            primary_trace, duplicate_trace, triplicate_trace, path_summary
        )

        return primary_trace, duplicate_trace, triplicate_trace, path_summary

    @staticmethod
    def analyze_combined_results(
        primary: torch.Tensor,
        duplicate: torch.Tensor,
        triplicate: torch.Tensor,
        summary: Dict,
    ) -> Dict:
        """
        Analyze combined routing traces to extract additional insights.

        Args:
            primary: Combined primary routing tensor
            duplicate: Combined duplicate routing tensor
            triplicate: Combined triplicate routing tensor
            summary: Combined path summary

        Returns:
            Dictionary containing additional analysis results
        """
        num_layers = primary.shape[1]
        num_tokens = primary.shape[0]

        analysis = {
            "layer_stats": [],
            "token_stats": {
                "tokens_with_full_path": len(summary["complete_paths"]["primary"]),
                "tokens_with_partial_path": 0,
                "tokens_with_no_path": 0,
            },
            "routing_patterns": {},
        }

        # Analyze each layer
        for layer in range(num_layers):
            layer_stats = {
                "layer": layer,
                "active_tokens": (primary[:, layer] != -1).sum().item(),
                "duplicated_tokens": (duplicate[:, layer] != -1).sum().item(),
                "triplicated_tokens": (triplicate[:, layer] != -1).sum().item(),
            }
            analysis["layer_stats"].append(layer_stats)

        # Count tokens with partial paths
        for token in range(num_tokens):
            path_length = (primary[token] != -1).sum().item()
            if path_length == 0:
                analysis["token_stats"]["tokens_with_no_path"] += 1
            elif path_length < num_layers:
                analysis["token_stats"]["tokens_with_partial_path"] += 1

        return analysis

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
