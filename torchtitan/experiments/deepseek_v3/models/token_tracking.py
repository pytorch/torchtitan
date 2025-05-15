import torch


class TokenExpertTracker:
    def __init__(self):
        self.layer_assignments = {}  # {layer_id: {token_id: [expert_ids]}}
        self.token_history = {}  # {token_id: {layer_id: [expert_ids]}}
        self.token_values = {}  # Store actual token values

    def record_tokens(self, tokens, layer_id=None):
        """Record the input tokens at the starting layer"""
        if tokens.dim() > 1:
            # Handle batched input - flatten for simplicity
            tokens = tokens.view(-1)

        for i, token_id in enumerate(tokens.cpu().tolist()):
            if token_id not in self.token_values:
                self.token_values[i] = token_id

        print(f"Stored tokens = {self.token_values=}")
        # assert False, "check"

    def record_expert_assignment(self, layer_id, topk_idx):
        """Record which experts were assigned to each token in a layer"""
        # Initialize layer if not seen before
        if layer_id not in self.layer_assignments:
            self.layer_assignments[layer_id] = {}

        # For each token position
        for token_pos, expert_ids in enumerate(topk_idx.cpu().tolist()):
            # Record in layer-centric view
            self.layer_assignments[layer_id][token_pos] = expert_ids

            # Record in token-centric view
            if token_pos not in self.token_history:
                self.token_history[token_pos] = {}
            self.token_history[token_pos][layer_id] = expert_ids

    def get_token_expert_history(self, token_pos):
        """Get the expert assignment history for a specific token position"""
        return self.token_history.get(token_pos, {})

    def get_layer_assignments(self, layer_id):
        """Get all token-expert assignments for a specific layer"""
        return self.layer_assignments.get(layer_id, {})

    def get_expert_load(self, layer_id):
        """Count how many tokens were assigned to each expert in a layer"""
        if layer_id not in self.layer_assignments:
            return {}

        expert_counts = {}
        for token_pos, expert_ids in self.layer_assignments[layer_id].items():
            for expert_id in expert_ids:
                if expert_id not in expert_counts:
                    expert_counts[expert_id] = 0
                expert_counts[expert_id] += 1

        return expert_counts

    def print_summary(self, top_n_tokens=10):
        """Print a summary of token-expert assignments"""
        print("=== Token Expert Assignment Summary ===")

        # Print layer-by-layer summary
        for layer_id in sorted(self.layer_assignments.keys()):
            expert_load = self.get_expert_load(layer_id)
            print(f"\nLayer {layer_id} Expert Load:")
            for expert_id, count in sorted(
                expert_load.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  Expert {expert_id}: {count} tokens")

        # Print token-centric view for a few tokens
        print(f"\nToken Assignment History (showing first {top_n_tokens} tokens):")
        for token_pos in sorted(list(self.token_history.keys()))[:top_n_tokens]:
            token_val = self.token_values.get(token_pos, "unknown")
            print(f"\nToken position {token_pos} (value={token_val}):")
            for layer_id in sorted(self.token_history[token_pos].keys()):
                experts = self.token_history[token_pos][layer_id]
                print(f"  Layer {layer_id}: assigned to experts {experts}")

    def export_numpy_report(self, filename="experts_assignments.npy"):
        """
        Save token-expert assignments as a 2D numpy array where:
        - Each row represents a token
        - Each column represents a layer
        - Each value is the index of the first expert assigned to that token in that layer

        Format:
        [
            [expert_for_token0_layer0, expert_for_token0_layer1, ...],
            [expert_for_token1_layer0, expert_for_token1_layer1, ...],
            ...
        ]
        """
        import numpy as np

        # Get all tokens and layers
        tokens = sorted(self.token_history.keys())
        layers = sorted(
            set(layer for token in tokens for layer in self.token_history[token].keys())
        )

        # Create the array with -1 as default (indicating no expert assignment)
        trace = np.full((len(tokens), len(layers)), -1, dtype=np.int32)

        # Fill in the expert assignments
        for token_idx, token_pos in enumerate(tokens):
            for layer_idx, layer_id in enumerate(layers):
                if layer_id in self.token_history[token_pos]:
                    # Use the first expert assigned to this token in this layer
                    trace[token_idx, layer_idx] = self.token_history[token_pos][
                        layer_id
                    ][0]

        # Save to file
        np.save(filename, trace)
        print(f"Expert trace saved to {filename}")

        return trace

    def export_csv_report(self, filename="expert_assignments.csv"):
        """
        Save token-expert assignments as a CSV file where:
        - Each row represents a token
        - Each column represents a layer
        - Each value is the index of the first expert assigned to that token in that layer
        """
        import csv

        import numpy as np

        # Get all tokens and layers
        tokens = sorted(self.token_history.keys())
        layers = sorted(
            set(layer for token in tokens for layer in self.token_history[token].keys())
        )

        # Create the data
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # Write header row with layer IDs
            writer.writerow(
                ["token_pos"] + [f"layer_{layer_id}" for layer_id in layers]
            )

            # Write data rows
            for token_pos in tokens:
                row = [token_pos]
                for layer_id in layers:
                    if layer_id in self.token_history[token_pos]:
                        # Use the first expert assigned to this token in this layer
                        row.append(self.token_history[token_pos][layer_id][0])
                    else:
                        row.append(-1)
                writer.writerow(row)

        print(f"Expert trace saved to {filename}")
