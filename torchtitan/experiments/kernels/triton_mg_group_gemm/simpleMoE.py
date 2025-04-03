# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import math
import time

from typing import Dict, List, Tuple

# import numpy as np
import torch  #
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from torchao_pr.mg_grouped_gemm import mg_grouped_gemm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Try to import the optimized MG GEMM implementation
try:
    from torchao_pr.mg_grouped_gemm import (  # grouped_gemm_backward,
        grouped_gemm_forward,
    )

    has_mg_gemm = True
except ImportError:
    logging.warning("MG GEMM implementation not found. Will use manual looping only.")
    has_mg_gemm = False


class Router(nn.Module):
    """
    Router module that assigns tokens to experts.
    """

    def __init__(self, input_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # Routing layer
        self.router = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Route input tokens to experts.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Tuple containing:
            - router_logits: Raw routing probabilities
            - dispatch_tensor: One-hot tensor indicating expert assignment
            - expert_indices: List of indices for each expert's tokens
        """
        batch_size, seq_len, _ = x.shape

        # Flatten batch and sequence dimensions
        x_flat = x.reshape(-1, self.input_dim)  # (batch_size * seq_len, input_dim)

        # Compute routing probabilities
        router_logits = self.router(x_flat)  # (batch_size * seq_len, num_experts)

        # Apply softmax to get probabilities
        router_probs = F.softmax(router_logits, dim=-1)

        # Get top-k experts for each token
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Create dispatch tensor (one-hot representation of assignments)
        dispatch_tensor = torch.zeros_like(router_probs)
        token_indices = (
            torch.arange(router_probs.size(0), device=router_probs.device)
            .unsqueeze(1)
            .expand(-1, self.top_k)
        )
        dispatch_tensor.scatter_(1, top_k_indices, top_k_probs)  # .unsqueeze(-1))

        # For each expert, get the indices of tokens routed to it
        expert_indices = []
        for expert_idx in range(self.num_experts):
            # Get indices of tokens that have non-zero probability for this expert
            indices = torch.nonzero(dispatch_tensor[:, expert_idx] > 0, as_tuple=True)[
                0
            ]
            expert_indices.append(indices)

        return router_logits, dispatch_tensor, expert_indices


class Expert(nn.Module):
    """
    Individual expert module.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts layer with support for both manual looping and grouped GEMM.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int,
        top_k: int = 2,
        use_mg_gemm: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_mg_gemm = use_mg_gemm and has_mg_gemm

        # Router
        self.router = Router(input_dim, num_experts, top_k)

        # Create expert modules
        if self.use_mg_gemm:
            # For MG GEMM, we need a single weight tensor for all experts
            # First layer (input -> hidden)
            self.expert_fc1_weight = nn.Parameter(
                torch.randn(num_experts * hidden_dim, input_dim) / math.sqrt(input_dim)
            )
            # self.expert_fc1_bias = nn.Parameter(torch.zeros(num_experts * hidden_dim))

            # Second layer (hidden -> output)
            self.expert_fc2_weight = nn.Parameter(
                torch.randn(num_experts * output_dim, hidden_dim)
                / math.sqrt(hidden_dim)
            )
            # self.expert_fc2_bias = nn.Parameter(torch.zeros(num_experts * output_dim))
        else:
            # For manual looping, create separate experts
            self.experts = nn.ModuleList(
                [Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)]
            )

    def forward_manual_loop(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using manual looping over experts.
        """
        batch_size, seq_len, _ = x.shape
        x_flat = x.reshape(-1, self.input_dim)  # (batch_size * seq_len, input_dim)

        # Get routing information
        router_logits, dispatch_tensor, expert_indices = self.router(x)

        # Initialize output tensor
        final_output = torch.zeros(
            batch_size * seq_len, self.output_dim, device=x.device
        )

        # Process each expert
        for expert_idx, indices in enumerate(expert_indices):
            if indices.numel() > 0:
                # Get tokens routed to this expert
                expert_inputs = x_flat[indices]  # (num_tokens_for_expert, input_dim)

                # Process tokens through expert
                expert_outputs = self.experts[expert_idx](
                    expert_inputs
                )  # (num_tokens_for_expert, output_dim)

                # Scale outputs by router probabilities
                scaled_outputs = expert_outputs * dispatch_tensor[
                    indices, expert_idx
                ].unsqueeze(1)

                # Add to final output
                final_output.index_add_(0, indices, scaled_outputs)

        # Reshape back to original dimensions
        output = final_output.reshape(batch_size, seq_len, self.output_dim)

        return output, router_logits

    def forward_mg_gemm(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x_flat = x.reshape(-1, self.input_dim)  # (batch_size * seq_len, input_dim)
        total_tokens = batch_size * seq_len

        # Get routing information
        router_logits, dispatch_tensor, expert_indices = self.router(x)

        # Get token counts for each expert
        token_counts = [indices.numel() for indices in expert_indices]
        m_sizes = torch.tensor(token_counts, dtype=torch.int32, device=x.device)

        print(f"Token counts per expert: {token_counts}")
        print(f"m_sizes: {m_sizes}")

        # Create the combined input tensor
        combined_input = torch.zeros(sum(token_counts), self.input_dim, device=x.device)

        start_idx = 0
        for expert_idx, indices in enumerate(expert_indices):
            if indices.numel() > 0:
                end_idx = start_idx + indices.numel()
                combined_input[start_idx:end_idx] = x_flat[indices]
                start_idx = end_idx

        print(f"combined_input shape: {combined_input.shape}")

        # First layer: input -> hidden
        fc1_weight_reshaped = self.expert_fc1_weight.reshape(
            self.num_experts, self.hidden_dim, self.input_dim
        )
        fc1_weight_combined = fc1_weight_reshaped.reshape(-1, self.input_dim)

        print(f"fc1_weight_combined shape: {fc1_weight_combined.shape}")

        # Run the grouped GEMM
        hidden_outputs = grouped_gemm_forward(
            combined_input, fc1_weight_combined, m_sizes
        )

        print(f"hidden_outputs shape after first GEMM: {hidden_outputs.shape}")

        # Apply activation
        hidden_outputs = F.gelu(hidden_outputs)

        print(f"hidden_outputs shape after activation: {hidden_outputs.shape}")

        # Second layer: hidden -> output
        # Reshape hidden_outputs to match expected dimensions
        reshaped_hidden_outputs = []
        start_idx = 0

        for expert_idx, count in enumerate(token_counts):
            if count > 0:
                end_idx = start_idx + count
                # Take this expert's outputs and reshape to [count, hidden_dim]
                expert_output = hidden_outputs[
                    start_idx:end_idx,
                    expert_idx * self.hidden_dim : (expert_idx + 1) * self.hidden_dim,
                ]
                reshaped_hidden_outputs.append(expert_output)
                start_idx = end_idx

        # Concatenate all reshaped outputs
        hidden_outputs = torch.cat(reshaped_hidden_outputs, dim=0)

        # Reshape expert weights for second layer
        fc2_weight_reshaped = self.expert_fc2_weight.reshape(
            self.num_experts, self.output_dim, self.hidden_dim
        )
        fc2_weight_combined = fc2_weight_reshaped.reshape(-1, self.hidden_dim)

        print(f"fc2_weight_combined shape: {fc2_weight_combined.shape}")

        # Run the second grouped GEMM
        expert_outputs_combined = grouped_gemm_forward(
            hidden_outputs, fc2_weight_combined, m_sizes
        )

        # Initialize final output tensor with correct shape
        final_output = torch.zeros(total_tokens, self.output_dim, device=x.device)

        # Distribute the outputs back to the original token positions
        start_idx = 0
        for expert_idx, indices in enumerate(expert_indices):
            if indices.numel() > 0:
                end_idx = start_idx + indices.numel()
                # Get this expert's outputs
                expert_outputs = expert_outputs_combined[start_idx:end_idx]

                print(
                    f"Expert {expert_idx} - indices shape: {indices.shape}, expert_outputs shape: {expert_outputs.shape}"
                )

                # Scale outputs by router probabilities
                scaled_outputs = expert_outputs * dispatch_tensor[
                    indices, expert_idx
                ].unsqueeze(1)

                # Ensure dimensions match before using index_add_
                if scaled_outputs.shape[1] != final_output.shape[1]:
                    # print(
                    #    f"Reshaping: Dimension mismatch: scaled_outputs {scaled_outputs.shape}, final_output {final_output.shape}"
                    # )
                    # Reshape if needed - make sure output_dim is correct
                    scaled_outputs = scaled_outputs[:, : self.output_dim]

                # Add to final output
                final_output.index_add_(0, indices, scaled_outputs)

                start_idx = end_idx

        # Reshape back to original dimensions
        output = final_output.reshape(batch_size, seq_len, self.output_dim)

        return output, router_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_mg_gemm and has_mg_gemm:
            return self.forward_mg_gemm(x)
        else:
            return self.forward_manual_loop(x)


class MoEModel(nn.Module):
    """
    Simple model using MoE layers.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 2,
        use_mg_gemm: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.moe_layer = MixtureOfExperts(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            num_experts=num_experts,
            top_k=top_k,
            use_mg_gemm=use_mg_gemm,
        )
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        moe_output, router_logits = self.moe_layer(
            embedded
        )  # (batch_size, seq_len, embed_dim)
        logits = self.output_layer(moe_output)  # (batch_size, seq_len, vocab_size)
        return logits, router_logits


def compute_load_balancing_loss(
    router_logits: torch.Tensor, num_experts: int
) -> torch.Tensor:
    """
    Compute the load balancing loss for MoE training.

    Args:
        router_logits (torch.Tensor): Router logits of shape (batch_size * seq_len, num_experts)
        num_experts (int): Number of experts

    Returns:
        torch.Tensor: Load balancing loss
    """
    # Get router probabilities
    router_probs = F.softmax(
        router_logits, dim=-1
    )  # (batch_size * seq_len, num_experts)

    # Compute fraction of tokens routed to each expert
    # Sum across the batch dimension and normalize
    router_probs_sum = router_probs.sum(dim=0)  # (num_experts,)
    router_probs_sum = router_probs_sum / router_probs_sum.sum()

    # Compute the mean probability per expert
    mean_prob = 1.0 / num_experts

    # Compute the fraction of tokens routed to each expert
    # The goal is to have uniform routing across experts
    load_balancing_loss = num_experts * torch.sum(router_probs_sum * router_probs_sum)

    return load_balancing_loss


def generate_sample_data(
    batch_size: int, seq_len: int, vocab_size: int, device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate sample data for training.

    Args:
        batch_size (int): Batch size
        seq_len (int): Sequence length
        vocab_size (int): Vocabulary size
        device (str): Device to use

    Returns:
        Tuple of input tokens and target tokens
    """
    # Generate random input tokens
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Generate random target tokens
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    return inputs, targets


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    num_batches: int,
    device: str,
    load_balance_coef: float = 0.01,
) -> Dict[str, float]:
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): Model to train
        optimizer (torch.optim.Optimizer): Optimizer
        batch_size (int): Batch size
        seq_len (int): Sequence length
        vocab_size (int): Vocabulary size
        num_batches (int): Number of batches per epoch
        device (str): Device to use
        load_balance_coef (float): Coefficient for load balancing loss

    Returns:
        Dict containing training metrics
    """
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    start_time = time.time()

    for i in range(num_batches):
        # Generate sample data
        inputs, targets = generate_sample_data(batch_size, seq_len, vocab_size, device)

        # Forward pass
        optimizer.zero_grad()
        logits, router_logits = model(inputs)

        # Compute loss
        # Reshape for cross entropy loss
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        # Cross entropy loss
        ce_loss = F.cross_entropy(logits_flat, targets_flat)

        # Load balancing loss
        lb_loss = compute_load_balancing_loss(
            router_logits, model.moe_layer.num_experts
        )

        # Combined loss
        loss = ce_loss + load_balance_coef * lb_loss

        # Backward pass
        loss.backward()
        optimizer.step()

        # Compute accuracy
        preds = logits_flat.argmax(dim=-1)
        correct = (preds == targets_flat).float().sum()
        acc = correct / (batch_size * seq_len)

        # Accumulate metrics
        total_loss += loss.item()
        total_acc += acc.item()

        # Log progress
        if (i + 1) % 10 == 0:
            logging.info(
                f"Batch {i + 1}/{num_batches} | "
                f"Loss: {loss.item():.4f} | "
                f"CE Loss: {ce_loss.item():.4f} | "
                f"LB Loss: {lb_loss.item():.4f} | "
                f"Acc: {acc.item():.4f}"
            )

    # Compute average metrics
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    epoch_time = time.time() - start_time

    return {"loss": avg_loss, "acc": avg_acc, "time": epoch_time}


def evaluate(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    num_batches: int,
    device: str,
) -> Dict[str, float]:
    """
    Evaluate the model.

    Args:
        model (nn.Module): Model to evaluate
        batch_size (int): Batch size
        seq_len (int): Sequence length
        vocab_size (int): Vocabulary size
        num_batches (int): Number of batches for evaluation
        device (str): Device to use

    Returns:
        Dict containing evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    with torch.no_grad():
        for i in range(num_batches):
            # Generate sample data
            inputs, targets = generate_sample_data(
                batch_size, seq_len, vocab_size, device
            )

            # Forward pass
            logits, router_logits = model(inputs)

            # Compute loss
            logits_flat = logits.reshape(-1, vocab_size)
            targets_flat = targets.reshape(-1)

            # Cross entropy loss
            loss = F.cross_entropy(logits_flat, targets_flat)

            # Compute accuracy
            preds = logits_flat.argmax(dim=-1)
            correct = (preds == targets_flat).float().sum()
            acc = correct / (batch_size * seq_len)

            # Accumulate metrics
            total_loss += loss.item()
            total_acc += acc.item()

    # Compute average metrics
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches

    return {"loss": avg_loss, "acc": avg_acc}


def measure_performance(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    num_batches: int,
    device: str,
) -> Dict[str, float]:
    """
    Measure forward and backward pass performance.

    Args:
        model (nn.Module): Model to evaluate
        batch_size (int): Batch size
        seq_len (int): Sequence length
        vocab_size (int): Vocabulary size
        num_batches (int): Number of batches for measurement
        device (str): Device to use

    Returns:
        Dict containing performance metrics
    """
    model.train()

    # Create dummy optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Warmup
    for _ in range(5):
        inputs, targets = generate_sample_data(batch_size, seq_len, vocab_size, device)
        logits, router_logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        optimizer.zero_grad()

    # Measure forward pass time
    torch.cuda.synchronize()
    forward_start = time.time()

    for _ in range(num_batches):
        inputs, targets = generate_sample_data(batch_size, seq_len, vocab_size, device)
        with torch.no_grad():
            logits, router_logits = model(inputs)

    torch.cuda.synchronize()
    forward_end = time.time()
    forward_time = (forward_end - forward_start) / num_batches

    # Measure backward pass time
    torch.cuda.synchronize()
    backward_start = time.time()

    for _ in range(num_batches):
        inputs, targets = generate_sample_data(batch_size, seq_len, vocab_size, device)
        logits, router_logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    backward_end = time.time()
    backward_time = (backward_end - backward_start) / num_batches

    return {
        "forward_time": forward_time * 1000,  # Convert to ms
        "backward_time": backward_time * 1000,  # Convert to ms
        "total_time": (forward_time + backward_time) * 1000,  # Convert to ms
    }


def compare_methods(args):
    """
    Compare manual looping and MG GEMM implementations.
    """
    device = torch.device(args.device)

    # Create models
    manual_model = MoEModel(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_experts=args.num_experts,
        top_k=args.top_k,
        use_mg_gemm=False,
    ).to(device)

    if has_mg_gemm:
        mg_model = MoEModel(
            vocab_size=args.vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_experts=args.num_experts,
            top_k=args.top_k,
            use_mg_gemm=True,
        ).to(device)
    else:
        mg_model = None

    # Measure performance
    logging.info("Measuring performance of manual looping method...")
    manual_perf = measure_performance(
        manual_model,
        args.batch_size,
        args.seq_len,
        args.vocab_size,
        args.perf_batches,
        device,
    )

    if mg_model is not None:
        logging.info("Measuring performance of MG GEMM method...")
        mg_perf = measure_performance(
            mg_model,
            args.batch_size,
            args.seq_len,
            args.vocab_size,
            args.perf_batches,
            device,
        )
    else:
        mg_perf = {"forward_time": 0, "backward_time": 0, "total_time": 0}

    # Log results
    logging.info("\n===== Performance Comparison =====")
    logging.info("Model Configuration:")
    logging.info(f"  - Batch Size: {args.batch_size}")
    logging.info(f"  - Sequence Length: {args.seq_len}")
    logging.info(f"  - Embed Dimension: {args.embed_dim}")
    logging.info(f"  - Hidden Dimension: {args.hidden_dim}")
    logging.info(f"  - Number of Experts: {args.num_experts}")
    logging.info(f"  - Top-K: {args.top_k}")
    logging.info("")

    logging.info("Manual Looping Method:")
    logging.info(f"  - Forward Time: {manual_perf['forward_time']:.2f} ms")
    logging.info(f"  - Backward Time: {manual_perf['backward_time']:.2f} ms")
    logging.info(f"  - Total Time: {manual_perf['total_time']:.2f} ms")
    logging.info("")

    if mg_model is not None:
        logging.info("MG GEMM Method:")
        logging.info(f"  - Forward Time: {mg_perf['forward_time']:.2f} ms")
        logging.info(f"  - Backward Time: {mg_perf['backward_time']:.2f} ms")
        logging.info(f"  - Total Time: {mg_perf['total_time']:.2f} ms")
        logging.info("")

        # Calculate speedup
        forward_speedup = (
            manual_perf["forward_time"] / mg_perf["forward_time"]
            if mg_perf["forward_time"] > 0
            else 0
        )
        backward_speedup = (
            manual_perf["backward_time"] / mg_perf["backward_time"]
            if mg_perf["backward_time"] > 0
            else 0
        )
        total_speedup = (
            manual_perf["total_time"] / mg_perf["total_time"]
            if mg_perf["total_time"] > 0
            else 0
        )

        logging.info("Speedup (MG GEMM vs Manual):")
        logging.info(f"  - Forward Speedup: {forward_speedup:.2f}x")
        logging.info(f"  - Backward Speedup: {backward_speedup:.2f}x")
        logging.info(f"  - Total Speedup: {total_speedup:.2f}x")
    else:
        logging.info("MG GEMM method not available.")


def train_model(args):
    """
    Train an MoE model.
    """
    device = torch.device(args.device)

    # Create model
    model = MoEModel(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_experts=args.num_experts,
        top_k=args.top_k,
        use_mg_gemm=args.use_mg_gemm and has_mg_gemm,
    ).to(device)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Log model information
    logging.info("Model configuration:")
    logging.info(f"  - Vocabulary Size: {args.vocab_size}")
    logging.info(f"  - Embedding Dimension: {args.embed_dim}")
    logging.info(f"  - Hidden Dimension: {args.hidden_dim}")
    logging.info(f"  - Number of Experts: {args.num_experts}")
    logging.info(f"  - Top-K: {args.top_k}")
    logging.info(f"  - Using MG GEMM: {args.use_mg_gemm and has_mg_gemm}")

    # Training loop
    for epoch in range(args.epochs):
        logging.info(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_metrics = train_epoch(
            model=model,
            optimizer=optimizer,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            num_batches=args.train_batches,
            device=device,
            load_balance_coef=args.load_balance_coef,
        )

        # Evaluate
        eval_metrics = evaluate(
            model=model,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            num_batches=args.eval_batches,
            device=device,
        )

        # Log metrics
        logging.info(
            f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['acc']:.4f}"
        )
        logging.info(
            f"Eval Loss: {eval_metrics['loss']:.4f} | Eval Acc: {eval_metrics['acc']:.4f}"
        )
        logging.info(f"Epoch Time: {train_metrics['time']:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MoE model")

    # Model parameters
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument(
        "--embed_dim", type=int, default=512, help="Embedding dimension"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=1024, help="Hidden dimension in experts"
    )
    parser.add_argument("--num_experts", type=int, default=8, help="Number of experts")
    parser.add_argument(
        "--top_k", type=int, default=2, help="Top-k experts to route to"
    )

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--train_batches",
        type=int,
        default=100,
        help="Number of training batches per epoch",
    )
    parser.add_argument(
        "--eval_batches", type=int, default=20, help="Number of evaluation batches"
    )
    parser.add_argument(
        "--perf_batches",
        type=int,
        default=50,
        help="Number of batches for performance testing",
    )
    parser.add_argument(
        "--load_balance_coef",
        type=float,
        default=0.01,
        help="Load balancing loss coefficient",
    )

    # Runtime parameters
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--use_mg_gemm",
        action="store_true",
        help="Use MG GEMM implementation if available",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare manual and MG GEMM implementations",
    )
    parser.add_argument("--train", action="store_true", help="Train the model")

    args = parser.parse_args()

    # Check for CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA not available, using CPU instead.")
        args.device = "cpu"

    # Log basic information
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"Device: {args.device}")
    logging.info(f"MG GEMM available: {has_mg_gemm}")

    # Run the requested action
    if args.compare:
        compare_methods(args)
    elif args.train:
        train_model(args)
    else:
        # Default to comparison if no action specified
        compare_methods(args)
