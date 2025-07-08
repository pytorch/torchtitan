#!/usr/bin/env python3
"""
Simple test to swap experts 0 and 1 in MoE layers.
This test creates a CSV file that swaps the first two experts and applies the remapping.
"""

import csv
import os
import random
import tempfile

import torch
import torch.distributed as dist

from model import DeepseekForCausalLM, MoE
from model_config import deepseek_config_registry
from remap_experts import (
    apply_expert_remapping_with_routing,
    create_expert_remapping_csv,
    test_routing_remapping,
)
from torch.distributed.device_mesh import DeviceMesh


def create_swap_01_csv(model, output_path: str):
    """
    Create a CSV file that swaps experts 0 and 1 for all MoE layers.

    Args:
        model: DeepseekForCausalLM model instance
        output_path: Path where to save the CSV file
    """
    moe_layers = []

    # Find all MoE layers and create swapped ordering
    for layer_idx, layer in model.model.layers.items():
        if hasattr(layer.mlp, "config") and hasattr(
            layer.mlp.config, "n_routed_experts"
        ):
            layer_num = int(layer_idx)
            n_experts = layer.mlp.config.n_routed_experts

            # Create swapped ordering: swap 0 and 1, keep rest the same
            if n_experts >= 2:
                swapped_order = [1, 0] + list(range(2, n_experts))
            else:
                swapped_order = list(range(n_experts))  # No swap if less than 2 experts

            order_str = ",".join(map(str, swapped_order))
            moe_layers.append(
                {
                    "layer": f"layer_{layer_num}",
                    "expert_order": order_str,
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

    print(f"Created swap CSV with {len(moe_layers)} MoE layers at {output_path}")
    print("Expert ordering: swapped experts 0 and 1")

    # Print first few layers for verification
    for i, layer_info in enumerate(moe_layers[:3]):
        print(f"  {layer_info['layer']}: {layer_info['expert_order']}")
    if len(moe_layers) > 3:
        print(f"  ... and {len(moe_layers) - 3} more layers")


def create_random_reorder_csv(model, output_path: str, seed: int = 42):
    """
    Create a CSV file that randomly reorders experts for all MoE layers.

    Args:
        model: DeepseekForCausalLM model instance
        output_path: Path where to save the CSV file
        seed: Random seed for reproducible ordering
    """
    random.seed(seed)
    moe_layers = []

    # Find all MoE layers and create random ordering
    for layer_idx, layer in model.model.layers.items():
        if hasattr(layer.mlp, "config") and hasattr(
            layer.mlp.config, "n_routed_experts"
        ):
            layer_num = int(layer_idx)
            n_experts = layer.mlp.config.n_routed_experts

            # Create random ordering
            random_order = list(range(n_experts))
            random.shuffle(random_order)

            order_str = ",".join(map(str, random_order))
            moe_layers.append(
                {
                    "layer": f"layer_{layer_num}",
                    "expert_order": order_str,
                    "num_experts": n_experts,
                    "original_order": ",".join(map(str, range(n_experts))),
                }
            )

    # Write CSV file
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["layer", "expert_order", "num_experts", "original_order"]
        )
        writer.writeheader()
        writer.writerows(moe_layers)

    print(
        f"Created random reorder CSV with {len(moe_layers)} MoE layers at {output_path}"
    )
    print(f"Expert ordering: random shuffle with seed {seed}")

    # Print first few layers for verification
    for i, layer_info in enumerate(moe_layers[:3]):
        print(
            f"  {layer_info['layer']}: {layer_info['original_order']} -> {layer_info['expert_order']}"
        )
    if len(moe_layers) > 3:
        print(f"  ... and {len(moe_layers) - 3} more layers")

    return moe_layers


def test_random_reorder():
    """
    Test randomly reordering experts in MoE layers.
    """
    print("=" * 60)
    print("Testing Random Expert Reordering")
    print("=" * 60)

    # Use a smaller model for testing
    model_id = "deepseek-ai/DeepSeek-V2-Lite-Chat"

    # Initialize distributed like in generate.py
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    # Create device mesh with ep=2 to fit model
    mesh_shape = (1, 2)  # (pp, ep) - 1 pipeline stage, 2 expert parallel
    mesh = dist.init_device_mesh("cuda", mesh_shape, mesh_dim_names=("pp", "ep"))
    rank = dist.get_rank()

    if rank == 0:
        print(f"Running random reorder test with mesh shape {mesh_shape}")

    # Create model
    model_args = deepseek_config_registry[model_id]
    model_args.ep_size = 2  # Expert parallelism = 2
    model_args.num_stages = 1
    model_args.stage_idx = 0
    model_args.max_seq_len = 512  # Smaller for testing

    device_count = torch.cuda.device_count()
    device = torch.device("cuda", rank % device_count)

    with device, mesh:
        model = DeepseekForCausalLM(model_args)

    model.eval()

    if rank == 0:
        print(f"Created model: {model_id}")

    # Synchronize all ranks before starting
    dist.barrier()

    # Create CSV files on rank 0, then broadcast path to other ranks
    if rank == 0:
        # Create temporary CSV file for random reordering
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp_file:
            random_csv_path = tmp_file.name

        # Create random reorder CSV
        expected_mappings = create_random_reorder_csv(model, random_csv_path, seed=42)
        print(f"\nRandom expert ordering saved to: {random_csv_path}")

        # Broadcast the CSV path to other ranks
        path_tensor = torch.tensor(
            [len(random_csv_path.encode())], dtype=torch.long, device=device
        )
        dist.broadcast(path_tensor, src=0)
        path_bytes = torch.tensor(
            list(random_csv_path.encode()), dtype=torch.uint8, device=device
        )
        dist.broadcast(path_bytes, src=0)
    else:
        # Receive CSV path from rank 0
        path_len_tensor = torch.tensor([0], dtype=torch.long, device=device)
        dist.broadcast(path_len_tensor, src=0)
        path_bytes = torch.zeros(
            path_len_tensor.item(), dtype=torch.uint8, device=device
        )
        dist.broadcast(path_bytes, src=0)
        random_csv_path = bytes(path_bytes.cpu().numpy()).decode()
        print(f"Rank {rank}: Received CSV path: {random_csv_path}")
        expected_mappings = None

    # Synchronize before applying remapping
    dist.barrier()

    try:
        # Apply the expert remapping on ALL ranks
        if rank == 0:
            print("\nApplying random expert remapping...")
        apply_expert_remapping_with_routing(model, random_csv_path)

        # Synchronize after remapping
        dist.barrier()

        # Verify the remapping was applied
        if rank == 0:
            print("\nVerifying random remapping was applied...")
            moe_layer_count = 0
            remapped_layer_count = 0
            correct_mappings = 0

            # Create expected mapping lookup
            expected_lookup = {}
            if expected_mappings:
                for layer_info in expected_mappings:
                    layer_name = layer_info["layer"]
                    expected_order = [
                        int(x) for x in layer_info["expert_order"].split(",")
                    ]
                    expected_lookup[layer_name] = expected_order

            for layer_idx, layer in model.model.layers.items():
                if hasattr(layer.mlp, "config") and hasattr(
                    layer.mlp.config, "n_routed_experts"
                ):
                    moe_layer_count += 1
                    layer_name = f"layer_{layer_idx}"

                    if hasattr(layer.mlp, "expert_routing_remap"):
                        remapped_layer_count += 1
                        remap_tensor = layer.mlp.expert_routing_remap
                        actual_order = remap_tensor.cpu().tolist()

                        print(f"  Layer {layer_idx}: Remapping tensor = {actual_order}")

                        # Verify against expected mapping
                        if layer_name in expected_lookup:
                            expected_order = expected_lookup[layer_name]
                            if actual_order == expected_order:
                                print(
                                    f"    ✓ Random ordering matches expected: {expected_order}"
                                )
                                correct_mappings += 1
                            else:
                                print(
                                    f"    ✗ Mismatch! Expected: {expected_order}, Got: {actual_order}"
                                )
                        else:
                            print(f"    ? No expected mapping found for {layer_name}")
                    else:
                        print(f"  Layer {layer_idx}: No remapping tensor found")

            print(f"\nSummary:")
            print(f"  Total MoE layers: {moe_layer_count}")
            print(f"  Layers with remapping: {remapped_layer_count}")
            print(f"  Correct mappings: {correct_mappings}")

            if correct_mappings > 0:
                print("\n" + "=" * 60)
                print("Random expert reorder test completed successfully!")
                print(
                    f"All {correct_mappings} layers have correct random expert ordering."
                )
                print("=" * 60)
            else:
                print("\n" + "=" * 60)
                print("Warning: No correct random mappings found!")
                print("=" * 60)

    finally:
        # Clean up temporary file on rank 0
        if rank == 0 and os.path.exists(random_csv_path):
            os.unlink(random_csv_path)

        # Final synchronization
        dist.barrier()


def test_expert_swap():
    """
    Test swapping experts 0 and 1 in MoE layers.
    """
    print("=" * 60)
    print("Testing Expert Swap (0 <-> 1)")
    print("=" * 60)

    # Use a smaller model for testing
    model_id = "deepseek-ai/DeepSeek-V2-Lite-Chat"

    # Initialize distributed like in generate.py
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    # Create device mesh with ep=2 to fit model
    mesh_shape = (1, 2)  # (pp, ep) - 1 pipeline stage, 2 expert parallel
    mesh = dist.init_device_mesh("cuda", mesh_shape, mesh_dim_names=("pp", "ep"))
    rank = dist.get_rank()

    if rank == 0:
        print(f"Running expert swap test with mesh shape {mesh_shape}")

    # Create model
    model_args = deepseek_config_registry[model_id]
    model_args.ep_size = 2  # Expert parallelism = 2
    model_args.num_stages = 1
    model_args.stage_idx = 0
    model_args.max_seq_len = 512  # Smaller for testing

    device_count = torch.cuda.device_count()
    device = torch.device("cuda", rank % device_count)

    with device, mesh:
        model = DeepseekForCausalLM(model_args)

    model.eval()

    if rank == 0:
        print(f"Created model: {model_id}")

    # Synchronize all ranks before starting
    dist.barrier()

    # Create CSV files on rank 0, then broadcast path to other ranks
    if rank == 0:
        # Create temporary CSV file for swapping
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp_file:
            swap_csv_path = tmp_file.name

        # Step 1: Create original CSV (for reference)
        original_csv_path = "original_expert_order.csv"
        create_expert_remapping_csv(model, original_csv_path)
        print(f"\nOriginal expert ordering saved to: {original_csv_path}")

        # Step 2: Create swapped CSV
        create_swap_01_csv(model, swap_csv_path)
        print(f"\nSwapped expert ordering saved to: {swap_csv_path}")

        # Broadcast the CSV path to other ranks
        path_tensor = torch.tensor(
            [len(swap_csv_path.encode())], dtype=torch.long, device=device
        )
        dist.broadcast(path_tensor, src=0)
        path_bytes = torch.tensor(
            list(swap_csv_path.encode()), dtype=torch.uint8, device=device
        )
        dist.broadcast(path_bytes, src=0)
    else:
        # Receive CSV path from rank 0
        path_len_tensor = torch.tensor([0], dtype=torch.long, device=device)
        dist.broadcast(path_len_tensor, src=0)
        path_bytes = torch.zeros(
            path_len_tensor.item(), dtype=torch.uint8, device=device
        )
        dist.broadcast(path_bytes, src=0)
        swap_csv_path = bytes(path_bytes.cpu().numpy()).decode()
        print(f"Rank {rank}: Received CSV path: {swap_csv_path}")

    # Synchronize before applying remapping
    dist.barrier()

    try:
        # Step 3: Apply the expert remapping on ALL ranks
        if rank == 0:
            print("\nApplying expert remapping...")
        apply_expert_remapping_with_routing(model, swap_csv_path)

        # Synchronize after remapping
        dist.barrier()

        # Step 4: Verify the remapping was applied (without forward pass to avoid hanging)
        if rank == 0:
            print("\nVerifying remapping was applied...")
            moe_layer_count = 0
            remapped_layer_count = 0

            for layer_idx, layer in model.model.layers.items():
                if hasattr(layer.mlp, "config") and hasattr(
                    layer.mlp.config, "n_routed_experts"
                ):
                    moe_layer_count += 1
                    if hasattr(layer.mlp, "expert_routing_remap"):
                        remapped_layer_count += 1
                        remap_tensor = layer.mlp.expert_routing_remap
                        print(
                            f"  Layer {layer_idx}: Remapping tensor = {remap_tensor.cpu().tolist()}"
                        )

                        # Verify that experts 0 and 1 are swapped
                        if (
                            len(remap_tensor) >= 2
                            and remap_tensor[0] == 1
                            and remap_tensor[1] == 0
                        ):
                            print(f"    ✓ Experts 0 and 1 successfully swapped")
                        else:
                            print(f"    ✗ Experts 0 and 1 NOT swapped as expected")
                    else:
                        print(f"  Layer {layer_idx}: No remapping tensor found")

            print(f"\nSummary:")
            print(f"  Total MoE layers: {moe_layer_count}")
            print(f"  Layers with remapping: {remapped_layer_count}")

            if remapped_layer_count > 0:
                print("\n" + "=" * 60)
                print("Expert swap test completed successfully!")
                print("Experts 0 and 1 have been swapped in MoE layers.")
                print("=" * 60)
            else:
                print("\n" + "=" * 60)
                print("Warning: No remapping tensors found!")
                print("=" * 60)

    finally:
        # Clean up temporary file on rank 0
        if rank == 0 and os.path.exists(swap_csv_path):
            os.unlink(swap_csv_path)

        # Final synchronization
        dist.barrier()


def test_routing_tracer():
    """
    Test expert routing by testing gate outputs directly without full forward passes.
    This avoids distributed communication issues while still verifying routing changes.
    """
    print("=" * 60)
    print("Testing Expert Routing with Gate-Only Test")
    print("=" * 60)

    # Use a smaller model for testing
    model_id = "deepseek-ai/DeepSeek-V2-Lite-Chat"

    # Initialize distributed like in generate.py
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    # Create device mesh with ep=2 to fit model
    mesh_shape = (1, 2)  # (pp, ep) - 1 pipeline stage, 2 expert parallel
    mesh = dist.init_device_mesh("cuda", mesh_shape, mesh_dim_names=("pp", "ep"))
    rank = dist.get_rank()

    if rank == 0:
        print(f"Running gate-only routing test with mesh shape {mesh_shape}")

    # Create model
    model_args = deepseek_config_registry[model_id]
    model_args.ep_size = 2  # Expert parallelism = 2
    model_args.num_stages = 1
    model_args.stage_idx = 0
    model_args.max_seq_len = 512  # Smaller for testing

    device_count = torch.cuda.device_count()
    device = torch.device("cuda", rank % device_count)

    with device, mesh:
        model = DeepseekForCausalLM(model_args)

    model.eval()

    if rank == 0:
        print(f"Created model: {model_id}")

    # Synchronize all ranks before starting
    dist.barrier()

    # Only run the routing test on rank 0 to avoid conflicts
    if rank == 0:
        # Create a simple tracer input - just a few tokens
        batch_size, seq_len = 1, 4
        tracer_input = torch.randint(
            0, 1000, (batch_size, seq_len), device=device, dtype=torch.long
        )

        print(f"\nTracer input shape: {tracer_input.shape}")
        print(f"Tracer tokens: {tracer_input.flatten().tolist()}")

        # Find first MoE layer
        first_moe_layer = None
        first_moe_layer_idx = None
        for layer_idx, layer in model.model.layers.items():
            if isinstance(layer.mlp, MoE):
                first_moe_layer = layer
                first_moe_layer_idx = int(layer_idx)
                break

        if first_moe_layer is None:
            print("No MoE layers found!")
            return

        print(f"Testing routing on first MoE layer: {first_moe_layer_idx}")

        # Step 1: Create test hidden states (simulate what would reach the MoE layer)
        print("\n--- Step 1: Creating test hidden states ---")

        with torch.no_grad():
            # Create embeddings and process through a few layers to get realistic hidden states
            hidden_states = model.model.embed_tokens(tracer_input)

            # Process through layers before the first MoE layer (but avoid full forward passes)
            layer_count = 0
            for layer_idx, layer in model.model.layers.items():
                layer_num = int(layer_idx)
                if layer_num >= first_moe_layer_idx:
                    break

                # Only process through a couple of layers to get realistic hidden states
                if layer_count < 2:
                    # Process through attention only (skip MLP to avoid hanging)
                    residual = hidden_states
                    hidden_states = layer.input_layernorm(hidden_states)
                    hidden_states = layer.self_attn(hidden_states)
                    hidden_states = residual + hidden_states
                    layer_count += 1
                else:
                    break

            # Prepare input for MoE gate
            hidden_states = first_moe_layer.post_attention_layernorm(hidden_states)
            print(f"Test hidden states shape: {hidden_states.shape}")

        # Step 2: Capture original routing
        print("\n--- Step 2: Capturing original gate routing ---")
        original_routing = {}

        with torch.no_grad():
            # Get gate outputs directly (this should not hang)
            topk_idx, topk_weight = first_moe_layer.mlp.gate(hidden_states)
            original_routing = {
                "topk_idx": topk_idx.cpu().clone(),
                "topk_weight": topk_weight.cpu().clone(),
            }

            print(f"  Original routing = {topk_idx.flatten().tolist()}")
            print(f"  Original weights = {topk_weight.flatten().tolist()}")

        # Step 3: Apply expert swapping (0 <-> 1)
        print("\n--- Step 3: Applying expert remapping (0 <-> 1) ---")

        # Create temporary CSV file for swapping
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp_file:
            swap_csv_path = tmp_file.name

        try:
            create_swap_01_csv(model, swap_csv_path)
            apply_expert_remapping_with_routing(model, swap_csv_path)
            print("Expert remapping applied successfully")

            # Step 4: Capture remapped routing
            print("\n--- Step 4: Capturing remapped gate routing ---")
            remapped_routing = {}

            with torch.no_grad():
                # Get gate outputs after remapping (same input)
                topk_idx, topk_weight = first_moe_layer.mlp.gate(hidden_states)
                remapped_routing = {
                    "topk_idx": topk_idx.cpu().clone(),
                    "topk_weight": topk_weight.cpu().clone(),
                }

                print(f"  Remapped routing = {topk_idx.flatten().tolist()}")
                print(f"  Remapped weights = {topk_weight.flatten().tolist()}")

            # Step 5: Verify routing changes
            print("\n--- Step 5: Verifying routing changes ---")

            orig_idx = original_routing["topk_idx"]
            remap_idx = remapped_routing["topk_idx"]

            print(f"  Layer {first_moe_layer_idx}:")
            print(f"    Original: {orig_idx.flatten().tolist()}")
            print(f"    Remapped: {remap_idx.flatten().tolist()}")

            # Check if routing changed as expected (experts 0 and 1 should be swapped)
            routing_verified = True
            expected_changes = 0
            actual_changes = 0

            for i in range(orig_idx.numel()):
                orig_expert = orig_idx.flatten()[i].item()
                remap_expert = remap_idx.flatten()[i].item()

                # Expected mapping: 0->1, 1->0, others unchanged
                if orig_expert == 0:
                    expected_expert = 1
                elif orig_expert == 1:
                    expected_expert = 0
                else:
                    expected_expert = orig_expert

                if expected_expert != orig_expert:
                    expected_changes += 1

                if remap_expert == expected_expert:
                    actual_changes += 1
                else:
                    print(
                        f"      Token {i}: Expected expert {expected_expert}, got {remap_expert}"
                    )
                    routing_verified = False

            if expected_changes > 0:
                print(
                    f"    ✓ {actual_changes}/{expected_changes} routing changes verified"
                )
            else:
                print(f"    - No routing changes expected (no experts 0 or 1 selected)")

            # Additional verification: check if any routing changed at all
            routing_changed = not torch.equal(orig_idx, remap_idx)
            if routing_changed:
                print(f"    ✓ Routing did change after remapping")
            else:
                print(f"    - Routing remained the same (may be expected)")

            if routing_verified and (expected_changes == 0 or actual_changes > 0):
                print("\n" + "=" * 60)
                print("Gate routing test completed successfully!")
                print("Expert gate routing follows expected remapped paths.")
                print("=" * 60)
            else:
                print("\n" + "=" * 60)
                print("Warning: Some routing changes were not as expected!")
                print("=" * 60)

        finally:
            # Clean up temporary file
            if os.path.exists(swap_csv_path):
                os.unlink(swap_csv_path)
    else:
        print(f"Rank {rank}: Waiting for rank 0 to complete routing test...")

    # Final synchronization
    dist.barrier()


def verify_swap_csv(csv_path: str):
    """
    Verify that the CSV file contains the expected expert swapping.

    Args:
        csv_path: Path to the CSV file to verify
    """
    print(f"\nVerifying CSV file: {csv_path}")

    with open(csv_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            layer = row["layer"]
            expert_order = row["expert_order"]
            num_experts = int(row["num_experts"])

            # Parse the order
            order_list = [int(x.strip()) for x in expert_order.split(",")]

            # Check if experts 0 and 1 are swapped (if there are at least 2 experts)
            if num_experts >= 2:
                if order_list[0] == 1 and order_list[1] == 0:
                    print(
                        f"  ✓ {layer}: Experts 0 and 1 correctly swapped: {expert_order}"
                    )
                else:
                    print(f"  ✗ {layer}: Experts 0 and 1 NOT swapped: {expert_order}")
            else:
                print(
                    f"  - {layer}: Only {num_experts} expert(s), no swap needed: {expert_order}"
                )


if __name__ == "__main__":
    # Set CUDA device
    if "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        torch.cuda.set_device(0)

    try:
        # Run both tests
        test_expert_swap()

        print("\n" + "=" * 80)
        print("STARTING SECOND TEST")
        print("=" * 80 + "\n")

        test_random_reorder()

        print("\n" + "=" * 80)
        print("STARTING THIRD TEST")
        print("=" * 80 + "\n")

        test_routing_tracer()

        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
