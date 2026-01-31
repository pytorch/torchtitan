"""
Device Mesh Visualizer for Distributed Training

Creates comprehensive visualization of how GPUs are allocated across
all parallelism dimensions: DP, PP, TP, CP, EP.
"""

import os
from typing import Dict

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.tools.logging import logger


def get_rank_info() -> Dict:
    """Get current rank's information across all process groups."""
    info = {
        "global_rank": dist.get_rank() if dist.is_initialized() else 0,
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
        "node_rank": int(os.environ.get("GROUP_RANK", os.environ.get("NODE_RANK", 0))),
    }
    return info


def visualize_mesh_structure(
    mesh: DeviceMesh,
    parallel_dims,
    rank: int = 0,
) -> str:
    """
    Create a detailed text visualization of the device mesh structure.

    Args:
        mesh: The DeviceMesh object
        parallel_dims: ParallelDims object with all parallelism settings
        rank: Current rank (only rank 0 prints full visualization)

    Returns:
        String visualization of the mesh
    """
    lines = []
    lines.append("=" * 100)
    lines.append("DEVICE MESH VISUALIZATION")
    lines.append("=" * 100)

    # Basic info
    lines.append("\n[CLUSTER INFO]")
    lines.append(f"  Total GPUs: {parallel_dims.world_size}")
    lines.append(f"  Nodes: {parallel_dims.world_size // 8} (assuming 8 GPUs/node)")

    # Parallelism dimensions
    lines.append("\n[PARALLELISM DIMENSIONS]")
    lines.append(f"  DP Replicate (HSDP): {parallel_dims.dp_replicate}")
    lines.append(f"  DP Shard (FSDP):     {parallel_dims.dp_shard}")
    lines.append(f"  Context Parallel:    {parallel_dims.cp}")
    lines.append(f"  Tensor Parallel:     {parallel_dims.tp}")
    lines.append(f"  Pipeline Parallel:   {parallel_dims.pp}")
    lines.append(f"  Expert Parallel:     {parallel_dims.ep}")
    lines.append(f"  Expert TP:           {parallel_dims.etp}")

    # Mesh structure
    lines.append("\n[MESH STRUCTURE]")
    lines.append(f"  Mesh dim names: {mesh.mesh_dim_names}")
    lines.append(f"  Mesh shape:     {mesh.mesh.shape}")

    # Log each dimension
    for i, (name, size) in enumerate(zip(mesh.mesh_dim_names, mesh.mesh.shape)):
        lines.append(f"    Dim {i}: {name:20s} = {size}")

    # EP-specific derived dimensions
    if parallel_dims.ep > 1:
        if parallel_dims.etp == parallel_dims.tp:
            dp_shard_mod_ep = (
                parallel_dims.dp_shard * parallel_dims.cp // parallel_dims.ep
            )
            dp_shard_in_ep = parallel_dims.ep // parallel_dims.cp
        else:
            dp_shard_mod_ep = (
                parallel_dims.dp_shard
                * parallel_dims.cp
                * parallel_dims.tp
                // parallel_dims.ep
            )
            dp_shard_in_ep = parallel_dims.ep // (parallel_dims.cp * parallel_dims.tp)

        lines.append("\n[EXPERT PARALLEL DERIVED DIMENSIONS]")
        lines.append(f"  dp_shard_mod_ep (DP for non-experts): {dp_shard_mod_ep}")
        lines.append(f"  dp_shard_in_ep (DP within EP group):  {dp_shard_in_ep}")
        lines.append(f"  ep_group_size (EP degree):            {parallel_dims.ep}")
        lines.append("")
        lines.append("  Formula: dp_shard = dp_shard_mod_ep * dp_shard_in_ep")
        lines.append(
            f"           {parallel_dims.dp_shard} = {dp_shard_mod_ep} * {dp_shard_in_ep}"
        )
        lines.append("")
        lines.append("  Formula: ep = dp_shard_in_ep * cp")
        lines.append(
            f"           {parallel_dims.ep} = {dp_shard_in_ep} * {parallel_dims.cp}"
        )

    # Submesh info
    lines.append("\n[SUBMESHES]")

    # Try to get submesh info
    submesh_names = ["dp", "dp_shard_cp", "dp_cp", "ep", "cp", "tp", "pp"]
    for name in submesh_names:
        try:
            submesh = mesh[name]
            lines.append(
                f"  {name:15s}: size={submesh.size():4d}, dim_names={submesh.mesh_dim_names}"
            )
        except (KeyError, RuntimeError):
            pass

    return "\n".join(lines)


def visualize_gpu_allocation(
    mesh: DeviceMesh,
    parallel_dims,
    rank: int = 0,
) -> str:
    """
    Create a grid visualization showing GPU allocation.

    For 16 nodes (128 GPUs) with EP=64, CP=8:
    - Shows how each GPU maps to (dp_shard_mod_ep, dp_shard_in_ep, cp) coordinates
    """
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("GPU ALLOCATION GRID")
    lines.append("=" * 100)

    world_size = parallel_dims.world_size
    num_nodes = world_size // 8

    # For EP-enabled config
    if parallel_dims.ep > 1:
        if parallel_dims.etp == parallel_dims.tp:
            dp_shard_mod_ep = (
                parallel_dims.dp_shard * parallel_dims.cp // parallel_dims.ep
            )
            dp_shard_in_ep = parallel_dims.ep // parallel_dims.cp
        else:
            dp_shard_mod_ep = (
                parallel_dims.dp_shard
                * parallel_dims.cp
                * parallel_dims.tp
                // parallel_dims.ep
            )
            dp_shard_in_ep = parallel_dims.ep // (parallel_dims.cp * parallel_dims.tp)

        lines.append(
            f"\nMesh: [{dp_shard_mod_ep}] x [{dp_shard_in_ep}] x [{parallel_dims.cp}] = {world_size} GPUs"
        )
        lines.append("      [dp_shard_mod_ep] x [dp_shard_in_ep] x [cp]")
        lines.append("")

        # Create mapping from global rank to mesh coordinates
        lines.append("GPU -> Mesh Coordinate Mapping:")
        lines.append("-" * 80)
        lines.append(
            f"{'Node':>6} | {'GPU':>4} | {'Rank':>5} | {'dp_mod_ep':>10} | {'dp_in_ep':>10} | {'cp':>4} | {'EP Group':>10}"
        )
        lines.append("-" * 80)

        # The mesh is laid out as: dp_shard_mod_ep (slowest) x dp_shard_in_ep x cp (fastest)
        for node in range(num_nodes):
            for local_gpu in range(8):
                global_rank = node * 8 + local_gpu

                # Compute mesh coordinates (assuming row-major ordering)
                # Total size = dp_shard_mod_ep * dp_shard_in_ep * cp
                cp_coord = global_rank % parallel_dims.cp
                dp_in_ep_coord = (global_rank // parallel_dims.cp) % dp_shard_in_ep
                dp_mod_ep_coord = global_rank // (parallel_dims.cp * dp_shard_in_ep)

                # EP group = dp_in_ep_coord * cp + cp_coord (within each dp_shard_mod_ep group)
                ep_group = dp_in_ep_coord * parallel_dims.cp + cp_coord

                row = (
                    f"{node:>6} | {local_gpu:>4} | {global_rank:>5} | "
                    f"{dp_mod_ep_coord:>10} | {dp_in_ep_coord:>10} | "
                    f"{cp_coord:>4} | {ep_group:>10}"
                )
                lines.append(row)

            if node < num_nodes - 1:
                lines.append("-" * 80)
    else:
        lines.append(
            f"\nMesh: [{parallel_dims.dp_shard}] x [{parallel_dims.cp}] = {world_size} GPUs"
        )
        lines.append("      [dp_shard] x [cp]")

    return "\n".join(lines)


def visualize_expert_parallel_groups(
    mesh: DeviceMesh,
    parallel_dims,
    rank: int = 0,
) -> str:
    """
    Visualize which GPUs belong to which Expert Parallel group.
    """
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("EXPERT PARALLEL GROUP ALLOCATION")
    lines.append("=" * 100)

    if parallel_dims.ep <= 1:
        lines.append("Expert Parallel is disabled (EP=1)")
        return "\n".join(lines)

    world_size = parallel_dims.world_size

    if parallel_dims.etp == parallel_dims.tp:
        dp_shard_mod_ep = parallel_dims.dp_shard * parallel_dims.cp // parallel_dims.ep
        dp_shard_in_ep = parallel_dims.ep // parallel_dims.cp
    else:
        dp_shard_mod_ep = (
            parallel_dims.dp_shard
            * parallel_dims.cp
            * parallel_dims.tp
            // parallel_dims.ep
        )
        dp_shard_in_ep = parallel_dims.ep // (parallel_dims.cp * parallel_dims.tp)

    lines.append(f"\nEP={parallel_dims.ep} experts distributed across GPUs")
    lines.append(
        f"Each EP group has {parallel_dims.ep} GPUs working on different experts"
    )
    lines.append(
        f"There are {dp_shard_mod_ep} such EP groups (for FSDP replication of experts)"
    )
    lines.append("")

    # Group GPUs by their dp_shard_mod_ep coordinate
    lines.append("EP Groups (GPUs that share the same set of experts):")
    lines.append("-" * 80)

    for dp_mod_ep_idx in range(dp_shard_mod_ep):
        # Find all ranks in this dp_shard_mod_ep group
        ranks_in_group = []
        for global_rank in range(world_size):
            dp_mod_ep_coord = global_rank // (parallel_dims.cp * dp_shard_in_ep)
            if dp_mod_ep_coord == dp_mod_ep_idx:
                ranks_in_group.append(global_rank)

        lines.append(f"\nDP_SHARD_MOD_EP group {dp_mod_ep_idx}:")
        lines.append(
            f"  GPUs: {ranks_in_group[:16]}{'...' if len(ranks_in_group) > 16 else ''}"
        )
        lines.append(f"  Total: {len(ranks_in_group)} GPUs")
        lines.append("  These GPUs have IDENTICAL expert parameters (FSDP sharded)")

    return "\n".join(lines)


def visualize_context_parallel_groups(
    mesh: DeviceMesh,
    parallel_dims,
    rank: int = 0,
) -> str:
    """
    Visualize Context Parallel groups - GPUs that work on different parts of the sequence.
    """
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("CONTEXT PARALLEL GROUP ALLOCATION")
    lines.append("=" * 100)

    if parallel_dims.cp <= 1:
        lines.append("Context Parallel is disabled (CP=1)")
        return "\n".join(lines)

    world_size = parallel_dims.world_size
    cp = parallel_dims.cp

    lines.append(f"\nCP={cp} - Each sequence is split into {cp} chunks")
    lines.append(
        "GPUs with the same (dp_shard, ep) coordinates but different cp coordinates"
    )
    lines.append("work on different parts of the same sequence.")
    lines.append("")

    # Show a few example CP groups
    lines.append("Example CP groups (first few):")
    lines.append("-" * 80)

    num_cp_groups = world_size // cp
    for cp_group_idx in range(min(4, num_cp_groups)):
        ranks_in_group = [cp_group_idx * cp + i for i in range(cp)]
        lines.append(f"\nCP group {cp_group_idx}:")
        lines.append(f"  GPUs: {ranks_in_group}")
        lines.append(f"  These {cp} GPUs process different chunks of the same sequence")

    if num_cp_groups > 4:
        lines.append(f"\n... and {num_cp_groups - 4} more CP groups")

    return "\n".join(lines)


def visualize_fsdp_sharding(
    mesh: DeviceMesh,
    parallel_dims,
    rank: int = 0,
) -> str:
    """
    Visualize FSDP sharding - which GPUs share which parameters.
    """
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("FSDP SHARDING VISUALIZATION")
    lines.append("=" * 100)

    if parallel_dims.ep > 1:
        if parallel_dims.etp == parallel_dims.tp:
            dp_shard_mod_ep = (
                parallel_dims.dp_shard * parallel_dims.cp // parallel_dims.ep
            )
            dp_shard_in_ep = parallel_dims.ep // parallel_dims.cp
        else:
            dp_shard_mod_ep = (
                parallel_dims.dp_shard
                * parallel_dims.cp
                * parallel_dims.tp
                // parallel_dims.ep
            )
            dp_shard_in_ep = parallel_dims.ep // (parallel_dims.cp * parallel_dims.tp)

        dp_shard_cp_size = parallel_dims.dp_shard * parallel_dims.cp

        lines.append("\n[NON-EXPERT PARAMETERS (Attention, Embeddings, etc.)]")
        lines.append("  FSDP mesh: dp_shard_cp")
        lines.append(f"  FSDP group size: {dp_shard_cp_size} GPUs")
        lines.append(f"  Each parameter is sharded across {dp_shard_cp_size} GPUs")
        lines.append(
            f"  All-gather buffer size per param: original_size / {dp_shard_cp_size}"
        )

        lines.append("\n[EXPERT PARAMETERS (MoE experts)]")
        lines.append("  FSDP mesh: dp_shard_mod_ep")
        lines.append(f"  FSDP group size: {dp_shard_mod_ep} GPUs")
        lines.append(
            f"  Each expert's parameters are sharded across {dp_shard_mod_ep} GPUs"
        )
        lines.append(
            f"  All-gather buffer size per expert param: original_size / {dp_shard_mod_ep}"
        )

        lines.append("\n[MEMORY IMPLICATIONS]")
        lines.append(
            f"  Non-expert params: sharded {dp_shard_cp_size}x -> small per-GPU footprint"
        )
        lines.append(
            f"  Expert params: sharded only {dp_shard_mod_ep}x -> larger per-GPU footprint"
        )
        lines.append("  ")
        lines.append("  As DP increases:")
        lines.append(
            "    - dp_shard_cp increases -> non-expert params get more sharded"
        )
        lines.append(
            "    - dp_shard_mod_ep increases -> expert params get more sharded"
        )
        lines.append(
            "    - BUT: all-gather/reduce-scatter buffers scale with group size!"
        )

    else:
        dp_shard_cp_size = parallel_dims.dp_shard * parallel_dims.cp
        lines.append("\n[ALL PARAMETERS]")
        lines.append("  FSDP mesh: dp_shard_cp")
        lines.append(f"  FSDP group size: {dp_shard_cp_size} GPUs")
        lines.append(f"  Each parameter is sharded across {dp_shard_cp_size} GPUs")

    return "\n".join(lines)


def create_full_visualization(
    mesh: DeviceMesh,
    parallel_dims,
    rank: int = 0,
) -> str:
    """Create a comprehensive visualization of the entire mesh structure."""
    parts = [
        visualize_mesh_structure(mesh, parallel_dims, rank),
        visualize_gpu_allocation(mesh, parallel_dims, rank),
        visualize_expert_parallel_groups(mesh, parallel_dims, rank),
        visualize_context_parallel_groups(mesh, parallel_dims, rank),
        visualize_fsdp_sharding(mesh, parallel_dims, rank),
    ]

    full_viz = "\n".join(parts)
    full_viz += "\n" + "=" * 100
    full_viz += "\nEND OF DEVICE MESH VISUALIZATION"
    full_viz += "\n" + "=" * 100

    return full_viz


def log_mesh_visualization(mesh: DeviceMesh, parallel_dims):
    """Log the full mesh visualization (only on rank 0)."""
    rank = dist.get_rank() if dist.is_initialized() else 0

    if rank == 0:
        viz = create_full_visualization(mesh, parallel_dims, rank)
        # Log each line separately for better formatting
        for line in viz.split("\n"):
            logger.info(f"[MESH-VIZ] {line}")
