# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections.abc import Callable


class Provisioner:
    """Allocates non-overlapping GPU ranges for Monarch proc meshes.

    Monarch's spawn_procs does not handle physical GPU binding, so each
    mesh on a shared host would claim GPUs 0..N-1 and collide. The
    Provisioner maintains a cursor that hands out sequential GPU ranges
    via CUDA_VISIBLE_DEVICES bootstrap callables.

    This is a workaround until Monarch exposes current_origin() or a
    gpu_offset parameter on spawn_procs.
    """

    def __init__(self, total_gpus: int = 8):
        self.total_gpus = total_gpus
        self.next_gpu = 0

    @property
    def available(self) -> int:
        return self.total_gpus - self.next_gpu

    def allocate(self, num_gpus: int) -> Callable[[], None]:
        if num_gpus > self.available:
            raise RuntimeError(
                f"Requested {num_gpus} GPUs but only {self.available} "
                f"available (total={self.total_gpus}, allocated={self.next_gpu})"
            )
        gpu_ids = list(range(self.next_gpu, self.next_gpu + num_gpus))
        self.next_gpu += num_gpus

        def _bootstrap():
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

        return _bootstrap


def create_meshes(
    host_mesh,
    gpus_per_node: int,
    gpu_requests: list[int],
    node_assignments: list[int] | None = None,
):
    """Create proc meshes with GPU isolation from a HostMesh.

    Each entry in gpu_requests becomes one ProcMesh. Entries with 0 GPUs
    produce CPU-only meshes (one process, no GPU dimension).

    Args:
        host_mesh: HostMesh to partition (from this_host(), a Job,
            or WorkerConnection).
        gpus_per_node: Physical GPUs per node.
        gpu_requests: GPUs needed per mesh. E.g. [4, 4, 0] for a
            4-GPU trainer, 4-GPU generator, and CPU-only grader.
        node_assignments: Dedicated nodes per mesh. When None (default),
            all meshes share the host_mesh and a single Provisioner
            partitions GPUs sequentially. When provided, each mesh gets
            its own host slice and Provisioner. Entries of 0 mean the
            mesh shares the host of the preceding non-zero entry.

    Returns:
        List of ProcMeshes, one per gpu_requests entry.
    """
    if node_assignments is None:
        return _create_shared_meshes(host_mesh, gpus_per_node, gpu_requests)
    return _create_dedicated_meshes(
        host_mesh, gpus_per_node, gpu_requests, node_assignments
    )


def _create_shared_meshes(host_mesh, gpus_per_node, gpu_requests):
    """All meshes share one host, GPUs partitioned by a single Provisioner."""
    provisioner = Provisioner(total_gpus=gpus_per_node)
    meshes = []
    for num_gpus in gpu_requests:
        if num_gpus == 0:
            meshes.append(host_mesh.spawn_procs())
        else:
            meshes.append(
                host_mesh.spawn_procs(
                    per_host={"gpus": num_gpus},
                    bootstrap=provisioner.allocate(num_gpus),
                )
            )
    return meshes


def _create_dedicated_meshes(host_mesh, gpus_per_node, gpu_requests, node_assignments):
    """Each mesh gets dedicated host nodes, sliced from the host_mesh."""
    # Build host slices: each mesh with nodes > 0 gets a contiguous range.
    # Meshes with nodes == 0 reuse the previous mesh's host slice and
    # its Provisioner so GPU allocations on that shared host remain disjoint.
    host_offset = 0
    host_slices = []
    provisioners = []
    last_slice = host_mesh
    last_provisioner = None
    for num_nodes in node_assignments:
        if num_nodes > 0:
            last_slice = host_mesh.slice(
                hosts=slice(host_offset, host_offset + num_nodes)
            )
            host_offset += num_nodes
            last_provisioner = Provisioner(total_gpus=gpus_per_node)
        host_slices.append(last_slice)
        provisioners.append(last_provisioner)

    meshes = []
    for num_gpus, num_nodes, h_mesh, provisioner in zip(
        gpu_requests, node_assignments, host_slices, provisioners
    ):
        if num_gpus == 0:
            meshes.append(h_mesh.spawn_procs())
        else:
            if provisioner is None:
                raise ValueError(
                    "node_assignments must begin with a positive node count "
                    "before any reused host slices"
                )
            gpus_this_node = num_gpus // max(num_nodes, 1)
            assert num_gpus % max(num_nodes, 1) == 0, (
                f"gpu_request ({num_gpus}) must be evenly divisible "
                f"by node_assignment ({num_nodes})"
            )
            meshes.append(
                h_mesh.spawn_procs(
                    per_host={"gpus": gpus_this_node},
                    bootstrap=provisioner.allocate(gpus_this_node),
                )
            )
    return meshes
