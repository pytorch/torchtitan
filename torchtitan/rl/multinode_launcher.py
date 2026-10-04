# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Multi-node RL training launcher for Monarch on XPU (PBS/mpiexec).

mpiexec launches one rank per node. Each rank starts a Monarch worker.
Rank 0 additionally runs the RL controller that attaches to all workers
and orchestrates GRPO training.

Configurable: number of nodes (from PBS) and GPUs per node (--gpus_per_node).
Parallelism auto-scales: nodes split evenly between trainer and generator,
trainer dp_shard and the generator replica count fill the available GPUs.

Unrecognized arguments are forwarded verbatim to ``ConfigLoader``, so the config
CLI (``--module``, ``--config``, ``--output-dir``, ``--override``, ...) works here
exactly as it does for ``torchtitan.rl.train``.

Usage:
    mpiexec -n $NUM_NODES -ppn 1 --hosts $ALL_NODES --cpu-bind none --envall \
        python3 -m torchtitan.rl.multinode_launcher \
        --num_nodes=2 --gpus_per_node=4 --all_nodes=$ALL_NODES \
        --module torchtitan_recipes.rl.alphabet_sort_xpu \
        --config rl_grpo_lora_qwen3_0_6b \
        --output-dir=outputs/rl_lora_multinode
"""

import argparse
import asyncio
import glob
import logging
import os
import socket
import sys
import threading
import time

# expandable_segments corrupts XPU's oneCCL USM pointers.
if "ZE_AFFINITY_MASK" not in os.environ:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logger = logging.getLogger(__name__)


def _newest_cray_libfabric() -> str:
    """Path to the newest /opt/cray/libfabric/*/lib64/libfabric.so.1, or "".

    Aurora system refreshes rotate the version directory: 1.22.0 disappeared in
    the 2026-09 refresh, and because a stale path surfaces only as an opaque
    oneCCL "failed to initialize ATL" cascade, resolve it instead of pinning.
    """

    def version_key(so_path: str) -> tuple[int, ...]:
        version = so_path.split("/opt/cray/libfabric/", 1)[1].split("/", 1)[0]
        return tuple(int(p) if p.isdigit() else -1 for p in version.split("."))

    return max(
        glob.glob("/opt/cray/libfabric/*/lib64/libfabric.so.1"),
        key=version_key,
        default="",
    )


MONARCH_PORT = 26600


def get_mpi_rank() -> int:
    """Get MPI rank from environment (Cray PALS, PMI, or MPICH)."""
    for var in ("PALS_RANKID", "PMI_RANK", "PMIX_RANK", "OMPI_COMM_WORLD_RANK"):
        if var in os.environ:
            return int(os.environ[var])
    return 0


def _configure_timeouts() -> None:
    """Increase Monarch spawn timeouts for multi-node XPU (device init is slow)."""
    from monarch._rust_bindings.monarch_hyperactor.channel import ChannelTransport
    from monarch._rust_bindings.monarch_hyperactor.config import configure

    configure(
        default_transport=ChannelTransport.TcpWithHostname,
        host_spawn_ready_timeout="300s",
        mesh_proc_spawn_max_idle="300s",
    )


def _spawn_generator_meshes(
    host_mesh,
    *,
    num_generators: int,
    per_generator_world_size: int,
    gpus_per_node: int,
) -> list:
    """Spawn one proc mesh per generator, packed onto the generator hosts.

    ``train.spawn_proc_mesh`` cannot do this: it builds a fresh
    ``PerHostProvisioner`` for every role mesh, so two generators sharing a node
    would both be handed tile 0 and collide. Here one provisioner per node hands
    out disjoint tile ranges, which is what makes the device visibility env var
    partition the node.
    """
    from monarch.actor import default_bootstrap_cmd

    from torchtitan.rl.train import _bootstrap_generator, PerHostProvisioner

    if per_generator_world_size > gpus_per_node:
        raise ValueError(
            f"a generator needs {per_generator_world_size} GPUs, more than the "
            f"{gpus_per_node} on one node; multi-node generators are not supported"
        )
    generators_per_node = gpus_per_node // per_generator_world_size
    num_nodes = len(host_mesh)
    if num_generators > generators_per_node * num_nodes:
        raise ValueError(
            f"{num_generators} generators of {per_generator_world_size} GPU(s) "
            f"do not fit on {num_nodes} node(s) of {gpus_per_node} GPUs"
        )

    provisioners: dict[int, PerHostProvisioner] = {}
    meshes = []
    for index in range(num_generators):
        node = index // generators_per_node
        provisioner = provisioners.setdefault(
            node, PerHostProvisioner(total_gpus=gpus_per_node)
        )
        env = provisioner.allocate(per_generator_world_size)
        node_mesh = host_mesh.slice(hosts=slice(node, node + 1))
        logger.info(f"generator {index} on node {node} with env {env}")
        meshes.append(
            node_mesh.spawn_procs(
                per_host={"gpus": per_generator_world_size},
                bootstrap=_bootstrap_generator,
                bootstrap_command=default_bootstrap_cmd().with_env(env),
            )
        )
    return meshes


def run_worker(address: str) -> None:
    """Start Monarch worker loop (blocks forever)."""
    from monarch.actor import run_worker_loop_forever

    _configure_timeouts()
    logger.info(f"Monarch worker starting on {address}")
    run_worker_loop_forever(address=address, ca="trust_all_connections")


async def run_controller(
    worker_addrs: list[str],
    num_nodes: int,
    gpus_per_node: int,
    train_argv: list[str],
    *,
    tensor_parallel_degree: int | None = None,
    data_parallel_replicate_degree: int | None = None,
) -> None:
    """Attach to all workers and run the RL training loop.

    Auto-scales parallelism to fill allocated resources:
    - Nodes split evenly: first half trainer, second half generator
    - Trainer dp_shard = trainer_gpus / (TP * PP * CP)
    - Generator dp = generator_gpus / gen_TP

    ``tensor_parallel_degree`` and ``data_parallel_replicate_degree`` override the
    recipe when given; ``None`` keeps the recipe's value. They are taken here
    rather than on the config CLI because dp_shard is derived from them below.
    """
    from dataclasses import replace

    from monarch._src.actor.bootstrap import attach_to_workers

    from torchtitan.config import ConfigLoader
    from torchtitan.observability import structured_logger as sl
    from torchtitan.rl.controller import Controller
    from torchtitan.rl.train import (
        _compute_generator_world_size,
        _compute_trainer_world_size,
        _preimport_torch,
        _spawn_proc_mesh,
        spawn_storage_volume_mesh_fn,
    )

    os.environ["MONARCH_ACTOR_QUEUE_DISPATCH"] = "0"

    # Parse the train config using the same CLI as train.py
    config = ConfigLoader().load(train_argv)
    assert isinstance(config, Controller.Config)

    if tensor_parallel_degree is not None:
        config.trainer.parallelism = replace(
            config.trainer.parallelism,
            tensor_parallel_degree=tensor_parallel_degree,
        )
    if data_parallel_replicate_degree is not None:
        config.trainer.parallelism = replace(
            config.trainer.parallelism,
            data_parallel_replicate_degree=data_parallel_replicate_degree,
        )

    sl.init_structured_logger(
        source="rl_controller",
        output_dir=config.dump_folder,
        rank=0,
        enable=config.trainer.debug.enable_structured_logging,
    )

    logger.info(f"Attaching to {num_nodes} workers: {worker_addrs}")
    host_mesh = attach_to_workers(
        name="grpo_xpu",
        ca="trust_all_connections",
        workers=worker_addrs,
    )

    # Split nodes: half trainer, half generator. For odd counts, give
    # the extra node to generator (more generation throughput is better).
    num_trainer_nodes = num_nodes // 2 or 1
    num_generator_nodes = num_nodes - num_trainer_nodes

    trainer_host_mesh = host_mesh.slice(hosts=slice(0, num_trainer_nodes))
    generator_host_mesh = host_mesh.slice(hosts=slice(num_trainer_nodes, num_nodes))

    # Scale parallelism to fill allocated GPUs.
    # Cap dp_shard at the LoRA rank (if LoRA is used) to avoid zero-sized FSDP
    # shards that TorchStore cannot handle. Excess GPUs go to dp_replicate.
    trainer_total_gpus = num_trainer_nodes * gpus_per_node
    tp = config.trainer.parallelism.tensor_parallel_degree
    pp = config.trainer.parallelism.pipeline_parallel_degree
    cp = config.trainer.parallelism.context_parallel_degree
    dp_replicate = config.trainer.parallelism.data_parallel_replicate_degree
    trainer_dp_shard = trainer_total_gpus // (tp * pp * cp * dp_replicate)

    # Detect LoRA rank from the model config tree. After LoRA conversion,
    # target Linear configs become LoRALinear.Config with `rank` + `alpha`.
    def _find_lora_rank(obj, depth=0):
        if depth > 5:
            return None
        if hasattr(obj, "rank") and hasattr(obj, "alpha"):
            return obj.rank
        if isinstance(obj, (list, tuple)):
            for item in obj:
                r = _find_lora_rank(item, depth + 1)
                if r is not None:
                    return r
        elif hasattr(obj, "__dataclass_fields__"):
            for field_name in obj.__dataclass_fields__:
                r = _find_lora_rank(getattr(obj, field_name), depth + 1)
                if r is not None:
                    return r
        return None

    # FSDP shards each param on dim 0, and the smallest LoRA tensor dim is the
    # LoRA rank, so dp_shard can grow up to the rank without producing zero-sized
    # shards (which TorchStore cannot handle). Cap at the rank -- NOT a hardcoded
    # 4 -- so all trainer GPUs go to dp_shard (dp_replicate stays 1) whenever the
    # rank allows; only spill to dp_replicate past that.
    lora_rank = _find_lora_rank(config.model)
    max_dp_shard = lora_rank or 4
    if trainer_dp_shard > max_dp_shard:
        dp_replicate = trainer_dp_shard // max_dp_shard
        trainer_dp_shard = max_dp_shard
        logger.info(
            f"Capping dp_shard at {max_dp_shard}, " f"dp_replicate={dp_replicate}"
        )

    # The requested dp_replicate x dp_shard split is used as-is, including when a
    # replicate group spans physical nodes.
    #
    # There used to be an auto-fold here that rewrote a cross-node replicate group
    # into dp_shard (dp_shard *= dp_replicate, dp_replicate = 1). It was removed on
    # 2026-07-31 after both of its premises were measured false; do not re-add it:
    #   1. CORRECTNESS. Cross-node dp_replicate looked broken on this Monarch/XPU
    #      (xccl/CXI) stack -- dp_shard4/rep2 exploded in bit_wise/logprob_diff and
    #      dp_shard1/rep2 x tp4 showed grad_norm 141-1352. Both were the CXI/oneCCL
    #      scale-out env, not the mesh: with that env and NO mesh change the same
    #      shapes train clean at grad_norm 0.058-0.17.
    #   2. PERF. The fold was then kept as the faster layout, but the unfolded runs
    #      are faster on identical allocations: dp_shard4 x rep2 7163 vs 6529 tok/s,
    #      rep2 x tp4 957 vs 900. dp_replicate all-reduces gradients once per step
    #      while dp_shard all-gathers parameters every layer, so folding cross-node
    #      DP onto dp_shard only adds fabric traffic.
    config.trainer.parallelism = replace(
        config.trainer.parallelism,
        data_parallel_shard_degree=trainer_dp_shard,
        data_parallel_replicate_degree=dp_replicate,
    )

    # Independent generator replicas are counted by num_generators. The
    # generator's own data_parallel_degree only supplies ranks to expert
    # parallelism, so one replica is tensor_parallel_degree GPUs.
    generator_total_gpus = num_generator_nodes * gpus_per_node
    per_generator_world_size = _compute_generator_world_size(
        config.generator.parallelism
    )
    config.num_generators = generator_total_gpus // per_generator_world_size

    logger.info(
        f"Mesh split: {num_trainer_nodes} trainer node(s) "
        f"({trainer_total_gpus} GPUs, dp_shard={trainer_dp_shard}, tp={tp}), "
        f"{num_generator_nodes} generator node(s) "
        f"({generator_total_gpus} GPUs, {config.num_generators} generator(s) of "
        f"{per_generator_world_size} GPU(s))"
    )

    rl_trainer: Controller = config.build()
    try:
        trainer_world_size = _compute_trainer_world_size(config.trainer.parallelism)
        trainer_mesh = _spawn_proc_mesh(
            trainer_host_mesh,
            trainer_world_size,
            gpus_per_node,
            bootstrap=_preimport_torch,
            role="trainer",
        )
        generator_meshes = _spawn_generator_meshes(
            generator_host_mesh,
            num_generators=config.num_generators,
            per_generator_world_size=per_generator_world_size,
            gpus_per_node=gpus_per_node,
        )
        await rl_trainer.setup_async(
            trainer_mesh=trainer_mesh,
            generator_meshes=generator_meshes,
            spawn_storage_volume_mesh=spawn_storage_volume_mesh_fn(
                trainer_host_mesh, trainer_world_size, gpus_per_node
            ),
        )
        await rl_trainer.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Interrupted; attempting graceful shutdown...")
    finally:
        await rl_trainer.close()


def main():
    parser = argparse.ArgumentParser(
        description="Multi-node GRPO launcher",
        allow_abbrev=False,
    )
    parser.add_argument("--num_nodes", type=int, required=True)
    parser.add_argument("--gpus_per_node", type=int, default=4)
    parser.add_argument(
        "--all_nodes",
        type=str,
        required=True,
        help="Comma-separated list of all hostnames (from PBS_NODEFILE)",
    )
    parser.add_argument(
        "--tensor_parallel_degree",
        type=int,
        help="Trainer TP degree; omit to keep the recipe's value.",
    )
    parser.add_argument(
        "--data_parallel_replicate_degree",
        type=int,
        help="Trainer dp_replicate degree; omit to keep the recipe's value.",
    )
    args, train_argv = parser.parse_known_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    rank = get_mpi_rank()
    hostname = socket.gethostname()

    # Strip PMI/PALS env vars so spawned child procs don't try to use
    # mpiexec's PMI wire protocol for torch.distributed rendezvous.
    for var in list(os.environ):
        if var.startswith(("PMI_", "PALS_")):
            del os.environ[var]

    # Force oneCCL to use OFI (libfabric) transport instead of MPI.
    # Without PMI vars, all spawned procs appear as MPI rank 0, causing
    # MPI_Group_incl to abort on duplicate ranks.
    os.environ.setdefault("CCL_ATL_TRANSPORT", "ofi")

    # Use Cray's libfabric (has CXI provider for native Slingshot RDMA).
    # Conda's libfabric only has tcp/psm2/psm3 -- no CXI support.
    cray_libfabric = _newest_cray_libfabric()
    if cray_libfabric:
        os.environ.setdefault("CCL_OFI_LIBRARY_PATH", cray_libfabric)
        os.environ.setdefault("FI_PROVIDER", "cxi")
        os.environ.setdefault("CCL_ATL_OFI_PROVIDER", "cxi")
    else:
        logger.warning(
            "No Cray libfabric under /opt/cray/libfabric. oneCCL will fall back "
            "to a provider without CXI, so cross-node transfers will be slow or "
            "fail outright."
        )

    all_nodes = list(dict.fromkeys(args.all_nodes.split(",")))

    # Worker address must use the same name form as all_nodes so that the
    # worker identity matches what attach_to_workers uses. Bind on 0.0.0.0
    # so spawned child procs can connect back via any interface.
    my_addr_name = hostname
    for node in all_nodes:
        if node.startswith(hostname):
            my_addr_name = node
            break

    logger.info(f"Rank {rank} on {hostname}, nodes={all_nodes}")

    worker_addr = f"tcp://{my_addr_name}:{MONARCH_PORT}@tcp://0.0.0.0:{MONARCH_PORT}"

    _configure_timeouts()

    if rank == 0:
        worker_thread = threading.Thread(
            target=run_worker, args=(worker_addr,), daemon=True
        )
        worker_thread.start()
        time.sleep(2)

        worker_addrs = [f"tcp://{node}:{MONARCH_PORT}" for node in all_nodes]
        try:
            asyncio.run(
                run_controller(
                    worker_addrs,
                    args.num_nodes,
                    args.gpus_per_node,
                    train_argv,
                    tensor_parallel_degree=args.tensor_parallel_degree,
                    data_parallel_replicate_degree=args.data_parallel_replicate_degree,
                )
            )
        except Exception:
            logger.exception("Controller crashed")
            sys.exit(1)
    else:
        run_worker(worker_addr)


if __name__ == "__main__":
    main()
