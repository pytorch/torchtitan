# Monarch actors

This package runs the standalone RL services on
[Monarch](https://github.com/meta-pytorch/monarch). Each adapter is a thin
Monarch `Actor` subclass of a service with no Monarch dependency of its own: it
declares the service's methods as `@concurrent_endpoint`s and supplies
Monarch-specific constructor arguments (rank, actor name, `Channel.open`).
Domain behavior stays in the service.

| Module | Purpose |
| --- | --- |
| `trainer.py` | `TrainerActor`, the endpoints of `rl/trainer/trainer.py`'s `Trainer`. |
| `generator.py` | `GeneratorActor`, the endpoints of `rl/generator/generator.py`'s `Generator`; supplies `Channel.open` for the data-parallel result fan-in. |
| `batcher.py` | `BatcherActor`, the endpoints of `rl/trainer/batcher.py`'s `Batcher`, in its own CPU process. |
| `rollout_worker.py` | `RolloutWorkerActor`, a rollout worker in the controller-host CPU pool. |
| `torch_elastic.py` | Sets the torch-elastic env vars (`RANK`, `MASTER_PORT`, ...) on a proc mesh before `init_process_group`; rank 0 keeps its rendezvous store port open so no other socket takes it. |

## Monarch outside this package

Some Monarch use still lives outside this package:

| File | What it uses Monarch for |
| --- | --- |
| `rl/train.py` | Provisioning: `HostMesh`, `ProcMesh`, `this_host`, `default_bootstrap_cmd`, `shutdown_context`, and `PerHostProvisioner`. |
| `rl/controller/controller.py` | Spawning the router and batcher processes (`this_host`), torch-elastic setup, spawning `TrainerActor`, `GeneratorActor` and `BatcherActor`, slicing the trainer by DP rank, stopping meshes, and `ProcMesh` type hints. |
| `rl/rollout/rollouter.py` | Spawning and stopping the rollout worker pool (`this_host`, `ProcMesh`, `RolloutWorkerActor`). |
| `rl/routing/inter_generator.py` | `InterGeneratorRouter` is itself a Monarch `Actor` (`concurrent_endpoint`, `current_size`). |
| `rl/trainer/rdma_microbatch.py` | `MonarchRDMAMicrobatch` on `monarch.rdma` (`RDMABuffer`, `RDMAAction`), which the batcher and the trainer use for `Batcher.Config.rdma_min_tensor_bytes`. |

The generator, its engine loop, dispatch and inference engines, and the
routing strategies do not import Monarch; the trainer and the batcher import it
only through `rl/trainer/rdma_microbatch.py`. The generator receives its
result channel as a plain `open_result_channel` callable.

The first three rows are spawning and teardown, which can move into this
package. The router can take the same split as the trainer and generator: a
plain class with the routing state and an actor adapter here. The RDMA
microbatch can move here behind an interface the batcher and the trainer use.
After these, no module outside this package imports Monarch.

What remains is the call convention: endpoint calls such as
`handle.endpoint.call()`, `.call_one()` and `.choose()`, and reading rank 0's
result with `.get(0)`, in the controller, the rollouter and the router. A second
actor framework (e.g. Ray) would add its own adapters next to this package
rather than abstract this convention.
