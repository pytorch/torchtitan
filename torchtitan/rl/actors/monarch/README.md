# Monarch actors

This package runs the standalone RL services on
[Monarch](https://github.com/meta-pytorch/monarch). `MonarchActorSystem`
implements the framework-agnostic `ActorSystem` (`rl/actors/system.py`), which
the controller builds from `Controller.Config.actor_system` (default
`MonarchActorSystem.Config()`) and spawns every service through. A launcher that
allocates its own hosts builds `MonarchActorSystem` with `host_meshes` (and
optionally `generator_env`) and passes it to `Controller.setup_async(actor_system=...)`.

Each adapter is a thin Monarch `Actor` subclass of a service with no Monarch
dependency of its own: it declares the service's methods as
`@concurrent_endpoint`s and supplies Monarch-specific constructor arguments
(rank, actor name, `Channel.open`). Domain behavior stays in the service.

| Module | Purpose |
| --- | --- |
| `system.py` | `MonarchActorSystem`: provisions the proc meshes and torch-elastic setup, spawns each service's actors on them, slices the trainer by DP rank, initializes TorchStore on the trainer mesh and every actor's client, and stops the meshes. |
| `trainer.py` | `TrainerActor`, the endpoints of `rl/trainer/trainer.py`'s `Trainer`. |
| `generator.py` | `GeneratorActor`, the endpoints of `rl/generator/generator.py`'s `Generator`; supplies `Channel.open` for the data-parallel result fan-in. |
| `router.py` | `InterGeneratorRouterActor`, the endpoints of `rl/routing/inter_generator.py`'s `InterGeneratorRouter`; enforces that the router is a singleton. |
| `batcher.py` | `BatcherActor`, the endpoints of `rl/trainer/batcher.py`'s `Batcher`, in its own CPU process; supplies `MonarchRDMAMicrobatch.register`. |
| `rdma_microbatch.py` | `MonarchRDMAMicrobatch`, the `RDMAMicrobatch` (`rl/trainer/rdma_microbatch.py`) on Monarch RDMA buffers: the batcher keeps a microbatch's large tensors and each trainer rank reads its column. |
| `rollout_worker.py` | `RolloutWorkerActor`, a rollout worker in the controller-host CPU pool. |
| `provisioning.py` | Proc meshes: the trainer and generator GPU meshes (`spawn_proc_mesh`, `PerHostProvisioner`), CPU procs on the controller host (`spawn_cpu_procs`), and the Monarch client's `shutdown`. |
| `torch_elastic.py` | Sets the torch-elastic env vars (`RANK`, `MASTER_PORT`, ...) on a proc mesh before `init_process_group`; rank 0 keeps its rendezvous store port open so no other socket takes it. |

## Outside this package

No module outside this package imports Monarch or this package, except the
controller's default for `Controller.Config.actor_system`, which imports
`MonarchActorSystem` lazily. The controller, the rollouter and `train.py` use
the abstract `ActorSystem`.

What remains outside is the call convention on the handles an actor system
returns (see `ActorSystem`): `.call()` with `.get(rank)` on its result,
`.call_one()`, `.choose()`, `.stream()` and `.broadcast()`. The trainer comes
back as a `TrainerHandle` with all-ranks, rank-0 and per-DP-rank handles, and
each generator as a `ServiceHandle` the router gets, so callers never slice
meshes themselves. The controller reaches the generators only through the
router. These calls appear in the controller, the rollouter (`.choose()` on the
worker pool) and the router (on generator handles). The batcher and the trainer
see RDMA microbatches only through the `RDMAMicrobatch` interface.

Another actor framework (e.g. Ray) would add its own `ActorSystem`
implementation and adapters, wrapping its handles to provide this call
convention, and be selected with `Controller.Config.actor_system`.
