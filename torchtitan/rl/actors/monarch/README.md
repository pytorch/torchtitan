# Monarch actors

This package runs the standalone RL services on
[Monarch](https://github.com/meta-pytorch/monarch). `MonarchActorSystem`
implements the framework-agnostic `ActorSystem` (`rl/actors/system.py`), which
the controller builds from `Controller.Config.actor_system` (default
`MonarchActorSystem.Config()`) and spawns every service through. A launcher that
allocates its own hosts subclasses `Controller` and overrides
`_build_actor_system` to build `MonarchActorSystem` with `host_meshes`. To get
hosts from a cluster scheduler, use a launch recipe (see
[Launchers](#launchers)).

Each adapter is a thin `TitanActor` (a Monarch `Actor`) subclass of a service with no Monarch
dependency of its own: it declares the service's methods as
`@concurrent_endpoint`s and supplies Monarch-specific constructor arguments
(rank, actor name, `Channel.open`). Domain behavior stays in the service.

| Module | Purpose |
| --- | --- |
| `system.py` | `MonarchActorSystem`: provisions the proc meshes and torch-elastic setup, places the generators, spawns each service's actors, slices the trainer by DP rank, initializes TorchStore on the trainer mesh and every actor's client (or the `ModelStateDictTransport`), and stops the meshes. |
| `actor.py` | `TitanActor`, the base of every actor here: drops a message it sends to a stopped or failed actor instead of failing (Monarch's default). |
| `trainer.py` | `TrainerActor`, the endpoints of `rl/trainer/trainer.py`'s `Trainer`. |
| `generator.py` | `GeneratorActor`, the endpoints of `rl/generator/generator.py`'s `Generator`; supplies `Channel.open` for the data-parallel result fan-in. |
| `router.py` | `InterGeneratorRouterActor`, the endpoints of `rl/routing/inter_generator.py`'s `InterGeneratorRouter`; enforces that the router is a singleton, spawns and stops the generators' proc meshes, and reports their failures from `__supervise__` (see [Generator fault tolerance](#generator-fault-tolerance)). |
| `batcher.py` | `BatcherActor`, the endpoints of `rl/trainer/batcher.py`'s `Batcher`, in its own CPU process; supplies `MonarchRDMAMicrobatch.register`. |
| `rdma_microbatch.py` | `MonarchRDMAMicrobatch`, the `RDMAMicrobatch` (`rl/trainer/rdma_microbatch.py`) on Monarch RDMA buffers: the batcher keeps a microbatch's large tensors and each trainer rank reads its column. |
| `rollout_worker.py` | `RolloutWorkerActor`, a rollout worker in the controller-host CPU pool. |
| `provisioning.py` | Proc meshes: the trainer's GPU mesh and each generator's `RolePlacement` (`spawn_proc_mesh`, `PerHostProvisioner`), CPU procs on the controller host (`spawn_cpu_procs`), and the Monarch client's `shutdown`. |
| `torch_elastic.py` | Sets the torch-elastic env vars (`RANK`, `MASTER_PORT`, ...) on a proc mesh before `init_process_group`; rank 0 keeps its rendezvous store port open so no other socket takes it. |
| `launcher/` | Scheduler launchers: `MonarchLauncher`, `MonarchControllerWithLauncher` and the shared layout planner (`base.py`), `SlurmLauncher` (`slurm.py`), and the submit command (`submit.py`). |

## Generator fault tolerance

Monarch reports a mesh's failures only to the actor that spawned it. The
generator router spawns every generator's proc mesh on that generator's
`RolePlacement` (hosts and GPUs), so it hears of their failures. With
`InterGeneratorRouter.Config.fault_tolerance` unset, it passes them on to the
controller, which exits. With it set (and `num_generators` >= 2), the router
restarts a failed generator in place while training goes on with the others:

```python
from torchtitan.rl.routing.inter_generator import GeneratorFaultToleranceConfig

config.generator_router.fault_tolerance = GeneratorFaultToleranceConfig()
```

```
           failure reported                stop its procs, spawn a new incarnation on
 healthy -------------------> dead ------> the same GPUs, connect it to the model state
    ^                                      dict source, start its engine loop
    |                                                     |
    +------ its first model state dict pull lands ---- joining
```

A generator fails as a whole when any of its ranks does. The router learns of it
from whichever signal comes first:

| Failure | Signal |
| --- | --- |
| A rank's process dies (OOM kill, crash) | `wait_engine_loop` on that rank fails at once; Monarch's `__supervise__` after `HYPERACTOR_MESH_DATA_SUPERVISION_TIMEOUT` (30s by default) |
| A rank's actor fails, its process alive | `__supervise__` |
| The engine loop crashes | `wait_engine_loop` |
| A model state dict pull fails | `pull_model_state_dict` |
| A call finds the generator gone (`SupervisionError`, `GeneratorUnavailableError`) | the router's own calls to it; a `generate` caller, through `release_reservation(error=...)` |

A `generate` call that fails because its generator is gone is retried on other
generators, in up to `num_generators` attempts. A restarted generator serves only
after the next model state dict pull lands, so it never serves stale weights.
The trainer never waits for a restart; a pull waits for a generator that fails
during it only until that generator is stopped. The run fails when a failure
leaves fewer than `min_healthy_generators` healthy generators, or when a
generator fails, or fails to restart, after `max_restarts_per_generator`
restarts.

Not covered: a lost host (Slurm ends the whole job), a generator failure before
the engine loops start (the run fails), a generator that hangs without failing,
and failures of the trainer, the router or the controller. A
request that crashes every engine it reaches is retried on each generator, so
it can fail several generators at once.

## Launchers

A launcher gets the job's hosts from a cluster scheduler. A launch recipe returns
a `MonarchControllerWithLauncher.Config`: an RL recipe plus the launcher that
runs it on one cluster, built with
`MonarchControllerWithLauncher.Config.from_controller_config(config,
launcher=...)`. `Controller.Config` itself has no launcher settings. Submit the
launch recipe with `python -m torchtitan.rl.actors.monarch.launcher.submit`,
which takes the same arguments as `torchtitan.rl.train`. The job runs
`python -m torchtitan.rl.train` with them, which builds a
`MonarchControllerWithLauncher`: it attaches to the job's hosts and passes them
to `MonarchActorSystem` as `host_meshes`. Add `--dry-run` to print the planned
nodes, roles and settings without submitting.
To support another scheduler, subclass `MonarchLauncher` (`launcher/base.py`) in
your own package.

**Layout.** The launcher places the trainer first, then the generators, filling
each node before the next:

- A role smaller than a node goes on the node being filled if it fits, else on
  the next node. It is never split across two nodes.
- A role using a node or more gets whole nodes to itself. Its world size must be
  a multiple of the node's GPU count.

Examples with 8-GPU nodes ("generators" is the number of generators x GPUs per
generator):

| Trainer | Generators | Layout | Idle |
| --- | --- | --- | --- |
| 8 GPUs | 8 x 1 GPU | node 0: trainer; node 1: generator_0..7 | 0 |
| 4 GPUs | 3 x 4 GPUs | node 0: trainer, generator_0; node 1: generator_1, generator_2 | 0 |
| 16 GPUs | 2 x 8 GPUs | nodes 0-1: trainer; node 2: generator_0; node 3: generator_1 | 0 |
| 6 GPUs | 2 x 4 GPUs | node 0: trainer; node 1: generator_0, generator_1 | 2 GPUs on node 0 |
| 4 GPUs | 1 x 8 GPUs | node 0: trainer; node 1: generator | 4 GPUs on node 0 |
| 12 GPUs | 1 x 4 GPUs | `ValueError`: 12 is not a multiple of 8 | |

### SLURM

Before you submit, set up the environment, model and dataset per the
[Quick Start](../../README.md#quick-start), and make sure every node can access
them at the same paths (see the
[`SlurmLauncher` docstring for more details](./launcher/slurm.py)). Submit from
the repository root, in the shell where the venv is active and `PYTHONPATH` is
set. JIT caches (Triton, vLLM, FlashInfer) default to node-local
`/tmp/torchtitan-<uid>`; set them in `launcher.env` or the submit shell to
override.

From a login node, this runs a complete example on two 4-GPU nodes: the trainer
on node 0, and four 1-GPU generators sharing node 1 (set `gpus_per_node` in the
recipe to your nodes' GPU count):

```bash
python -m torchtitan.rl.actors.monarch.launcher.submit \
  --module torchtitan_recipes.rl.alphabet_sort --config rl_grpo_qwen3_0_6b_flex_slurm
```

Site settings such as the account and partition go in `slurm_args`, in your own
launch recipe:

```python
def my_rl_slurm():
    config = rl_grpo_qwen3_0_6b_flex_slurm()
    config.launcher.slurm_args += ["--account=my-account", "--partition=gpu"]
    return config
```

The launcher's `env` values are part of the config, which the controller may log
and save (`print_config`, `save_config_file`), so don't put secrets there; set them
in the submit shell, which the job inherits.

## Outside this package

No module outside this package imports Monarch or this package, except the
controller's default for `Controller.Config.actor_system` and launch recipes,
which import them lazily. The controller, the rollouter and `train.py` use
the abstract `ActorSystem`.

What remains outside is the call convention on the handles an actor system
returns (see `ActorSystem`): `.call()` with `.get(rank)` on its result,
`.call_one()`, `.choose()`, `.stream()` and `.broadcast()`. The trainer comes
back as a `TrainerHandle` with all-ranks, rank-0 and per-DP-rank handles, and
each generator as a `ServiceHandle` the router gets, so callers never slice
meshes themselves. Generator handles come only from the router: its `reserve`
returns the one a generate call goes to, and every other generator call goes
through the router's endpoints. These calls appear in the controller (and its
`GenerateFn`), the rollouter (`.choose()` on the worker pool) and the router (on
generator handles). The batcher and the trainer see RDMA microbatches only
through the `RDMAMicrobatch` interface.

Another actor framework (e.g. Ray) would add its own `ActorSystem`
implementation and adapters, wrapping its handles to provide this call
convention, and be selected with `Controller.Config.actor_system`.
