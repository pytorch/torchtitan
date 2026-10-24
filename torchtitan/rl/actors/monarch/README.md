# Monarch actors

This package runs the standalone RL services on
[Monarch](https://github.com/meta-pytorch/monarch). `MonarchActorSystem`
implements the framework-agnostic `ActorSystem` (`rl/actors/system.py`), which
the controller builds from `Controller.Config.actor_system` (default
`MonarchActorSystem.Config()`) and spawns every service through. A launcher that
allocates its own hosts builds `MonarchActorSystem` with `host_meshes` (and
optionally `generator_env`) and passes it to `Controller.setup_async(actor_system=...)`.
To get hosts from a cluster scheduler, set `MonarchActorSystem.Config.launcher`
(see [Launchers](#launchers)).

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
| `launcher/` | Scheduler launchers: `MonarchLauncher` and the shared layout planner (`base.py`), `SlurmLauncher` (`slurm.py`), and the submit command (`submit.py`). |

## Launchers

A launcher gets the job's hosts from a cluster scheduler. Set it on
`MonarchActorSystem.Config.launcher` in your recipe, then submit the recipe with
`python -m torchtitan.rl.actors.monarch.launcher.submit`, which takes the same
arguments as `torchtitan.rl.train`. Add `--dry-run` to print the planned nodes,
roles and settings without submitting. To support another scheduler, subclass
`MonarchLauncher` (`launcher/base.py`) in your own package.

**Layout.** Roles do not share nodes, so a role smaller than a node leaves the
rest of its node idle.

TODO: support packing roles onto shared nodes, e.g. with 8-GPU nodes and 4-GPU
roles, `trainer` and `generator_0` on node 0 and `generator_1` and `generator_2`
on node 1.

### SLURM

Before you submit, set up the environment, model and dataset per the
[Quick Start](../../README.md#quick-start), and make sure every node can access
them at the same paths (see the
[`SlurmLauncher` docstring for more details](./launcher/slurm.py)). Submit from
the repository root, in the shell where the venv is active and `PYTHONPATH` is
set.

From a login node, this runs a complete example on three nodes, one for the
trainer and one for each of two generators:

```bash
python -m torchtitan.rl.actors.monarch.launcher.submit \
  --module torchtitan_recipes.rl.alphabet_sort --config rl_grpo_qwen3_0_6b_flex_slurm
```

Site settings such as the account and partition go in `slurm_args`, in your own
recipe:

```python
def my_rl_slurm():
    config = rl_grpo_qwen3_0_6b_flex_slurm()
    config.actor_system.launcher.slurm_args += ["--account=my-account", "--partition=gpu"]
    return config
```

## Outside this package

No module outside this package imports Monarch or this package, except the
controller's default for `Controller.Config.actor_system` and recipes that set a
launcher, which import them lazily. The controller, the rollouter and `train.py` use
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
