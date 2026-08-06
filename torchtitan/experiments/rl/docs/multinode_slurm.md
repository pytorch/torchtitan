# Multi-node RL on SLURM

This document covers running the RL loop across more than one node with the
`slurm_launcher` entry point: the environment variables it reads, which process
the controller runs in, and where the logs land. For the minimal invocation, see
the "Multi-node on SLURM" section of `torchtitan/experiments/rl/README.md`.

## Submitting a job

For configurations whose total GPU footprint exceeds a single node (e.g.
`rl_grpo_qwen3_14b`: trainer TP=8 + generator TP=8 = 16 GPUs), invoke the
`slurm_launcher` entry point and let it submit one sbatch covering the trainer
plus one mesh per generator on disjoint nodes:

```bash
RL_SLURM_PARTITION=h100 \
RL_SLURM_GPUS_PER_NODE=8 \
RL_SLURM_TIME=02:00:00 \
RL_SLURM_QOS=h100_dev \
RL_SLURM_ACCOUNT=pytorch \
python -m torchtitan.experiments.rl.slurm_launcher \
    --module alphabet_sort --config rl_grpo_qwen3_14b
```

Every world size must be divisible by `RL_SLURM_GPUS_PER_NODE`. `RL_SLURM_QOS`
and `RL_SLURM_ACCOUNT` are optional, passed through as `#SBATCH --qos=...` /
`--account=...`; substitute values for your cluster.

Add `RL_SLURM_BATCH=1` to detach: the login-node process then exits as soon as
the sbatch is in, and the controller runs inside the allocation instead. See
[Where the controller runs](#where-the-controller-runs) for what moves where.

For single-node SLURM runs, invoke `train` directly inside a standard
allocation (`salloc` or `sbatch --wrap "python -m ..."`); `this_host()`
partitions GPUs between trainer and generator with no launcher involved.

## Where the controller runs

The allocation only ever holds *workers*: one Monarch worker process per node,
with nodes handed out in allocation order to `trainer` first, then
`generator_0`, `generator_1`, ... The *controller* -- the process that parses the
config, builds the actor meshes on those workers, and runs the training loop --
is a separate process, and `RL_SLURM_BATCH` picks which side of the allocation it
lives on.

| | controller process | survives logout | run ends when |
| --- | --- | --- | --- |
| external (default) | your login-node shell | no | the controller exits, or the allocation hits its time limit |
| `RL_SLURM_BATCH=1` | inside the allocation, on its first node | yes | the controller exits (the runner then frees the allocation) |

**External controller (default).** The generated sbatch body is only
`srun <python> -c <worker bootstrap>`. The controller is the login-node process
you typed the command into: it submits, waits for the allocation to start,
attaches to the workers over TCP, and drives training in the foreground. Kill it
or drop the ssh session and the run goes with it. Logs stream to your terminal,
which makes this the mode to iterate in.

**Batch (`RL_SLURM_BATCH=1`).** The login-node process does nothing but submit.
The sbatch body becomes `python -m monarch._src.job._slurm_batch <the same
launcher command>`, which SLURM runs on the allocation's first node -- the same
node the `trainer` mesh's first worker lands on. No extra node is requested for
the controller, so it shares that node's CPU and host memory with a worker. That
in-allocation runner:

1. seeds one worker per node with `srun --ntasks-per-node=1`;
2. re-executes your launcher command with `MONARCH_BATCH_JOB=1`, which tells
   `slurm_launcher` to skip resubmission and reconnect to *this* allocation
   through the `BatchJob` that the submit step cached in
   `./.monarch/job_state.pkl`;
3. terminates the workers in a `finally`, so the allocation is released with the
   controller's exit status instead of lingering to its time limit.

Because step 2 is a literal re-execution, the in-allocation controller inherits
the argv, the environment (sbatch propagates the submit environment), and the
working directory of your login-node command. Two consequences: relative paths
such as `--hf_assets_path` must resolve from the directory you submitted from,
and `.monarch/job_state.pkl` is resolved relative to that same directory on both
sides -- so two concurrent batch submissions from one directory clobber each
other's cached job. Submit them from separate directories.

Worker and controller output both land in the job's stdout/stderr,
`slurm_<jobid>_torchtitan_rl_<pid>.{out,err}`, written to the directory you
submitted from (Monarch's `log_dir` defaults to the submitting process's cwd).
Follow a detached run with `tail -f` on the `.out`.
