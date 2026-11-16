---
name: parallel_autokernel
description: Run many autokernel runs concurrently on disjoint problem subsets of one substitute-kernels cache_dir, as independent, resumable workers (for example one preemptible cluster job per pattern), then monitor them and collect the promoted kernels. Use when asked to tune a whole kernel cache in parallel, on a cluster, or across many GPUs or jobs.
---

# Parallel autokernel

The [autokernel](../autokernel/SKILL.md) skill tunes one problem set with one
coordinator. This skill runs many such coordinators at once, each on a disjoint
subset of the problems (`pat_<l1>/<l2>.py` files) of one `cache_dir`, as workers
that can be killed and restarted at any time. Everything about how kernels are written,
verified and promoted stays in autokernel; this skill only adds the orchestration.

There are two roles. The **launcher** is the interactive session that plans,
starts, monitors and collects. A **worker** is one unattended autokernel
coordinator for one shard, kept running by a wrapper (see Running workers).

## Workspace

One shared directory, visible to every worker under the same path:

```
<workspace>/
  cache/              # the cache_dir all workers tune: a working copy, not the original
                      #   (several workspaces may instead share one cache)
  runs/<shard>/       # one autokernel run directory per shard: board, candidates,
                      #   backups, promotions.md, summary.md, transcripts, DONE / STOP;
                      #   written by the launcher: PROBLEMS, NOTES (optional, see Launcher)
  logs/               # wrapper logs
  STOP                # if present, every worker wraps up
```

Tune a copy of the cache, not the original (for example a committed
`kernel_cache/`): that keeps the original clean, and the result can be diffed and
copied back once reviewed.

## Launcher

1. **Get the inputs.** Ask for the source `cache_dir` and the workspace path
   before anything else, as autokernel requires, and copy the cache into
   `<workspace>/cache/`.
2. **Plan shards.** A shard is a set of problems (`pat_<l1>/<l2>.py` files); all
   problems of one pattern directory per shard is the default. Shards must be
   disjoint: that is what makes concurrent promotion safe without cross-worker
   locking, since each `<l2>.py` file has exactly one writer. Several shards may
   share a pattern directory (for example problems of different models in one
   shared cache); each writes only its own files. Exclude problems whose contract
   cannot be gated bytewise (for example outputs with uninitialized bytes) and record
   why. For a shard that is not a whole pattern directory, list its problem IDs
   (`pat_<l1>/<l2>`, one per line) in `runs/<shard>/PROBLEMS`. Facts a worker needs
   about its problems go into `runs/<shard>/NOTES`: how to build valid inputs for
   them (for example index-valued inputs or symbolic sizes), how often they run per
   training step, and where earlier kernels for them are.
3. **Check the environment** before scaling out: the worker's Python environment
   imports every kernel's dependencies, an existing-kernel exactness check passes
   on a worker machine, and the agent CLI runs headless on a worker machine with API
   access. Then run one short pilot worker and read its board and transcript
   before starting the rest.
4. **Start** one worker per shard, at most one live worker per shard at any time
   (two workers on one shard would both promote into the same files).
5. **Monitor**: worker state, promotions per shard, board freshness. A board that
   stops changing while its worker is alive means a stuck worker.
6. **Collect.** Each promotion is in `runs/<shard>/promotions.md` with old and new
   sha256 and a backup of the replaced file. When done, re-gate every kernel in
   `cache/` in one place (autokernel's exactness gate on every `<l2>.py` whose `run`
   is not `reference`), review, and copy back to the original cache.

## Worker

A worker is started with this prompt, filled in by its wrapper:
[scripts/worker_prompt.md](scripts/worker_prompt.md). It invokes autokernel with
`cache_dir` = `<workspace>/cache` (or the shared cache), problem subset = the shard's
problem IDs, run directory = `runs/<shard>`, the GPUs it was given, and these
rules on top of autokernel:

- **Ownership.** Write and promote only the shard's `<l2>.py` files; other
  workers own the rest of `cache_dir`, including other files in the shard's
  pattern directories, and may replace them at any time (atomically).
- **Unattended.** Nobody answers questions; decide, record on the board, continue.
- **Killable.** The worker may be killed at any moment and restarted later, likely
  on another machine. Keep the board current after every meaningful step and never
  leave temporary files in `cache_dir` (autokernel's promotion is already atomic).
- **Resume.** If the run directory already holds a board, mark the leases and
  in-flight trials of earlier attempts abandoned (those agents are gone), re-read
  the shard's current `<l2>.py` files and their sha256 (an incumbent promoted
  by an earlier attempt is the one to beat), and continue from the recorded next
  experiments without redoing finished work.
- **Do not exit early.** In a headless session, ending your turn ends the session,
  and that kills every agent you spawned, mid-trial. Never end your turn while
  agents are running. Either launch agents in the foreground (several in one
  message run in parallel, and you wait for all of them), or launch them in the
  background and wait with a Bash poll loop (`sleep 300`, then check their outputs,
  the board, the stop files and the deadline) until they finish. Scheduled wakeups
  do not keep a headless session alive. The wrapper relaunches you after an exit,
  but in-flight agent work is lost.
- **Timing.** Never compare against numbers recorded by an earlier attempt; time
  candidate and incumbent interleaved in the current process on the current GPU.
- **Stopping.** Between iterations, check `<workspace>/STOP`, `runs/<shard>/STOP`
  and the deadline the wrapper passes in. On any of them, or when the shard is
  finished (every problem at its roofline with measurements, or with its remaining
  experiments recorded as not worth it with evidence), wrap up as autokernel
  describes (board, `promotions.md`, `summary.md`). Only when the shard is
  finished, also create `runs/<shard>/DONE`, so the wrapper stops relaunching.

## Running workers

Each worker runs under a wrapper (a scheduler job, or a shell loop per GPU on one
host). Whatever runs it must provide:

- **One live worker per shard.** Use the scheduler's unique-job mechanism or a lock;
  starting a shard twice must not produce two workers.
- **The prompt.** Render [scripts/worker_prompt.md](scripts/worker_prompt.md) by
  replacing `@@SKILLS@@` (the skills directory), `@@WS@@`, `@@SHARD@@`, `@@CACHE@@`,
  `@@PROBLEMS@@` (the contents of `runs/<shard>/PROBLEMS`, or the shard's pattern
  directory), `@@SHARD_NOTES@@` (`runs/<shard>/NOTES`), `@@GPUS@@`, `@@DEADLINE@@`
  (unix time), `@@ATTEMPT@@` (0 for the first start, counting up after each kill or
  restart) and `@@ENV_NOTES@@` (facts about the environment: missing tools,
  unavailable inputs).
- **A deadline with margin.** Pass a deadline some minutes before the wrapper's own
  end, so the worker can wrap up before it is killed.
- **Relaunch until a stop condition.** Run the agent CLI headless in
  `runs/<shard>`, and relaunch it after every exit (it resumes from its board) until
  `STOP`, `DONE` or the deadline. Give up after a few quick exits in a row, which
  mean a broken environment rather than an interruption.
- **Restart after preemption.** A killed worker is restarted later with the next
  attempt number; shards with no live worker and no `DONE` or `STOP` are started again.
- **Private caches.** Point compiler caches (Triton, Inductor, CUDA) and `TMPDIR`
  at per-worker local scratch, and set `PYTHONDONTWRITEBYTECODE=1`, so concurrent
  workers do not share them and nothing lands in `cache/`.
- **Shared storage.** The workspace must be on a filesystem every worker mounts at
  the same path. `os.replace` within one directory is atomic, also on NFS, so
  promotions are safe; disjoint shards avoid needing cross-machine locks.
