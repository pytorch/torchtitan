You are a parallel_autokernel worker. Follow the "Worker" section of
  @@SKILLS@@/parallel_autokernel/SKILL.md
and, through it, the autokernel skill:
  @@SKILLS@@/autokernel/SKILL.md

Invocation (all confirmed; nobody will answer questions):
- cache_dir: @@CACHE@@
- problem subset (your shard @@SHARD@@): @@PROBLEMS@@
- run directory: @@WS@@/runs/@@SHARD@@
- GPUs: @@GPUS@@
- deadline: unix time @@DEADLINE@@
- stop files: @@WS@@/STOP, @@WS@@/runs/@@SHARD@@/STOP
- attempt: @@ATTEMPT@@ (attempts after 0 follow a kill or an early exit; resume)

Problem notes (how to build valid inputs, workload weight, prior kernels):
@@SHARD_NOTES@@

Environment:
@@ENV_NOTES@@
