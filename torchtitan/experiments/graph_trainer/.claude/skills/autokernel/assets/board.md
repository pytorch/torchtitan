# Autokernel task board

Use board.lock for every claim and read-modify-write update. Replace this file
atomically. Keep detailed trial logs and profiles in the linked candidate folders.

## Run contract

- Run ID:
- Coordinator:
- Substitute-kernels cache_dir (confirmed by the user), if any:
- Problem manifest:
- Immutable references and source hashes:
- Target GPU, CUDA, CuTe DSL, and reference math/backend settings:
- Validation and benchmark commands:
- Workload frequency or critical-path evidence:
- Resource/time limits supplied by the user:

## Problems

States: queued, writing, validating, review, verifying, accepted, blocked.

| Problem ID / hypothesis | Occurrences / priority | Owner | State | Best accepted candidate + hash | Exactness coverage | Reference median us | Best median us / spread | Next experiment or blocker |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- |

## GPU leases

One active benchmarking/profiling owner per GPU. Record physical identity as well
as process-visible indexing. The coordinator confirms an owner has stopped before
releasing or reassigning its lease.

| GPU UUID / visible index | Owner | Problem / trial | Activity | Process or job ID | Last update |
| --- | --- | --- | --- | --- | --- |

## Handoffs and findings

Each handoff links source and environment hashes, exactness results, reproduction
commands, timing samples, and profiler artifacts. The cross-reviewer reports an
independent verdict to the coordinator and records it here.

| Time | Problem / trial | Writer -> reviewer | Evidence path | Finding / independent verdict | Coordinator action |
| --- | --- | --- | --- | --- | --- |

## Verifications

One newly spawned verifier per claimed win. For substitute-kernels problems, the
coordinator replaces `<l2>.py` only after a pass with a measurable
speedup, and records the replaced file's backup path, if any.

| Time | Problem / trial + hash | Claimed by | Verifier | Exactness / contract | Median us vs current | Verdict | Replaced <l2>.py (backup path) |
| --- | --- | --- | --- | --- | ---: | --- | --- |
