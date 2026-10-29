# Experimental last-words integration

This WIP targets the current TorchTitan TrainingEngine. It ports a split
implementation originally exercised with TorchTitan 0.3.0; those results do not
validate this Main port. The design discussion is in [the RFC](rfc_last_words.md).

## Ownership

TorchFT owns the `torchft_last_words` service and SDK, native quorum/commit
decisions and peer healing. TorchTitan owns application completeness,
checkpoint/cursor/RNG bindings, publication, restore preflight, and its Linux
supervisor. Backend plugins own device and communicator recovery.

`torchtitan.last_words` contains `control`, `client`, `file_backend`, `validation`,
`checkpoint`, `schema`, `rng`, `training`, `job` and `external`. Service code is not copied
into TorchTitan. The FT experiment wires `LastWordsTrainingCoordinator` into its
TrainingEngine, optimizer, checkpoint and trainer adapters. Ordinary training
uses the existing path when the feature is disabled.

## Paired installation

The proposed SDK is not available merely by installing a released TorchFT wheel
with the same version number. Build the paired TorchFT PR and its native
extension against its required PyTorch, then install this branch:

```bash
python -m pip install -e ../torchft
python -m pip install -e .
python -c 'from torchft.last_words import LastWordsManager; from torchft_last_words.client import OwnerClient'
```

The upstream TorchFT Manager now uses native process-group reconfiguration.
Do not substitute an old native extension from earlier candidate experiments
and claim that the resulting environment validates the new upstream branch.

## Configuration and lifecycle

Use `FaultTolerantTrainer.Config` and set its `fault_tolerance` fields explicitly:
`enable=True`, `use_async_quorum=False`, at least two logical replica slots, and
`last_words.enable=True` with a unique launch `run_id`. Fixed last-words requires
`min_replica_size == group_size`; elastic last-words requires service coordination.
Full checkpointing and FT dataloader checkpoints must be enabled. Semi-sync
methods and optimizer CUDA graphs are unsupported in this combination.
Complete last-words recovery rejects `exclude_from_loading` entries for model,
optimizer, scheduler, counters, dataloader, EMA and native TorchFT progress.
Model-only warm starts are a separate workflow, not committed-state recovery.

The training coordinator begins the quorum before consuming the attempt,
requires a single commit authorization before persistent updates, and marks
completion only after the optimizer/scheduler/auxiliary update. Unknown commit,
partial restoration and an interrupted update prohibit emergency publication.
The checkpoint adapter records full state, logical cursors and rank-local RNG.
The Main RNG layout is not an implicit upgrade path for old 0.3.0 checkpoints.
Nonzero recovery candidates must include required-state schemas. Metadata checks
cover required tensors, serialized fields and complete nonoverlapping storage
chunks; restore also compares an independently derived live application schema
before changing model or cursor values. Candidate declarations alone cannot
relax the current application's requirements. These checks do not hash tensor
contents or prove that arbitrary serialized payloads can be deserialized.

Workers persist DCP contents and atomically publish candidate metadata. The
supervisor then validates the candidate and records the save outcome. Candidate
publication alone is not acceptance. Restart independently checks compatibility
and completeness before restoring live state. A bound cursor may lag model
progress; this permits data replay and does not promise an identical trajectory.

The independent service runs on loopback HTTP. Job supervision requires Linux
process groups, `/proc`, `fcntl`, and a shared POSIX checkpoint filesystem:

```bash
python -m torchft_last_words --help
torchtitan_last_words_job run --spec job.json --state-dir /private/job-control
torchtitan_last_words_external --help
```

Equivalent source entrypoints are `scripts/last_words_job.py` and
`scripts/last_words_external.py`. The JSON supervisor modes are `elastic`,
`last_words`, and `elastic_last_words`. Replica commands use `{run_id}` and
`{resume_path}` substitutions. External admission requires an explicit logical
slot, generation and owned PID/start token; an admission response does not itself
create a worker. Session response files contain private credentials.

Retention is explicit and previews its plan by default:

```bash
torchtitan_last_words_job cleanup --checkpoint-root /path/to/checkpoints --num-replicas 2 --keep 2
```

`--apply` executes the plan. Retained checkpoints protect their referenced
cursor branches; unusable candidates are preserved for diagnosis. Publication,
independent storage validation and all owned process exits are separate checks.
Periodic checkpoints, terminal candidates and elastic `events/event-*` candidates
use the same explicit read-only recovery preflight. Cleanup does not instantiate
an incomplete training manager and is separate from base `keep_latest_k` policy.

## Tests and WIP gates

With a compatible PyTorch and the paired TorchFT installed, run:

```bash
python -m pytest tests/unit_tests/cpu/experiments/torchft -v
python tests/integration_tests/last_words_rng_checkpoint.py --help
python tests/integration_tests/last_words_service_checkpoint.py --help
pre-commit run --all-files
```

The distributed CPU integration programs exercise real DCP and service
publication, not accelerator recovery. The paired manual workflow takes an exact
TorchFT commit and fails if the added API is unavailable. It must be dispatched
against the pushed revision before its result can be claimed.

Before removing WIP: build both wheels from their actual PR heads; pass CPU
transaction/DCP and disabled-path regressions; run end-to-end failure/rejoin and
shutdown cases; and compare full state, numerical results and actual inputs.
The current validation plan contains three paired normal/numerical modes plus
cross-replica periodic recovery, native shrink/rejoin and emergency cold recovery.
Performance is deferred. Historical 0.3.0 runs cannot validate this Main revision.
Model-only Inductor training does not establish loss compilation or bitwise
equivalence. Nondeterministic functional recovery reports measured differences.
