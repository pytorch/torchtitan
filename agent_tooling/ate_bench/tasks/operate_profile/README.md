# ATE-Bench — Operate-and-Profile tasks (4 tasks)

These tasks drive a real training workflow end-to-end. They require GPUs, a
pre-staged MoE checkpoint, pre-tokenized data (TorchTitan uses C4; paper used
DCLM), vLLM + lm-evaluation-harness, and Nsight Systems. The harness is **wired
up**: prompts (`op1..op4`), runner (`runner/run_operate.py`), and programmatic
checks (`runner/checks/op*.py`) — but passing them needs the GPU substrate above.
Correctness is checked on the *artifact the agent produces*, not the path taken —
a mix of programmatic checks and human inspection.

Fixed evaluation config (shared with new-feature tasks): parallelism mesh
`PP=4, EP=2, DP=1`, sequence length `2048`, global batch size `1024`, precision
`BF16`, on 8 GPUs.

## Tasks (Appendix B.2.1)

### OP1 — Getting Started
Set up the Python environment for the framework and run the provided 5-step smoke
training script. Success means the script reaches step 5 with a finite loss. The
agent must install all dependencies the script needs so that running `bash
train.sh` as-is succeeds; `train.sh` itself documents best practices for training
MoE models and is read-only. Pre-tokenized DCLM and the converted base-model
checkpoint are pre-staged.
- **Check:** after the agent finishes installing deps, the harness re-runs the
  (read-only) training script and parses the log; accepted only if step 5 prints
  a finite loss value.

### OP2 — Train and Evaluate
Drive the full setup → train → export → evaluate pipeline for the base model. The
agent trains from random initialization for 25 steps, exports the resulting
checkpoint to HuggingFace format, and runs lm-evaluation-harness HellaSwag
(zero-shot) on it via vLLM. Tests pipeline correctness, not model quality:
HellaSwag is expected to be near-random after 25 steps from random init.
Initialization must be random; everything outside the fixed mesh and step count
(LR, optimizer, scheduler, data preprocessing) is left to the agent.
- **Check:** a read-only `evaluate.sh` consumes the agent's pipeline output, loads
  the export into vLLM, runs HellaSwag (zero-shot). Satisfied when `evaluate.sh`
  completes and writes a finite score (no quality threshold).

### OP3 — Collect Routing Trace
Instrument training to dump the per-token MoE routing trace for the first 8M
training tokens. For each MoE layer, the top-k expert IDs and their gating weights
must be captured for every token in the global batch. Training resumes from the
released HuggingFace checkpoint (router is in its trained, load-balanced regime;
no warmup required). Output schema: one `step-<step_id:08d>.npz` file per step
under `workspace/<framework>/routing-traces/`, each containing the expert-ID and
gating-weight arrays.
- **Check (programmatic):** four `step-<step_id:08d>.npz` files present; each
  carries expert-ID and gating-weight arrays of the correct shape for every MoE
  layer over every token in the global batch; expert-ID values within
  `[0, num_experts)`; gating weights non-negative and sum to 1 over the top-k
  selected experts for every token.

### OP4 — Report Heavy Kernels
Profile a 7-step training run with Nsight Systems and identify the top 3 most
expensive CUDA kernels by total GPU time, aggregated across all 8 ranks. Training
resumes from the released HuggingFace checkpoint; profile only step 7 (steps 1–6
are warmup for cudagraph capture, NCCL handshake, and allocator priming). Output:
a single CSV at `workspace/<framework>/heavy-kernels/top-kernels.csv` with header
`kernel_name,total_time_ms,instances,mean_time_ms` and exactly three rows sorted
by `total_time_ms` descending, plus the raw `profile.nsys-rep`.
- **Check:** CSV validated programmatically against the schema (header, exactly
  three rows sorted by total time descending). Kernel names validated by a human
  reader who opens `profile.nsys-rep` in the Nsight Systems GUI (CUDA GPU Kernel
  Summary, Stats System view) and confirms the top three match.
