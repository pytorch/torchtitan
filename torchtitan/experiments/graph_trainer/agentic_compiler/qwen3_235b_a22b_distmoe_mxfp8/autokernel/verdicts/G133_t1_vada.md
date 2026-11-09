# G133 eve t1 verification (vada): PASS under the bucket contract

- Problem: pat_4a7hkgorlp4plsn7ryjmy3ozp5ylhkn5prme2yhp46me5vak2l6 / ag3ngj62uhiomowuwrayumsbymx6vvx3eju45bthx67fcxqiva44
  (FSDP all-gather bucket prep: 3 fp32->bf16 `_to_copy`, 3 zero-width pads, `bucketing._pre_bucket_all_gather`, slice)
- Candidate: candidates/G133/eve/t1/kernel.py, copied to `cand.py`;
  sha256 22c64141a21cf191ef74d5a7567de2bec9795849a59d05e3eda0a9801331f116 (matches the handoff)
- Reference: `reference.py` here, sha256 cf5e97c241edaf28... (matches cache_dir and the board)
- No current `<kernel_name>` file in the pattern dir: compared against reference.py only.
- GPU 0 only. Correctness, sanitizer and graph runs used no lock (revised brief, 16:58); timing held gpu0.lock
  only around the timed loop. Scripts are in `scripts/`, logs in `logs/` and `sanitizer/`.

## Contract caveat: resolved independently, writer's claim confirmed

- The op source (`logs/op_source.txt`, torch/_inductor/fx_passes/bucketing.py lines 1252-1287):
  `new_ag_out = torch.empty(ag_input_numel * group_size)`, `new_ag_in = new_ag_out.narrow(0, ag_input_numel * rank,
  ag_input_numel)`, then `_foreach_copy_` per group into split views of `new_ag_in`. With rank=0 and group_size=256,
  only out0[:282658] is written. The other 72077790 elements (144155580 bytes) stay as allocated. out1 =
  `aten.slice(out0, 0, 0, 282658)`, a view at offset 0. `_ALL_DTYPES[1]` is torch.bfloat16 on this build
  (`logs/contract.log`).
- Official harness on a verbatim copy of reference.py (`refcopy_as_candidate.py`): every case is
  `MISMATCH out0: 144155580 bytes differ`. That is exactly (72360448 - 282658) x 2, the unwritten tail, and the copy
  fails the same way the candidate does (`logs/harness_official.log`). The official gate cannot be passed on this
  problem by anything. I used the coordinator's bucket contract instead.
- In every baited run of my own gate, the reference's tail equals the bait byte for byte under 4 patterns. So the
  reference writes nothing outside the slot.

## Checks

| # | Check | Command | Result |
|---|---|---|---|
| 1 | sha256, signature, standalone import | `sha256sum`; `python -I scripts/contract.py` (fresh process, from /tmp) | match; `def run(in0, in1, in2, in3, in4, in5, in6)` == spec; imports and runs standalone; out0 (72360448,) (1,) bf16 cuda:0 offset 0, out1 (282658,) (1,) offset 0 |
| 2a | Official harness | `python harness.py <cache spec> cand.py --seeds 0 1 2 3 4 5 6 7 --real fusion_G133.pt --time` | seeds 0-7 + real: `out0: 144155580 bytes differ` only (the tail; out1 and alias structure are not flagged; no rerun mismatch). The reference copy gives the identical result. FAIL by construction (see caveat) |
| 2b | Bucket-contract gate, every byte | `python scripts/gate.py` (`logs/gate_full.log`) | ALL PASS. Before each reference run and each candidate run, a buffer of out0's size is baited (0xFF, 0x5A, 0x00, random bytes) and freed. Both runs then land on the baited block, and ALL 144.7 MB of out0 plus out1 are bytewise identical. Also identical to an independent model (per-input `.to(bf16)` concatenated in bucket order). Seeds 0-7 randn, real, 6 random-bit-pattern sets, 2 special sets |
| 2c | Metadata / aliasing / mutation / rerun | gate.py (every baited case) | shape, stride, dtype, device, storage offsets 0/0; out1 `_base is out0`, same storage ptr, offset 0; out0 storage nbytes = 144720896 (same as the reference); inputs unmutated (reference and candidate); 2 candidate reruns per pattern bytewise identical |
| 2d | Allocator state (contract item 2) | gate.py `allocator accounting`, `fragmented large-pool state`; `scripts/extra.py` | Per warm call: exactly 1 large-pool allocation of 144720896 bytes for both reference and candidate; 0 new segments; 0 retries. Segment/block snapshots (`memory_snapshot`) are identical before the call, with outputs alive, and after freeing them, alternating ref/cand/ref/cand. out0 lands on the same block. From a fragmented large pool (free 140/146/145 MiB blocks), both land in the same block at the same offset, and the tail equals the setup fill. Candidate small-pool allocations: 0. The reference makes 6 (18432 B, v7..v12), all freed before return |
| 2e | Writes stay inside the slot (white-box) | gate.py `guarded direct kernel call` | the compiled kernel writes a slot view inside a guarded buffer (4096 guard elements on each side, 3 patterns): guards untouched |
| 3 | Sanitizers (`PYTORCH_NO_CUDA_MEMORY_CACHING=1`, `scripts/run_sanitizers.sh`, `san_driver.py`: 2 input sets x 2 calls, compares out1 only) | compute-sanitizer memcheck / racecheck / synccheck / initcheck | candidate: 0 errors on all four (driver EXACT). initcheck is clean, so the candidate never reads uninitialized memory, including out0's tail. SASS agrees: only 2 LDG.E.128 + 2 LDG.E, all from input pointers |
| 3 | initcheck on the tail, by design | `san_driver.py cand.py --read-tail` (reads 4096 tail elements after each call) | 8192 errors = 4 calls x 2048 vector reads, all in aten `converting_copy_kernel` (the driver's read), none in the candidate kernel. Expected under this contract, not a failure |
| 3 | Sanitizer negative controls (`negctl/san_*.py`) | same tools | memcheck in6 read +16 elements: 10 errors (Invalid __global__ read, candidate kernel); racecheck smem write/read without a barrier: 4 hazards; synccheck `bar.warp.sync 0x0000ffff`: 34816 errors; initcheck read of the output before the store: 139264 errors. All flagged |
| 4 | CUDA graphs | `python scripts/graph_test.py` (fresh process) | The first-ever candidate call is inside capture. One graph holds neighbour `copy_` writes to the inputs, then 3 candidate calls (set A, set B, then A again after a neighbour overwrites it with set C), with readers of out1 and out0[:282658] after each call. 60 replays with sources refilled in place (random bits; specials every 5th): 0 mismatches against the eager reference; live outputs are distinct buffers. 20 eager iterations on a side stream with in-stream neighbours: 0 mismatches. Negative control `negctl/graph_cached_buf.py` (cached output buffer): 180 mismatches + shared-buffer flag |
| 5 | Adversarial values | gate.py, `gate.py --only-exhaustive`, `extra.py` | All EXACT: 60 fp32 specials tiled over all inputs at 8 rotations (NaN payloads incl. 0x7F800001/0x7FBFFFFF/0xFFFFFFFF, inf, -0, subnormals, RNE ties 0x3F808000/0x3F818000/0x00008000, overflow tie 0x7F7F8000, max finite); all 3600 ordered special pairs through the cross-tensor (in0, in1) bf16x2 pair; all 2^16 high halves x low halves {0000, 0001, 7FFF, 8000, 8001, FFFF, random} through in2 in both bf16x2 lanes (224 calls); in5/in6 sweeps (3072 calls); **every 2^32 fp32 bit pattern through in3/in4 in both bf16x2 lanes (30842 calls, 0 mismatching)**; 16 constant fills; one-hot markers at the first and last element of every input land at the bucket-order position. Outside the spec: misaligned ptr%16 = 4/8/12 and non-contiguous inputs are EXACT (copy-first path). No reductions: each output element is one rounding of one input element |
| 5 | Gate negative controls (`negctl/nc_*.py`, `gate.py --quick`, `logs/negctl_summary.log`) | | 12/12 rejected: tail_overwrite (4 tail bytes), rz, int_round_nan, ftz, lane_swap, swap56, in2_boundary (off by one pair), zero_tail (144155580 tail bytes), scratch_alive and scratch_transient (extra large-pool allocation; caught only by the allocator accounting, which is why that check exists), bigger_alloc (storage size, `_base`), out1_copy (alias). extra.py also localizes swap56 and in2_boundary by marker position |
| 6 | Special cases | SASS: `CUTE_DSL_KEEP=ptx,cubin`, `cuobjdump -sass` (`sass/`) | Not GEMM-adjacent. No PDL (no griddepcontrol in PTX). No `.nc`/CONSTANT loads, no barriers, no shared memory. Output device = `in0.device`; stream = TVM-FFI env stream, i.e. the current torch stream (confirmed by the side-stream and capture tests). Run on cuda:0 only; multi-device not tested |
| 7 | Timing | `python scripts/bench.py` (gpu0.lock only around the timed loop; 41 interleaved rounds x 10 replays, order alternating) | see below |

## Timing (CUDA graph, interleaved, us per call)

| Graph | Reference median (p10-p90) | Candidate median (p10-p90) | Saving |
|---|---:|---:|---|
| 8 calls/graph, 4 rotating input sets, all outputs live | 16.98 (16.96-17.01) | **1.59** (1.57-1.61) | 15.39 us/call = **1.45 ms/step** at 94 calls/step (10.7x) |
| 1 call/graph (harness style) | 20.84 (20.64-21.19) | 4.44 (4.08-4.97) | 16.41 us/call = 1.54 ms/step |

Other verifiers' correctness processes were resident on GPU 0 during timing (no-lock rule). Reference and candidate
ran interleaved, so both saw the same conditions, and the spreads are tight. Official harness (one call per graph,
not interleaved): reference 21.6 us, candidate 8.6 us.

## Notes for the coordinator (none blocks the verdict)

1. The official harness cannot be used for this problem: a verbatim copy of reference.py fails it. The verdict uses
   the coordinator's bucket contract, checked with baited allocator state plus the allocator accounting and
   snapshot checks above.
2. The reference also makes six small-pool temporaries (v7..v12, 18432 B per call). They are freed before return,
   and the block structure afterwards is identical (checked), but the candidate never writes them, so the contents
   of those freed small-pool blocks differ. This matters only if a later op in the fake-PG step reads uninitialized
   small-pool memory (for example, a fake reduce-scatter output under 1 MiB). Every fused candidate that removes
   intermediates has the same effect. The end-to-end bitwise training run is the check for it.
3. Outside the spec: misaligned in3/in4 or strided inputs are cloned first. These are small-pool allocations, so the
   large pool is untouched, and the results are verified EXACT. A wrong dtype or shape raises ValueError; the
   reference would accept other float dtypes.
