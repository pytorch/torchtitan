# Ideas that measured wrong

Check these before spending time on them; each was measured, not guessed.

- **Per-site RMSNorm regions** on 7168-wide norms: ~0.4 ms per layer (0.7%), and at 4k the region's host time (330 vs 141 µs per norm) exceeded the kernel saving.
- **Fused residual-add + RMSNorm region:** -0.7% at 16k, slightly slower at 4k (two more region calls per layer cost +0.35 ms host). Cross-block fusion needs a block restructure.
- **Folding the residual add into the MoE combine:** wrong under TP (the MoE output is partial before its all-reduce).
- **Global cat-lowering option:** faster for one RoPE, 24-26% slower for another; scope per region.
- **Forcing unrolling via a big global threshold** (the first thing people try): has no effect on symbolic sizes.
- **`assume_32bit_indexing` on large regions:** hard failure once T·N·D > int32.
- **Coordinate descent on every region:** +2x cold compile and decode regressions; we kept it to the one region where it measured a clear gain, and turned it off in deterministic mode because its benchmark-picked configs can differ run to run.
- **Gathers instead of scatters, assumed to win compiled too:** EP permute/unpermute as gathers was 2.7x in eager. Compiled, the permute half still gained and the pair was a net win, but the unpermute half was 7% slower (Inductor's scatter was already fast and the in-graph inverse permutation cost ~18 µs). Measure each half compiled.
- **A library's fused permute+dispatch:** bitwise but slower than separate kernels.
- **Passing dense top-k ids instead of a routing map:** bitwise, not faster within one node (the win is a smaller allgather at large scale).
- **Removing a host sync:** worth it for overlap and CUDA graphs, but often within noise for wall time at one rank; frame it as a sync removal, not a speedup.
- **Making a rewrite the single path because it won under compile:** check eager with inputs requiring grad; one "single path" verdict reversed (+5.75 ms eager) once the harness stopped detaching inputs.
- **Wrapping compiled regions for SAC** (`wrap_inductor_compiled_regions`): crashed MoE models after recompiles (SAC storage keyed per cache entry).
- **Block compile as a deliverable:** a measurement baseline only (not a ceiling: rewrites plus regions beat it); production configs use regions or CUDA graphs.
