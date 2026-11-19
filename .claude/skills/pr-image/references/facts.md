# pr-image: where facts come from

The rule (SKILL.md §1): every fact in the image is checkable from the PR's body, its diff, or the repo code at its
base. These are the cases.

In GitHub Actions, skip the fetches, downloads and probes below: you can run only pr-review's `gh` commands and
the two render commands in SKILL.md "GitHub Actions". The checkout is the PR head, and the base is the `-` and context lines of `gh pr diff`. A fact
you can't check there leaves the image.

- Code outside the diff: read it at the PR's base, after a fetch. An open PR: `git merge-base upstream/main <head>`.
  A stacked PR (`gh pr view <n> --json baseRefName` is not `main`): the parent PR's head. A merged PR: its parent
  commit.
- Library code (vLLM, FA4): the image draws only what the body states, and you check it at the version the PR
  pins. Read its tag on GitHub, or run `pip download <pkg>==<pin> --no-deps --only-binary=:all: -d <new empty dir>`
  (any pip; `--only-binary` keeps a source package's build code from running) and read the files there. If you
  can't read it, log "per the PR body" in `SOURCES.md`. Another version's fields stay out, even with the version
  named (example: a library's 0.1.12 fields drawn on a 0.1.11 pin).
  Library line numbers carry their version (example: lines 218 and 221 came from 0.6.0, line 250 from 0.7.0).
- Stacked PR: check each "what is sent / what runs" claim against this PR's diff, not the stack's end state. A later
  PR in the stack may be the one that changes it (one image claimed a change that only the next PR in the stack made).
- Probes: a quick CPU probe (`CUDA_VISIBLE_DEVICES=""`, also for the PR's own tests) helps you understand the code.
  The image still draws only what the body, the diff or the repo code at the base shows: reviewers marked probe
  facts down even when true and labelled (op counts, matmul counts). If the body needs one anyway, put
  it there first. Otherwise the image goes without it. Op lists and autograd graphs carry to GPU. Kernel counts and numerics don't (a CPU probe
  stored sigma in bf16; the GPU kernel keeps fp32). A toy runs on the real code (the merged class, not a copy).
- A probe that disagrees with the body: fix the body in the same change (example: the probe said 20, the body 19). On a PR you
  can't edit, leave the number out and log the gap in `SOURCES.md`.

## Numbers in the image

- Label the axes ("log" on a log axis). Two rows on one axis come from the same run. Don't chart a comparison the
  PR calls meaningless.
- Main vs this PR uses main's measured row, even when the TLDR compares against another (example: the TLDR says 55.7 ms against another build,
  main's row 77.5 ms). A number sits in the box of the build it was measured on.
- A derived number keeps its arithmetic in `SOURCES.md`, and the image says the result in words. Or it goes.
  Bytes worked out from shapes say "nominal bytes". A nominal bar next to a measured one says "nominal" on the bar.
  Nominal reservations that never coexist are not stacked (example: four 90% reservations drawn as 360%). Don't draw
  a count to scale when its area implies a cost the PR didn't measure (63 empty rows are not a 64x cost).
- A product holds only when every combination occurs. Example: 4 call shapes x 2 grad modes x 2 = 16 possible, and
  the code comment says 12 graphs: draw 12. Recompute totals after applying the diff.
- A made-up example number says "illustrative". Kernel counts, launch order and intermediate tensors need a profile
  or a trace. A count read off the code says "from the code". A timeline without a trace says "schematic". Say
  each of these once, in the subtitle or a panel header, not on every label.
- A schematic lane may shrink time, but not change how often an event happens. Label a rare event's rate ("7 per
  2048-step batch", not a pause every 4 steps). No made-up events on a real time axis.
- A result measured with another PR applied says so, by what that PR does ("with the detokenize change"). A number
  whose comparator the body omits (main vs this PR, or start vs end of one run) leaves the image.
- An estimate is text, never a bar next to measured ones. A sum of separate timings is an estimate. "Bitwise
  identical" names its scope.
- No notes about the PR body in the image ("times from the PR body"): those go in `SOURCES.md`.
