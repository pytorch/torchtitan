---
name: pr-image
description: Draw one PNG that shows a PR's problem and fix at a glance, for your own message: the @claude review (pr-review calls it in GitHub Actions) or your reply to a person. HTML rendered by headless Chrome.
---

# pr-image

One PNG for a PR. From the PR title and the image alone, a reader can say what was wrong and what changes, and
nothing in it is false. The text around it holds the details.

The reader is an engineer about to read the code and approve or request changes:
- People read pictures better than walls of text. The drawing carries the meaning. Text is welcome, but of two
  images with the same message, the one with less text wins.
- Factually correct above all: one wrong arrow, label or number undoes the rest.
- It is not a sales image. Results are fine, small and with their setup; no hype.
- Dense in drawing, light in words: no one should need to read it 3 times. Write every word for a tired reader
  who is new to the code (§3 Text).

Two modes:
- Local: a person asked for an image.
- GitHub Actions: `pr-review` called this skill (the prompt has `<formatted_context>` tags). Read "GitHub Actions"
  at the end first: it changes the folder, the commands and how the image is published.

Never edit the PR body or post anything yourself. The image goes in your own message: the review in GitHub
Actions, your reply locally.

## Output

```text
<folder>        /tmp/pr_image/<pr>/ locally, pr_image/ in the checkout in GitHub Actions
  SKETCHES.md   line 1: the PR head sha you drew from. Then 6 sketches, the 2 you drew, the pick and why (local only)
  <form>.html   the source of each drawn image (locally also <form>.py for matplotlib, <form>.dot for graphviz)
  <form>.png    1800 px wide (900 CSS px at 2x), up to ~1100 CSS px tall; <form>_900px.png is the copy to check
  SOURCES.md    each label, number and arrow -> the body line, diff hunk or path:line it comes from
  no image      (§2 says when) say why: in SKETCHES.md locally, to pr-review in GitHub Actions
```

## 1. Read

The PR body (TLDR, Problem, Solution, Results) and the diff. Names and code come from the diff, not the body's
snippets. The reader sees only the PR, so every fact in the image must be checkable there. The cases (code
outside the diff, library versions, stacked PRs, probes) are in [references/facts.md](references/facts.md):
- A body line the code contradicts: the image follows the code. Name the stale line in your message, and log it
  in `SOURCES.md`.
- A fact the body and diff don't state (a probe result, a library field, git history) leaves the image. Repo code at
  the base is fine, cited by `path:line`.
- Numbers the diff already holds, like a test's expected values, beat probe numbers.

## 2. Pick the form before drawing

Write 6 sketches. A sketch is a paragraph of at most 6 lines: the form, what each lane, cell or column holds
(the PR's real names and values), what the reader sees on main and on this PR, how it shows the cause, and what
stays out. Drawings of the PR's own data and timelines work best. Pick by the cause:

```text
the cause is...                    draw
an amount or a layout (tokens,     the PR's own example as cells, rows or pages, small and real; main -> this PR.
bytes, slots, objects, ids)        One group drawn, "x4" for the rest. The diff's test values first
timing (a hang, a race, a lag,     lanes (event loop, thread, process, GPU stream). Hatch the wait. Grey out what
a lifetime)                        never runs. A lifetime: one lane per holder (each local, each buffer)
the state each call sees (a cache  one column per call, in time order. Rows: the cache, the graph that runs, what
or dispatch bug, a reorder)        is saved. A reorder: the same calls in two columns, one line per moved call
an order or a placement            the data in its slots (rows into slots, roles onto GPUs) and what sets the order
what runs where (kernels, eager    two lanes, main | this PR, only the changed boxes coloured. Code moved into a
vs compiled)                       compile region: forward and backward op lanes, one red box per eager op vs one
                                   blue band per pass. If the PR lists no ops, the function moving into the region
a value two sides disagree on      each side's path to the value and the step or decision where they part (the ÷T
(trainer vs generator)             one path skips, the near-tie at the top-k cutoff); the fix as the same drawing
a recompile limit                  one numbered cell per compiled graph, bars at the old and new limit
memory                             what holds the bytes and for how long (lifetime lanes, or slots). A to-scale
                                   stacked bar only as a small strip under it
a changed default or config        many recipes gain: a grid of recipes x the setting. Few gain: the mechanism the
                                   new value turns on
speed                              what changes in the work. Measured times: one small bar under it, or one text
                                   line when the change is under ~10%
```

No image when nothing runs differently: a pin, import, golden, log level, pass-through or test-only change, or a
new flag whose default keeps today's behaviour. A one-line config that changes what runs gets one, and so does a
pin that fixes a bug in what runs (a deadlock, a crash). A race or hang the body explains only in prose gets
lanes. If someone asks for an image anyway, use the row that fits the cause.

These work badly: results first, text tables, numbered step lists, code listings, and call chains as boxes.

Checklist for each sketch:
- Does it show something the TLDR's code block can't: the data, the timing or the cause? If not, drop it.
- Does it show why main fails, not only the symptom? Draw the link from the changed value to its effect, not only
  the place where the value changes.
- The cause gets the most space. The path that carries it is one grey line. Other hunks may get a small mark; most
  stay in the text. Panels differing only in colour or label redraw the diff. Draw each case small, not one big.
- Does the diff make a choice a reviewer will question ("why not just concat?")? Draw that rejected option as one
  small strip, failing on the PR's real numbers. One extra strip per image at most; the rest stay in the text.
- Does the PR change behaviour a reviewer must accept (a log now dropped, a prompt that changes, answers that now
  score 0)? Give it a mark in the drawing, not a footnote, and a line in the text.
- Hide the sentences: do the shapes and short labels still tell it? More than ~8 crossing lines: draw less.

A sketch, for a library pin that fixes a hang:

```text
lanes. One thread, one lane per torchmonarch version, time left to right. 0.6.0: freeing tensor A (an RDMABuffer)
takes Monarch's lock. A's cleanup frees tensor B on the same thread, and B's cleanup waits for that lock: a hatched
wait that never ends, marked "hang". 0.7.0: the lock is an RLock, so B's cleanup takes it again and both return.
Strip under it: a CPU stress of the trainer-pull batcher hung in 3 of 10 runs on 0.6.0, 0 of 10 on 0.7.0. Shows the
cause (one thread taking the lock twice), which the version line can't. Stays out: the rest of the batcher.
```

Draw your top 2, then pick one yourself. Check the pick with §4 as if no one checks it after you.

## 3. Draw

- Size: 900 CSS px wide, rendered at 2x. Good images are 500-1040 CSS px tall; stay under ~1100.
- Opaque light background (`#fbf8f3`), so it reads in dark mode too. Smallest text 13 CSS px. Code in monospace.
- Text: each item is a name or a number with a plain label (1-4 words: "64 KiB", "14.7 ms"), or one short sentence
  with a subject and a verb. No colon chains, semicolons, "X = ..." or code in place of a sentence. A count may be a
  product ("16 × 1,024 tokens"). No term that needs defining. A code name comes with words for what it is.
- Order: a one-sentence subtitle says what you look at. Each panel has a header sentence that names its side and
  the set it holds for, at most 4 notes, and a last sentence on what it means. About 150-200 words in all.
- Too much text: write better before you cut. Keep the numbers an engineer checks: sizes, counts, times and the
  baseline (main, vLLM). Cut provenance first (to `SOURCES.md`), then setup lists, test names and limits (to the
  body). Say "illustrative", "schematic" or a scope once, in a header. A claim needing its own qualifier leaves the
  image.
- Key: under the title, naming every mark style used (~3 at most). A mark covers only what its key says.
- Code: at most 2 lines, the ones that decide the change, tokens copied exactly. You may de-indent, wrap, join and
  cut with `...` (logged in `SOURCES.md`), but never drop a wrapper call or argument without `...` or add a comment.
  Generated code (Inductor, Triton) stays in the text: draw the cells each computation reads and writes instead.
- Title: the whole change as one claim in under 10 words, no semicolon (not one model, pass or field, not the
  speedup). Trace it like any label. One symbol, one meaning (not T for temperature and `[T]` for tokens).
- Footer: at most 3 short sentences, no result cards or big numbers. One result in words with its baseline, model
  and GPUs ("Each step takes 60 ms instead of 80 ms on <GPUs>."), plus any condition it needs. Then a cost, if any.
- Colours and marks. Each compared panel names its side, and red or green always comes with a word ("main"):
  ```text
  muted red   main, before, broken           green   this PR, after, fixed
  grey        unchanged, never runs, never read. Other comparisons and data categories: neutral tints, outlines
  amber       code outside the repo that causes or carries the change (the repo's own code is not outside)
  pale blue   a band around a compiled region
  dashed      an option not taken, or a thing absent. Dashed + ✕: removed by this diff, on the exact token removed
  ```
- Unchanged parts: thin grey boxes with only a name. A run of unchanged boxes or calls collapses to one grey line.
- Each arrow means one thing (data moves, calls, waits for); label it when unclear. Number steps only in real order.
- Numbers (axes, derived, nominal, "schematic", rates): [facts.md](references/facts.md#numbers-in-the-image).

Tools: HTML/CSS + inline SVG, rendered by `python3 -I .claude/skills/pr-image/scripts/render_html.py <form>.html`.
Locally also matplotlib or graphviz. Gotchas in [references/tools.md](references/tools.md). Winning sources:
[examples/](examples/README.md).

## 4. Check

- Open `<form>_900px.png`, which `render_html.py` writes next to the PNG (resize a matplotlib or graphviz PNG
  yourself): GitHub shows the image at that size. Fix overlaps and text clipped at the right edge.
- Read test: read every text item once, top to bottom, as a tired reader new to the code. Rewrite or cut what you
  read twice.
- `SOURCES.md` traces the title and every label, number, arrow, error name and date to a body line, a diff hunk or
  a `path:line`. Drop what you can't trace.
- Each code token is in the diff (`+`, `-` or context lines), at the base's `path:line`, or in the body (library
  code, checked at the pinned version). Each ✕ box holds only the token the diff deletes.
- Each count equals the marks drawn ("12 ops" over 8 boxes is wrong) and the source, digit by digit.
- Each arrow points at its real callee (the router, not the generator it forwards to).
- Each main-side claim holds for the set its panel header names (which models, which mode). A value that differs
  per case gets one label per case. Check "only", "never", "every", platforms and prefill vs decode.
- Stacked PR (its base is not `main`): no "main" in the image. Label that side "before" or by the parent PR.
- Glance test: cover the PR text. Can you say the problem and the fix in 10 seconds?
- Alt text: one sentence with the image's message and numbers. A later review checks it against the new diff.

## 5. Reply (local)

- Reply with the PNG path, the alt text and `<!-- pr-image drawn at <PR head sha> -->`. The person posts it:
  github.com hosts a PNG dragged into a PR description or comment.
- Asked again after the PR head moved past the "drawn at" sha: run §4 on the new diff. Passes: reply with the new
  sha. Fails: redraw.

## GitHub Actions

`pr-review` calls this skill and posts the image in its review. What changes:

- Folder: `pr_image/` in the checkout, because writes elsewhere are denied. Never commit it. No `SKETCHES.md`:
  the sketches stay in your reasoning. `SOURCES.md` means the image sources that the review lists under Details.
- HTML only. Name the page `[a-z0-9_-]{1,40}.html` (it becomes part of the image URL), e.g. `lanes.html`. Run each
  command from the checkout root exactly as written, with no `cd`, pipe or redirect, or it is denied:
  ```bash
  python3 -I /tmp/pr-image/render_html.py pr_image/<form>.html
  python3 -I /tmp/pr-image/render_html.py pr_image/<form>.html --upload
  ```
  The first writes `<form>.png` and `<form>_900px.png`. The second renders the page again, refuses unless the result
  is pixel-equal to `<form>.png`, uploads it and prints its URL. `/tmp/pr-image/` holds the default branch's copy of
  `scripts/render_html.py`, put there by the workflow's setup step.
- Facts: no probes or downloads. The only commands are pr-review's `gh` ones and the two above. The checkout is the
  PR head; the base is the `-` and context lines of `gh pr diff`.
- The review's **Example** is the code block: the image shows what it can't.
- At most 6 renders before `--upload`, both drawings together. Read the picked `<form>_900px.png` after your
  last fix, then upload that unchanged page. Over budget, or a command fails: no image, and tell pr-review why.
