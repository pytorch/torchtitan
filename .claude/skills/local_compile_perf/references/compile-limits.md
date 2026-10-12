# What torch.compile can't do today

Limits with a known rewrite are in symptoms.md and patterns.md. These are the rest: design around them.
Format: **limit**: symptom -> workaround.

- **FMA contraction**: compiled differs from eager by ~1 bf16 ulp (seen on GB300; H100 was bitwise) -> tolerant tests (~1 ulp); bitwise only for batch invariance.
- **Masked loads with symbolic T (GB300)**: many-load kernels ~2x slower -> keep small dims static; T stays dynamic (one graph), so expect this on broadcast-heavy kernels.
- **`maybe_mark_dynamic` forbidden in graphs**: trace error -> mark in eager before the region (pattern 5).
- **size-0/1 dims always specialize**: extra graphs -> avoid size-1 call signatures, or budget for them.
- **make_fx specializes Python ints and captures class-attribute tensors as constants**: stale values under GraphTrainer -> graph inputs, or update a persistent tensor in place.
- **Region options dropped under an outer compile**: an option has no effect under block compile or GraphTrainer -> express the fix in code, not only as an option.
- **Functional regions can't alias outputs or update grads in place**: extra copies -> hand-written override when it matters.
- **SAC replays regions; saving them (`wrap_inductor_compiled_regions`) hits a storage bug after a recompile**: SAC recompute cost, or a crash -> measure under SAC; region-level SAC (gating-and-options.md).
- **Store to a new attribute of an outer-scope object inside a checkpoint HOP**: generic "Observed exception" -> create the state outside (e.g. a thread-local's `__init__`).
- **`autograd.Function` reading `w._base` / `grad_dtype` of a weight view**: AOTAutograd crash -> avoid in traced code.
- **Mesh push/pop inside traced regions**: hard error -> keep mesh context outside regions.
- **Tensor caches (`functools.cache`, module-level dicts) under FakeTensorMode**: fake tensors cached and reused in real runs -> key caches on real tensors only, or skip under fake mode.
- **Inductor's FX graph cache ignores a changed custom-op fake**: the old (wrong) fake is still used -> clear the cache dir after changing a fake.

A new limit hit during a run: add it here (symptom, workaround, a minimal repro command), and report it per pr-template.md. Don't file it outside the repo without asking the user.
