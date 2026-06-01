"""The per-candidate gate: the 4-step pipeline (ARCHITECTURE.md section 5.3).

    admit -> run+measure -> verify-routes-quality -> decide

This is the harness's judgment. The agent cannot influence it. The gate is pure
orchestration over the injectable `Executor`, so it is fully testable on CPU.
"""

from __future__ import annotations

from torchtitan_autoresearch import quality as Q
from torchtitan_autoresearch import significance as sig
from torchtitan_autoresearch import workload_guard as wg
from torchtitan_autoresearch.constitution import Rules
from torchtitan_autoresearch.executor import Executor, classify_crash
from torchtitan_autoresearch.ledger import Ledger, Record
from torchtitan_autoresearch.state import HarnessState
from torchtitan_autoresearch.types import Candidate, Quality, Verdict


def _family(c: Candidate) -> str:
    if c.family:
        return c.family
    if c.addresses:
        return c.addresses[0]
    return (c.label.split() or ["misc"])[0].lower()


def _record(c: Candidate, v: Verdict) -> Record:
    return Record(
        commit=c.commit[:7],
        tps_mean=v.throughput_mean,
        tps_cv=v.throughput_cv,
        quality_margin=v.quality.margin,
        verify=v.verify,
        verdict=v.verdict,
        crash_class=v.crash_class,
        status=v.status,
        label=c.label,
        addresses=c.addresses,
        rationale=c.rationale,
    )


def gate(
    c: Candidate,
    *,
    rules: Rules,
    state: HarnessState,
    ledger: Ledger,
    executor: Executor,
    statefile: str,
    session=None,
    mode: str = "screen",
    max_reruns: int = 1,
) -> Verdict:
    """Judge one candidate end to end; append exactly one ledger row.

    When a `session` is provided, the harness owns git: it commits the candidate's
    file edits to the isolated branch before running, and resets the branch to the
    champion on any non-promotion. The agent never touches git.
    """
    family = _family(c)
    budget = state.family_budget(rules.family_defer_substrate, rules.family_defer_other)
    prev_tip = state.champion_commit or (session.base_commit if session else "")

    def finish(v: Verdict) -> Verdict:
        if session is not None and v.status != "keep" and prev_tip:
            session.reset_to(prev_tip)  # discard the candidate commit; tip stays at champion
        state.absorb_budget(budget)
        state.save(statefile)
        ledger.append(_record(c, v))
        return v

    none_q = Quality(checked=False, passed=False, margin=0.0)

    # 1. Admissible? (locked invariants + editable scope + not a deferred family)
    if budget.is_deferred(family):
        return finish(Verdict(False, 0, 0, none_q, "invalid", "invalid", "-",
                              f"family '{family}' deferred (repeated substrate failures)"))
    ok, reason = wg.admissible(c, rules)
    if not ok:
        return finish(Verdict(False, 0, 0, none_q, "invalid", "invalid", "-", reason))

    # Harness commits the candidate's edits to the isolated branch (agent has no git).
    if session is not None:
        c.commit = session.commit_candidate(c, rules)

    # 2. SPEED TEST FIRST: run + measure throughput, decide significance vs the
    #    champion (with one optional rerun near the boundary). A candidate that is
    #    not clearly faster is rejected here -- with NO quality eval -- so the
    #    expensive eval only ever runs on candidates that actually win on speed.
    steps, window = rules.steps(mode), rules.window(mode)
    bootstrap = not state.champion_tps
    samples: list[float] = []
    last_cv = 0.0
    tps_recommend, tps_detail = "promote", "bootstrap champion (no prior best)"
    for attempt in range(max_reruns + 1):
        tr = executor.run_throughput(c, steps, window)
        if not tr.ok:
            cv = classify_crash(tr.crash_text)
            budget.record(family, cv)
            st = "oom" if cv.crash_class == "OOM" else "crash"
            return finish(Verdict(True, 0, 0, none_q, "-", st, cv.crash_class,
                                  f"run failed: {cv.crash_class}/{cv.stage}"))
        samples.append(tr.tps_mean)
        last_cv = tr.tps_cv
        if bootstrap:
            break
        sv = sig.decide(samples, state.champion_tps, state.tps_noise())
        tps_recommend, tps_detail = sv.recommend, sv.detail
        if sv.recommend != "rerun":
            break
    budget.record(family, None)  # it ran fine; reset the family streak

    if not bootstrap and tps_recommend != "promote":
        return finish(Verdict(True, samples[-1], last_cv, none_q, tps_recommend, "discard",
                              "-", f"not faster than champion: {tps_detail}", verify="n/a"))

    # The bootstrap baseline defines the workload; it has no champion to beat and
    # no golden to check against yet, so it is kept as the first champion.
    if bootstrap:
        state.champion_commit = c.commit
        state.champion_tps = samples
        return finish(Verdict(True, samples[-1], last_cv, none_q, "promote", "keep", "-",
                              "bootstrap champion (no prior best)", verify="n/a"))

    # 3. QUALITY (only reached when the candidate is faster): verify routes --
    #    faithful changes are quality-preserved (no eval); affecting changes must
    #    clear the held-out eval floor.
    qo = Q.verify_routes_quality(c, executor, state, rules)
    if qo.verify == "fail":
        cv = classify_crash(qo.eval_crash_text or "")
        budget.record(family, cv)
        st = "oom" if cv.crash_class == "OOM" else "crash"
        return finish(Verdict(True, samples[-1], last_cv, qo.quality, "-", st,
                              cv.crash_class, f"eval failed: {cv.crash_class}", verify="fail"))
    if qo.quality.checked and not qo.quality.passed:
        return finish(Verdict(True, samples[-1], last_cv, qo.quality, "reject", "discard",
                              "-", f"faster but quality below floor (margin {qo.quality.margin:+.4f})",
                              verify="affecting"))

    # 4. Faster AND quality preserved -> promote.
    state.champion_commit = c.commit
    state.champion_tps = samples
    return finish(Verdict(True, samples[-1], last_cv, qo.quality, "promote", "keep", "-",
                          f"faster ({tps_detail}) and quality preserved",
                          verify=("affecting" if qo.quality.checked else "faithful")))
