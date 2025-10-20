Lesson 19: Evaluation at scale and canarying models

0) outcomes

ship canaries/A-B safely (no fake wins)

collect feedback signals (explicit + implicit)

run tests with proper statistics (peeking-safe)

close the loop: rank + preference training from production data

dashboards + auto-rollback when things go south

1) experiment taxonomy (pick the right one)

Shadow: duplicate live requests to candidate model; don’t return its answer. Use for safety/latency shakeout.

Canary: route 5–10% of traffic to candidate; rollback if KPIs breach.

A/B: randomized 50/50 (or other split), sticky per user/session. Prefer when you care about quality KPIs.

Interleaving (ranking/RAG): mix citations/snippets from both variants in one answer; infer winner from clicks—faster than A/B when UI supports it.

Sticky assignment (cookie/header):

bucket = hash(user_id) % 100
variant = "B" if bucket < 10 else "A"   # 10% canary

2) success metrics (define before launch)

Core

Quality: task pass rate (goldens online), user thumbs-up rate, answer reuse (copy/submit).

Safety: refusal-where-required rate, flagged-content rate.

Latency: TTFT p95, tokens/sec p50.

Reliability: 5xx rate, timeout rate.

Business: resolution rate, deflection, conversion, CSAT.

Guardrails (hard stops)

Safety fail rate > threshold

p95 TTFT > SLO for N minutes

5xx > 1% for M minutes

Write to EXPERIMENT.md for each test.

3) logging schema (keep it lean & analyzable)

Each response log row:

{
  "ts": "2025-10-14T09:12:33Z",
  "rid": "uuid",
  "user_id": "anon-123",        // pseudonymized
  "variant": "A",
  "prompt_hash": "sha256:...",
  "domain": "code_review|email|rag",
  "safety_mode": "on|off",
  "latency_ms": 742,
  "ttft_ms": 186,
  "input_tokens": 1543,
  "output_tokens": 212,
  "answer_hash": "sha256:...",
  "feedback": null,             // or { "thumb": 1, "comment": "..." }
  "implicit": { "copy": true, "edit_chars": 9, "clicked_citation": [2] },
  "eval": { "golden_ok": true, "regex_ok": false },  // if applicable
  "errors": null               // or { "type":"timeout" }
}


Privacy: store hashes, not raw content; redact PII before logging.

4) statistics you can trust (no p-hacking)

Main test: difference in means/props with Welch’s t-test (continuous) or two-proportion z-test (binary). Report effect size + 95% CI (e.g., Newcombe/Wilson).

Sequential monitoring: avoid peeking inflation. Use:

Group sequential (e.g., O’Brien-Fleming boundaries), or

Always-valid methods (e.g., e-values) if you need continuous looks.

Variance reduction: CUPED (pre-period covariate) for metrics like thumbs rate.

Sample size (binary approx):

n per arm ≈ 16 * p*(1-p) / δ^2


where p ≈ baseline rate, δ = desired absolute lift.
Example: baseline thumbs 0.35, want +0.03 → n ≈ 160.350.65 / 0.0009 ≈ 4044 per arm.

Power checks: aim for ≥80% at your minimal detectable effect (MDE).

5) online goldens & passive evals

Online goldens: 1–5% of traffic gets embedded, hidden test prompts (non-user content). Pass/fail logged by variant.

Passive evals: regex checks for format, presence of citations, refusal template, JSON validity, etc.

Canary gate: require online-golden pass rate ≥ offline baseline − ε.

6) safety canarying

Mirror lesson-12 refusals as online tests (prompt set rotated daily).

Run a red team sampler batch each day; log any violations with full trace to a quarantined store.

Auto-rollback rule example:

If safety fails ≥ X in last Y minutes and z-test vs control shows ↑ with p<0.01 → switch 100% back to A.

7) feedback loop → preference data

Collect:

Explicit: thumbs up/down, 5-star, quick reasons (“helpful”, “incorrect”, “unsafe”, “too long”).

Implicit: copy, time-to-edit, ignore, follow-up rate, citation clicks.

Outcome: did unit tests pass? did the email get sent? did the ticket close?

Turn into DPO pairs:

Prompt = user input (+ context).

Chosen = high-engagement/approved answer.

Rejected = low-engagement/rolled-back answer.

Add tags: domain, safety flags, latency bucket.

Nightly job:

Ingest last 24h feedback.

Filter (dedupe, privacy).

Construct pairs; balance by domain.

Re-train small preference model or run DPO on last aligned checkpoint.

Run offline eval suite; if ≥ thresholds → candidate to canary.

8) bandits & routing (optional but powerful)

Epsilon-greedy or UCB per domain to route between A/B (or small/large model).

Reward = thumbs / task pass / engagement composite (define weights).

Converges faster than fixed A/B when user mix shifts.

9) rollout playbook (copy/paste)

Shadow for 24–48h → ensure latency/safety good.

Canary 10% sticky → primary metrics + safety gates live; sequential analysis set up.

If stable 24–72h → A/B 50/50 to confirm lifts with power.

Promote if CI + offline evals + online KPIs green.

Keep kill switch (feature flag) and automatic rollback rule.

Archive experiment report (metrics, CIs, decisions).

10) dashboards (minimal)

Overview: traffic by variant, p50/p95 TTFT, errors, safety fails.

Quality: thumbs rate with CI, online-golden pass, JSON validity, citation presence.

Engagement: copy/edit rates, time-to-task.

By domain: code/email/RAG slices.

Experiment view: effect size over time with CIs; peeking indicators.

11) pitfalls & fixes

Metric drift from user mix → stratify by domain, use CUPED or covariate adjustment.

Peeking false positives → commit to sequential method; don’t eyeball daily winners.

Feedback bias (only angry users vote) → combine explicit + implicit; randomize feedback nudge.

Canary looks fine, A/B loses → underpowered canary; re-run with enough sample or run interleaving for retrieval UX.

Safety regression only in niche slice → add slice-level guardrails + auto-slice rollback (e.g., “model=B for email only”).