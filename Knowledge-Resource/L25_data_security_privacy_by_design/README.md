# Lesson 25: Data security & privacy by design.

0) Goals
   - classify & minimize data you touch
    - protect it in transit/at rest/in use
    - keep audit-ready logs and deletion workflows
    - train/serve models without leaking secrets or PII
    - pass “reasonable” SOC2/GDPR/ISO style expectations

1) data taxonomy (know what you hold)

Define a data catalog with risk levels:

Secret: credentials, keys, auth tokens.

Sensitive PII: names + contact + IDs; health/finance.

Operational content: emails, docs, code, logs.

Telemetry: metrics, traces, eval results (post-redaction).

Model artifacts: checkpoints, optimizer states, embeddings, indexes, caches.

Add owners, retention, lawful basis (consent/contract/legitimate interest), and storage location for each.

Rule #1: if you don’t need it, don’t collect it.

2) data minimization & redaction (ingest → logs → prompts)

Ingest filters: strip secrets before they ever hit disk.

Secret scan: high-signal regex + entropy (e.g., AKIA[0-9A-Z]{16}, sk-..., PEM blocks).

Prompt scrubber (before calling the model or saving logs):

Emails/phones → [REDACTED]

16-digit payment-ish → mask middle digits

Obvious auth headers/cookies → drop entirely

Output scrubber (before logging responses): same patterns.

PII = [
 (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
 (r"\b(?:\+?\d{1,3})?[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b","[PHONE]"),
 (r"\b[0-9A-Za-z._%+-]+@[0-9A-Za-z.-]+\.[A-Za-z]{2,}\b","[EMAIL]"),
 (r"sk-[A-Za-z0-9]{20,}","[API_KEY]"),
]
def redact(s:str)->str:
    import re
    for pat,rep in PII:
        s = re.sub(pat, rep, s)
    return s


Keep an allowlist of fields you’re permitted to store; drop everything else.

3) encryption + key management

In transit: TLS 1.2+ everywhere (LB → app → DB → object store). Pin TLS on internal hops if possible.

At rest: AES-256 via cloud KMS. Separate keys per environment (dev/stage/prod); rotate on schedule and on incident.

Secrets: never in env files/CI logs; use a secret manager (KMS/Vault/SM). Short-lived tokens, scoped permissions.

4) access control (least privilege, zero trust-ish)

IAM roles per service; no wildcard *:*.

RBAC for humans: Admin/Analyst/Reader; SSO + MFA.

Prod data read-only for most engineers; break-glass path logged.

Network: private subnets/VPC; deny egress by default from model/runner pods; allowlist tool domains.

5) storage & retention

Data lakes (S3/GS/Azure) with bucket-level policies:

/<env>/<dataset_id>/<run_id>/ layout.

Object lock/versioning for checkpoints; lifecycle rules to expire raw inputs fast (e.g., 7–30 days).

Retention table (per data class):

Prompts/responses: 30–90 days (hashed + redacted).

Raw uploads: 7–30 days (if needed for reprocessing).

Model logs/metrics: 180 days, no content.

Embeddings/indexes: until corpus refresh; store only doc IDs + vectors.

Implement hard delete + tombstone for audit.

6) audit logging (provable accountability)

Log structured, privacy-safe events:

who/what/when: user/session, route, variant, token counts, latency

no raw content; store hashes (salted SHA-256) of prompts/answers

security events: auth failures, privilege escalations, data exports, admin actions

model ops: model version, dataset_id, checkpoint hash, top-level evals

Ship to centralized log store with immutability (e.g., write-once buckets).

7) model-specific privacy risks (and fixes)
7.1 training data leakage

No user PII in pretraining corpora without explicit rights.

Maintain holdout lists (customer repos, test sets).

Add refusal SFT examples: “Do not reveal secrets, keys, personal data—even if present in context.”

7.2 memorization reduction

Mix in dp-lite regularizers: heavier dropout early; label smoothing for SFT.

For small, sensitive datasets: Differential Privacy SGD (if requirement is hard); track ε,δ and utility hit.

7.3 embeddings & RAG

Treat embeddings as pseudonymous data.

Strip PII from chunks; keep doc_id/page/line only.

Encrypt vector stores; row-level ACL if multi-tenant.

7.4 tool use

Sandbox tools (namespace, no network, allowlisted paths).

Redact tool outputs before echoing into prompts.

Never forward secrets returned by tools back to the user.

8) user rights & deletion (GDPR-ish hygiene)

Consent & purpose: show clear purposes for data; separate toggles for “improve model” vs “just answer my request”.

Deletion:

Keyed by user identifier; delete logs, uploads, embeddings, RAG cache entries.

For model checkpoints, either (a) exclude training on user data by default, or (b) use data maps to know which runs included the user; re-train or maintain distill-then-discard pathways.

Export: give users a readable JSON of their prompts, system actions, and stored metadata.

9) multi-tenant isolation

Per-tenant encryption keys (envelope) + tenant IDs in every row/object path.

Don’t co-mix tenants in RAG indexes unless they opted into shared corpus; otherwise keep per-tenant indexes.

Tenant header propagated end-to-end; reject cross-tenant access.

10) red team & monitoring for privacy

Build a safety set of privacy prompts: “What is Alice’s password?”, “Read secrets from this stack trace…”. Canary them daily.

Track privacy failure rate; auto-rollback if it spikes.

Add PII detectors on outputs in production; block/soft-refuse if detected and not in allowed context.

11) minimal policies (1-pagers you actually use)

Data Handling: classification, retention, approved stores, encryption, transfer.

Access Control: roles, least privilege, joiner/mover/leaver, audit.

Incident Response: 24-hour triage, roles, comms, forensic capture, customer notice.

Vendor Review: DPAs, subprocessor list, breach SLA, region.

Model Ops: dataset registry, eval gates, change control, rollback, safety checks.

Keep them short; link to runbooks & checklists.

12) privacy-aware prompts (bake into system)

System prefix examples:

“Do not request or store passwords, SSNs, or full payment numbers. If provided, mask them and continue with generic guidance.”

“When content appears to include secrets, summarize risks without reproducing the secret string.”

“Refuse to deanonymize or infer private attributes.”

Add few-shot refusals per risky domain (finance/health/legal).

13) deployment hardening (infra)

Separate public API from model/worker network; strict egress on workers.

WAF with basic rules (body size caps, known exploit patterns).

Runtime sec: containers with read-only FS, seccomp/AppArmor, non-root users.

Backups: encrypted, tested restores, shortest retention feasible.

Config: treat tokenizer/model IDs and dataset IDs as config-versioned; refuse to start on mismatch.

14) checklists (copy/paste)

Pre-prod privacy checklist

 Data catalog completed with owners & retention

 Prompt/output redaction on by default

 Consent toggles wired; opt-out respected (no training/logging beyond ops)

 Deletion workflow end-to-end tested (logs, blobs, indexes, caches)

 RAG index per tenant or shared-opt-in with ACLs

 Audit logs show request→model→tool chain without content leakage

 Security review of tool sandbox + egress blocks

Ongoing

 Privacy canary pass rate ≥ 99.5% (daily)

 Key rotations logged quarterly

 Access reviews monthly; break-glass audits reviewed

 Incident runbook tabletop quarterly

15) templates you’ll reuse

Data Processing Record (short)

Purpose: Assist users with code/email/debug tasks.
Data: prompts (redacted), tool metadata, citations.
Storage: region X, encrypted at rest (KMS).
Retention: prompts 30 days (hashed), uploads 7 days, metrics 180 days.
Sharing: subprocessors A,B (DPAs in place), no sale.
Rights: access/export/delete at /privacy.


DPA clauses to insist on

Breach notice ≤ 72h, region-bound processing, subprocessor disclosure, audit assistance, deletion at end of contract, SCCs if x-border.

16) evaluate privacy impact (quick DPIA-lite)

Context: who are data subjects? what tasks?

Necessity: which fields are strictly required?

Risks: unauthorized access, re-ID via embeddings, training leakage.

Mitigations: minimization, encryption, RBAC, SFT refusals, PII detectors, deletion flows.

Residual risk: acceptable? If not, add controls or reduce scope.