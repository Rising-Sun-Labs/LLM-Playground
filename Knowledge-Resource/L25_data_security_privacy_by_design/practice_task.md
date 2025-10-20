# Practice

    - Ship the prompt/output redaction middleware; flip it on by default in all routes.
    - Write your Data Catalog (spreadsheet or YAML) with classes, owners, retention, keys.
    - Add deletion endpoint that fans out to logs, blob store, embeddings, caches; test with a seed user.
    - Lock down egress on model/tool pods and add an allowlist for RAG fetches.
    - Enable audit logging with hashed content & model metadata; verify you can trace a request end-to-end.
    - Add a privacy canary set to your nightly evals; set an auto-rollback rule.

Stretch

    - Per-tenant keys + indexes; envelope encryption.
    - DP-SGD experiment on a small sensitive corpus; record ε,δ and utility.
    - Build a tiny Privacy Dashboard: deletion requests, canary pass rate, data volumes by class.