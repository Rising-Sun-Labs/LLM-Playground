# Lesson 24: Verification - first code: (tests -> patch -> run -> auto-fix)
    write tests, localize bugs, generate minimal patches, run the suite, and iterate untill green. this is how you turn your assistant into a reliable code-fixer, not a code-dumper.


0) Goals
    - build a **sandbox runner** (pytest/lint/typecheck) with timeouts.
    - add a **patch protocol** (unified diff) the model must output
    - teach a **loop**: read failure -> localize -> propose small patch -> re-run
    - harden with **guardrails** (allowed paths, byte caps, anti-"rewrite project")
    - evaluate pass@k, path size, and time-to-green


1) test-first loop (overview)

    - Prep: you have a project dir with tests.
    - Run tests → capture failing tests (names, file:line, traceback tail).
    - Localize likely fault lines (simple heuristics first).
    - Propose patch: minimal diff only in touched files.
    - Apply & re-run tests.
    - Iterate ≤ N rounds or until green.
    - Return a change summary + remaining failures if any.

2) sandbox runner (FastAPI tool you already use)

Add two endpoints or expand prior ones:

# tools_server.py (additions)
```
class TestResult(BaseModel):
    code: int
    stdout: str
    stderr: str
    sec: float

@app.post("/pytest_summary")
def pytest_summary(req: DirReq) -> TestResult:
    cwd = safe_path(req.project_dir)
    return run_cmd(["pytest", "-q", "--maxfail=1", "--disable-warnings", "-rA"], cwd)

@app.post("/mypy")
def mypy(req: DirReq) -> TestResult:
    cwd = safe_path(req.project_dir)
    return run_cmd(["mypy", "--hide-error-context", "--no-error-summary", "."], cwd)
```

Guardrails

timeouts (≤ 30s), output cap (~200k chars), allowlist path.

3) failure parser & localization (cheap and effective)
```
import re, pathlib

FAIL_RE = re.compile(r"^_+ (test[^\s]+) _+$|E\s+(.*)$", re.M)
TRACE_RE = re.compile(r"File \"([^\"]+)\", line (\d+), in ([^\n]+)")

def summarize_fail(stdout: str):
    tests = []
    for m in TRACE_RE.finditer(stdout)[-10:]:
        path, line, func = m.group(1), int(m.group(2)), m.group(3)
        tests.append({"path": path, "line": line, "func": func})
    return tests[-5:]  # last frames are highest signal near failure
```

Heuristics:

Rank frames by project path (ignore venv/site-packages).

Within a file, propose editing the function enclosing the failing line; if unknown, show ±30 lines around.

4) patch protocol (strict)

Model must output only a unified diff, inside markers:
```
<patch>
*** BEGIN PATCH
*** UPDATE: path/to/file.py
@@
...context...
- old line
+ new line
@@
*** END PATCH
</patch>
```

Rules the model sees in system prompt:

Only edit listed files in failing frames.

Prefer 1–10 line changes.

Keep imports intact unless necessary.

Include enough context for git apply -3 to work (3 lines typical).

No cosmetic reformatting outside hunks.

5) applying patches safely
```
import subprocess, tempfile, os, textwrap, json

def apply_patch(project_dir: str, patch_text: str) -> dict:
    pt = textwrap.dedent(patch_text).encode()
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(pt); tmp = f.name
    try:
        p = subprocess.run(["git", "apply", "-3", "--unsafe-paths", tmp],
                           cwd=project_dir, capture_output=True, text=True, timeout=10)
        ok = (p.returncode == 0)
        return {"ok": ok, "stdout": p.stdout[-4000:], "stderr": p.stderr[-4000:]}
    finally:
        os.remove(tmp)

```
Use a clean git worktree; git reset --hard between attempts.

Track the patch size (loc added/removed) to discourage over-edits.

6) agent loop (planner → call tools → patch → verify)

Pseudocode:
```
def fix_project(project_dir, max_rounds=3):
    trace = []
    # 1) initial run
    t = post("/pytest_summary", {"project_dir": project_dir})
    if t["code"] == 0: return {"status":"green", "trace":[t]}

    for round in range(1, max_rounds+1):
        # 2) localize
        frames = summarize_fail(t["stdout"] + "\n" + t["stderr"])
        context = collect_context(project_dir, frames)  # read 80-line windows

        # 3) ask model for patch
        prompt = render_patch_prompt(frames, context)   # includes strict rules
        patch = call_model_for_patch(prompt)            # must return <patch> ... </patch>

        # 4) apply
        res = apply_patch(project_dir, extract_patch(patch))
        if not res["ok"]:
            t = t | {"stderr": t["stderr"] + "\nPATCH APPLY FAILED:\n" + res["stderr"]}
            break

        # 5) re-run tests
        t = post("/pytest_summary", {"project_dir": project_dir})
        trace.append({"round": round, "patch_bytes": len(patch), "pytest": t})
        if t["code"] == 0: break

    # optional: mypy/lint pass when green
    return {"status": "green" if t["code"]==0 else "failed", "trace": trace, "final": t}
```
7) prompts (battle-tested)
Patch generation (system)
You produce MINIMAL, CORRECT patches in unified diff format. Constraints:
- Change only files listed under FAILING FRAMES.
- Keep changes minimal (1–10 lines).
- Do not reformat unrelated code.
- Return ONLY the patch inside <patch> ... </patch>.

Patch generation (user)
```
Failing frames:
{frames_table}

Relevant code (windows):
{snippets}

Tests output (tail):
{pytest_tail}

Write a patch to fix the failure and improve robustness.
```

Post-run critic (optional)

If tests still fail, ask a critic model to analyze the new failure vs the previous patch and suggest a follow-up patch or rollback.

8) extras that raise reliability

    - Typecheck gate: run mypy after green; if errors appear, allow one more patch.
    - Linter gate: run ruff --fix as a tool call (or ask model to patch lints).
    - Snapshot tests: for string outputs, generate/update snapshot files carefully (require explicit “update snapshot” intent).
    - Flaky tests: re-run failing tests with -k name -q -x and mark flaky if pass-on-rerun; don’t chase ghosts.

9) safety & containment

    - Allowlist edits to project_dir only; reject creating new files outside it.
    - Output cap on tool logs; redact secrets before echoing into prompts.
    - Step/round caps (≤3); if not green, return best diagnosis + partial fix.
    - Never run networked commands; test runner has no internet.

10) evaluation you should track

    - pass@1 / pass@3 on a suite of small projects (e.g., 50–200 tasks).
    - Median patch size (added+removed LOC).
    - Attempts-to-green (rounds).
    - Time-to-green (sec).
    - Rollback rate (patch didn’t apply or made more tests fail).
    - Automate with a harness that:
    - copies a project sandbox,
    - runs the loop with N=1 and N=3 attempts,
    - records metrics and stores final diffs.

11) tiny helpers

Collect context windows
```
from pathlib import Path
def collect_context(project_dir, frames, radius=40):
    out=[]
    for f in frames:
        p = Path(project_dir) / f["path"]
        if not p.exists() or p.stat().st_size > 200_000: continue
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        i = max(0, f["line"]-1)
        lo, hi = max(0, i-radius), min(len(lines), i+radius)
        snippet = "\n".join(f"{n+1:>4} {lines[n]}" for n in range(lo, hi))
        out.append({"path": str(p), "snippet": snippet})
    return out[:5]


Extract <patch>

import re
def extract_patch(text: str) -> str:
    m = re.search(r"<patch>(.*?)</patch>", text, re.S)
    if not m: raise ValueError("patch block not found")
    return m.group(1).strip()
```

12) training data to make this work even better

    - Record successful fix sessions: (frames, context, failure tail) → patch.
    - Build SFT pairs (good patch) and DPO pairs (minimal-correct vs bloated/incorrect).
    - Add negative examples: “patch applies but breaks other tests” with rejected label.
    - For code robustness, include tests where the fix is to add input validation or edge-case handling, not just adjust expectations.