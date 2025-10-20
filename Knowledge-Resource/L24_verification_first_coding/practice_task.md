# Practice
    - Add /pytest_summary and patch apply to your tool server (with timeouts).
    - Implement the fix loop (max 3 rounds) and run it on 2–3 toy projects; paste traces here and we’ll tune prompts.
    - Create 10 verification tasks (small buggy repos with tests) and wire them into your eval harness; measure pass@1 and median patch LOC.
    - Start logging (failure → patch → outcome) to grow an SFT/DPO dataset for patching.

Stretch

    - Add fault localization with coverage (pytest --cov) to rank candidate files/lines probabilistically.
    - Support multi-file patches (cap at 2 files/round) and track complexity penalties.
    - Teach the agent to write a new unit test to prevent regressions after a fix (only when asked).