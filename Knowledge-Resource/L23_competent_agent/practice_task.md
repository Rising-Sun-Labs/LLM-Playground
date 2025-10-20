# Practice
    - Implement Plan-then-exec with the JSON schema above; wire to your existing tools.
    - Add a critic verifier for your top 2 domains (code & RAG citations).
    - Write 10 agent goldens and run them in CI (pass rate target ≥ 0.7 to start).
    - Log plans + outcomes; curate 50 good plans → create a tiny planner SFT set.
    - Add loop guards (step/tool caps) and safety goldens; confirm they trip correctly.

Stretch

    - Add beam-2 planning for hard tasks with a 2× step budget cap; pick the plan that yields the better verifier score.
    - Teach the agent tool schema discovery: if a tool fails validation, request the correct shape once (not infinitely).