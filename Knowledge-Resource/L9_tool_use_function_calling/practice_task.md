### Practice:

- Launch **tool_server** and extend your model server with `/agent`.
- Add dummy prompts like 
```
# prompts that trigger tools (examples)

Zip review

“I’ve uploaded /srv/projects/demo/repo.zip. Find Python files with ‘TODO’ and read app.py to propose fixes.”

Diff explain

“Explain how to fix this function. Produce a minimal patch diff.”

Tests

“Run tests in /srv/projects/demo/project and summarize failures with file:line and likely fix.”

Lint

“Run the linter in /srv/projects/demo/project and list top 10 issues with a one-line fix each.”
```
- Add **tools eval** to your CI (pass rate threshold >= 70% to start).
- Bonus: Implement `zip_search` (grep-like search within zip) to speed up "find then read".
