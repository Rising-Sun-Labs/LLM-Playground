### Practice:

- Create eval/config.yaml, val.txt, and goldens.jsonl.
- Run python -m eval.run_eval and inspect eval/report.json.
- Add tests/test_regression.py and run pytest.
- Break something on purpose (e.g., higher LR) and confirm the test fails.

Stretch: add a “safety” JSONL with prompts that should be refused; assert refusal rate ≥ threshold.
