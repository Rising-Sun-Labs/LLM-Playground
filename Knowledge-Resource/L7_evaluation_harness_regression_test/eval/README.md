### Install

cd eval
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

### Initialize the baseline once after a good run:

```

python run_eval.py --ckpt miniLM.pt --tokenizer tokenizer.json
# suppose it wrote results/results_1712345678.json
cp results/results_1712345678.json baselines/baseline.json

```

### Usage Local:

```
# 1) run eval
python eval/run_eval.py --ckpt miniLM.pt --tokenizer tokenizer.json

# 2) compare against baseline & thresholds (fails with exit code if regression)
python eval/compare_results.py --current eval/results/results_*.json --baseline eval/baselines/baseline.json

```

- Determinism:

  - we set seeds + disable cuDNN benchmark; still, small nondeterminism can remain → use relative tolerances (e.g., allow ±1%).

- Golden tests strategy:

  - Use contains checks for phrasing-flexible tasks, and exact only when the output must be exact.

  - Add a few negative cases (model must not say X). You can extend runner.py to support "type": "not_contains".

- Scaling up:

  - Store eval datasets under version control (tiny) or fetch from internal storage.

  - Keep baselines per branch if you have multiple active model lines.

  - Log run metadata (git SHA, model hash) in results\_\*.json so you can bisect regressions quickly.

- Speed:

  - For PPL on big sets, compute on batches and avoid very long contexts if your model is small.

  - For classification, plug in your actual classifier head if you have one (accuracy improves, runs faster).

### Run the tests
- From the repo root where `/eval` and `/tests` live:
```
# ensure dependencies
cd eval
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
deactivate
cd ..

# run fast test (weight-perturb regression)
pytest -q

# optional slower “high LR” demonstration
pytest -q -k high_lr -m slow

```


### General Python 
```
🧩 What __init__.py Does

In simple terms:
__init__.py tells Python:
“This folder is a package, and you can import from it.”
Without it, Python just sees the folder as a regular directory — not one you can import from.
```


