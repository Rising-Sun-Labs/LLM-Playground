# Lesson: 7 - How you measure your model and prevent regressions.

- we'll build a small, reusable evaluation harness for LMs + classifiers, and golden tests, and wire it to CI.
    

## 0) Goals

- Compute perplexity (PPL) for your LM on held-out set.
- add task evals (e.g., classification accuracy; prompt -> expected o/p checks)
- Create a golden test suite with pass/fail thresholds
- log results, compare to baselines, and block bad releases

