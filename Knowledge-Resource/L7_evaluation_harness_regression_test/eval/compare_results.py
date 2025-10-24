import argparse, sys, math
from harness.utils import load_json

def rel_delta(curr, base):
    if base == 0: return 0.0 if curr==0 else float("inf")
    return (curr - base) / abs(base)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", required=True)       # results_*.json
    ap.add_argument("--baseline", default="baselines/baseline.json")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    curr = load_json(args.current)["metrics"]
    base = load_json(args.baseline).get("metrics", {})
    import yaml
    cfg = yaml.safe_load(open(args.config,"r",encoding="utf-8"))
    policy = cfg.get("regression_policy", {}).get("max_relative_drop", {})
    absolute = cfg.get("absolute_thresholds", {})

    def fail(msg): print("REGRESSION:", msg); sys.exit(1)

    # Check absolute gates
    for k, thresh in absolute.items():
        if k not in curr: continue
        v = curr[k]
        if "ppl" in k or "loss" in k:
            if v > thresh: fail(f"{k}={v:.4f} exceeds max {thresh}")
        else:
            if v < thresh: fail(f"{k}={v:.4f} below min {thresh}")

    # Compare with baseline (if present)
    for k, limit in policy.items():
        if k in curr and k in base:
            c = curr[k]; b = base[k]
            rd = rel_delta(c, b)
            # For metrics where lower-is-better (ppl), a positive rd means worse
            lower_is_better = ("ppl" in k) or ("loss" in k)
            if lower_is_better:
                if rd > limit:
                    fail(f"{k} relative increase {rd:.4%} exceeds allowed {limit:.2%} (curr={c:.4f}, base={b:.4f})")
            else:
                # higher-is-better -> negative rd means drop
                if rd < limit:
                    fail(f"{k} relative drop {rd:.4%} worse than allowed {limit:.2%} (curr={c:.4f}, base={b:.4f})")

    print("No regressions detected.")
    sys.exit(0)

if __name__ == "__main__":
    main()
