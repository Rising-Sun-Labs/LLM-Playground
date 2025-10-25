# tests/test_regression.py
import json, os, sys, copy, math
from pathlib import Path
import numpy as np
import torch
import pytest

# Make project importable when tests run from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.harness.simple_transformer import MiniTransformerLM
from eval.harness.utils import write_json
from eval.harness.runner import run_all
import eval.compare_results as cmp

# ---- Tiny tokenizer + data helpers ----
def write_tiny_tokenizer(path: Path):
    vocab = {"<pad>":0,"<bos>":1,"<eos>":2,"<unk":3,"<unk>":3,"</w>":4,
             "hello":5,"world":6,"good":7,"bad":8}
    write_json(path, {"vocab": vocab})

def write_tiny_heldout(path: Path):
    # simple repetitive corpus so PPL is measurable
    text = "hello world hello world good world hello good\n" * 20
    path.write_text(text, encoding="utf-8")

def write_tiny_config(path: Path, results_dir: Path, heldout_path: Path, classify_path: Path, golden_path: Path):
    cfg = {
        "seed": 123,
        "results_dir": str(results_dir),
        "regression_policy": {
            "max_relative_drop": {
                "lm.ppl": 0.00,          # no PPL increase allowed
                "classify.acc": -0.01,   # allow <=1% drop
                "golden.pass_rate": 0.00
            }
        },
        "absolute_thresholds": {
            "lm.ppl": 200.0,            # generous absolute gates for the tiny toy
            "classify.acc": 0.0,
            "golden.pass_rate": 0.0
        },
        "tasks": {
            "lm_perplexity": {
                "enabled": True, "data_path": str(heldout_path),
                "max_len": 32, "batch_size": 8
            },
            "classification": {
                "enabled": False, "data_path": str(classify_path), "normalize_labels": True
            },
            "golden": {
                "enabled": False, "data_path": str(golden_path),
                "max_new_tokens": 32, "temperature": 0.9, "top_k": 20
            }
        }
    }
    write_json(path, cfg)

def save_ckpt(path: Path, model: MiniTransformerLM, vocab_size: int, d_model=64, n_layers=1, n_heads=4, d_mlp=128, max_len=32):
    obj = {
        "model": model.state_dict(),
        "config": {"vocab_size": vocab_size, "d_model": d_model, "n_layers": n_layers,
                   "n_heads": n_heads, "d_mlp": d_mlps, "max_len": max_len}
    }
    # fix small typo if any
    obj["config"]["d_mlp"] = obj["config"].pop("d_mlps", d_mlps) if "d_mlps" in obj["config"] else d_mlps
    torch.save(obj, str(path))

# ---- Fixtures ----
@pytest.fixture()
def tmp_eval_env(tmp_path):
    """
    Build a tiny self-contained eval environment in a temp dir:
    - tokenizer.json
    - config.json (yaml not needed here; runner accepts yaml but we use json writer for simplicity)
    - data files
    - good checkpoint and baseline.json
    """
    work = tmp_path / "e2e"
    (work / "eval" / "data").mkdir(parents=True, exist_ok=True)
    (work / "eval" / "results").mkdir(parents=True, exist_ok=True)
    (work / "eval" / "baselines").mkdir(parents=True, exist_ok=True)

    tok = work / "tokenizer.json"
    write_tiny_tokenizer(tok)

    heldout = work / "eval" / "data" / "heldout_lm.txt"
    write_tiny_heldout(heldout)

    # dummy CSV & golden files (disabled in this test, but keep present)
    classify = work / "eval" / "data" / "classify.csv"
    classify.write_text("text,label\nhello world,positive\nbad world,negative\n", encoding="utf-8")

    golden = work / "eval" / "data" / "golden_prompts.jsonl"
    golden.write_text('{"prompt":"say hello","expect":"hello","type":"contains"}\n', encoding="utf-8")

    # Build config (JSON via our helper, runner loads YAML; for test we’ll write YAML-like JSON)
    cfg = {
        "seed": 123,
        "results_dir": str(work / "eval" / "results"),
        "regression_policy": { "max_relative_drop": {"lm.ppl": 0.00}},
        "absolute_thresholds": { "lm.ppl": 9999.0 },
        "tasks": {
            "lm_perplexity": {
                "enabled": True, "data_path": str(heldout),
                "max_len": 32, "batch_size": 8
            },
            "classification": { "enabled": False, "data_path": str(classify), "normalize_labels": True },
            "golden": { "enabled": False, "data_path": str(golden), "max_new_tokens": 16, "temperature": 0.9, "top_k": 10 }
        }
    }
    # write as YAML-compatible text
    import yaml
    (work / "eval" / "config.yaml").write_text(yaml.dump(cfg), encoding="utf-8")

    # Build a tiny random model & save as "good" (we’ll also overfit it a bit so PPL lowers)
    vocab_size = len(json.load(open(tok))["vocab"])
    good = MiniTransformerLM(vocab_size=vocab_size, d_model=64, n_layers=1, n_heads=4, d_mlp=128, max_len=32)
    # quick micro-train on the heldout to make it slightly better than random
    text_ids = []
    # mimic SimpleBPE encode: tokens are words + '</w>' markers
    words = "hello world hello world good world".split()
    # build a trivial id map from tokenizer
    v = json.load(open(tok))["vocab"]
    for _ in range(30):
        seq=[]
        for w in words:
            seq.append(v.get(w, v.get("<unk>",3))); seq.append(v.get("</w>",4))
        text_ids.extend(seq)
    x = torch.tensor([text_ids[:31]], dtype=torch.long)
    y = torch.tensor([text_ids[1:32]], dtype=torch.long)
    opt = torch.optim.AdamW(good.parameters(), lr=5e-3)
    ce = torch.nn.CrossEntropyLoss()
    for _ in range(40):
        opt.zero_grad()
        logits = good(x)
        loss = ce(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward(); opt.step()

    good_ckpt = work / "miniLM_good.pt"
    torch.save({"model": good.state_dict(),
                "config": {"vocab_size": vocab_size, "d_model":64,"n_layers":1,"n_heads":4,"d_mlp":128,"max_len":32}},
               str(good_ckpt))

    # Create a baseline by running the harness with the good checkpoint
    from eval.harness.runner import run_all
    out_path, metrics, _ = run_all(str(work/"eval"/"config.yaml"), str(good_ckpt), str(tok), device="cpu")
    # Use the produced result as baseline
    baseline = work / "eval" / "baselines" / "baseline.json"
    Path(baseline).write_text(Path(out_path).read_text(), encoding="utf-8")

    return {
        "work": work, "tok": tok, "heldout": heldout, "config": work/"eval"/"config.yaml",
        "good_ckpt": good_ckpt, "baseline": baseline
    }

def perturb_weights(model: MiniTransformerLM, std=0.5):
    with torch.no_grad():
        for p in model.parameters():
            if p.dtype.is_floating_point:
                p.add_(std * torch.randn_like(p))

# ---- Tests ----

def test_regression_detection_by_weight_perturbation(tmp_eval_env, monkeypatch):
    """
    1) Use the 'good' checkpoint to create a baseline (already in fixture).
    2) Create a 'bad' checkpoint by adding strong noise to weights -> higher PPL.
    3) Run eval on bad checkpoint; compare to baseline; expect SystemExit(1) from compare_results.
    """
    env = tmp_eval_env
    tok = env["tok"]
    good_ckpt = env["good_ckpt"]
    config = env["config"]
    baseline = env["baseline"]

    # Load good model and perturb
    obj = torch.load(good_ckpt, map_location="cpu")
    cfg = obj["config"]
    bad = MiniTransformerLM(**cfg)
    bad.load_state_dict(obj["model"])
    perturb_weights(bad, std=1.0)  # strong noise to ensure worse PPL
    bad_ckpt = env["work"] / "miniLM_bad.pt"
    torch.save({"model": bad.state_dict(), "config": cfg}, str(bad_ckpt))

    # Run eval on bad checkpoint
    out_path, curr_metrics, _ = run_all(str(config), str(bad_ckpt), str(tok), device="cpu")

    # Expect compare_results to FLAG regression (SystemExit with code 1)
    with pytest.raises(SystemExit) as e:
        cmp.main.__wrapped__ if hasattr(cmp.main, "__wrapped__") else cmp.main()  # support pytest wrapping
    # If main() expects CLI args, simulate them:
    # We'll call it via argument injection:
    # But cmp.main reads argparse directly; emulate by setting sys.argv
    # Instead, call it in a subprocess-free way:
    import importlib, types
    import argparse
    # Rebuild parser call
    parser = argparse.ArgumentParser()
    # Re-import for clean state with custom args
    import importlib, importlib.util
    # Simpler: call compare_results via its functions by re-parsing:
    # Fallback: Call via CLI style
    old_argv = sys.argv[:]
    try:
        sys.argv = ["compare_results.py", "--current", str(out_path), "--baseline", str(baseline), "--config", str(config)]
        with pytest.raises(SystemExit) as exc:
            cmp.main()
        assert exc.value.code == 1
    finally:
        sys.argv = old_argv

@pytest.mark.slow
def test_regression_detection_by_high_lr(tmp_eval_env):
    """
    Optional realistic demo:
    Train a 'good' tiny model with reasonable LR, then a 'bad' one with too-high LR & too few steps.
    Expect regression flagged. Marked slow; run with:  pytest -k high_lr
    """
    env = tmp_eval_env
    tok = env["tok"]
    baseline = env["baseline"]
    config = env["config"]

    vocab_size = len(json.load(open(env["tok"]))["_vocab"],) if False else len(json.load(open(env["tok"]))["vocab"])

    # Train a 'bad' model quickly with high LR
    bad = MiniTransformerLM(vocab_size=vocab_size, d_model=64, n_layers=1, n_heads=4, d_mlp=128, max_len=32)
    # Reuse same tiny training loop but with high LR + fewer steps
    words = "hello world bad bad bad world".split()
    v = json.load(open(env["tok"]))["vocab"]
    seq=[]
    for _ in range(10):
        for w in words:
            seq.append(v.get(w, v.get("<unk>",3))); seq.append(v.get("</w>",4))
    x = torch.tensor([seq[:31]], dtype=torch.long)
    y = torch.tensor([seq[1:32]], dtype=torch.long)
    opt = torch.optim.AdamW(bad.parameters(), lr=0.2)  # intentionally high
    ce = torch.nn.CrossEntropyLoss()
    for _ in range(10):
        opt.zero_grad()
        logits = bad(x)
        loss = ce(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward(); opt.step()

    bad_ckpt = env["work"] / "miniLM_bad_lr.pt"
    torch.save({"model": bad.state_dict(),
                "config": {"vocab_size": vocab_size, "d_model":64,"n_layers":1,"n_heads":4,"d_mlp":128,"max_len":32}},
               str(bad_ckpt))

    out_path, curr_metrics, _ = run_all(str(config), str(bad_ckpt), str(tok), device="cpu")

    # Compare with baseline; expect regression (SystemExit 1)
    old_argv = sys.argv[:]
    try:
        sys.argv = ["compare_results.py", "--current", str(out_path), "--baseline", str(baseline), "--config", str(config)]
        with pytest.raises(SystemExit) as exc:
            cmp.main()
        assert exc.value.code == 1
    finally:
        sys.argv = old_argv
