import yaml, time, os
from .utils import set_seed, ensure_dir, read_jsonl, write_json, load_json
from .tasks_lm import eval_lm_ppl
from .tasks_classify import eval_classify_acc
from .metrics import contains_norm, exact_match

def run_all(config_path="config.yaml", ckpt="miniLM.pt", tokenizer="tokenizer.json", device="cpu"):
    cfg = yaml.safe_load(open(config_path,"r",encoding="utf-8"))
    set_seed(cfg.get("seed",123))
    results_dir = cfg.get("results_dir","results")
    ensure_dir(results_dir)
    all_metrics = {}
    # 1) LM PPL
    t_lm = cfg["tasks"].get("lm_perplexity", {})
    if t_lm.get("enabled", True):
        m = eval_lm_ppl(ckpt, tokenizer, t_lm["data_path"], max_len=t_lm.get("max_len",256),
                        batch_size=t_lm.get("batch_size",8), device=device)
        all_metrics.update({"lm.ppl": m["lm.ppl"]})
    # 2) Classification
    t_clf = cfg["tasks"].get("classification", {})
    if t_clf.get("enabled", True):
        m = eval_classify_acc(ckpt, tokenizer, t_clf["data_path"], device=device,
                              normalize_labels=t_clf.get("normalize_labels",True))
        all_metrics.update({"classify.acc": m["classify.acc"]})
    # 3) Golden
    t_gold = cfg["tasks"].get("golden", {})
    if t_gold.get("enabled", True):
        total=0; passed=0
        for rec in read_jsonl(t_gold["data_path"]):
            total+=1
            # naive generation using runner-time adapter: we'll call LM generate via loaders.load_lm
            from .loaders import load_lm, load_tokenizer
            model,_ = load_lm(ckpt, device=device)
            tok = load_tokenizer(tokenizer)
            # generate
            ids = tok.encode(rec["prompt"], add_special=True, max_len=256)
            import torch
            x = torch.tensor([ids], dtype=torch.long, device=device)
            y = model.generate(x, max_new_tokens=t_gold.get("max_new_tokens",64),
                               temperature=t_gold.get("temperature",0.9),
                               top_k=t_gold.get("top_k",40))[0].tolist()
            out = tok.decode(y)
            typ = rec.get("type","contains")
            exp = rec["expect"]
            ok = exact_match(out,exp) if typ=="exact" else contains_norm(out,exp)
            passed += 1 if ok else 0
        pass_rate = passed/max(total,1)
        all_metrics.update({"golden.pass_rate": pass_rate})

    # Save
    stamp=int(time.time())
    out_path=os.path.join(results_dir, f"results_{stamp}.json")
    write_json(out_path, {"timestamp": stamp, "metrics": all_metrics})
    return out_path, all_metrics, cfg
