import csv, torch
from tqdm import tqdm
from .metrics import accuracy
from .loaders import load_tokenizer, load_lm

def _predict_label_from_lm(model, tok, text, device="cpu"):
    """
    Simple heuristic classifier using LM as a scorer for label-conditioned prompts.
    Replace this with your real classifier head if you have one.
    """
    labels = ["positive","negative"]
    best_label=None; best_score=float("inf")
    for lab in labels:
        prompt = f"Text: {text}\nLabel: {lab}\n"
        ids = tok.encode(prompt, add_special=True, max_len=128)
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)
            ce = torch.nn.CrossEntropyLoss(reduction="mean")
            score = ce(logits.view(-1, logits.size(-1)), y.view(-1)).item()
        if score < best_score: best_score=score; best_label=lab
    return best_label

def eval_classify_acc(ckpt_path, tokenizer_json, csv_path, device="cpu", normalize_labels=True):
    model, cfg = load_lm(ckpt_path, device=device)
    tok = load_tokenizer(tokenizer_json)
    y_true=[]; y_pred=[]
    with open(csv_path,"r",encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in tqdm(r, desc="Classify"):
            text=row["text"]; lab=row["label"]
            if normalize_labels: lab=lab.strip().lower()
            pred=_predict_label_from_lm(model,tok,text,device=device)
            y_true.append(lab); y_pred.append(pred)
    acc = accuracy(y_true, y_pred)
    return {"classify.acc": acc}
