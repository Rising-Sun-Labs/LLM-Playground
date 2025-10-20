## LLM_project structure:
        /llm-project
        ├── README.md
        ├── docs/
        ├── data/
        ├── src/
        │   ├── tokenizer/
        │   ├── transformer/
        │   ├── multimodal/
        │   ├── training/
        │   └── inference/
        ├── examples/
        └── requirements.txt


### How to run:

0) cd llm-project
```
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

1) Train tokenizer
```
python src/tokenizer/train_bpe.py --text data/tiny_corpus.txt --size 800 --out data/tokenizer.json
```
2) Train tiny LM
```
python src/training/train_lm.py --text data/tiny_corpus.txt \
  --tokenizer data/tokenizer.json --save runs/lm_demo --steps 300 --device cpu
```
3) Generate email
```
python src/inference/generate.py --tokenizer data/tokenizer.json \
  --ckpt runs/lm_demo/best.pt --prompt "Write a helpful onboarding email for new users:" --device cpu
```
4) Summarize an email
```
python src/inference/summarize.py --tokenizer data/tokenizer.json \
  --ckpt runs/lm_demo/best.pt --text "Hey team, here are the updates..." --device cpu
```
5) Multimodal caption demo
```
python src/training/train_mm_caption.py \
  --captions data/captions/captions.jsonl --images_root data/captions \
  --tokenizer data/tokenizer.json --save runs/mm_demo --steps 150 --device cpu
```
6) Local API (offline)
```
uvicorn src.server.fastapi_server:app --reload
```