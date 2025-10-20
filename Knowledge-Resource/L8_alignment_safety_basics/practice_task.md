### Practice:

- Create **10-50 SFT** chat examples targeting **your use-cases** (code review, debugging, email).
- Sample a few outputs.
- Create `200 - 1000 DPO` pairs (small is okay if high quality); 
- Add safety goldens and a **refusal policy**; run evals and confirm pass rules. 
- Update your **FastAPI** server to load `dop.pt` and enforce **max tokens + stop sequences**. 

### Stretch:
 - Add **function-calling** examples to SFT and extend server to accept detected tool calls.
 - Build a small pipeline that auto-redacts secrets from tool outputs before passing to the model.
