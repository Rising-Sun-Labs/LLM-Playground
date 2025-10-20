### Practice:
    - Export **TorchScript** and **ONNX** successfully from your trained checkpoint.
    - Launch the FastAPI server and call **/generate** from **curl** or a tiny client.
    - Try the streaming endpoint; confirm tokens arrive incrementally.
    - If CPU-only qunatize and measure latency before / after.

- Stretch: add a `/logits` endpoint that returns raw logits for research, or add **top-p** nucleus sampling to the server.


