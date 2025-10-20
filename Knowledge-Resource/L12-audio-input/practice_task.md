### Practice
    - Reseve <audio> special id in your tokenizer; document it.
    - Collect 50-200 short audio clips (16 kHZ, mono) + transcripts/instructions, build am_sft.jsonl.
    - Train Stage 1 SFT miniAM_sft_strage1.pt and transcript 10 clips; spot-check WER.
    - Add /transcribe to your server; time a 30-sec clip end-to-end.
    - unfreeze top 2 encoder layers; train stage 2 with a small LR (5e-5).

Stretch

    - Implement streaming with VAD; return partial transcripts with timestamps.
    - Add speaker diarization (basic): use an off-the-shelf embedding (x-vectors) to cluster speakers; prepend speaker tags to transcript.