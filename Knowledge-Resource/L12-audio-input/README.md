# Lesson 12: Audio input (speech & logs)
    - Audio for Assistant can listen to speech, logs, and calls. 
    - Feature extraction (MFCC/log-mel),
    - ASR encoder
    - projector -> LM
    - Streaming
    - VAD/diarization basics
    - training/eval
    - server



0) mental model (2 min)

    - Keep your decoder-only LM for text reasoning.
    - Add an audio encoder (e.g., small Conformer or QuartzNet-style CNN, or a simple stack of 1D convs + Transformer).
    - Convert raw waveform → log-mel spectrogram frames.
    - Map encoder outputs through a projector → LM hidden size; inject as a sequence of audio tokens after a <audio> marker, just like images.