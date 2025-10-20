# Practice Task:
    - Create dataflow/config.yaml for your sources and thresholds.
    - Implement ingest → dedupe → filter → normalize → split on a 10k-sample sandbox; produce metrics & a manifest.
    - Build SFT and DPO packs (separate manifests); record dataset IDs in a lightweight registry.
    - Wire your trainer to stream shards from the manifest and log loss by source.
    - Write a DATA_CARD.md from the metrics (template above).

# Stretch
    - Add curriculum packs (short→long).
    - Integrate active learning: send low-confidence model outputs to the HITL queue.
    - Stand up nightly pipeline that rebuilds datasets and runs evals; compare to prior run with a change report.