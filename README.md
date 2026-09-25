# Amazon ML Challenge 2026 — Business Entity Resolution

Team repo for the entity-resolution challenge (match noisy Source 2 /
Source 3 business records to Source 1 reference entities).

## Layout

```
.
├── dataset/
│   ├── sample/       tiny synthetic fixture, committed — for smoke-testing only
│   ├── train/        drop the real train_*.tsv files here (gitignored)
│   └── test/         drop the real test_*.tsv files here (gitignored)
├── code/
│   └── business_entity_resolution/
│       ├── src/            all pipeline code — see its own README
│       ├── README.md       exact run instructions
│       └── requirements.txt
├── output/            matching_results.tsv + candidate_pairs.tsv land here
├── utils/
│   └── local_validator.py   local sanity check (NOT the official validator)
└── Documentation_template.md   methodology write-up — grab the real one from
                                 your official challenge kit and drop it in here
```

The `dataset/train/` and `dataset/test/` real data files, and
generated `output/*.tsv` files, are gitignored — they're either large
provided data or regenerable, not something to carry in git history.
They **do** need to be in the final submission zip though (see the
challenge's *Final Submission Package* spec) — that's a separate,
manual "zip it up" step at the end, not something this repo tracks.

## Quickstart

```bash
cd code/business_entity_resolution
pip install -r requirements.txt
cd src
python3 pipeline.py --sample     # smoke test — no real data needed
```

Once the real dataset is dropped into `dataset/train/` and
`dataset/test/`:

```bash
python3 pipeline.py              # full run
```

Full instructions, the F0.5 metric explanation, and how to extend
each stage: see [`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md).

## Team workflow

- One branch per teammate/experiment, PR into `main`.
- Land improvements to `blocking.py` (recall), `features.py`
  (signal), and `model.py`/threshold tuning (precision-recall
  trade-off) independently — the pipeline stitches them together.
- Before any leaderboard upload: run `utils/local_validator.py`, then
  the official `utils/validate_submission.py` from the challenge kit.
