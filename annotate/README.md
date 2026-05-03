# Annotate Review App

`annotate/` is a React/Vite fingering review UI.

It can now consume the rule-based hard-note output from `ManualCheck/` through
`prepare_review_data.py`.

## Quick start

1. Prepare a review bundle.

```bash
python annotate/prepare_review_data.py \
  --midi testvideo/sample.mid \
  --video testvideo/sample.mp4 \
  --audio testvideo/sample.wav
```

If you already have a fingering TSV, add it so the review queue is driven by
`ManualCheck` hard-note rules:

```bash
python annotate/prepare_review_data.py \
  --midi /path/to/recording.mid \
  --video /path/to/recording.mp4 \
  --audio /path/to/recording.wav \
  --tsv /path/to/recording.tsv \
  --piece "My Piece" \
  --difficulty hard
```

If you have a `fingering_pp.zip` with per-frame `{key_index, hand, finger}`
pickles, you can feed that directly. The script will auto-pick the best-matching
`.pkl` entry for the given MIDI/video pair and derive note-level seed labels:

```bash
python annotate/prepare_review_data.py \
  --midi testvideo/sample.mid \
  --video testvideo/sample.mp4 \
  --audio testvideo/sample.wav \
  --fingering-zip testvideo/fingering_pp.zip \
  --trial 2024-02-14_19-44-26 \
  --piece 2024-02-14_19-44-26
```

2. Launch the app.

```bash
cd annotate
npm install
npm run dev
```

3. Open the Vite URL, usually `http://localhost:5174`.

## Notes

- The script writes `annotate/public/data/notes.json`.
- Seed verdicts are written to `annotate/public/data/human_verdicts.json`.
- With a TSV, notes flagged by `ManualCheck/hard_part_selector.py` appear in the
  app's `Hard notes` review queue.
- With `--fingering-zip`, note-level labels are inferred from the selected
  pickle entry and then passed through the same hard-note rules.
- Without a TSV, the app still works as a full-piece manual annotation UI, but
  there is no hard-note prioritization.
