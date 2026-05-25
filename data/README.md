# Data

Keep raw datasets, text corpora, and checkpoints out of Git.

Suggested local files for this repository:

- `input.txt` - training text for the decoder-only language model.
- `shakespeare.txt` - source text for the speaker-classification experiment.

Large datasets should be downloaded locally and referenced in the project
scripts using `--data-file` or `--input-file` arguments.
