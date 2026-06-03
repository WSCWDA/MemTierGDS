# Criteo data download note

This benchmark does **not** automatically download Criteo data.

- Kaggle Display Advertising Challenge: download `train.txt` from Kaggle after
  accepting the dataset terms, then pass it to `scripts/preprocess_criteo_trace.py
  --input /path/to/train.txt`.
- Criteo Terabyte: download day files (`day_0` ... `day_23`) from the official
  source after accepting the terms, then pass them with
  `--input-glob "/path/to/day_*"`.

The preprocessing scripts stream text rows and use only the 26 categorical
features for embedding lookup trace generation.
