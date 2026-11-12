# Data

The dataset is **not published** in this repository (comments collected via the
Google API and partly derived from ViHSD, which is licensed for research only).

To reproduce, place the splits here:

```
data/raw/train.csv   # columns: text,label  (label ∈ {clean, offensive, hate})
data/raw/val.csv
data/raw/test.csv
```

The 2026 YouTube comments used for test set 2 and for the v6/v7 training rows are not stored here
either. Their public YouTube ids and labels are in [`labels/`](../labels/), so the texts can be
fetched again with `scripts/build_testset_pool.py` and a YouTube Data API key.
