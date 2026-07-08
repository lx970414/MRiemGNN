# GitHub Release Checklist

Use this checklist before publishing the repository.

## Required Before Public Release

- Confirm the MIT license copyright holder in `LICENSE`.
- Confirm the paper title, author list, venue, and citation format.
- Decide whether `EXPERIMENTS.md` should include all local exploratory results or a shorter reproducibility-focused subset.
- Verify that no private paths, credentials, or manually downloaded raw datasets are committed.
- Keep `data/`, `external/`, and `runs/` out of Git unless you intentionally publish small sample artifacts.

## Recommended Validation

```bash
python -m py_compile main.py train.py model_mriemgnn.py layer_mriemgnn.py data.py utils.py
python main.py --help
```

## Data

- `DATASETS.md` documents how to rebuild local IMDB, Alibaba, Taobao, and MAG inputs.
- Large generated `.pt` and `.mat` files are ignored by `.gitignore`.
- If you want a runnable demo without external downloads, add a small toy dataset under a separate tracked directory and update `README.md`.

## Reported Results

Current best local node-classification results:

| Dataset | Macro-F1 | Micro-F1 | Status |
| --- | ---: | ---: | --- |
| Taobao | 0.4176 | 0.4552 | exceeds paper |
| MAG | 0.5668 | 0.7054 | exceeds paper |
| IMDB | 0.7826 | 0.7757 | Macro exceeds paper, Micro below |
| Alibaba | 0.5345 | 0.5882 | Macro exceeds paper, Micro below |
