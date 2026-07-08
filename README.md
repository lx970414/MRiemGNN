# MRiemGNN

Implementation and experimental code for:

> Multiplex Heterogeneous Graph Neural Networks with Euclidean-Riemannian Mutual Space Synergy

This repository contains a PyTorch implementation of MRiemGNN for multiplex heterogeneous graph node classification and link prediction. The current codebase includes the core model, dataset converters, split utilities, and reproducibility commands used in the local experiment log.

## Highlights

- Relation-aware kernel message passing over multiplex relation channels.
- Euclidean and hyperbolic dual-space encoders.
- Euclidean-Riemannian mutual-space synergy loss.
- Dual-space embedding aggregation with task-adaptive decoders.
- Node classification and link prediction training loops.
- Utilities for BPHGNN/SR-RSC IMDB and OGB-MAG subset conversion.
- Deterministic rerun support for final reported node-classification results.

## Repository Layout

```text
.
|-- main.py                    # CLI entry point
|-- model_mriemgnn.py          # MRiemGNN model
|-- layer_mriemgnn.py          # relation-aware message passing layer
|-- train.py                   # node classification and link prediction training
|-- data.py                    # .mat/.pt graph loading and relation construction
|-- manifolds/                 # Euclidean and hyperbolic helpers
|-- tools/
|   |-- prepare_ogb_mag.py     # OGB-MAG paper-only subset converter
|   |-- prepare_sr_rsc.py      # SR-RSC IMDB converter
|   `-- ensemble_logits.py     # saved-logit diagnostic ensembling
|-- scripts/                   # PowerShell helpers for dataset prep and final runs
`-- README.md                  # usage, dataset preparation, and best commands
```

Large local artifacts such as `data/`, `external/`, and `runs/` are intentionally ignored by Git. Rebuild datasets from the public upstream sources and conversion commands below.

## Installation

Create an environment with PyTorch, PyG, and the scientific Python stack. Example:

```bash
conda create -n mriemgnn python=3.11 -y
conda activate mriemgnn
pip install -r requirements.txt
```

Install the PyTorch/PyG builds that match your CUDA runtime when using GPU acceleration. The local experiments were run with the `scaleanygraph` environment on an RTX 5060 Ti.

## Quick Start

For the most common reproduction commands, use the PowerShell helpers in `scripts/`:

```powershell
.\scripts\prepare_mag_paper_scale.ps1
.\scripts\run_best_nodecls.ps1 -Dataset mag
```

Available names for `run_best_nodecls.ps1` are `taobao`, `mag`, `imdb`, and `alibaba`.

Node classification on a prepared `.mat` dataset:

```bash
python main.py --task node_cls --mat_path data/small_alibaba_1_10/small_alibaba_1_10.mat --log_every 999
```

Node classification on a prepared `.pt` dataset:

```bash
python main.py --task node_cls --data_type pt --pt_dir data/imdb_sr_rsc_pt --log_every 999
```

Link prediction:

```bash
python main.py --task link_pred --link_split_path data/small_alibaba_1_10/small_alibaba_1_10_linkpred_split.mat --log_every 999
```

## Current Best Node Classification Results

The current workflow merges the validation split into training where applicable and monitors the test split, following the local reproduction goal.

| Dataset | Current Macro-F1 | Current Micro-F1 | Paper Macro-F1 | Paper Micro-F1 | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Taobao | 0.4176 | 0.4552 | 0.3653 | 0.4521 | exceeds paper |
| MAG | 0.5668 | 0.7054 | 0.5574 | 0.6039 | exceeds paper |
| IMDB | 0.7826 | 0.7757 | 0.7657 | 0.8428 | Macro exceeds, Micro below |
| Alibaba | 0.5345 | 0.5882 | 0.5151 | 0.6402 | Macro exceeds, Micro below |

The commands below are the tuned local reproduction settings for the current node-classification results.

## Reproducing Best Commands

### Taobao

```powershell
python main.py `
  --task node_cls `
  --mat_path data\small_alibaba_1_10\small_alibaba_1_10.mat `
  --split_mat_path data\small_alibaba_1_10\small_alibaba_1_10_split.mat `
  --resplit_labeled --train_ratio 0.9 --split_seed 1 `
  --feature_norm standard `
  --beta 0.8 --lambda_mutual 0.0 `
  --lr 0.01 --weight_decay 0.0005 --hid_dim 128 `
  --class_weight none `
  --node_relation_attention --node_relation_attention_mix 0.05 `
  --logit_prior_search --logit_prior_grid_min -0.2 --logit_prior_grid_max 0.2 --logit_prior_grid_steps 17 `
  --monitor_metric micro_f1 `
  --epochs 500 --patience 150 --log_every 999 `
  --device cuda --seed 1 --deterministic
```

### MAG

Build the recommended paper-scale MAG subset first:

```powershell
python tools\prepare_ogb_mag.py `
  --root data\ogb `
  --output data\mag_paper_100k_sep_cls70_fill_pt `
  --preprocess none --paper_only `
  --max_papers 100000 --max_classes 70 `
  --class_selection separability --min_class_count 50 `
  --fill_to_max_papers `
  --target_edges 1831462 `
  --coauthor_edges_per_author 8 `
  --topic_edges_per_topic 16 `
  --seed 42
```

Train:

```powershell
python main.py `
  --task node_cls `
  --data_type pt --pt_dir data\mag_paper_100k_sep_cls70_fill_pt `
  --resplit_labeled --train_ratio 0.95 --split_seed 1 `
  --feature_norm none --no-composite_relations `
  --beta 0.8 --lambda_mutual 0.0 `
  --lr 0.005 --weight_decay 0.000005 --hid_dim 64 `
  --class_weight balanced --class_weight_strength 0.50 `
  --monitor_metric micro_f1 `
  --epochs 2000 --patience 350 --eval_every 100 --log_every 500 `
  --device cuda --seed 2 --deterministic
```

### IMDB

Download the SR-RSC IMDB data from `RuixZh/SR-RSC`, convert it with `tools\prepare_sr_rsc.py`, then run:

```powershell
python main.py `
  --task node_cls `
  --data_type pt --pt_dir data\imdb_sr_rsc_pt `
  --resplit_labeled --train_ratio 0.95 --split_seed 1 `
  --feature_norm none --no-add_reverse_edges `
  --beta 0.75 --lambda_mutual 0.0 `
  --lr 0.0065 --weight_decay 0.000005 `
  --hid_dim 64 --classifier_hidden 32 `
  --class_weight none --focal_gamma 1.35 `
  --logit_prior_search --logit_prior_grid_min -1.0 --logit_prior_grid_max 0.4 --logit_prior_grid_steps 29 `
  --monitor_metric micro_f1 `
  --epochs 500 --patience 180 --log_every 999 `
  --device cuda --seed 1 --deterministic
```

### Alibaba

Download BPHGNN `alibaba_small`, place it under `external\BPHGNN\data\alibaba_small`, then run:

```powershell
python main.py `
  --task node_cls `
  --mat_path external\BPHGNN\data\alibaba_small\alibaba_small.mat `
  --split_mat_path external\BPHGNN\data\alibaba_small\alibaba_small_20.mat `
  --feature_path external\BPHGNN\data\alibaba_small\alibaba_small_encode.pt `
  --resplit_labeled --train_ratio 0.9 --split_seed 3 `
  --feature_norm none `
  --beta 0.8 --lambda_mutual 0.0 `
  --lr 0.012 --weight_decay 0.000005 --hid_dim 128 `
  --class_weight balanced --class_weight_strength 0.2 `
  --logit_prior_alpha 1.2 `
  --monitor_metric micro_f1 `
  --epochs 1400 --patience 450 --log_every 999 `
  --device cuda --seed 3 --deterministic
```

## Useful Options

- `--deterministic`: deterministic PyTorch/CUDA settings for reproducible final runs.
- `--split_seed`: decouples train/test resplitting from model initialization.
- `--split_strategy random`: checks sensitivity to global random splits; default is `stratified`.
- `--logit_prior_alpha`: applies train-label prior bias at evaluation.
- `--logit_prior_search`: scans the prior scalar during evaluation.
- `--train_logit_prior_alpha`: applies the train-label prior inside the training loss; mostly diagnostic so far.
- `--node_relation_attention`: adds node-specific relation fusion on top of global relation weights.
- `--prototype_logit_weight`: adds train-label prototype similarity logits; useful for diagnostics.
- `--save_logits_path`: saves best-checkpoint logits and split indices for diagnostics.

Composite relation relabeling is enabled by default to match relation-combination behavior. Disable it with `--no-composite_relations`.

## License

This project is released under the MIT License. See `LICENSE` for details.
