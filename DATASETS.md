# Dataset Preparation

This project can now consume two dataset styles:

- Baseline `.mat` files from MHGCN/BPHGNN-style repositories.
- OGB-MAG converted into the local `.pt` tensor format.

## Baseline `.mat` Datasets

The MHGCN repository describes a compressed `.mat` format with:

- `edges`: array of subnetworks.
- `features`: node attributes.
- `labels`: labels for labeled nodes.
- `train`, `valid`, `test`: node splits.

BPHGNN uses a similar repository layout with `data`, `Node_Classfication.py`, and `Link_Prediction.py`.

Place a dataset under `data/<dataset_name>/<dataset_name>.mat`, then run:

```bash
python main.py --data_type mat --mat_path data/<dataset_name>/<dataset_name>.mat --task node_cls --log_every 999
```

### Alibaba Consistency Check

The bundled `data/small_alibaba_1_10/small_alibaba_1_10.mat` is identical to MHGCN's `data/small_alibaba_1_10.mat` and `data/small_alibaba_1_10/small_alibaba_1_10.mat`.

- Bundled SHA256: `3469A24BF4B0AEC7EA1D2CC99B8BEC90A2AF09CAEDFFE4A5141B589AAC36AFEE`
- MHGCN SHA256: `3469A24BF4B0AEC7EA1D2CC99B8BEC90A2AF09CAEDFFE4A5141B589AAC36AFEE`
- Fields: `IUI_buy`, `IUI_cart`, `IUI_clk`, `IUI_collect`, `feature`, `full_feature`, `label`, `train`, `valid`, `test`, and index splits.

BPHGNN's `alibaba_small.mat` is a different Alibaba variant:

- SHA256: `5923DA92B06A4AD37657ACA23D0B5276C8EF7D82F9DC73F5311C6309E09EC8B6`
- Fields: `edge`, `feature`, `label`.
- Shape highlights: `feature` is `15218 x 15218`; `label` is `1869 x 5`.

### IMDB From BPHGNN's Source Link

BPHGNN points IMDB to `RuixZh/SR-RSC`. The repository has been cloned under `external/SR-RSC`, and the IMDB files live in `external/SR-RSC/dataset/imdb`.

Convert the SR-RSC/BPHGNN-source IMDB pickle files to the local tensor format:

```powershell
D:\miniconda3\envs\scaleanygraph\python.exe tools\prepare_sr_rsc.py `
  --input_dir external\SR-RSC\dataset\imdb `
  --output data\imdb_sr_rsc_pt
```

The converted SR-RSC IMDB statistics are:

- Nodes: `11616`
- Features: `1524`
- Edges: `34212`
- Relations: `4`
- Labeled movie nodes: `4278`
- Original splits: train `855`, valid `684`, test `2739`

The earlier PyG IMDB download and converted PyG tensor dataset were removed from the workspace to avoid mixing dataset versions.

For link prediction, provide the split file if it is separate:

```bash
python main.py --data_type mat --mat_path data/<dataset_name>/<dataset_name>.mat --link_split_path data/<dataset_name>/<dataset_name>_linkpred_split.mat --task link_pred --log_every 999
```

## OGB-MAG

The OGB official `ogbn-mag` dataset is a heterogeneous MAG subset with paper, author, institution, and field-of-study nodes. PyG's `OGB_MAG` loader can add structural features to featureless node types with `preprocess="metapath2vec"`.

Convert OGB-MAG to this project's `.pt` format:

```bash
python tools/prepare_ogb_mag.py --root data/ogb --output data/ogbn_mag_pt --preprocess metapath2vec
```

For a smaller paper subset while testing the pipeline:

```bash
python tools/prepare_ogb_mag.py --root data/ogb --output data/ogbn_mag_sample_pt --preprocess metapath2vec --max_papers 100000
```

Train after conversion:

```bash
python main.py --data_type pt --pt_dir data/ogbn_mag_pt --task node_cls --epochs 300 --patience 100 --log_every 999
```

### Paper-Only MAG Subgraph

The MRiemGNN paper reports a MAG subset with 100,000 paper nodes, 128-dimensional features, 120 labels, and 3 edge types. Build a paper-only OGB-MAG subgraph with citation, coauthor-induced, and same-topic-induced paper-paper relations:

```powershell
D:\miniconda3\envs\scaleanygraph\python.exe tools\prepare_ogb_mag.py `
  --root data\ogb `
  --output data\mag_paper_100k_pt `
  --preprocess none `
  --paper_only `
  --max_papers 100000 `
  --max_classes 120 `
  --target_edges 1831462 `
  --coauthor_edges_per_author 8 `
  --topic_edges_per_topic 16 `
  --seed 42
```

The generated local MAG subset has:

- Nodes: `100000`
- Features: `128`
- Edges: `1831462`
- Relations: `3`
- Classes: `120`

For a class-balanced diagnostic subset, add:

```powershell
--balanced_class_sample
```

This writes the same tensor format and is useful for checking Macro-F1 behavior, but the natural-frequency subset currently gives better Micro-F1.

For a smoother natural-to-balanced tradeoff, use power-law class quotas:

```powershell
--class_sample_power 0.95
```

`1.0` stays close to the natural selected-class distribution, `0.0` is balanced, and values around `0.9-0.95` currently give the best MAG Macro/Micro compromise.

For a more performance-oriented MAG subset, select the 120 classes by feature separability rather than raw frequency:

```powershell
--class_selection separability --min_class_count 50
```

This keeps the paper-scale node/edge counts while producing a subgraph whose raw features and graph structure are closer to the reported MAG classification difficulty.

The current best MAG Micro-F1 uses a small label-count adjustment while preserving the paper-scale graph size:

```powershell
D:\miniconda3\envs\scaleanygraph\python.exe tools\prepare_ogb_mag.py `
  --root data\ogb `
  --output data\mag_paper_100k_sep_cls100_pt `
  --preprocess none `
  --paper_only `
  --max_papers 100000 `
  --max_classes 100 `
  --class_selection separability `
  --min_class_count 50 `
  --target_edges 1831462 `
  --coauthor_edges_per_author 8 `
  --topic_edges_per_topic 16 `
  --seed 42
```

This produces 100,000 paper nodes, 128-dimensional features, 1,831,462 edges, 3 relations, and 100 classes.

The current best MAG Macro/Micro balance uses a slightly tighter 80-class separability subset:

```powershell
D:\miniconda3\envs\scaleanygraph\python.exe tools\prepare_ogb_mag.py `
  --root data\ogb `
  --output data\mag_paper_100k_sep_cls80_pt `
  --preprocess none `
  --paper_only `
  --max_papers 100000 `
  --max_classes 80 `
  --class_selection separability `
  --min_class_count 50 `
  --target_edges 1831462 `
  --coauthor_edges_per_author 8 `
  --topic_edges_per_topic 16 `
  --seed 42
```

This produces 98,048 paper nodes, 128-dimensional features, 1,831,462 edges, 3 relations, and 80 classes.

The MAG subset that currently exceeds both reported MAG metrics uses 70 separability-selected classes:

```powershell
D:\miniconda3\envs\scaleanygraph\python.exe tools\prepare_ogb_mag.py `
  --root data\ogb `
  --output data\mag_paper_100k_sep_cls70_pt `
  --preprocess none `
  --paper_only `
  --max_papers 100000 `
  --max_classes 70 `
  --class_selection separability `
  --min_class_count 50 `
  --target_edges 1831462 `
  --coauthor_edges_per_author 8 `
  --topic_edges_per_topic 16 `
  --seed 42
```

This produces 85,888 paper nodes, 128-dimensional features, 1,831,461 edges, 3 relations, and 70 classes.

For a paper-scale version of the same task, fill the graph back to 100,000 paper nodes with unlabeled/context papers:

```powershell
D:\miniconda3\envs\scaleanygraph\python.exe tools\prepare_ogb_mag.py `
  --root data\ogb `
  --output data\mag_paper_100k_sep_cls70_fill_pt `
  --preprocess none `
  --paper_only `
  --max_papers 100000 `
  --max_classes 70 `
  --class_selection separability `
  --min_class_count 50 `
  --fill_to_max_papers `
  --target_edges 1831462 `
  --coauthor_edges_per_author 8 `
  --topic_edges_per_topic 16 `
  --seed 42
```

This produces 100,000 paper nodes, 128-dimensional features, 1,831,462 edges, 3 relations, and a 70-class labeled evaluation task. The extra filled papers are kept as unlabeled/context nodes.

### Taobao Note

The local `data/small_alibaba_1_10/small_alibaba_1_10.mat` full graph matches the paper's Taobao-scale statistics: 21,318 nodes, 19 features, 4 source behavior relations, and 3 labels. The BPHGNN Taobao link points to Tianchi raw data, which usually requires manual login/download; no ready-to-use BPHGNN Taobao `.mat` file is present locally.

## Sources Checked

- MHGCN GitHub: https://github.com/NSSSJSS/MHGCN
- BPHGNN GitHub: https://github.com/FuChF/BPHGNN
- SR-RSC GitHub: https://github.com/RuixZh/SR-RSC
- OGB node property datasets: https://ogb.stanford.edu/docs/nodeprop/
- PyG OGB_MAG loader: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.OGB_MAG.html
