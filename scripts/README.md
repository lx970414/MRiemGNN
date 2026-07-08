# Reproduction Scripts

These PowerShell scripts collect the most useful dataset preparation and best-result commands from `README.md`, `DATASETS.md`, and `EXPERIMENTS.md`.

Run them from the repository root:

```powershell
cd path\to\MRiemGNN
.\scripts\prepare_mag_paper_scale.ps1
.\scripts\run_best_nodecls.ps1 -Dataset imdb
```

The scripts assume dependencies are installed and that required upstream raw datasets have already been downloaded where needed.

Available datasets for `run_best_nodecls.ps1`:

- `taobao`
- `mag`
- `imdb`
- `alibaba`

