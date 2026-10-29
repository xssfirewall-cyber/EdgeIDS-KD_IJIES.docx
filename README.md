# EdgeIDS-KD

Code and results for a study of knowledge distillation and INT8 quantization
for resource-constrained IoT intrusion detection, with a focus on what
compression does to rare attack classes.

The manuscript is under preparation and is not included in this repository.

## Contents

| Folder | What it holds |
|---|---|
| `src/data/` | dataset loaders and the leakage-controlled pipeline (identifier removal, de-duplication before splitting, overlap checks) |
| `src/` | teacher and student models, training objectives (CE, WCE, LA, KD, KD+WCE, RAP-KD, LA-KD, BT-KD), training loop, ONNX/INT8 utilities |
| `scripts/` | every experiment, statistic, table and figure |
| `results/` | per-run results (`runs/<dataset>/*.json`), summaries, quantization, cost-model and latency CSVs, statistics |
| `config.py` | dataset paths (overridable by environment variables) and global settings |

## Data

Download the five public datasets from their publishers and place them under
`data/raw/` (or point the environment variables in `config.py` to them):
TON_IoT (network subset), UNSW-NB15 (training and testing CSVs), N-BaIoT,
CICIoT2023 (official CSV parts) and the Edge-IIoTset ML subset.

## Setup

```bash
pip install -r requirements.txt
```

## Reproducing the results

| Step | Command | Produces |
|---|---|---|
| 1. Leakage-controlled splits | `python scripts/prepare_data.py` | `data_cache/*.npz`, `*_report.json` |
| 2. Teachers and CE/WCE students | `python scripts/run_week2.py` | `results/runs/`, teacher checkpoints |
| 3. Size sweep (7 sizes x 7 objectives x 3 seeds) | `python scripts/run_sweep.py` | `results/runs/` |
| 4. Balanced-teacher study | `python scripts/run_balanced.py` | `results/runs/` |
| 5. INT8 quantization | `python scripts/run_quant.py`, then `python scripts/run_quant_fpin.py` | `results/quant/` |
| 6. Hardware-independent cost | `python scripts/cost_model.py` | `results/cost_model.csv` |
| 7. Single-core latency | `python scripts/bench_edge.py` | `results/edge_bench.csv` |
| 8. Statistics | `python scripts/stats.py` | `results/stats.csv` |
| 9. Summary tables and figures | `python scripts/summarize.py`, `python scripts/make_tables.py`, `python scripts/plot_evidence.py` | tables and figures |

Steps 2-5 need a CUDA GPU (an 8 GB laptop GPU was used). Batch scripts skip
runs whose result file already exists, and `scripts/run_retry.ps1` restarts
a run after a CUDA or memory error.
