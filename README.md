# Quantifying evaluation protocol bias and label-space dependence in few-shot crop mapping

Code, data partitions and trained weights for the manuscript submitted to the
*International Journal of Remote Sensing*.

The repository measures what self-supervised pre-training actually contributes to
few-shot crop mapping once two artefacts of evaluation are separated from it: whether
support and query parcels come from the same split, and how many classes the probe
must separate.

| Finding | Value |
|---|---|
| Accuracy inflation from same-split sampling (5 / 10 / 20-shot) | 29.6 / 34.2 / 31.7 points |
| Pre-training gain, 5-way 20-shot vs full 124-class split | 7.7 vs 0.8 points |
| Reconstruction-loss reduction from the phenology-aware mask | 14% (0.412 to 0.355), no accuracy gain |
| Linear probe on raw time series vs pre-trained encoder | within 2.0 points |
| Peak GPU memory, all runs | 34 MB on a single RTX 2060 (6 GB) |

## Repository layout

```
configs/            hyper-parameters (data / pretrain / eval); no magic numbers in code
data/
  dataset.py        loading, resampling, normalisation, phenology mask, episodic sampling
  raw/latvia/       split.json, sampled_ids.json, stats.json   <- partition indices
  raw/estonia/      split.json, fewshot_5/10/20shot.json       <- partition indices
  raw/portugal/     split.json, fewshot_5/10/20shot.json       <- partition indices
models/ltae.py      L-TAE encoder, tiny = 9,472 / medium = 102,272 parameters
models/heads.py     masked-reconstruction head, linear probe head
utils/              seeding, metrics (OA / macro-F1), masked loss, AMP, OOM back-off
scripts/            entry points (see below)
checkpoints/        pre-trained weights, seed 42 plus seeds 43-45, norm_stats.json
outputs/            every JSON behind the tables and figures
docs/paper/         LaTeX sources, bibliography, figures, compiled PDF
```

The raw EuroCropsML time series are **not** in this repository. The `.npz` parcels are
about 1.9 GB and are redistributed by Zenodo; only the partition indices that select
them are archived here, which is what the Data Availability Statement refers to.

## Data

EuroCropsML, Zenodo record `13789558` (doi:10.5281/zenodo.13789558). The Latvian
pre-training subset and the Estonian/Portuguese evaluation splits are reconstructed by
`scripts/prepare_real_data.py`, which streams individual members out of
`preprocess.zip` without unpacking the 1.4 GB archive:

```bash
python scripts/prepare_real_data.py fetch-splits --record 13789558 --out data/raw/_official
python scripts/prepare_real_data.py prepare --record 13789558 \
    --splits data/raw/_official/split.zip \
    --latvia-per-class 250 --latvia-max-total 30000 \
    --estonia-shots 5 10 20 --out data/raw
python scripts/prepare_real_data.py verify --out data/raw
```

`data/raw/*/split.json` and the `fewshot_*shot.json` files in this repository are the
exact partitions used for every number in the paper.

## Reproduce

```bash
python -m venv .venv && .venv\Scripts\activate        # Windows
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Pre-train on Latvia, then probe on Estonia:

```bash
# phenology-aware mask, tiny encoder, seed 42
python scripts/pretrain.py --config configs/pretrain.yaml \
    --data data/raw/latvia --split data/raw/latvia/split.json \
    --mask-type phenology --model-size tiny --epochs 30 --seed 42

# 5/10/20-shot, both protocols, 100 episodes each
python scripts/evaluate.py --ckpt checkpoints/pretrain_phenology_tiny_best.pt \
    --data data/raw/estonia --shots 5 10 20 --episodes 100 --protocol both
```

The full 2 x 2 x 3 ablation grid (mask x model size x shots) runs through:

```bash
python scripts/run_experiments.py --dry-run
python scripts/run_experiments.py --epochs 30 --episodes 100
```

Figures are generated from the JSON in `outputs/`:

```bash
python scripts/figs/fig1_masking.py
python scripts/figs/fig2_budget.py
python scripts/figs/fig3_protocols.py
```

## Checkpoints

| File | Configuration |
|---|---|
| `pretrain_phenology_tiny_best.pt` | phenology mask, tiny, seed 42 (headline run) |
| `pretrain_uniform_tiny_best.pt` | uniform mask, tiny |
| `pretrain_phenology_medium_best.pt` | phenology mask, medium |
| `pretrain_uniform_medium_best.pt` | uniform mask, medium |
| `pretrain_phen_tiny_s43/s44/s45_best.pt` | seed-robustness reruns (seeds 43-45) |
| `phen_aligned_best.pt` | aligned-mask control |
| `norm_stats.json` | channel-wise normalisation statistics |

`*_last.pt` are the final-epoch weights; `*_best.pt` are the early-stopping weights
used in the paper.

## Reproducibility notes

- Every number in the manuscript is read from `outputs/*.json`; confidence intervals
  are 95% from 100 episodes (`OA_ci95` field).
- Seeds are fixed per run; the three extra pre-training seeds exist because a single
  draw cannot support a claim about gain size.
- The `interact.cls` layout used for submission is in `docs/paper/`; the compiled
  manuscript PDF is included there as well.

## License

Code: MIT (see `LICENSE`). Pre-trained weights and partition indices: CC BY 4.0.
EuroCropsML itself remains under its own license on Zenodo.

## Contact

Yulin Zhou — ethanlynx@foxmail.com
Wolverhampton University Institute of Agricultural Engineering of Yunnan Agricultural University, Kunming 650201, China
