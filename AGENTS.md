# AGENTS.md

## Environment

- The system `python` is Python 2.7. Always use the conda env:
  - `/data/qiuyk/.miniconda/envs/biai/bin/python` (Python 3.12, torch 2.5.0+cu121)
- 3x RTX 3090 (24 GB) available; `cuda` works out of the box.
- No test suite, linter, CI, or pre-commit config exists. Do not invent one.

## Commands

Run everything from the repository root; all config/data paths are CWD-relative.

```bash
PY=/data/qiuyk/.miniconda/envs/biai/bin/python

$PY train.py --model resnet     # training; --epochs/--batch-size/--lr/--patience available
$PY test.py --model resnet --split test  # evaluate a checkpoint
$PY scripts/generate_splits.py  # regenerate deterministic 7:1:2 split CSVs
$PY scripts/ml_baselines.py --source image --model random_forest --pca-components 256
PYTHON=$PY bash scripts/run_ml_baselines.sh  # batch ML baselines -> results/traditional_ml
$PY scripts/gradcam.py --split test --index 0
bash scripts/run_all_experiments.sh  # 6 models x 5 seeds; logs in results/runs
$PY scripts/collect_results.py       # -> results/final_metrics.md (mean±std) and best_metrics.md
$PY scripts/plot_losses.py --from-csv results/loss_history.csv --max-epoch 25
$PY scripts/attention_ad.py  # AD axial/coronal/sagittal Grad-CAM -> assets/mvst_attention_ad_{view}.png
```

Focused verification (no tests exist):

```bash
$PY -m compileall -q models utils scripts train.py test.py
$PY -c "from utils.dataset import ADNIDataset; print(len(ADNIDataset('resnet').load_data('val')))"  # expect 21
```

## Architecture

- `utils/dataset.py: ADNIDataset(model_name)` loads `configs/training/<model_name>.yaml` and dispatches the six supported models.
- `ADNI3DDataset` -> `(1, 32, 112, 112)`; `ADNI2DDataset` -> `(3, 224, 224)` from mid axial/coronal/sagittal slices.
- Root `train.py` and `test.py` are the primary CLIs; `--model {resnet,vgg16,hcct_vit,vit_3d,swin_3d_cnn,mvst}` selects the model via `models.build_model`. Config defaults to `configs/training/<model>.yaml` and CLI flags override batch size, epochs, lr, patience, and save path.
- `ADNIDataset` supports `vgg16`, `resnet`, `hcct_vit`, `vit_3d`, `swin_3d_cnn`, and `mvst` (3D loaders map to the same volume cache).
- `models/mvst.py` is the added SOTA attempt: 12 slices (4 per view) -> shared ResNet18 tokens -> transformer encoder.
- Saved weights live under `checkpoints/<model>/`. `configs/training/` is the source of truth for paths/hyperparameters.
- Labels: `CN=0, MCI=1, EMCI=2, LMCI=3, AD=4` (`utils/dataset.py`).
- Metrics in `utils/metrics.py` use macro-AUC. The test split has only 32 scans (EMCI/LMCI = 1 each), so seed variance is inherently large; report n and do not over-interpret small differences.
- Training selects the checkpoint by validation macro-F1 by default (`--monitor val_f1`), uses label smoothing 0.1, and ResNet has dropout 0.3. Do not switch back to val-loss selection on the 14-scan val split.
- Keep `batch_size: 8` identical in every training config. Loss plots are truncated to the same epoch range (25) for visual comparison.

## Data / artifacts

- Label table: `dataset/5_class_10_12_2025.csv` (205 rows). NIfTI files under `dataset/ADNI/ADNI/<Subject>/` are matched by `Image Data ID` via recursive glob; do not assume filenames equal IDs.
- Split CSVs in `configs/split_results/` are subject-level 159/14/32 (train/val/test), no subject overlap, seed 42. `scripts/generate_splits.py` guarantees every class in every split.
- `.gitignore` excludes `dataset/ADNI/`, `dataset/5_class_metadata/`, `checkpoints/`, `cache/`, `outputs/`, and model weight formats. Never commit these large artifacts.
- Preprocessed caches: `cache/3d/<split>.npy` (3 files) and `cache/2d/<split>_{triple,single}.npy` (6 files). Build with `$PY scripts/build_cache.py`. Do not create per-sample `.npy` files.
- `scripts/run_all_experiments.sh` exports `OMP_NUM_THREADS=1`; keep this for parallel GPU jobs or PyTorch thread oversubscription makes GPUs sit idle.
- `results/` and the small label CSV are trackable; `figures/` is version-controlled, `outputs/` is ignored.
- Results: `results/final_metrics.md` (mean ± std), `results/best_metrics.md` (best seed), `results/deep_metrics.csv`, and `results/loss_history.csv`.
- Tracked figures: `figures/age_distribution_blue_dual_axis.png`, `figures/loss_curves.png`, and `assets/mvst_attention_ad_{axial,coronal,sagittal}.png`.

## Conventions

- Root holds only the primary CLIs (`train.py`, `test.py`); helper tools belong under `scripts/`, importable code under `models/` and `utils/`.
- Runnable code lives in `scripts/`. Scripts that import project modules bootstrap `sys.path` with `Path(__file__).resolve().parents[1]` so `from utils...` works when run as `python scripts/foo.py`.
- Use ASCII English for filenames, comments, docstrings, and README. No Chinese characters.
- `scripts/ml_baselines.py` handles image and metadata pipelines; `scripts/metadata_ml_baseline.py` is the Age/Sex-only baseline. Keep filenames descriptive (old names such as `ml.py`, `atrr.py`, `tu1.py` were renamed).
