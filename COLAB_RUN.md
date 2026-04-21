# Run in Google Colab (Kvasir-SEG + ResUNet++/TransFuse/WDFFNet + Weighted Ensemble)

Use these cells in order.

## 1) Runtime + dependencies

```python
# Colab: Runtime -> Change runtime type -> GPU
!nvidia-smi

!pip -q install timm albumentations opencv-python-headless scikit-learn matplotlib pandas
```

## 2) Clone this repo

```python
%cd /content
!git clone <YOUR_GITHUB_REPO_URL> ensembleArchitectureBalkees
%cd /content/ensembleArchitectureBalkees
```

## 3) Download Kvasir-SEG

```python
!mkdir -p /content/data
!wget -q --no-check-certificate https://datasets.simula.no/downloads/kvasir-seg.zip -O /content/data/kvasir-seg.zip
!unzip -qq -o /content/data/kvasir-seg.zip -d /content/data/
!ls /content/data/Kvasir-SEG
```

Expected structure:
- `/content/data/Kvasir-SEG/images/*.jpg`
- `/content/data/Kvasir-SEG/masks/*.jpg` (or `.png`)

## 4) Prepare ResUNet++ source file (required by wrapper)

The wrapper imports `build_resunetplusplus` from DebeshJha's source file.

```python
!git clone https://github.com/DebeshJha/ResUNetPlusPlus.git /content/ResUNetPlusPlus
!ls /content/ResUNetPlusPlus/resunet++_pytorch.py
```

## 5) (Optional) Upload your checkpoints

If you already have checkpoints for any model, upload and pass their paths.

```python
from google.colab import files
# files.upload()  # optional
```

## 6) Run end-to-end pipeline

```python
!python colab_pipeline.py \
  --data_dir /content/data/Kvasir-SEG \
  --resunet_source /content/ResUNetPlusPlus/resunet++_pytorch.py \
  --epochs 12 \
  --ensemble_epochs 8 \
  --batch_size 8
```

With checkpoints:

```python
!python colab_pipeline.py \
  --data_dir /content/data/Kvasir-SEG \
  --resunet_source /content/ResUNetPlusPlus/resunet++_pytorch.py \
  --resunet_ckpt /content/best_resunetpp_model.pth \
  --transfuse_ckpt /content/transfuse.pth \
  --wdff_ckpt /content/best_wdffnet.pth \
  --epochs 12 \
  --ensemble_epochs 8 \
  --batch_size 8
```

## 7) Outputs you should see

- Best checkpoints:
  - `best_resunetpp.pth`
  - `best_transfuse.pth`
  - `best_wdffnet.pth`
  - `best_ensemble.pth`
- CSV table: `comparison_table.csv`
- Plots:
  - training curves
  - prediction panels (input, GT, each model, ensemble)
  - Dice bar chart

## 8) Common troubleshooting

1. **OOM / CUDA out of memory**
   - Reduce `--batch_size` from 8 to 4 or 2.
2. **Missing ResUNet++ file**
   - Ensure `--resunet_source /content/ResUNetPlusPlus/resunet++_pytorch.py` exists.
3. **Checkpoint key mismatch logs**
   - This is expected when architecture differs; loader keeps only shape-matching layers.
4. **Slow run on CPU**
   - Confirm GPU runtime is enabled (`!nvidia-smi`).

## 9) Recommended quick smoke test

Run fewer epochs first to verify pipeline:

```python
!python colab_pipeline.py \
  --data_dir /content/data/Kvasir-SEG \
  --resunet_source /content/ResUNetPlusPlus/resunet++_pytorch.py \
  --epochs 2 \
  --ensemble_epochs 1 \
  --batch_size 4
```
