# Training Module

This folder contains all scripts and configuration necessary to train the speaker verification model (hybrid ECAPA‑TDNN + handcrafted features).

## 📁 Structure

- `config.py` &ndash; centralized hyperparameters, paths and experiment options.
- `dataset.py` &ndash; utilities to build training/validation dataloaders from pre‑extracted features.
- `model.py` &ndash; factory function and network definitions (ECAPA‑TDNN branch, handcrafted branch, fusion, AAM‑Softmax loss).
- `train.py` &ndash; main training driver with loops, logging, early stopping and checkpointing.
- `metrics.py` &ndash; helpers for evaluation during/after training.
- `inference.py` &ndash; example inference pipeline using saved model checkpoints.
- `main.ipynb` &ndash; notebook sample showing how to launch training, visualize history and gating analysis.

## ✅ Prerequisites

1. Python 3.8+ environment with these packages installed (see workspace `requirements.txt` if available):
   ```bash
   torch torchvision torchaudio
   numpy tqdm matplotlib seaborn scikit-learn
   tensorboard torchinfo
   ```
2. GPU recommended (CUDA) but CPU works; set `DEVICE` in `config.py` or via CLI argument.
3. Pre‑extracted features stored in folders matching `TRAIN_VAL_DIR` structure:
   ```text
   train_val/
   ├── MFBE + Pitch
   │   ├── spk0001.pt
   │   ├── spk0002.pt
   |   └── ...
   ├── FBank/...
   └── ...
   ```
   - `FBank` and one of the handcrafted subfolders (`MFBE + Pitch`, `Only MFBE`, etc.).
   - `TEST_DIR` for inference/validation.

## 🔧 Configuration

Adjust settings in `config.py` or override via command‑line arguments when running `train.py`.
Important fields:

- Data paths: `TRAIN_VAL_DIR`, `TEST_DIR`.
- `MODE` – 1 (FBank only), 2 (handcrafted only), 3 (hybrid).
- `FUSION_METHOD` – `concat`, `cross_attention`, or `gating` for MODE 3.
- Learning hyperparameters (batch size, epochs, lr, scheduler, early stopping, etc.).
- `AAM_MARGIN` / `AAM_SCALE` – parameters for additive angular margin softmax loss.
- `DEVICE` & `MIXED_PRECISION` – training device options.

## 🚀 Running Training

From the `train` directory or workspace root, execute:

```bash
python train/train.py \
    --base_dir /path/to/data/train_val \
    --mode 3 \
    --feature_mode mfbe_pitch \
    --batch_size 64 \
    --num_epochs 100 \
    --learning_rate 0.001 \
    --output_dir ./checkpoints
```

You can view all available CLI options by running `python train/train.py --help`.

Training will automatically split the data according to `TRAIN_RATIO` and save checkpoints in:

```
<output_dir>/experiments/<exp_name>/
    config.json
    model_summary.txt
    tensorboard_logs/
    best_model.pth
    best_mindcf_model.pth
    final_model.pth            (saved at end)
    results.json
    gating_analysis/          (if using gating fusion)
```

The script supports mixed precision, LR schedulers (`cosine` or `plateau`), early stopping, gradient clipping, and tensorboard logging.

### Monitoring

Launch TensorBoard to inspect loss/accuracy curves:

```bash
tensorboard --logdir ./checkpoints/experiments/<exp_name>/tensorboard_logs
``` 

Progress bars also print loss and accuracy every `LOG_INTERVAL` batches.

## 🧪 Inference & Evaluation

`inference.py` provides a simple example of loading a saved checkpoint for prediction on the `TEST_DIR`.

Call it like:

```bash
python train/inference.py --checkpoint ./checkpoints/experiments/<exp_name>/best_model.pth \
    --base_dir /path/to/data/test \
    --mode 3 --feature_mode mfbe_pitch
```

## 📘 Notebook Example

Open `main.ipynb` for an interactive walkthrough: building dataloaders, running a short training session, plotting histories, and analyzing gating weights.

## 🧾 Notes

- Always ensure `FEATURE_MODE` matches the handcrafted folder names defined in `HANDCRAFTED_FOLDERS`.
- Random seed is fixed (`RANDOM_SEED`) for reproducibility.
- Checkpoints are saved when validation EER/MinDCF improve; early stopping tracks validation EER.

---

Use this module as the core of your speaker verification experiments. Tweak hyperparameters and fusion strategies to find the best configuration for your dataset.