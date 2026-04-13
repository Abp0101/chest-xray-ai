# Chest X-Ray Disease Classifier

Multi-label thoracic disease classifier trained on the [NIH Chest X-Ray 14](https://www.kaggle.com/datasets/nih-chest-xrays/data) dataset.  
Fine-tunes a pretrained **DenseNet-121** to predict 14 conditions from a single frontal X-ray.  
Includes Grad-CAM visualisations showing which regions of the image drove each prediction.

> **Disclaimer:** This project is for research and educational purposes only.  
> It is **not** a medical device and must **not** be used for clinical decision-making.

---

## What it does

| Step | Script | Description |
|---|---|---|
| Download data | `src/download_data.py` | Pulls ~45 GB NIH CXR-14 from Kaggle |
| Explore data | `src/explore_data.py` | Class distribution, sample images, missing-data report |
| Train | `src/train.py` | Two-stage fine-tuning with W&B logging |
| Evaluate | `src/evaluate.py` | AUC-ROC table, ROC curves, Grad-CAM figures |
| Demo | `app.py` | Gradio web app (local or HF Spaces) |

---

## Dataset

**NIH Chest X-Ray 14** — Wang et al. 2017  
- 112,120 frontal-view X-ray images from 30,805 unique patients  
- 14 disease labels (multi-label — one image can have multiple conditions)  
- Official `train_val_list.txt` / `test_list.txt` splits used (patient-level — no leakage)

| Label | Prevalence |
|---|---|
| Infiltration | 17.7% |
| Effusion | 11.8% |
| Atelectasis | 10.3% |
| Nodule | 5.6% |
| Mass | 5.1% |
| Pneumonia | 1.3% |
| Hernia | 0.2% |
| … | … |

---

## Model Architecture

- **Backbone:** DenseNet-121 pretrained on ImageNet-1K  
- **Head:** `Dropout(0.5) → Linear(1024 → 14)`  
- **Loss:** `BCEWithLogitsLoss` with per-class `pos_weight` to handle class imbalance  
- **Output:** 14 independent sigmoid probabilities (multi-label, not softmax)

### Two-stage fine-tuning

| Stage | Epochs | Backbone | LR |
|---|---|---|---|
| 1 — head warm-up | 3 | Frozen | 1e-3 |
| 2 — full fine-tune | 12 | Unfrozen | 1e-4 |

Stage 1 gets the randomly-initialised head to a sensible scale before gradients flow into the pretrained backbone.  
Stage 2 adapts the full network to X-ray statistics with a `ReduceLROnPlateau` scheduler (patience=2, factor=0.5).

### Key training decisions

- **No horizontal flip** — the heart must stay on the left; flipping creates anatomically impossible images  
- **Conservative crop** — `RandomResizedCrop(scale=0.9–1.0)` preserves clinically relevant regions (apices, costophrenic angles)  
- **WeightedRandomSampler** — up-samples rare-disease images so the model doesn't collapse to predicting "No Finding" for everything  
- **Gradient clipping** — `max_norm=1.0` prevents explosive updates at the start of stage 2

---

## Results

Evaluated on the official NIH test split (~25,000 images, patient-level separation from train).  
The standard metric is **mean AUC-ROC** across all 14 classes.

| Metric | Value |
|---|---|
| Mean AUC-ROC | *(run `src/evaluate.py` after training)* |
| CheXNet baseline (Rajpurkar et al. 2017) | 0.841 |

Per-class AUC table and ROC curves are saved to `outputs/figures/roc_curves.png` after evaluation.

---

## Grad-CAM

Gradient-weighted Class Activation Mapping highlights the spatial regions that drove each prediction.

- Hook point: `features.denseblock4` (1024 channels, 7×7 spatial resolution)  
- Gradient of the top-predicted class score is backpropagated to the feature maps  
- Channels are weighted by their spatially-averaged gradient magnitude  
- The resulting 7×7 map is upsampled to 224×224 and blended over the original X-ray

Sample Grad-CAM outputs are saved to `outputs/figures/gradcam_*.png`.

---

## How to Run

### Prerequisites

```bash
# Clone and set up the venv
git clone https://github.com/Abp0101/chest-xray-ai
cd chest-xray-ai
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Make sure your Kaggle credentials are at `~/.kaggle/kaggle.json`.

### 1 — Download the dataset

```bash
python src/download_data.py
# Downloads ~45 GB to data/raw/
```

### 2 — Explore the dataset (optional)

```bash
python src/explore_data.py
# Saves class_distribution.png and sample_images.png to outputs/figures/
```

### 3 — Train

```bash
# Default: 3 warm-up + 12 fine-tune epochs, batch 32, W&B logging on
python src/train.py

# Disable W&B
python src/train.py --no-wandb

# Custom settings
python src/train.py --epochs-stage1 5 --epochs-stage2 20 --batch 64
```

Checkpoints are saved to `outputs/checkpoints/best_model.pth` and `last_model.pth`.

### 4 — Evaluate

```bash
python src/evaluate.py
# Prints AUC table, saves roc_curves.png and gradcam_*.png to outputs/figures/
```

### 5 — Run the demo

```bash
pip install gradio
python app.py
# Opens at http://localhost:7860
```

---

## Weights & Biases

Training logs train loss, val loss, and learning rate every epoch.  
After training, the best checkpoint is reloaded to compute test AUC, and the ROC curves figure and 3 Grad-CAM samples are uploaded as W&B images.

```bash
wandb login   # first time only — enter your API key from wandb.ai/authorize
python src/train.py --wandb-project my-project
```

---

## Project Structure

```
chest-xray-ai/
├── src/
│   ├── download_data.py   # Kaggle API download
│   ├── explore_data.py    # EDA and visualisation
│   ├── dataset.py         # ChestXrayDataset, DataLoaders, augmentation
│   ├── model.py           # DenseNet-121 with 14-class head
│   ├── train.py           # Two-stage training loop + W&B
│   └── evaluate.py        # AUC-ROC, ROC curves, Grad-CAM
├── app.py                 # Gradio demo (HF Spaces entry point)
├── requirements.txt
├── data/
│   ├── raw/               # NIH CXR-14 (gitignored, ~45 GB)
│   └── processed/
└── outputs/
    ├── checkpoints/       # best_model.pth, last_model.pth (gitignored)
    └── figures/           # roc_curves.png, gradcam_*.png
```

---

## References

- Wang et al. (2017). *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks.* CVPR.  
- Rajpurkar et al. (2017). *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning.* arXiv:1711.07837.  
- Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV.
