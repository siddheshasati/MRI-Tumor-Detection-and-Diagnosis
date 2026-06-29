# 🧠 Brain Tumor Detection & Diagnosis

A deep learning project that detects brain tumors from MRI images using a custom Convolutional Neural Network (CNN) built with PyTorch.

---

## 📌 Overview

This project implements a binary image classifier to detect the presence of brain tumors in MRI scans. The model is trained on labeled MRI images (`Brain Tumor` vs `Healthy`) and evaluated using accuracy, classification report, and confusion matrix.

---

## 📁 Dataset

Dataset: https://www.kaggle.com/datasets/siddheshasati/mri-brain-tumor?select=Brain+Tumor+Data+Set

- **Classes:** `Brain Tumor` | `Healthy`
- **Split:** 80% Training / 20% Validation
- **Format:** JPEG images organized in class-named folders (compatible with `torchvision.datasets.ImageFolder`)

---

## 🏗️ Model Architecture — `CNN_TUMOR`

A custom 4-block CNN with the following structure:

| Layer | Type | Details |
|---|---|---|
| Conv Block 1 | Conv2d → ReLU → MaxPool | `in_channels → 8`, kernel 3×3 |
| Conv Block 2 | Conv2d → ReLU → MaxPool | `8 → 16`, kernel 3×3 |
| Conv Block 3 | Conv2d → ReLU → MaxPool | `16 → 32`, kernel 3×3 |
| Conv Block 4 | Conv2d → ReLU → MaxPool | `32 → 64`, kernel 3×3 |
| FC1 | Linear → ReLU → Dropout | `4096 → 100` |
| FC2 | Linear → LogSoftmax | `100 → 2` |

**Model Parameters:**
```python
params_model = {
    "shape_in": (3, 256, 256),   # RGB input
    "initial_filters": 8,
    "num_fc1": 100,
    "dropout_rate": 0.25,
    "num_classes": 2
}
```

---

## ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| Input Size | 256 × 256 RGB |
| Batch Size | 64 |
| Epochs | 60 |
| Optimizer | Adam (lr = 3e-4) |
| Loss Function | NLLLoss |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=20) |
| Device | CUDA / CPU (auto-detected) |

---

## 🔄 Data Augmentation & Preprocessing

```python
transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

---

## 📊 Evaluation

After training, the model is evaluated using:

- **Classification Report** — Precision, Recall, F1-Score per class
- **Confusion Matrix** — Visual heatmap showing TP, TN, FP, FN with percentages

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install torch torchvision split-folders torch-summary scikit-learn matplotlib seaborn tqdm Pillow
```

### 2. Prepare Dataset

```
Brain Tumor Data Set/
├── Brain Tumor/
│   ├── image1.jpg
│   └── ...
└── Healthy/
    ├── image1.jpg
    └── ...
```

Run the notebook cells to auto-split into `train/` and `val/` directories.

### 3. Train the Model

Run all cells sequentially in `brain-tumor-detection-and-diagnosis.ipynb`. The best model weights are saved automatically to `weights.pt`.

### 4. Save Final Model

```python
torch.save(cnn_model, "Brain_Tumor_model.pt")
```

---

## 📦 Tech Stack

| Tool | Purpose |
|---|---|
| PyTorch | Model building & training |
| torchvision | Dataset loading & transforms |
| scikit-learn | Metrics (confusion matrix, classification report) |
| splitfolders | Train/val dataset splitting |
| torchsummary | Model architecture summary |
| matplotlib / seaborn | Visualization |

---

## 📂 Project Structure

```
├── brain-tumor-detection-and-diagnosis.ipynb   # Main notebook
├── weights.pt                                   # Best model weights (saved during training)
├── Brain_Tumor_model.pt                         # Final saved model
└── README.md
```

---

## 📈 Training Outputs

The training loop logs and plots:
- Train vs Validation **Loss** over epochs
- Train vs Validation **Accuracy** over epochs

Best model weights are restored automatically if validation loss improves.

---

## 📝 Notes

- The notebook was developed on **Kaggle** (GPU environment).
- Dataset path references (`/kaggle/input/...`) are Kaggle-specific — update them for local runs.
- Model input expects **RGB images resized to 256×256**.
