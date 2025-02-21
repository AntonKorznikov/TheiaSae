# ViT Autoencoder Training

This repository contains the implementation of a sparse autoencoder (SAE) trained on Vision Transformer (ViT) activations. The autoencoder is designed to learn a dictionary of sparse representations from large-scale image datasets.

## Features
- Implements a **BatchTopK Sparse Autoencoder (SAE)**
- Uses **Vision Transformer (ViT) activations** as input
- Supports training on **large-scale datasets** (e.g., ImageNet)
- Utilizes **WandB** for logging and model tracking
- Efficient activation storage with dataset streaming

---

## Installation

### Prerequisites
Ensure you have Python installed along with the required dependencies:

```bash
pip install torch torchvision transformers datasets tqdm wandb
```

Clone the repository:

```bash
git clone <repository_url>
cd <repository_name>
```

---

## Usage

### 1. Configure the Autoencoder
Modify `config.py` to set parameters such as:
- `batch_size`
- `learning_rate (lr)`
- `dict_size`
- `top_k`
- `device` (e.g., `cuda:0`)

### 2. Train the Autoencoder
Run the training script:

```bash
python main.py
```

This will:
- Load the dataset
- Extract activations from the ViT model
- Train the autoencoder
- Log training metrics to Weights & Biases (WandB)

### 3. Checkpoints and Logging
Model checkpoints are saved in `checkpoints/`, and training logs are stored in WandB under the specified project name in `config.py`.

---

## Repository Structure
```
.
├── sae.py                # Autoencoder model implementation
├── logs.py               # WandB logging utilities
├── training.py           # Training loop
├── config.py             # Configuration settings
├── main.py               # Entry point for training
├── activation_store.py   # Handles dataset streaming & feature extraction
```

---

## Model Details
The `BatchTopKSAE` model is a sparse autoencoder that:
- Uses **ReLU activations**
- Applies a **top-k thresholding** strategy
- Ensures decoder weights remain unit-normed

### Key Components:
- `compute_activations()`: Computes the sparse activations
- `encode()`: Encodes input activations
- `decode()`: Decodes sparse representations back to the original space
- `update_threshold()`: Dynamically updates activation thresholds

---

## Dataset
The default dataset is **ImageNet**, streamed using the `datasets` library. The dataset path can be modified in `config.py`:
```python
cfg["dataset_path"] = "evanarlian/imagenet_1k_resized_256"
```

Images are preprocessed using **torchvision.transforms** before extracting activations.

---

## Logging and Checkpointing
- **Weights & Biases (WandB)**: Used for real-time monitoring of loss, sparsity, and feature utilization.
- **Checkpoints**: Models are saved at `checkpoints/` and include:
  - `sae.pt` (model weights)
  - `config.json` (training configuration)

---
