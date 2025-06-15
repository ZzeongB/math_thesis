# Object-Level Vision-Language Alignment

This repository contains the full experimental pipeline and analysis code used for object-level vision-language alignment experiments, as presented in the thesis. We investigate the alignment structure between DINOv2 vision embeddings and CLIP language embeddings on object crops extracted from COCO 2017.

---

## Repository Structure

```

MATH\_THESIS/
├── reports/               # Experimental results (figures, reports)
│   ├── alignment\_hist.png
│   └── alignment\_report.txt
├── src/
│   ├── experiments/
│   │   └── run\_experiment.py       # Main experiment pipeline: alignment & evaluation
│   ├── preprocess/
│   │   ├── download\_dataset.py     # Download COCO 2017 dataset
│   │   ├── sample\_data.py          # Randomly sample 3000 object instances for experiments
│   │   ├── text\_encoder.py         # Extract CLIP text embeddings from object category labels
│   │   └── segment/
│   │       ├── sam\_dino.py         # Crop objects using SAM, extract DINO embeddings (single GPU)
│   │       └── sam\_dino\_multi.py   # Multi-GPU version for DINO embedding extraction
├── README.md

````

---

## Pipeline Overview

### 1. Dataset Preparation

- **download_dataset.py**  
  Download and organize COCO 2017 dataset into `./data/COCO2017/`.

- **sample_data.py**  
  Randomly sample 3000 object instances from the COCO training set for alignment experiments.

### 2. Embedding Extraction

- **Object segmentation (SAM):**  
  Use `sam_dino.py` or `sam_dino_multi.py` to crop object instances using Segment Anything Model (SAM), and extract DINOv2 vision embeddings (`o_i ∈ ℝ⁷⁶⁸`).

- **Language embedding (CLIP):**  
  Use `text_encoder.py` to encode object category labels via CLIP-ViT-B/32 (`t_i ∈ ℝ⁵¹²`).

### 3. Alignment Experiment

- **run_experiment.py**  
  Perform PCA-based dimensionality matching, apply linear ridge regression and kernel ridge regression for alignment, compute residual errors, and generate evaluation reports.

- Results are saved in `./reports/`, including:
  - `alignment_hist.png` (alignment distribution plots)
  - `alignment_report.txt` (alignment performance summary)

---

## Model Checkpoint Download Instructions

### 1. SAM (Segment Anything)

- Download pretrained SAM ViT-H checkpoint:

```bash
mkdir -p ./data/pretrained
wget -P ./data/pretrained https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
````

* The checkpoint will be saved as:

  ```
  ./data/pretrained/sam_vit_h_4b8939.pth
  ```

### 2. DINOv2

* DINOv2 models are automatically downloaded via `torch.hub`:

```python
import torch
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
```

* The checkpoint will be cached locally under `~/.cache/torch/hub/`.

### 3. CLIP

* CLIP models are automatically downloaded via OpenAI `clip` package:

```python
import clip
model, preprocess = clip.load("ViT-B/32")
```

* The model is cached locally under `~/.cache/clip/`.

---

## Key Dependencies

* Python 3.9+
* numpy
* scipy
* scikit-learn
* matplotlib
* tqdm
* Segment Anything Model (SAM)
* DINOv2
* CLIP

> ⚠ Pretrained model checkpoints for SAM must be manually downloaded as described above.

---

## Experiment Summary

* Dataset: COCO 2017
* Sample size: 3000 object-level pairs
* Vision embeddings: DINOv2-ViT-L/14 (768D)
* Language embeddings: CLIP-ViT-B/32 (512D)
* Alignment models: PCA + Ridge Regression + Kernel Ridge Regression
