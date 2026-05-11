#  Skin Lesion Classification using Multimodal Transformer-Based Models


A multimodal deep learning pipeline for dermatological skin lesion classification, combining **Vision Transformers (ViT)** for image features with a **metadata fully-connected network (MetaNet)** — fused into a unified classifier. Achieves **83.2% validation accuracy** on the HAM10000 benchmark with explainability via Grad-CAM and LIME.

---

##  Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│              Multimodal Fusion Classifier                       │
│                                                                 │
│   Dermoscopy Image                   Patient Metadata           │
│        │                                    │                   │
│   ┌────▼───────┐                   ┌────────▼──────┐           │
│   │  ViT-B/16  │                   │    MetaNet     │           │
│   │ (pretrained│                   │  (FC + BN +   │           │
│   │  ImageNet) │                   │   Dropout)    │           │
│   └────────────┘                   └───────────────┘           │
│        │  768-d                          │  64-d                │
│        └──────────────┬─────────────────┘                      │
│                  ┌────▼────┐                                    │
│                  │  Fusion │  (concat → FC → Dropout)           │
│                  └────┬────┘                                    │
│                  ┌────▼────┐                                    │
│                  │Classifier│ (7 lesion classes)                │
│                  └─────────┘                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

##  Lesion Classes (HAM10000)

| ID | Class | Full Name |
|----|-------|-----------|
| 0 | akiec | Actinic Keratoses |
| 1 | bcc | Basal Cell Carcinoma |
| 2 | bkl | Benign Keratosis |
| 3 | df | Dermatofibroma |
| 4 | mel | Melanoma  |
| 5 | nv | Melanocytic Nevi |
| 6 | vasc | Vascular Lesions |

---

##  Dataset Setup

 Download [HAM10000](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection) from Kaggle.
```



---

##  Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | 83.2% |
| Macro F1-Score | 0.79 |
| AUC (OvR) | 0.96 |
| Model | ViT-B/16 + MetaNet |
| Dataset | HAM10000 (10,015 images) |
| Epochs | 30 |
| Optimizer | AdamW + CosineAnnealingLR |

---

##  Explainability

- **Grad-CAM**: Highlights image regions most responsible for the prediction.
- **LIME**: Perturbs superpixels to identify which image segments influenced the decision.

Both methods are visualized and saved as overlaid heatmaps.

---

##  Running Tests

```bash
pytest tests/ -v
```

---

##  Tech Stack

- **Deep Learning**: PyTorch 2.x, timm (ViT-B/16)
- **XAI**: pytorch-grad-cam, LIME
- **Data**: pandas, scikit-learn, albumentations
- **Metrics**: torchmetrics
- **Logging**: TensorBoard, CSV

---
