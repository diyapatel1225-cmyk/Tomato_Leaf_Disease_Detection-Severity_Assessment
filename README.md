# Tomato Leaf Disease Detection & Severity Assessment

AI-based system for detecting tomato leaf diseases and estimating disease severity from a single leaf image, using multi-task deep learning. The project benchmarks three CNN backbones — **MobileNetV2**, **EfficientNetB0**, and **DenseNet121** — under identical experimental conditions to identify the best architecture for accuracy vs. efficiency trade-offs.

## Overview

- **Dataset:** PlantVillage tomato subset — 10,000 images across 10 classes (9 diseases + healthy)
- **Tasks:** Multi-task classification
  - Disease classification (10 classes)
  - Severity grading (4 classes: Healthy / Mild / Moderate / Severe)
- **Approach:** Transfer learning with a shared CNN backbone, two dense task-specific heads, two-phase training (frozen base → fine-tuning), and Grad-CAM explainability

Since PlantVillage provides disease labels only, severity labels were auto-generated using an HSV color-thresholding heuristic that estimates the proportion of diseased (brown/yellow) pixels vs. healthy green pixels per leaf image.

## Model Architecture

```
Input Image (224x224x3)
        |
   CNN Backbone (pretrained on ImageNet)
        |
   Global Average Pooling
        |
   Shared Dense Layer
      /        \
Disease Head   Severity Head
 (10 classes)   (4 classes)
```

The same head architecture, loss weights, optimizer, and data split (seed=42) are used across all three backbone variants so that any performance difference is attributable only to the backbone.

## Results

| Backbone | Params (total) | Disease Test Acc | Disease Macro F1 | Severity Test Acc | Inference Speed |
|---|---|---|---|---|---|
| MobileNetV2 | 2.64M | ~84%* | — | — | — |
| EfficientNetB0 | 4.43M | **90%** | **0.90** | **75%** | **14.2 ms/img** |
| DenseNet121 | 7.35M | 88% | 0.87 | 74% | 28.3 ms/img |

\*From training-curve notes; confirm exact test-set figure from your MobileNetV2 run before publishing.

**Key findings:**
- EfficientNetB0 delivered the best accuracy with roughly 40% fewer parameters than DenseNet121 and about half its inference time — the strongest candidate for deployment.
- DenseNet121 showed the smallest train/validation gap (least overfitting) but was the slowest to train and run due to its dense feature-reuse connections.
- MobileNetV2 was fastest and lightest but showed a larger train/test accuracy gap.
- Spider mite infestation was the hardest disease class to classify across all backbones; viral diseases (Yellow Leaf Curl Virus, Mosaic Virus) were the easiest.
- Severity grading is inherently harder than disease classification (F1 ≈ 0.52–0.57 for Mild/Moderate), likely reflecting noise in the rule-based severity labels rather than a model limitation.

## Pipeline

1. **Setup & imports** — TensorFlow/Keras, OpenCV, scikit-learn, seaborn
2. **Dataset loading** — PlantVillage tomato subset extracted from a zip archive
3. **Severity labeling** — Rule-based HSV color thresholding to bucket diseased-pixel ratio into 4 severity levels
4. **Label encoding & split** — Stratified 70/15/15 train/val/test split (seed=42), constant across all backbone runs
5. **Preprocessing & augmentation** — Resize to 224x224, random flip/brightness/contrast/rotation for training, backbone-specific `preprocess_input`
6. **Model construction** — Shared backbone + GAP + shared dense layer + two task heads
7. **Phase A training** — Frozen backbone, train new heads only (15 epochs)
8. **Phase B fine-tuning** — Unfreeze last 30 backbone layers, train at a lower learning rate (20 epochs)
9. **Evaluation** — Classification reports and confusion matrices for both tasks on the held-out test set
10. **Explainability** — Grad-CAM heatmaps to visualize model focus regions on the leaf
11. **Export** — Saved as `.keras` and converted to `.tflite` for mobile/edge deployment

## Repository Structure

```
.
├── notebooks/
│   ├── Tomato_Disease_Detection_MobileNetV2.ipynb
│   ├── Tomato_Disease_Detection_EfficientNet.ipynb
│   └── Tomato_Disease_Detection_DenseNet121.ipynb
├── models/              # Saved .keras / .tflite models (not tracked in git — see .gitignore)
├── results/             # Confusion matrices, training curves, classification reports
├── requirements.txt
└── README.md
```

## Tech Stack

`Python` · `TensorFlow / Keras` · `OpenCV` · `scikit-learn` · `pandas` / `NumPy` · `Matplotlib` / `Seaborn` · `TensorFlow Lite`

## Limitations & Future Work

- Severity labels are rule-based pseudo-labels (HSV color thresholding), not expert-annotated — manual validation on a sample subset is a recommended next step.
- PlantVillage images are captured under controlled lab conditions; real-field accuracy may be lower due to lighting, occlusion, and background variation.
- Future work: validate on field-collected images, explore ensembling backbones, and expand severity grading with expert-labeled data.

## Acknowledgments

Built on the [PlantVillage](https://plantvillage.psu.edu/) tomato leaf dataset, using ImageNet-pretrained backbones from `keras.applications`.
