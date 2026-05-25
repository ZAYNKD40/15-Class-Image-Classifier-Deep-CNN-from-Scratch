# 15-Class Image Classifier — Deep CNN from Scratch

Tiny ImageNet Subset · PyTorch · No pretrained weights

## Overview
Built a deep convolutional neural network entirely from scratch to classify 64×64 RGB 
images across 15 categories. No pretrained weights, no borrowed architectures — every 
weight trained on the chosen dataset only (~5,775 training images).
Built under constraints: no pretrained weights, no skip connections (no ResNet/VGG), trained solely on the provided dataset.
## Results

| Inference Mode | Accuracy |
|---|---|
| Standard (single pass) | 77.58% |
| **TTA — 8 augmented passes** | **79.76%** |

*+2.18 pp gained from Test-Time Augmentation at zero training cost.*  
*Only benchmark higher (~84%) likely used pretrained weights under prior-semester rules.*

## Architecture — 13-Layer Custom CNN

    Input (3×64×64)
    → Block 1: Conv×2 + BN + ReLU + MaxPool + Dropout2d   [64→32]
    → Block 2: Conv×2 + BN + ReLU + MaxPool + Dropout2d   [32→16]
    → Block 3: Conv×3 + BN + ReLU + MaxPool + Dropout2d   [16→8]
    → Block 4: Conv×3 + BN + ReLU + MaxPool + Dropout2d   [8→4]
    → Block 5: Conv×3 + BN + ReLU + Dropout2d             [4→4, no pool]
    → Global Average Pooling → Linear(512→256) → Linear(256→15)

~8M parameters · Kaiming He weight initialization

## Training Stack
- **Optimizer:** AdamW (weight decay 1e-4)
- **Scheduler:** OneCycleLR — warm-up → cosine anneal
- **Augmentation:** RandomCrop, HorizontalFlip, ColorJitter, RandomErasing, Mixup (α=0.4)
- **Loss:** CrossEntropyLoss with label smoothing (0.1)
- **Regularization:** Dropout2d per block, gradient clipping (max norm 1.0)
- **Early stopping:** patience = 20 epochs, best checkpoint saved

## Test-Time Augmentation (TTA)
At inference, each image is passed through 8 augmented versions (original + flips + 
crops + color jitter). Probabilities are averaged before argmax — no retraining required.

## Tech Stack
Python · PyTorch · torchvision · scikit-learn · matplotlib · seaborn

## To view html
https://zaynkd40.github.io/15-Class-Image-Classifier-Deep-CNN-from-Scratch/CNN_From_Scratch_final.html
