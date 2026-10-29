# Pneumonia Classification from Chest X-rays Using CNN and PyTorch

## Overview

An educational medical-imaging computer vision project that trains a Convolutional Neural Network (CNN) to classify grayscale chest X-ray images into two dataset-defined categories: **Normal** and **Pneumonia**.

Developed in Python using PyTorch and Google Colab with NVIDIA T4 GPU acceleration, the project covers image preprocessing, model training, validation, test-set evaluation, threshold analysis, and checkpoint saving.

## Problem Statement

Explore whether a CNN can learn patterns in labeled chest X-ray images and evaluate its classification performance. The goal is to demonstrate a reproducible deep-learning workflow and investigate false positives, false negatives, and class-specific performance.

This is a technical learning project, not a clinically validated diagnostic tool.

## Dataset

- **Dataset:** [PneumoniaMNIST — MedMNIST](https://medmnist.com/)
- **Training images:** 4,708
- **Validation images:** 524
- **Test images:** 624
- **Image dimensions:** 28 × 28 pixels, grayscale
- **Task:** Binary image classification

The dataset contains predefined splits and imbalanced class distributions. The low-resolution images limit the conclusions that can be drawn about real-world radiology workflows.

## Tech Stack

- Python
- PyTorch
- NumPy
- Matplotlib
- scikit-learn
- MedMNIST
- Google Colab and NVIDIA T4 GPU

## Methodology

1. Loaded and inspected the predefined PneumoniaMNIST dataset splits.
2. Converted images to tensors and prepared batched data loaders.
3. Implemented a CNN with convolution, ReLU activation, max pooling, dropout, and fully connected layers.
4. Trained for five epochs using Adam and binary cross-entropy with logits.
5. Monitored training and validation loss and accuracy.
6. Evaluated classification performance using precision, recall, F1-score, ROC-AUC, and a confusion matrix.
7. Compared decision thresholds using validation-set balanced accuracy.
8. Saved the model checkpoint and experiment metadata to Google Drive.

## Reported Results

### Training and validation

| Metric | Result |
|---|---:|
| Training epochs | 5 |
| Final training accuracy | 93.2% |
| Final validation accuracy | 96.0% |
| Final validation loss | 0.1469 |

### Held-out test set

| Metric | Normal | Pneumonia |
|---|---:|---:|
| Precision | 0.90 | 0.85 |
| Recall | 0.72 | 0.95 |
| F1-score | 0.80 | 0.90 |
| Support | 234 | 390 |

- Reported test accuracy: **87%**
- Reported ROC-AUC: **0.9237**
- Selected validation threshold: **0.50**
- Reported validation balanced accuracy at the selected threshold: **0.9571**

These are results from the reported experiment and should be reproduced against the notebook before being treated as final. Validation and test metrics represent different evaluation steps.

## Error Analysis

In the reported test confusion matrix:

| Actual class | Predicted Normal | Predicted Pneumonia |
|---|---:|---:|
| Normal | 169 | 65 |
| Pneumonia | 19 | 371 |

The model misclassified 65 normal-labeled images as pneumonia and 19 pneumonia-labeled images as normal. These errors demonstrate why class-specific recall and false-negative analysis matter in medical-imaging research.

## Limitations

- Images are 28 × 28 pixels and do not represent full-resolution clinical imaging workflows.
- Results from one dataset do not establish generalization across hospitals, scanners, or patient populations.
- The model has not undergone clinical validation or prospective testing.
- DICOM processing, segmentation, object detection, OCR, clinical entity detection, and de-identification are not implemented in this version.
- HIPAA compliance and production deployment have not been established.

## Future Work

- Compare the baseline CNN with transfer-learning architectures.
- Evaluate higher-resolution datasets with appropriate access and licensing.
- Improve experiment reproducibility with fixed random seeds and recorded dependency versions.
- Explore Grad-CAM for model interpretability.
- Investigate DICOM preprocessing, privacy-aware data handling, and automated testing as separate extensions.

These are proposed improvements, not completed features.

## Reproducibility

The notebook contains the data-loading workflow, model architecture, training loop, evaluation, and threshold analysis. The trained checkpoint was saved to Google Drive.

To reproduce the experiment, run the notebook cells in order and verify the software versions, evaluation metrics, and confusion matrix.

## Disclaimer

For educational and research purposes only. This model must not be used to diagnose patients, guide treatment, or replace professional medical judgment.
