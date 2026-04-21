# Chapter 3: Methodology

## 3.1 Overview

This chapter presents the methodological framework used to develop and evaluate an automated polyp segmentation system on the Kvasir-SEG dataset. The core design objective is to ensure methodological rigor through architectural consistency, preprocessing standardization, reproducible training, and objective comparative evaluation across multiple model families. To this end, three heterogeneous segmentation architectures—ResUNet++, TransFuse, and WDFFNet—are trained under a unified protocol, followed by a learnable weighted ensemble that exploits model complementarity.

The end-to-end pipeline consists of five stages: (i) data preparation and split generation, (ii) model-specific forward design with output-shape harmonization, (iii) supervised optimization of individual base models, (iv) frozen-backbone ensemble-head training, and (v) quantitative and qualitative performance analysis.

## 3.2 Dataset and Experimental Setup

### 3.2.1 Dataset

Experiments are conducted on Kvasir-SEG, a benchmark colonoscopy dataset for binary polyp segmentation. Each sample comprises an RGB image and its corresponding binary mask.

### 3.2.2 Data Partitioning

To avoid information leakage and to maintain experimental reproducibility, the dataset is partitioned into training, validation, and test sets using seed-controlled randomized splitting. The train split is used for parameter optimization, the validation split for model selection and early checkpointing, and the test split for final reporting.

### 3.2.3 Unified Preprocessing

All models receive identical input statistics to ensure fairness:

1. Spatial resizing to 256×256.
2. Intensity normalization using ImageNet mean and standard deviation.
3. Mask conversion to floating-point [0, 1], followed by binary thresholding.

A single preprocessing policy is applied to all architectures so observed differences reflect architectural behavior rather than inconsistent data treatment.

## 3.3 Base Segmentation Architectures

## 3.3.1 ResUNet++

ResUNet++ is integrated through a wrapper that preserves the original network implementation while enforcing interface compatibility with the shared training framework. Inputs are three-channel images of size 256×256, and outputs are constrained to one-channel 256×256 logits via interpolation when required. This wrapper-based strategy preserves architectural fidelity and resolves checkpoint-interface inconsistencies.

## 3.3.2 TransFuse

The TransFuse implementation employs a dual-stream feature extraction strategy, where two feature backbones are fused at multiple semantic levels via BiFusion blocks. Deep and intermediate fused representations are upsampled and concatenated with shallow features before a final projection layer generates segmentation logits. To guarantee compatibility with supervision masks, predictions are explicitly resized to 256×256.

## 3.3.3 WDFFNet

WDFFNet adopts a dual-backbone design and performs stage-wise cross-branch fusion using learnable weighted fusion modules. Fused features are further refined with object-aware attention (channel and spatial mechanisms) and decoded progressively using upsampling and skip concatenation. Final logits are projected to a one-channel segmentation map and normalized to 256×256 spatial resolution.

## 3.4 Shape and Checkpoint Consistency Controls

Two practical reliability controls are introduced:

1. **Output normalization utility**: Converts model outputs to a canonical binary-segmentation tensor format and enforces target spatial dimensions.
2. **Robust partial checkpoint loading**: Loads only parameters with matching key names and tensor shapes; incompatible entries are skipped and logged.

These controls prevent silent failures caused by architecture drift, naming mismatches, and inconsistent output tensor formats.

## 3.5 Optimization Strategy

### 3.5.1 Loss Function

Each base model is trained using a composite objective balancing region overlap and pixel-wise calibration:

\[
\mathcal{L}_{total} = 0.5\,\mathcal{L}_{BCE} + 0.5\,\mathcal{L}_{Dice}
\]

with Dice loss defined as:

\[
\mathcal{L}_{Dice}=1-\frac{2\sum_i p_i g_i + \epsilon}{\sum_i p_i + \sum_i g_i + \epsilon}
\]

where \(p_i\) and \(g_i\) denote predicted probability and ground-truth label at pixel \(i\).

### 3.5.2 Optimizer and Training Regime

All base models are trained independently using Adam (learning rate \(10^{-4}\)). The best checkpoint is selected based on validation Dice score. This model-selection criterion aligns optimization with the primary overlap objective used in medical segmentation literature.

## 3.6 Learnable Weighted Ensemble

After base-model training, a fusion module is trained while freezing all backbone parameters. Let \(P_r\), \(P_t\), and \(P_w\) denote predictions from ResUNet++, TransFuse, and WDFFNet, respectively. These predictions are stacked and passed through a lightweight convolutional head that outputs softmax-normalized weights \(w_r, w_t, w_w\), with:

\[
w_r + w_t + w_w = 1
\]

The final ensemble prediction is computed as:

\[
\hat{Y} = w_r P_r + w_t P_t + w_w P_w
\]

Unlike arithmetic averaging, this mechanism learns context-dependent contribution weights and is therefore expected to improve robustness in heterogeneous visual conditions.

## 3.7 Evaluation Protocol

Performance is quantified using pixel-level metrics:

- Dice score,
- Intersection-over-Union (IoU),
- Precision,
- Recall,
- Accuracy.

In addition to segmentation quality, computational characteristics are profiled via:

- total parameter count,
- trainable parameter count,
- inference throughput (FPS).

This dual reporting provides both clinical relevance (segmentation accuracy) and deployment relevance (efficiency).

## 3.8 Qualitative and Statistical Reporting

Three visualization modes are used to complement scalar metrics:

1. Input/ground-truth/prediction panels for each model and the ensemble,
2. Training and validation learning curves,
3. Cross-model bar plots for Dice comparison.

A unified comparison table consolidates segmentation accuracy and computational metrics for transparent model ranking.

## 3.9 Reproducibility Considerations

To maximize reproducibility, the pipeline enforces deterministic seeds, standardized preprocessing, architecture-specific output harmonization, checkpoint-loading logs, and consistent metric computation. All stages are organized into modular scripts to support exact reruns in Google Colab and local GPU environments.

## 3.10 Chapter Summary

This chapter established a controlled and reproducible methodology for fair comparison of heterogeneous segmentation architectures and their adaptive ensemble. The next chapter reports quantitative outcomes, qualitative visual evidence, and an interpretation of strengths, failure modes, and practical implications for clinical AI deployment.
