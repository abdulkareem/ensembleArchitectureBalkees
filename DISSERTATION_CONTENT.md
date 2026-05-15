# Dissertation Content Draft: Polyp Segmentation with ResUNet++, TransFuse, WDFFNet, and a Learnable Weighted Ensemble on Kvasir-SEG

## 1. Problem Statement

Automatic colorectal polyp segmentation from colonoscopy frames is a dense prediction task where each pixel is classified as polyp or background. Accurate segmentation supports downstream clinical decision-making, including lesion boundary estimation and potential malignancy assessment.

This dissertation investigates three complementary deep models—ResUNet++, TransFuse, and WDFFNet—and a learnable weighted ensemble for robust binary segmentation on Kvasir-SEG.

---

## 2. Dataset and Preprocessing Protocol

### 2.1 Dataset
Kvasir-SEG contains RGB endoscopic images and corresponding binary masks. Images and masks are loaded from:

- `images/*.jpg`
- `masks/*.jpg` or `masks/*.png`

### 2.2 Unified Preprocessing
To ensure fair comparison across architectures, all models use an identical data pipeline:

1. Resize image and mask to **256 × 256**.
2. Normalize RGB images with ImageNet statistics:
   - mean = (0.485, 0.456, 0.406)
   - std = (0.229, 0.224, 0.225)
3. Convert mask to float and scale to [0, 1].
4. Binarize mask with threshold 0.5.

### 2.3 Splitting Strategy
Data is shuffled with fixed seed and split into train/validation/test subsets using reproducible random splitting.

---

## 3. Architecture Details

## 3.1 ResUNet++ (Wrapper-based Integration)

ResUNet++ is imported from the original source implementation via dynamic module loading to preserve architectural fidelity. The wrapper:

- instantiates `build_resunetplusplus()` from the external file,
- accepts 3-channel 256×256 input,
- forces output to 1-channel 256×256 using bilinear interpolation when needed.

This solves practical output-size mismatch without modifying the original core architecture.

### Architectural motivation
ResUNet++ combines residual learning with multi-scale U-Net style encoding-decoding and skip pathways, improving boundary-aware segmentation over plain U-Net in challenging low-contrast medical imagery.

---

## 3.2 TransFuse (CNN + Lightweight Transformer-style Dual Stream)

The TransFuse implementation uses two feature extractors:

- CNN stream: `efficientnet_b0` (features-only),
- second stream: `mobilenetv3_large_100` (features-only),

with BiFusion blocks at multiple semantic scales:

- Deep fusion: 256 channels,
- Mid fusion: 128 channels,
- Shallow fusion: 64 channels.

Fused deep/mid features are upsampled to shallow spatial resolution, concatenated, and projected to one-channel logits with a 1×1 convolution.

Finally, prediction is upsampled/interpolated to fixed 256×256 output.

### Why TransFuse helps
The dual-path design mixes local texture cues from convolutional features with broader semantic representations, reducing missed polyps and improving context robustness.

---

## 3.3 WDFFNet (Dual Backbone + Weighted Fusion + Attention Decoder)

WDFFNet integrates two backbones:

- EfficientNet-B0 branch,
- ResNet-50 branch,

using matched projection layers and per-scale **WeightedFusion** units. Each fusion output is refined by **Object-Aware Attention**:

1. Channel attention via global pooling and bottleneck MLP-like convs,
2. Spatial attention via max/avg channel maps and 3×3 conv.

Decoder design:

- progressive transposed-convolution upsampling,
- skip concatenation with fused encoder features,
- staged 3×3 refinement,
- final 1×1 projection to binary logits.

Output is interpolated to 256×256.

### Why WDFFNet helps
Combining two heterogeneous encoders improves feature diversity. Weighted fusion and attention increase saliency for small or ambiguous lesions.

---

## 4. Ensemble Architecture

A learnable weighted ensemble is constructed from frozen base-model outputs:

1. Generate logits from ResUNet++, TransFuse, WDFFNet.
2. Stack into 3-channel prediction tensor.
3. Pass through a lightweight ensemble head:
   - Conv(3→16, 3×3) + ReLU,
   - Conv(16→3, 1×1),
   - Softmax across model-channel dimension.
4. Weighted sum of stacked predictions gives final fused map.

Mathematically:

\[
\hat{Y} = \sum_{m=1}^{3} w_m(X) \cdot P_m(X), \quad \sum_m w_m(X)=1
\]

where \(P_m\) are base-model predictions and \(w_m\) are input-adaptive softmax weights.

This is superior to plain averaging because contribution weights adapt spatially and sample-wise.

---

## 5. Loss Function and Optimization

All base models are trained independently with:

- **Adam** optimizer, learning rate = 1e-4.
- Composite loss:

\[
\mathcal{L} = 0.5\,\mathcal{L}_{BCE} + 0.5\,\mathcal{L}_{Dice}
\]

Dice term (from probabilities) is:

\[
\mathcal{L}_{Dice}=1-\frac{2\sum_i p_i g_i + \epsilon}{\sum_i p_i + \sum_i g_i + \epsilon}
\]

Best checkpoint is selected by validation Dice. Ensemble training freezes base models and optimizes only ensemble-head parameters.

---

## 6. Evaluation Metrics

For each model and the ensemble, compute:

- Dice score,
- Intersection-over-Union (IoU),
- Precision,
- Recall,
- Accuracy.

Definitions (TP/FP/FN/TN pixel-wise):

\[
\text{Dice} = \frac{2TP}{2TP + FP + FN}
\]
\[
\text{IoU} = \frac{TP}{TP + FP + FN}
\]
\[
\text{Precision} = \frac{TP}{TP + FP},\;
\text{Recall} = \frac{TP}{TP + FN},\;
\text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN}
\]

Additionally, report:

- Total and trainable parameter counts,
- Throughput (FPS) measured with warmup and timed inference loops.

---

## 7. Reproducibility and Engineering Controls

1. Fixed random seeds across Python/NumPy/PyTorch.
2. Unified preprocessing and split logic for all models.
3. Shape normalization utility to guarantee binary BCHW outputs.
4. Robust checkpoint loader that:
   - loads only key/shape-matching tensors,
   - skips incompatible tensors,
   - logs loaded/skipped layer summaries.
5. Debug prints for tensor I/O shapes at first forward pass.

These controls reduce experimental variance and prevent silent shape/configuration failures.

---


## 8. Ablation Analysis

The ablation analysis isolates the contribution of the most important methodological decisions in the proposed segmentation framework. All ablation experiments should use the same Kvasir-SEG split, the same 256×256 evaluation resolution, the same metric definitions, and the same post-processing threshold as the main comparison. This ensures that each reported change reflects the removed or replaced component rather than changes in evaluation protocol.

### 8.1 Preprocessing Standardization

This ablation compares a non-unified preprocessing setup against the proposed unified pipeline. The unified setup applies identical resizing, ImageNet normalization, mask scaling, and binary thresholding to ResUNet++, TransFuse, WDFFNet, and the ensemble.

| Setting                          | Dice | IoU | Precision | Recall | Accuracy |
|----------------------------------|------|-----|-----------|--------|----------|
| Non-unified preprocessing        | [ ]  | [ ] | [ ]       | [ ]    | [ ]      |
| Unified preprocessing (proposed) | [ ]  | [ ] | [ ]       | [ ]    | [ ]      |

**Interpretation:** This ablation demonstrates whether fair data handling improves model comparability. A gain in Dice and IoU under the unified protocol would indicate that preprocessing consistency reduces avoidable variance. If the scores remain similar, the result still supports the validity of the remaining experiments by showing that the proposed gains are not caused by hidden data-pipeline differences.

### 8.2 Base-Model Contribution

Each base model is evaluated independently before fusion. This establishes a baseline for the ensemble and identifies the unique behavior of each architecture.

| Model     | Dice | IoU | Precision | Recall | Accuracy |
|-----------|------|-----|-----------|--------|----------|
| ResUNet++ | [ ]  | [ ] | [ ]       | [ ]    | [ ]      |
| TransFuse | [ ]  | [ ] | [ ]       | [ ]    | [ ]      |
| WDFFNet   | [ ]  | [ ] | [ ]       | [ ]    | [ ]      |

**Interpretation:** ResUNet++ is expected to contribute boundary-sensitive local structure, TransFuse contributes broader semantic context through its dual stream, and WDFFNet contributes feature diversity through dual-backbone weighted fusion and attention. Reporting these models independently shows whether the ensemble is combining genuinely complementary predictors.

### 8.3 Fusion Strategy

This ablation compares the proposed learnable weighted ensemble with a simple non-learned average of the same base-model predictions.

| Fusion Strategy                        | Dice | IoU | Precision | Recall | Accuracy |
|----------------------------------------|------|-----|-----------|--------|----------|
| Simple average `(r + t + w) / 3`       | [ ]  | [ ] | [ ]       | [ ]    | [ ]      |
| Learnable weighted ensemble (proposed) | [ ]  | [ ] | [ ]       | [ ]    | [ ]      |

**Interpretation:** If the learnable ensemble outperforms simple averaging, the result supports the claim that spatially adaptive weighting is more effective than treating all base models as equally reliable for every pixel and image. Improvements in Recall should be discussed as better lesion coverage, while improvements in Precision should be discussed as better suppression of false-positive mucosal regions.

### 8.4 Checkpoint Loading and Output Normalization

Because the framework combines heterogeneous architectures, this ablation documents the effect of the engineering safeguards used for reproducible evaluation.

| Strategy                           | Dice | IoU | Notes |
|------------------------------------|------|-----|-------|
| Strict checkpoint loading          | [ ]  | [ ] | [ ]   |
| Shape-safe partial load (proposed) | [ ]  | [ ] | [ ]   |

**Interpretation:** Shape-safe partial loading and output normalization should be described as reliability controls rather than as performance-enhancing tricks. They allow compatible learned weights to be reused while preventing shape mismatches and inconsistent output formats from invalidating the comparison.

### 8.5 Ablation Conclusion

Overall, the ablation study should show that the final system benefits from three separable factors: a standardized evaluation protocol, complementary base architectures, and a learnable adaptive fusion mechanism. This strengthens the dissertation by demonstrating that the proposed ensemble improvement is not merely a consequence of implementation details or a single high-performing base model.

---

## 9. Suggested Dissertation Results Table Template

| Model     | Dice | IoU | Precision | Recall | Accuracy | Params | Trainable Params | FPS |
|-----------|------|-----|-----------|--------|----------|--------|------------------|-----|
| ResUNet++ |      |     |           |        |          |        |                  |     |
| TransFuse |      |     |           |        |          |        |                  |     |
| WDFFNet   |      |     |           |        |          |        |                  |     |
| Ensemble  |      |     |           |        |          |        |                  |     |

---

## 10. Suggested Dissertation Figures

1. **Pipeline overview**: data flow from preprocessing → three base models → ensemble head.
2. **Model architecture diagrams**:
   - ResUNet++ macro-structure,
   - TransFuse dual-branch fusion,
   - WDFFNet dual-backbone fusion + attention decoder.
3. **Training curves**: train/val loss, val Dice per epoch (all models).
4. **Qualitative segmentation panel**: input, GT, each model output, ensemble output.
5. **Bar plot**: Dice/IoU comparison across models.

---

## 11. Discussion Points for PhD Writing

1. **Why standardization matters**: architecture-level gains are obscured when preprocessing differs.
2. **Error modes**: small-flat polyps, specular highlights, blurred boundaries, stool artifacts.
3. **Model complementarity**:
   - ResUNet++: strong local contour learning,
   - TransFuse: improved global-context sensitivity,
   - WDFFNet: feature diversity via dual-backbone fusion and attention.
4. **Why ensemble improves generalization**: adaptive weighting mitigates model-specific failure modes.
5. **Limitations**: dataset size, single-dataset evaluation, possible domain shift across hospitals/devices.
6. **Future work**: cross-dataset validation, uncertainty estimation, semi-supervised learning, test-time adaptation.

---

## 12. Example Methodology Paragraph (Ready to paste)

> We implemented a unified segmentation framework in PyTorch for Kvasir-SEG using ResUNet++, TransFuse, and WDFFNet under identical preprocessing conditions (256×256 resize and ImageNet normalization). Each model was trained independently with a balanced Dice-BCE objective and Adam optimization (1e-4), selecting checkpoints by best validation Dice. To leverage model complementarity, we introduced a learnable weighted ensemble that freezes base networks and estimates adaptive per-pixel fusion weights via a lightweight softmax head. Evaluation was performed on a held-out test split using Dice, IoU, Precision, Recall, and Accuracy, supplemented with computational profiling (parameter counts and FPS). This design ensures architectural fairness, reproducibility, and robust comparative analysis for medical image segmentation.

---

## 13. Example Contributions List (Ready to paste)

1. A fully standardized training/evaluation protocol for three heterogeneous segmentation models on Kvasir-SEG.
2. Practical resolution of architecture-checkpoint mismatches through robust, shape-safe partial weight loading.
3. A learnable weighted ensemble that improves robustness over naive averaging by adaptive model weighting.
4. A reproducible Colab-compatible codebase with unified metrics, visualization, and profiling utilities.

