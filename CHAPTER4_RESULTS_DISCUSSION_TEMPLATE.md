# Chapter 4: Results and Discussion (Template)

> **How to use this template:** Replace all bracketed placeholders such as `[XX.XX]` and `[MODEL_NAME]` with your measured results from the final experiment logs.

## 4.1 Chapter Overview

This chapter presents the quantitative and qualitative evaluation of ResUNet++, TransFuse, WDFFNet, and the proposed learnable weighted ensemble on Kvasir-SEG. Results are organized into: (i) primary segmentation metrics, (ii) computational efficiency metrics, (iii) visual quality assessment, and (iv) critical discussion of strengths, limitations, and implications.

## 4.2 Experimental Configuration for Final Reporting

- Input resolution: **256×256**
- Preprocessing: ImageNet normalization (mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
- Loss: \(0.5\,\text{BCE} + 0.5\,\text{Dice}\)
- Optimizer: Adam, learning rate \(1\times10^{-4}\)
- Best model selection criterion: validation Dice
- Ensemble training: frozen base models, learnable softmax weighting head
- Hardware: `[GPU model / CPU details]`
- Framework versions: `[PyTorch version]`, `[CUDA version]`, `[timm version]`

## 4.3 Quantitative Results

### 4.3.1 Primary Segmentation Metrics

Table 4.1 summarizes the test-set performance of all methods.

**Table 4.1. Segmentation performance on Kvasir-SEG test set.**

| Model     | Dice | IoU | Precision | Recall | Accuracy |
|-----------|------|-----|-----------|--------|----------|
| ResUNet++ | [ ]  | [ ] | [ ]       | [ ]    | [ ]      |
| TransFuse | [ ]  | [ ] | [ ]       | [ ]    | [ ]      |
| WDFFNet   | [ ]  | [ ] | [ ]       | [ ]    | [ ]      |
| Ensemble  | [ ]  | [ ] | [ ]       | [ ]    | [ ]      |

**Narrative example (replace values):**

The proposed ensemble achieved the highest Dice score of **[XX.XX]**, exceeding the best individual model (**[MODEL_NAME]**, Dice **[XX.XX]**) by **[ΔXX.XX]** absolute points. A similar trend was observed for IoU, where the ensemble reached **[XX.XX]** compared with **[XX.XX]** for the strongest base model.

### 4.3.2 Computational Profile

**Table 4.2. Model complexity and throughput.**

| Model     | Total Params | Trainable Params | FPS |
|-----------|--------------|------------------|-----|
| ResUNet++ | [ ]          | [ ]              | [ ] |
| TransFuse | [ ]          | [ ]              | [ ] |
| WDFFNet   | [ ]          | [ ]              | [ ] |
| Ensemble  | [ ]          | [ ]              | [ ] |

**Narrative example:**

Although the ensemble improved segmentation quality, it incurred additional inference cost relative to single-model inference, with FPS changing from **[XX.XX]** (best base model) to **[XX.XX]** (ensemble). This trade-off is acceptable for `[real-time / near-real-time / offline]` deployment scenarios.

## 4.4 Training Dynamics

### 4.4.1 Loss Curves

Figure 4.1–4.4 should show per-model train/validation loss trajectories.

**Interpretation template:**

- ResUNet++ converged by epoch `[N]` with stable validation behavior.
- TransFuse showed `[faster/slower]` convergence and `[higher/lower]` variance.
- WDFFNet displayed `[steady/fluctuating]` validation Dice, likely due to `[reason]`.
- Ensemble-head training converged rapidly within `[N]` epochs, indicating effective reuse of frozen base predictions.

### 4.4.2 Dice Progression

Report best-validation Dice and epoch of peak performance for each model.

| Model     | Best Val Dice | Epoch at Best |
|-----------|---------------|---------------|
| ResUNet++ | [ ]           | [ ]           |
| TransFuse | [ ]           | [ ]           |
| WDFFNet   | [ ]           | [ ]           |
| Ensemble  | [ ]           | [ ]           |

## 4.5 Qualitative Results

Include representative examples showing:

1. Input image,
2. Ground-truth mask,
3. ResUNet++ prediction,
4. TransFuse prediction,
5. WDFFNet prediction,
6. Ensemble prediction.

**Discussion template:**

Qualitative inspection indicates that the ensemble better preserves fine polyp boundaries in challenging frames with specular highlights and low contrast. In difficult cases where individual models under-segment (e.g., `[case description]`), the ensemble provides more complete lesion coverage while reducing false positives in background mucosal regions.

## 4.6 Comparative Discussion

### 4.6.1 Why the Ensemble Improves

The weighted fusion mechanism learns context-adaptive model contributions, allowing the system to rely on:

- contour-sensitive predictions from ResUNet++,
- context-enriched outputs from TransFuse,
- multi-branch fused representations from WDFFNet.

This complementary behavior explains improvements in overlap metrics over uniform averaging.

### 4.6.2 Error Analysis

Common failure modes observed in `[N]` difficult samples:

- tiny flat polyps with weak contrast,
- blur/motion artifacts,
- specular reflections,
- stool-like distractors.

Add a short per-failure-case analysis and mention whether ensemble mitigates or amplifies each issue.

### 4.6.3 Clinical and Deployment Perspective

From a practical standpoint, the system’s Dice of **[XX.XX]** and Recall of **[XX.XX]** suggest strong lesion-capture behavior. However, deployment in real-world endoscopy pipelines requires external validation on multi-center data and robustness testing under device/domain shift.

## 4.7 Ablation Study Template (Optional but Recommended)

### 4.7.1 Impact of Standardized Preprocessing

| Setting                               | Dice | IoU |
|---------------------------------------|------|-----|
| Non-unified preprocessing             | [ ]  | [ ] |
| Unified preprocessing (proposed)      | [ ]  | [ ] |

### 4.7.2 Ensemble Strategy Comparison

| Ensemble Type                         | Dice | IoU |
|--------------------------------------|------|-----|
| Simple average (r+t+w)/3             | [ ]  | [ ] |
| Learnable weighted ensemble (proposed)| [ ] | [ ] |

### 4.7.3 Checkpoint Loading Strategy

| Loading Strategy                      | Dice | Notes |
|---------------------------------------|------|-------|
| Strict load only                      | [ ]  | [ ]   |
| Shape-safe partial load (proposed)    | [ ]  | [ ]   |

## 4.8 Threats to Validity

1. Single-dataset evaluation may overestimate generalization.
2. Split sensitivity may influence absolute scores.
3. Throughput values depend on hardware/software stack.
4. Visual conclusions are partly subjective unless supported by blinded review.

## 4.9 Chapter Conclusion

This chapter demonstrated that the learnable weighted ensemble provides `[improved/comparable]` segmentation accuracy relative to individual architectures, with a manageable computational overhead. The results validate the proposed standardized methodology and motivate future extensions toward cross-dataset generalization, uncertainty-aware inference, and clinically robust deployment.

---

## Appendix: Ready-to-paste Result Summary Paragraph

> On the Kvasir-SEG test split, the proposed weighted ensemble achieved the strongest overall performance, yielding Dice/IoU values of **[XX.XX]/[XX.XX]**, compared with **[XX.XX]/[XX.XX]** for the best standalone model. Precision and Recall were measured at **[XX.XX]** and **[XX.XX]**, respectively, indicating a favorable balance between false-positive control and lesion coverage. Although ensemble inference introduced additional computational overhead, the measured throughput of **[XX.XX] FPS** remained feasible for `[target setting]`. Qualitative visualization further confirmed improved boundary delineation and robustness in challenging low-contrast frames.
