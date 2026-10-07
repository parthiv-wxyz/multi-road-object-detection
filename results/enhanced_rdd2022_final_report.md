# Enhanced RDD2022 Final — Training & Evaluation Report

## 1. Experiment Overview

**Experiment name:** `enhanced_rdd2022_final`  
**Model:** Enhanced YOLOv5-based road-damage detector  
**Training duration:** 100 epochs  
**Input size:** 640 × 640  
**Task:** Multi-class road-damage object detection

This report documents the final training run and the evaluation artifacts supplied for the `enhanced_rdd2022_final` experiment.

---

## 2. Final Performance

The final recorded evaluation values are:

| Metric | Final value |
|---|---:|
| **Precision** | **61.09%** |
| **Recall** | **55.69%** |
| **mAP@0.5** | **57.38%** |
| **mAP@0.5:0.95** | **28.94%** |
| **Best F1** | **≈ 0.58** |
| **F1-optimal confidence** | **≈ 0.194** |

The final mAP@0.5 is **0.5738**, while the stricter COCO-style mAP@0.5:0.95 is **0.2894**.

The F1-confidence curve reaches approximately **0.58 at a confidence threshold of 0.194**, indicating that the precision/recall balance is strongest at a relatively low confidence threshold.

> **Important:** The F1-optimal threshold is an evaluation result. It should not automatically be used as the final deployment threshold without testing false positives and false negatives on representative images.

---

## 3. Training Configuration

The available training curves indicate a **100-epoch** YOLOv5 training run.

The training process shows:

- progressive reduction in bounding-box loss;
- progressive reduction in objectness loss;
- progressive reduction in classification loss;
- increasing precision;
- increasing recall;
- increasing mAP throughout training.

The model therefore demonstrates stable convergence rather than a severe training collapse.

---

## 4. Training and Validation Loss Analysis

### Training losses

Approximate final training losses:

| Loss | Final value |
|---|---:|
| Train box loss | **0.0346** |
| Train objectness loss | **0.00467** |
| Train classification loss | **0.00415** |

### Validation losses

Approximate final validation losses:

| Loss | Final value |
|---|---:|
| Validation box loss | **0.0411** |
| Validation objectness loss | **0.00289** |
| Validation classification loss | **0.00872** |

The training losses decrease consistently across the 100 epochs. Validation losses also decrease substantially. The validation classification loss shows a small late-stage increase, but the detection metrics continue improving, so the run does not show evidence of catastrophic overfitting.

---

## 5. Precision, Recall and mAP

The training results show the following final values:

- **Precision = 0.6109**
- **Recall = 0.5569**
- **mAP@0.5 = 0.5738**
- **mAP@0.5:0.95 = 0.2894**

The relatively small gap between precision and recall indicates a more balanced detector than a model that achieves high precision by rejecting a large number of true detections.

The mAP@0.5 result is substantially higher than the stricter mAP@0.5:0.95 value, which is expected because the latter requires accurate localization across multiple IoU thresholds.

---

## 6. Class-wise Average Precision

The supplied Precision–Recall curve reports the following class-wise AP@0.5 values:

| Class | AP@0.5 |
|---|---:|
| **Other corruption** | **0.730** |
| **Alligator crack** | **0.646** |
| **Longitudinal crack** | **0.530** |
| **Transverse crack** | **0.523** |
| **Pothole** | **0.440** |
| **Overall** | **0.574** |

### Interpretation

**Other corruption** is the strongest class with an AP of approximately **73.0%**.

**Alligator crack** performs comparatively well at approximately **64.6%**.

**Longitudinal crack** and **transverse crack** are moderate at approximately **53.0%** and **52.3%**, respectively.

**Pothole** is the weakest class with an AP of approximately **44.0%**.

The class-wise results indicate that pothole detection is the main remaining performance bottleneck.

---

## 7. Confusion Matrix Analysis

The normalized confusion matrix reports approximately:

| True class | Correct prediction |
|---|---:|
| Longitudinal crack | **0.49** |
| Transverse crack | **0.42** |
| Alligator crack | **0.56** |
| Other corruption | **0.70** |
| Pothole | **0.37** |

The strongest diagonal value is associated with **other corruption (0.70)**.

The weakest is **pothole (0.37)**.

The confusion matrix also shows a substantial background component, indicating that some ground-truth damage instances are not being detected at the selected evaluation conditions.

This agrees with the PR-curve result where pothole has the lowest AP.

---

## 8. F1-Confidence Analysis

The F1-confidence curve reports:

> **Maximum overall F1 ≈ 0.58 at confidence ≈ 0.194**

The curve rises rapidly at low confidence values, reaches its maximum around 0.19–0.20, and then decreases as the confidence threshold becomes more restrictive.

This indicates a precision–recall trade-off:

- very low thresholds retain more detections but may introduce false positives;
- higher thresholds improve confidence filtering but remove lower-confidence true detections;
- approximately **0.194** provides the best F1 balance in the supplied validation curve.

For deployment, a threshold around **0.20–0.30** should be tested experimentally rather than selecting a threshold solely from the curve.

---

## 9. Dataset Distribution

The supplied label-distribution plot shows an imbalanced distribution of road-damage instances.

Approximate relative distribution visible in the plot:

- **Longitudinal crack:** more than 18,000 instances
- **Transverse crack:** approximately 8,500 instances
- **Alligator crack:** approximately 7,500 instances
- **Other corruption:** approximately 7,500 instances
- **Pothole:** approximately 4,600 instances

Thus, pothole has substantially fewer examples than longitudinal crack.

This imbalance is consistent with the lower pothole AP and confusion-matrix performance.

---

## 10. Bounding-Box Distribution

The supplied label-analysis plots show:

- many small bounding boxes;
- a strong concentration of low normalized width/height values;
- elongated bounding-box distributions;
- crack instances concentrated toward particular image regions;
- significant variation in object size.

These characteristics are challenging for object detectors because crack damage can be thin, elongated, low contrast and spatially small.

The bounding-box distribution therefore supports the need for effective multi-scale feature extraction and localization.

---

## 11. Validation Predictions

The validation prediction examples show that the model is able to detect visible road damage, including:

- longitudinal cracks;
- potholes;
- other road-surface damage.

However, the supplied predictions also show cases where multiple overlapping predictions can occur around the same damaged region.

This suggests that inference-stage optimization should be considered, particularly:

- confidence threshold tuning;
- NMS IoU threshold tuning;
- class-specific threshold analysis;
- inspection of overlapping predictions.

This issue should be separated from the training-performance assessment because NMS and confidence filtering directly affect the displayed detections.

---

## 12. Overall Assessment

### Strengths

1. **Stable 100-epoch convergence**
2. **mAP@0.5 of 57.38%**
3. **mAP@0.5:0.95 of 28.94%**
4. **Precision of 61.09%**
5. **Recall of 55.69%**
6. Strong performance on **other corruption**
7. Good performance on **alligator crack**
8. Training and validation curves do not show catastrophic divergence

### Limitations

1. **Pothole AP is only 44.0%**
2. Recall remains below 60%
3. Thin/elongated crack instances remain challenging
4. Class imbalance affects the minority class
5. Some overlapping/duplicate detections are visible in validation predictions
6. More evaluation is required before selecting a final deployment threshold

---

# 13. Comparison With the Original RDD4D Paper

The supplied reference paper is:

**A. M. Alkalbani et al., “RDD4D: 4D-Attention-Guided Road Damage Detection and Classification,” IEEE Access, 2026.**

The paper introduces a **Diverse Road Damage Dataset (DRDD)** and an **Attention4D** architecture. The paper reports an overall **mAP of 0.446** on its proposed dataset and reports an AP of **0.458** for large-sized road cracks.

The paper's ablation study reports:

| Configuration | mAP | AP50 | AP75 | FPS |
|---|---:|---:|---:|---:|
| Single Attention4D block | 0.425 | — | — | — |
| Top-down placement | **0.446** | **0.687** | **0.451** | **26.8** |
| Bottom-up placement | 0.412 | — | — | — |
| Both paths | 0.455 | — | — | **25.1** |

The paper reports that placing Attention4D blocks in the top-down path provides a strong balance between accuracy and computational efficiency.

### Important comparability note

The comparison above is **not a direct benchmark comparison**.

The paper evaluates RDD4D on its **DRDD dataset**, while this project result is the **`enhanced_rdd2022_final` experiment** using the project's road-damage dataset and class configuration.

Therefore:

> **57.38% mAP@0.5 from this project must not be presented as a direct replacement for or direct apples-to-apples comparison against the paper's 0.446 mAP.**

The metrics use different datasets, evaluation settings and model architectures.

The correct academic wording is that the project result provides a **reference comparison with the published RDD4D work**, rather than claiming superiority or equivalence.

---

## 14. Paper vs. Project — Reference Comparison

| Aspect | Original RDD4D Paper | `enhanced_rdd2022_final` |
|---|---|---|
| Main architecture | RDD4D with Attention4D | Enhanced YOLOv5-based detector |
| Dataset | DRDD | Project road-damage dataset |
| Overall reported metric | mAP = **0.446** | mAP@0.5 = **0.574** |
| Stricter metric | Paper's reported mAP | mAP@0.5:0.95 = **0.289** |
| AP50 | **0.687** in reported top-down configuration | **0.574 overall** |
| Precision | Paper-specific evaluation | **0.611** |
| Recall | Paper-specific evaluation | **0.557** |
| F1 | Paper-specific evaluation | **≈0.58** |
| Best class in this run | Paper reports strong crack categories | **Other corruption: 0.730 AP50** |
| Weak class in this run | Potholes identified as challenging in paper | **Pothole: 0.440 AP50** |

### Interpretation

The original paper identifies potholes as a challenging category because of their irregular shapes and appearance variation. The current project shows a similar pattern: **pothole is also the weakest class in the supplied evaluation, with AP@0.5 = 0.440**.

This is a useful qualitative agreement between the project observations and the published study, although the numerical results should not be directly compared because the datasets and experimental setups differ.

---

# 15. Relation to the Original Paper

The original paper identifies several road-damage detection challenges:

- scale variation;
- different damage shapes and appearances;
- occlusion;
- image-quality variation;
- shadows;
- labeling noise;
- unwanted artifacts;
- patching noise.

The supplied project plots show related challenges, especially:

- small and elongated damage regions;
- class imbalance;
- pothole detection difficulty;
- background/missed detections;
- overlapping predictions.

The original paper also proposes future improvements including semantic augmentation, copy-paste augmentation, multi-scale training, focal/class-aware weighting, illumination normalization, temporal aggregation and model compression.

These techniques provide useful directions for improving the current project.

---

# 16. Recommended Next Experiments

Before another full training run, the following should be tested.

### 16.1 Inference threshold tuning

Test:

```text
0.10
0.15
0.20
0.25
0.30
0.35
0.40
0.50
```

Measure precision, recall and F1 for each threshold.

### 16.2 NMS tuning

Test multiple NMS IoU thresholds to reduce overlapping predictions.

### 16.3 Pothole-focused analysis

Investigate:

- pothole image diversity;
- pothole bounding-box size;
- difficult lighting conditions;
- partially visible potholes;
- class imbalance;
- false positives involving potholes.

### 16.4 Class-balanced augmentation

If retraining is required, consider targeted augmentation for underrepresented pothole examples rather than indiscriminate augmentation.

### 16.5 Best vs. last checkpoint

Evaluate both:

```text
best.pt
last.pt
```

on the same validation/test set.

The checkpoint with the strongest independent evaluation should be selected for deployment.

---

# 17. Final Result Statement

The **`enhanced_rdd2022_final`** experiment completed 100 epochs with stable convergence.

The final recorded results are:

> **Precision: 61.09%**  
> **Recall: 55.69%**  
> **mAP@0.5: 57.38%**  
> **mAP@0.5:0.95: 28.94%**  
> **Best F1: approximately 0.58 at confidence 0.194**

The detector performs best on **other corruption** and **alligator crack**, while **pothole** remains the primary weakness.

The results are suitable for inclusion in the project report as the **final `enhanced_rdd2022_final` training experiment**, with the qualification that comparison with the published RDD4D paper is a reference comparison rather than a direct benchmark.

---

## 18. Final Report Checklist

- [x] 100-epoch training results
- [x] Training loss curves
- [x] Validation loss curves
- [x] Precision curve
- [x] Recall curve
- [x] mAP@0.5
- [x] mAP@0.5:0.95
- [x] F1-confidence analysis
- [x] Precision–Recall analysis
- [x] Confusion matrix
- [x] Class distribution
- [x] Bounding-box distribution
- [x] Validation labels
- [x] Validation predictions
- [x] Class-wise AP analysis
- [x] Original RDD4D paper reference comparison
- [x] Limitations
- [x] Recommended future improvements

