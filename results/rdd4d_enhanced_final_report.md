# RDD4D Enhanced Final — Training & Evaluation Report

## 1. Experiment Overview

**Experiment:** `rdd4d_enhanced_final`  
**Task:** Multi-class road-damage object detection  
**Training length:** 100 epochs  
**Evaluation artifacts:** `results(2).csv`, confusion matrix, F1/Precision/Recall curves, PR curve, label-distribution plots, validation labels and predictions, and the supplied `best(2).pt` / `last(2).pt` checkpoints.

The experiment uses five road-damage classes:

1. Alligator
2. Block
3. Longitudinal
4. Transversal
5. Pot Hole

The analysis below is based on the supplied artifacts. Architecture-specific claims are intentionally not added where the checkpoint could not be inspected reliably in the current environment.

---

# 2. Final Training Results

The CSV contains epochs **0–99**, corresponding to **100 training epochs**.

### Final epoch — epoch 99

| Metric | Value |
|---|---:|
| Precision | **0.54042** |
| Recall | **0.46427** |
| mAP@0.5 | **0.38368** |
| mAP@0.5:0.95 | **0.19792** |
| Train box loss | **0.03203** |
| Train objectness loss | **0.01523** |
| Train classification loss | **0.00511** |
| Validation box loss | **0.03582** |
| Validation objectness loss | **0.00966** |
| Validation classification loss | **0.00628** |

Therefore, the final-epoch detector achieves:

> **Precision = 54.04%**  
> **Recall = 46.43%**  
> **mAP@0.5 = 38.37%**  
> **mAP@0.5:0.95 = 19.79%**

---

# 3. Best Metric Values During Training

The best value for each metric does not necessarily occur at the final epoch.

| Metric | Best value | Epoch |
|---|---:|---:|
| Precision | **0.78954** | **64** |
| Recall | **0.46427** | **99** |
| mAP@0.5 | **0.39289** | **96** |
| mAP@0.5:0.95 | **0.20219** | **91** |

### Important distinction

The **best mAP@0.5 = 0.39289** occurs at epoch **96**, whereas the final epoch reaches **0.38368**.

The final epoch therefore is not the best epoch according to mAP@0.5.

Similarly, the best mAP@0.5:0.95 is **0.20219 at epoch 91**, slightly above the final value of **0.19792**.

For reporting model performance, the checkpoint corresponding to the selected best validation metric should be preferred over automatically reporting the last epoch.

---

# 4. Training Convergence

The training curves show a clear reduction in all three training losses.

### Training box loss

The box loss decreases from approximately **0.097** at the beginning of training to approximately **0.032** at epoch 99.

### Training objectness loss

The objectness loss decreases overall from approximately **0.019–0.020** to approximately **0.0152**.

### Training classification loss

The classification loss falls rapidly during the early epochs and stabilizes around **0.005–0.006** toward the end.

Overall, the training curves demonstrate that the model learned progressively during the 100-epoch run.

---

# 5. Validation Loss

The validation curves also show substantial reduction:

| Validation loss | Approx. initial | Final |
|---|---:|---:|
| Box loss | ~0.092 | **0.03582** |
| Objectness loss | ~0.0118 | **0.00966** |
| Classification loss | ~0.043 | **0.00628** |

The validation losses remain relatively stable during the later epochs.

There is no obvious catastrophic divergence between training and validation losses in the supplied curves.

---

# 6. Precision and Recall

The precision curve rises rapidly during the early training period and reaches approximately **0.79** around the later-middle part of training.

The highest recorded precision is:

> **Precision = 0.78954 at epoch 64**

However, precision subsequently fluctuates toward the end of training.

Recall improves more gradually and reaches its highest recorded value at the final epoch:

> **Recall = 0.46427 at epoch 99**

This indicates that the model becomes better at recovering true damage instances as training progresses, but the overall recall remains a significant limitation.

---

# 7. mAP Performance

### mAP@0.5

The mAP@0.5 curve increases steadily throughout training.

Best recorded value:

> **mAP@0.5 = 0.39289 at epoch 96**

Final value:

> **mAP@0.5 = 0.38368**

The difference between the best and final values is small:

> **0.39289 − 0.38368 = 0.00921**

Thus, the model reaches a relatively stable plateau near the end of training.

### mAP@0.5:0.95

Best recorded value:

> **0.20219 at epoch 91**

Final value:

> **0.19792**

The lower value under the stricter IoU range indicates that localization accuracy remains considerably harder than detection at IoU = 0.5.

---

# 8. Precision–Recall Curve

The supplied PR curve reports the following class-wise AP values:

| Class | AP@0.5 |
|---|---:|
| **Alligator** | **0.522** |
| **Block** | **0.257** |
| **Longitudinal** | **0.605** |
| **Transversal** | **0.579** |
| **Pot Hole** | **0.000** |
| **Overall mAP@0.5 shown by curve** | **0.392** |

### Interpretation

**Longitudinal** is the strongest class:

> AP = **0.605**

**Transversal** follows:

> AP = **0.579**

**Alligator** achieves:

> AP = **0.522**

**Block** is considerably weaker:

> AP = **0.257**

The most serious issue is:

> **Pot Hole AP = 0.000**

This means the supplied PR evaluation produced no meaningful average precision for the Pot Hole class at the displayed operating/evaluation conditions.

This should be treated as the primary weakness of this experiment.

---

# 9. F1–Confidence Curve

The overall F1 curve reports:

> **Maximum F1 ≈ 0.36 at confidence ≈ 0.138**

This is substantially lower than the precision value achievable at very high confidence.

The curve shows that F1 reaches its best balance at a relatively low confidence threshold and then gradually declines as the confidence threshold increases.

### Practical interpretation

At high confidence thresholds, precision can become very high, but recall falls sharply.

Therefore:

- **High confidence → fewer predictions, higher precision**
- **Low confidence → more predictions, higher recall**
- **Best overall F1 → approximately 0.138 confidence**

The threshold of **0.138** is an evaluation result and should not automatically be used as the deployment threshold. A separate test-set threshold study should be performed.

---

# 10. Precision–Confidence Curve

The overall precision curve reaches:

> **Precision = 1.00 at confidence ≈ 0.934**

This does **not** mean that the detector has perfect overall performance.

At such a high confidence threshold, very few predictions may remain. Consequently, recall can be extremely low.

The precision–confidence and recall–confidence curves must therefore be interpreted together.

The more meaningful operating point for balanced detection is closer to the F1 optimum.

---

# 11. Recall–Confidence Curve

The overall recall curve reports:

> **Maximum overall recall ≈ 0.67 at confidence = 0.000**

As the confidence threshold increases, recall decreases continuously.

This behavior is expected: increasing the threshold removes lower-confidence detections.

The supplied curve therefore reinforces the main problem of this experiment:

> **The model is losing a substantial number of true road-damage detections when stricter confidence filtering is applied.**

---

# 12. Confusion Matrix

The normalized confusion matrix is arranged with:

- **True classes on the horizontal axis**
- **Predicted classes on the vertical axis**

The most visible diagonal values are approximately:

| Class | Correct prediction / diagonal |
|---|---:|
| Alligator | **0.58** |
| Block | **0.00 / no visible correct cell** |
| Longitudinal | **0.65** |
| Transversal | **0.65** |
| Pot Hole | **0.00 / no visible correct cell** |

The matrix also shows strong background-related components, including approximately:

- Alligator → background: **0.41**
- Block → background: **0.92**
- Longitudinal → background: **0.34**
- Transversal → background: **0.34**
- Pot Hole → background: **1.00**

There are also notable cross-class/background prediction components, including values around:

- Longitudinal prediction against background: **0.51**
- Transversal prediction against background: **0.45**

### Main interpretation

The detector recognizes **Longitudinal** and **Transversal** substantially better than **Block** and **Pot Hole**.

The Pot Hole result is particularly problematic because its PR curve reports **AP = 0.000** and the confusion matrix does not show a meaningful diagonal detection component.

---

# 13. Class Distribution

The supplied label-distribution plot shows strong class imbalance.

Approximate instance counts visible in the plot are:

| Class | Approximate instances |
|---|---:|
| Alligator | ~250 |
| Block | ~70 |
| Longitudinal | ~1,600 |
| Transversal | ~2,100 |
| Pot Hole | ~20 |

The exact counts should be taken from the dataset audit if available; the values above are approximate readings from the supplied plot.

### Imbalance severity

The distribution is highly skewed toward:

1. **Transversal**
2. **Longitudinal**

while:

- **Block** is underrepresented;
- **Pot Hole** is extremely underrepresented.

This provides a strong explanation for the poor Block and Pot Hole performance.

---

# 14. Bounding-Box Distribution

The label plots show substantial variation in:

- object location;
- object width;
- object height;
- object scale.

The width/height distributions are strongly concentrated toward smaller values.

The road-damage objects are therefore frequently:

- small;
- thin;
- elongated;
- irregularly positioned.

This is especially challenging for crack detection because a crack can occupy a relatively small region while extending over a large distance.

The supplied bounding-box plots support the need for strong multi-scale feature extraction and careful localization.

---

# 15. Validation Labels

The validation-label images demonstrate examples containing:

- Longitudinal damage;
- Transversal damage;
- Alligator damage;
- multiple damage instances within the same image.

The annotations show that a single road image can contain multiple damage types and multiple damage instances.

This is an appropriate object-detection problem rather than a simple image-classification problem.

---

# 16. Validation Predictions

The validation prediction image shows that the model can detect several road-damage regions.

However, the predictions also reveal practical weaknesses:

### 16.1 Duplicate / overlapping detections

Some regions receive multiple predictions.

For example, in the first image, several **Transversal** detections appear around nearby road regions.

### 16.2 Classification confusion

Some detections can be assigned different crack categories in visually similar regions.

### 16.3 Low-confidence predictions

Several detections are around the **0.3–0.7** confidence range, indicating uncertainty.

### 16.4 Missing detections

Some visible road damage does not receive a corresponding prediction.

This is consistent with the relatively low overall recall.

---

# 17. Main Failure Case: Pot Hole

The strongest conclusion from the supplied artifacts is:

> **Pot Hole detection is currently not successful.**

Evidence:

1. **PR AP = 0.000**
2. No meaningful diagonal value is visible for Pot Hole in the confusion matrix.
3. The label-distribution plot shows Pot Hole as the rarest class.
4. The dataset therefore provides very little training evidence for this class.

This is not primarily an inference-threshold problem. The class requires additional training data and/or targeted training strategies.

---

# 18. Main Failure Case: Block

Block also performs poorly compared with the major crack classes.

The PR curve reports:

> **Block AP = 0.257**

The confusion matrix shows a very large background component for Block:

> approximately **0.92**

The class distribution also shows relatively few Block instances.

Therefore, Block suffers from the same fundamental issue of insufficient representation.

---

# 19. Strongest Classes

The best-performing classes are:

### Longitudinal

> **AP = 0.605**

### Transversal

> **AP = 0.579**

### Alligator

> **AP = 0.522**

These classes have substantially more training instances than Block and Pot Hole.

The model therefore learns the dominant crack patterns more effectively.

---

# 20. Overall Technical Assessment

## Strengths

- 100-epoch training completed.
- Training losses decrease consistently.
- Validation losses also decrease and stabilize.
- Precision reaches approximately **0.79** during training.
- Best mAP@0.5 reaches approximately **0.393**.
- Longitudinal AP reaches **0.605**.
- Transversal AP reaches **0.579**.
- Alligator AP reaches **0.522**.
- The detector successfully identifies multiple road-damage instances in several validation images.

## Weaknesses

- Final recall is only **0.464**.
- Final mAP@0.5 is **0.384**.
- Final mAP@0.5:0.95 is only **0.198**.
- Block AP is only **0.257**.
- Pot Hole AP is **0.000**.
- Severe class imbalance exists.
- Duplicate/overlapping predictions are visible.
- Some damage instances are missed.
- Localization performance drops significantly under stricter IoU evaluation.

---

# 21. Comparison With the Original RDD4D Paper

The supplied reference paper is:

**A. M. Alkalbani et al., “RDD4D: 4D-Attention-Guided Road Damage Detection and Classification,” IEEE Access, 2026.**

The paper proposes:

- the Diverse Road Damage Dataset (DRDD);
- the RDD4D detection model;
- the Attention4D feature-refinement mechanism.

The paper reports an overall:

> **mAP = 0.446**

on its proposed dataset.

It also reports:

> **AP = 0.458** for large-sized road cracks.

The paper's Attention4D placement ablation reports:

| Configuration | mAP | AP50 | AP75 | FPS |
|---|---:|---:|---:|---:|
| Single Attention4D block | 0.425 | — | — | — |
| Top-down path | **0.446** | **0.687** | **0.451** | **26.8** |
| Bottom-up path | 0.412 | — | — | — |
| Both paths | **0.455** | — | — | **25.1** |

The paper identifies potholes as a challenging category because of their irregular shape and appearance.

The current experiment shows a similar practical weakness:

> **Pot Hole AP = 0.000**

---

# 22. RDD4D Paper vs. Current Experiment

| Aspect | Published RDD4D | `rdd4d_enhanced_final` |
|---|---|---|
| Dataset | DRDD | Project road-damage dataset |
| Architecture | RDD4D + Attention4D | Current project enhanced model |
| Reported mAP | **0.446** | **0.38368 final** |
| Best mAP@0.5 | Paper uses its own evaluation protocol | **0.39289** |
| Final precision | Dataset/model-specific | **0.54042** |
| Final recall | Dataset/model-specific | **0.46427** |
| Strongest current class | — | **Longitudinal: 0.605 AP** |
| Weakest current class | Potholes identified as difficult | **Pot Hole: 0.000 AP** |

### Critical academic qualification

This is **not an apples-to-apples benchmark**.

The published RDD4D result and this experiment use different datasets and experimental conditions. Therefore, the current result should not be written as:

> “Our model is worse/better than RDD4D by X%.”

Instead, write:

> “The `rdd4d_enhanced_final` experiment achieved a final mAP@0.5 of 0.3837, while the published RDD4D study reports a mAP of 0.446 on its proposed DRDD dataset. Since the datasets and evaluation configurations differ, these values are provided as reference points rather than a direct benchmark comparison.”

---

# 23. Recommended Improvements

## Priority 1 — Fix Pot Hole data scarcity

This is the highest-impact issue.

Actions:

- increase the number of Pot Hole training instances;
- verify that Pot Hole labels are correctly formatted;
- inspect whether Pot Hole annotations are too small or inconsistent;
- add more diverse pothole images;
- use targeted augmentation rather than simply duplicating the same images.

## Priority 2 — Improve Block representation

Increase Block examples and inspect label quality.

## Priority 3 — Reduce duplicate predictions

Tune:

- confidence threshold;
- NMS IoU threshold;
- class-specific confidence thresholds.

The F1 curve indicates an overall optimum around **confidence = 0.138**, but this should be validated rather than blindly adopted.

## Priority 4 — Improve small/elongated object detection

The label plots indicate many small and thin objects.

Useful experiments include:

- higher input resolution;
- multi-scale training;
- small-object-focused augmentation;
- optimized anchors where appropriate;
- stronger feature fusion.

## Priority 5 — Improve localization

The difference between:

> mAP@0.5 = **0.39289 best**

and:

> mAP@0.5:0.95 = **0.20219 best**

shows that localization becomes substantially harder under stricter IoU requirements.

---

# 24. Recommended Next Experiment

Do **not** immediately run another 100-epoch training session with exactly the same dataset and configuration.

The current bottleneck is clearly class imbalance.

A better next experiment is:

### `rdd4d_enhanced_v2`

Focus on:

1. correcting/validating Pot Hole labels;
2. increasing Pot Hole samples;
3. increasing Block samples;
4. applying targeted augmentation;
5. testing higher input resolution;
6. tuning confidence and NMS thresholds;
7. comparing `best.pt` against `last.pt`;
8. evaluating class-wise AP after each change.

The objective should be to improve:

> **Pot Hole AP → meaningful non-zero AP**

while preserving the strong:

> **Longitudinal / Transversal performance**

---

# 25. Final Result Statement

The **`rdd4d_enhanced_final`** experiment completed 100 epochs and demonstrated stable training convergence.

The key final results are:

> **Precision = 54.04%**  
> **Recall = 46.43%**  
> **mAP@0.5 = 38.37%**  
> **mAP@0.5:0.95 = 19.79%**

The best validation values during training were:

> **Precision = 78.95% at epoch 64**  
> **Recall = 46.43% at epoch 99**  
> **mAP@0.5 = 39.29% at epoch 96**  
> **mAP@0.5:0.95 = 20.22% at epoch 91**

The class-wise PR analysis identifies:

> **Longitudinal = 0.605 AP**  
> **Transversal = 0.579 AP**  
> **Alligator = 0.522 AP**  
> **Block = 0.257 AP**  
> **Pot Hole = 0.000 AP**

The primary technical limitation is therefore **minority-class detection, especially Pot Hole and Block**, rather than failure of the model to learn the dominant crack categories.

---

# 26. Report-Ready Summary

**RDD4D Enhanced Final achieved a final precision of 54.04%, recall of 46.43%, mAP@0.5 of 38.37%, and mAP@0.5:0.95 of 19.79% after 100 epochs. The best mAP@0.5 of 39.29% was obtained at epoch 96. Class-wise analysis showed the strongest performance for Longitudinal cracks (AP@0.5 = 60.5%) and Transversal cracks (57.9%), followed by Alligator cracks (52.2%). Block detection remained weak (25.7%), while Pot Hole detection achieved 0% AP. The results indicate that the major limitation is severe class imbalance, particularly the very small number of Pot Hole and Block instances. Further improvement should prioritize minority-class data expansion, targeted augmentation, label verification, and inference-threshold/NMS optimization.**

---

## 27. Experiment Status

**Status:** Completed — baseline/final evaluation available.

**Recommended next stage:** `rdd4d_enhanced_v2` with targeted minority-class improvement.

