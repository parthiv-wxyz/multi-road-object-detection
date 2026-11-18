# IDD Object Detection — Final Training Report

## Project: Road Object Detection using YOLOv5

> **Report status:** Training-results draft. Dataset class information is verified from the uploaded `data.yaml`. Dataset image counts, hardware, exact training command, and independent test-set metrics still need to be inserted if required by the final report.

---

# 1. INTRODUCTION

## 1.1 Objective

The objective of this stage of the project is to develop and evaluate a YOLOv5-based object detection model for identifying important road-scene objects in the Indian Driving Dataset (IDD).

The model is trained to detect seven road-scene object categories: person, car, autorickshaw, truck, bus, motorcycle, and bicycle.

## 1.2 Scope of the Project

The IDD training stage covers dataset preparation, YOLO-format annotation handling, model training, validation, quantitative performance evaluation, class-wise analysis, error analysis, and qualitative inspection of predicted detections.

The trained model provides a baseline for subsequent improvements to the overall road-object detection system.

## 1.3 Organization of the Report

This section documents the IDD dataset, training methodology, implementation, quantitative results, class-wise analysis, qualitative results, limitations, and possible future improvements.

---

# 2. PROBLEM STATEMENT

Road environments contain multiple object categories with substantial variation in scale, appearance, illumination, viewpoint, occlusion, and background complexity. A practical road-object detection system must identify these objects accurately while maintaining a useful balance between precision and recall.

The objective of the IDD experiment is therefore to train a YOLOv5 detector capable of recognizing multiple road-scene objects and to quantitatively evaluate its detection performance.

---

# 3. LITERATURE / TECHNICAL BACKGROUND

The project uses YOLOv5 as the object-detection framework. YOLO-based detectors formulate object detection as a single-stage prediction problem, allowing object localization and classification to be performed efficiently.

For this experiment, the IDD data is represented in YOLO format and the detector is evaluated using precision, recall, mAP@0.5, and mAP@0.5:0.95.

The final report should include the project's selected research papers and other sources in the References section.

---

# 4. PROJECT METHODOLOGY

## 4.1 Dataset Preparation

The uploaded IDD `data.yaml` confirms the following YOLO dataset configuration:

- Training images: `E:/Parthiv/multi-road-object-detection/datasets/idd/train/images`
- Validation images: `E:/Parthiv/multi-road-object-detection/datasets/idd/val/images`
- Number of classes: 7
- A test path is present in the configuration as a commented entry and therefore is **not currently active**.

The dataset configuration identifies seven object classes:

| ID | Class |
|---:|---|
| 0 | person |
| 1 | car |
| 2 | autorickshaw |
| 3 | truck |
| 4 | bus |
| 5 | motorcycle |
| 6 | bicycle |

The class names and ordering above are taken directly from the uploaded `data.yaml`. fileciteturn1file0L9-L17

## 4.2 Model Training

The YOLOv5 detector was trained on the IDD training set and evaluated on the configured validation set.

The training experiment contains **100 epochs**, corresponding to epochs 0–99 in the supplied `results.csv`.

The training process monitored:

- Training box loss
- Training objectness loss
- Training classification loss
- Validation box loss
- Validation objectness loss
- Validation classification loss
- Precision
- Recall
- mAP@0.5
- mAP@0.5:0.95

## 4.3 Evaluation

The model was evaluated using both quantitative and qualitative measures.

Quantitative evaluation included:

- Precision
- Recall
- mAP@0.5
- mAP@0.5:0.95
- Precision–Recall curves
- F1-confidence curve
- Precision-confidence curve
- Recall-confidence curve
- Confusion matrix

Qualitative evaluation used validation images with ground-truth annotations and model predictions.

---

# 5. IMPLEMENTATION

## 5.1 Dataset Configuration

The final IDD dataset configuration contains seven classes:

```yaml
nc: 7

names:
  0: person
  1: car
  2: autorickshaw
  3: truck
  4: bus
  5: motorcycle
  6: bicycle
```

The configured training and validation directories are:

```text
train: E:/Parthiv/multi-road-object-detection/datasets/idd/train/images
val: E:/Parthiv/multi-road-object-detection/datasets/idd/val/images
```

The test path shown in the uploaded configuration is commented out, so it should not be described as an active test split without additional evidence. fileciteturn1file0L1-L6

## 5.2 Model Checkpoints

Two trained model checkpoints were supplied:

- `best.pt` — best model checkpoint saved during training.
- `last.pt` — model checkpoint from the final training epoch.

The exact epoch at which `best.pt` was saved should be confirmed from the training log before the final report is submitted.

---

# 6. RESULT AND ANALYSIS

## 6.1 Final Performance

The final row of the supplied `results.csv` corresponds to epoch 99.

| Metric | Final Value |
|---|---:|
| Precision | **82.16%** |
| Recall | **57.84%** |
| mAP@0.5 | **65.24%** |
| mAP@0.5:0.95 | **42.98%** |

Therefore, the final validation performance is:

- **Precision = 0.82162**
- **Recall = 0.57838**
- **mAP@0.5 = 0.65238**
- **mAP@0.5:0.95 = 0.42982**

### Best Recorded Metrics

| Metric | Best Epoch | Best Value |
|---|---:|---:|
| mAP@0.5 | 93 | **0.65301** |
| mAP@0.5:0.95 | 93 | **0.43031** |

## 6.2 Loss Analysis

The final losses recorded at epoch 99 are:

| Loss | Value |
|---|---:|
| Training Box Loss | **0.03714** |
| Training Objectness Loss | **0.03161** |
| Training Classification Loss | **0.00783** |
| Validation Box Loss | **0.03654** |
| Validation Objectness Loss | **0.02303** |
| Validation Classification Loss | **0.01034** |

The training curves show a substantial decrease in all three training losses during the early stages of training. The rate of decrease becomes smaller as the number of epochs increases.

The validation box and classification losses also decrease and become relatively stable toward the end of training. This indicates that the model has reached a comparatively stable training state.

## 6.3 Precision Analysis

The final precision is approximately **82.16%**.

The precision curve increases rapidly during the early training period and remains around the low-to-mid 0.8 range during the later epochs.

A high precision indicates that a large proportion of the detections produced by the model correspond to correct object predictions.

## 6.4 Recall Analysis

The final recall is approximately **57.84%**.

Recall improves strongly during the early epochs and then gradually approaches a plateau.

The difference between precision and recall indicates that the model is more successful at producing correct detections than at recovering every ground-truth object. This suggests that missed detections remain an important area for improvement.

## 6.5 mAP@0.5 Analysis

The final mAP@0.5 is approximately **65.24%**.

The mAP@0.5 curve rises quickly during the initial training period and gradually approaches a plateau near 0.65.

This indicates that the detector achieves moderate overall object-detection performance when a 0.5 IoU threshold is used.

## 6.6 mAP@0.5:0.95 Analysis

The final mAP@0.5:0.95 is approximately **42.98%**.

This metric is more stringent because it averages Average Precision across IoU thresholds from 0.5 to 0.95.

The lower value compared with mAP@0.5 indicates that localization performance becomes more challenging at stricter IoU requirements.

---

# 7. CLASS-WISE ANALYSIS

## 7.1 Precision–Recall Results

The supplied Precision–Recall curve reports the following class-wise AP values at IoU 0.5:

| Class | AP@0.5 |
|---|---:|
| person | **0.511** |
| car | **0.717** |
| autorickshaw | **0.740** |
| truck | **0.669** |
| bus | **0.752** |
| motorcycle | **0.696** |
| bicycle | **0.485** |
| **All classes** | **0.653** |

The highest AP values are observed for:

1. **bus — 0.752**
2. **autorickshaw — 0.740**
3. **car — 0.717**

The lowest AP values are observed for:

1. **bicycle — 0.485**
2. **person — 0.511**

The class-wise results show that the model performs differently across object categories. Vehicle classes such as bus, autorickshaw, and car achieve stronger AP values, whereas bicycle and person are more challenging for this trained model.

---

# 8. F1-CONFIDENCE ANALYSIS

The supplied F1-confidence curve shows an overall maximum F1 value of approximately **0.68** at a confidence threshold of approximately **0.326**.

This indicates that a confidence threshold near 0.326 provides a useful balance between precision and recall for the evaluated validation predictions.

The threshold should not automatically be treated as the final deployment threshold. The appropriate threshold depends on the application's priority:

- Increasing the confidence threshold generally favors precision.
- Decreasing the confidence threshold generally favors recall.

For the final system, the selected threshold should be validated on an independent test set.

---

# 9. CONFUSION MATRIX ANALYSIS

The supplied confusion matrix shows relatively strong diagonal responses for several classes.

Approximate diagonal values shown are:

| Class | Matrix Value |
|---|---:|
| person | 0.47 |
| car | 0.66 |
| autorickshaw | 0.67 |
| truck | 0.62 |
| bus | 0.67 |
| motorcycle | 0.67 |
| bicycle | 0.48 |

The matrix also contains background-related errors. The relatively lower diagonal values for person and bicycle are consistent with their lower AP values in the Precision–Recall curve.

The confusion matrix therefore indicates that the model has difficulty with some object instances even though it performs relatively well on several vehicle classes.

---

# 10. QUALITATIVE ANALYSIS

The supplied validation prediction images demonstrate detections of objects including:

- truck
- car
- person
- motorcycle

The prediction images show confidence scores associated with detected objects.

Examples include high-confidence detections such as truck and motorcycle instances, while some person detections are produced with lower confidence.

The qualitative results demonstrate that the trained detector can identify multiple road-scene objects in realistic Indian road environments.

However, qualitative examples alone cannot establish overall performance. They must be interpreted together with precision, recall, mAP, and confusion-matrix results.

---

# 11. DISCUSSION

The IDD training experiment produced a functional multi-class road-object detector.

### Main strengths

- Final precision of approximately **82.16%**.
- Final mAP@0.5 of approximately **65.24%**.
- Strong performance for bus, autorickshaw, and car classes.
- Stable loss reduction during training.
- Successful detection of multiple road-scene object categories.
- Training performance approaches a plateau during later epochs.

### Main limitations

- Recall is approximately **57.84%**, substantially lower than precision.
- Bicycle and person have lower AP values than several vehicle classes.
- mAP@0.5:0.95 is approximately **42.98%**, indicating stricter localization remains challenging.
- Background-related errors remain visible in the confusion matrix.
- The uploaded `data.yaml` does not currently define an active test path; therefore the reported metrics should be described as validation metrics unless a separate test evaluation is performed.

---


# 10. COMPARISON WITH THE ORIGINAL RESEARCH PAPER

## 10.1 Comparison with the Paper's YOLOv5s Baseline

The original paper *Enhanced YOLOv5s Model for Improved Multi-Sized Object Detection in Road Scenes* reports a YOLOv5s baseline on the IDD dataset with **precision = 0.60, recall = 0.35, and mAP@0.5 = 0.28** in its anchor-box experiment. fileciteturn2file6

| Metric | Paper YOLOv5s Baseline | Current IDD Training | Improvement |
|---|---:|---:|---:|
| Precision | 0.6000 | **0.8216** | **+0.2216 (+36.94%)** |
| Recall | 0.3500 | **0.5784** | **+0.2284 (+65.25%)** |
| mAP@0.5 | 0.2800 | **0.6524** | **+0.3724 (+132.99%)** |
| mAP@0.5:0.95 | Not reported for this comparison | **0.4298** | — |

The current model therefore performs substantially better than the reported paper baseline on all three directly comparable metrics. In particular, mAP@0.5 increases from **0.28 to 0.6524**.

This should be described as a **comparison against the paper's reported baseline**, not as a strict reproduction, unless the training configuration, preprocessing, dataset split, hardware, and evaluation protocol have been verified to be identical.

## 10.2 Comparison with the Paper's Proposed Enhanced Model

The original paper's proposed model combines percentile-based anchor refinement, modified kernel sizes and channel configuration, ECA attention, BiFPN feature fusion, and CutMix augmentation. The paper reports approximately **85% precision, 79% recall, and 77% mAP@0.5** for its final enhanced model. fileciteturn2file9turn2file2

| Metric | Current IDD Training | Paper Proposed Model | Difference |
|---|---:|---:|---:|
| Precision | **0.8216 (82.16%)** | 0.8500 (85%) | **-0.0284** |
| Recall | **0.5784 (57.84%)** | 0.7900 (79%) | **-0.2116** |
| mAP@0.5 | **0.6524 (65.24%)** | 0.7700 (77%) | **-0.1176** |

The current IDD model is below the paper's fully enhanced model, particularly in recall and mAP@0.5. This is consistent with treating the current experiment as the **baseline IDD training stage** before implementing the full set of proposed enhancements.

## 10.3 Runtime Comparison

The paper reports approximately **9.8 ms inference time** for its baseline and **14.6 ms** for its proposed model. It also reports approximately **7.2M parameters** for the baseline and **8.9M parameters** for the proposed model, with the proposed model reported at approximately **86 FPS**. fileciteturn2file2

A runtime comparison for the current model is not included because the supplied training results do not contain a verified inference benchmark. It should be added after benchmarking `best.pt` under documented hardware and inference settings.

## 10.4 Overall Comparison

| Model | Precision | Recall | mAP@0.5 |
|---|---:|---:|---:|
| Paper — YOLOv5s baseline | 0.60 | 0.35 | 0.28 |
| **Current IDD training** | **0.8216** | **0.5784** | **0.6524** |
| Paper — proposed enhanced model | 0.85 | 0.79 | 0.77 |

### Key Finding

The current IDD training result is **clearly stronger than the original paper's reported YOLOv5s baseline**, but it does **not yet reach the performance reported by the paper's fully enhanced model**.

This establishes a quantitative baseline for the next project stage, where the planned architectural enhancements can be implemented and evaluated against the current model.

> **Note:** The paper reports different baseline figures in different experiments. Its anchor-box experiment reports **0.60 precision, 0.35 recall, and 0.28 mAP@0.5**, while its broader runtime/model comparison discusses a baseline mAP of approximately **0.31**. fileciteturn2file6turn2file5 The comparison above uses **0.60 / 0.35 / 0.28** because these are the paper's directly reported three-metric baseline in the anchor-box experiment.

# 12. CONCLUSION

The IDD training stage successfully trained a YOLOv5-based multi-class road-object detector for seven object categories: person, car, autorickshaw, truck, bus, motorcycle, and bicycle.

After 100 training epochs, the final validation results were:

- **Precision: 82.16%**
- **Recall: 57.84%**
- **mAP@0.5: 65.24%**
- **mAP@0.5:0.95: 42.98%**

The results show that the detector can identify a range of road-scene objects with good precision. The strongest class-wise AP values are obtained for bus, autorickshaw, and car, while bicycle and person remain comparatively difficult classes.

The trained IDD model can therefore serve as a baseline for subsequent improvements to the project's road-object detection architecture.

---

# 13. FUTURE WORK

The following improvements can be considered:

1. Improve recall while preserving the current precision level.
2. Improve small-object detection.
3. Improve person and bicycle detection.
4. Optimize the confidence threshold using an independent test set.
5. Investigate data augmentation strategies.
6. Improve localization accuracy at higher IoU thresholds.
7. Compare the baseline YOLOv5 model with enhanced model architectures.
8. Perform controlled ablation studies.
9. Evaluate inference speed and computational cost.
10. Perform final evaluation on a separate test set.

---

# 14. APPENDIX — FIGURES

The following figures should be included in the final report:

1. **Training and validation curves**
   - `results.png`

2. **F1-confidence curve**
   - `F1_curve.png`

3. **Precision–Recall curve**
   - `PR_curve.png`

4. **Precision-confidence curve**
   - `P_curve.png`

5. **Recall-confidence curve**
   - `R_curve.png`

6. **Confusion matrix**
   - `confusion_matrix(1).png`

7. **Validation ground-truth examples**
   - `val_batch0_labels.jpg`

8. **Validation prediction examples**
   - `val_batch0_pred.jpg`

---

# 15. INFORMATION TO ADD BEFORE FINAL SUBMISSION

The following items are not established by the supplied `data.yaml` and should be filled from the project records rather than guessed:

- Exact number of IDD training images.
- Exact number of IDD validation images.
- Exact number of annotated instances per class.
- Exact model YAML/configuration.
- Exact training command.
- Batch size.
- Optimizer.
- Learning rate.
- Hardware/GPU used.
- Training time.
- Exact epoch corresponding to `best.pt`.
- Independent test-set evaluation.
- Inference speed/FPS, if required.
- Final references used in the project.

## Important reporting rule

The metrics in this document come from the training/validation results. They should **not** be described as test-set accuracy.

A separate test evaluation should be performed if the final report requires claims about performance on unseen test data.
