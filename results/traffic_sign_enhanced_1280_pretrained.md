# Traffic Sign Enhanced 1280 Pretrained --- Final Evaluation

## Model

**Model:** `traffic_sign_enhanced_1280_pretrained`\
**Task:** Traffic Sign Detection\
**Framework:** YOLOv5\
**Training:** 100 epochs\
**Input resolution:** 1280 × 1280

------------------------------------------------------------------------

## 1. Final Performance

  Metric           Final Value
  -------------- -------------
  Precision         **89.25%**
  Recall            **77.65%**
  mAP@0.5           **80.19%**
  mAP@0.5:0.95      **45.86%**

### Best observed values

  Metric                 Best Value
  ------------------- -------------
  Best Precision        **100.00%**
  Best Recall            **78.32%**
  Best mAP@0.5           **80.30%**
  Best mAP@0.5:0.95      **47.17%**

------------------------------------------------------------------------

## 2. F1-Confidence Analysis

The F1-confidence curve indicates:

-   **Maximum overall F1 ≈ 0.80**
-   Optimal confidence ≈ **0.431**

Therefore, a confidence threshold around **0.43** is a strong starting
point for inference.

Recommended confidence values for controlled testing:

``` text
0.30
0.35
0.40
0.43
0.45
0.50
0.55
```

The final deployment threshold should be selected after checking missed
detections and duplicate detections on representative road scenes.

------------------------------------------------------------------------

## 3. Class-wise AP@0.5

  Class                      AP@0.5 Assessment
  --------------------- ----------- ------------
  Regulatory              **0.935** Excellent
  Mandatory               **0.953** Excellent
  Informatory             **0.502** Weak
  General                 **0.849** Good
  Warning                 **0.733** Moderate
  **Overall mAP@0.5**     **0.795** Good

The strongest classes are **Mandatory** and **Regulatory**.

The principal weakness is **Informatory**, followed by **Warning**.

------------------------------------------------------------------------

## 4. Confusion Matrix Analysis

The diagonal values indicate strong classification for several classes:

``` text
Regulatory     → 0.91
Mandatory      → 0.98
Informatory    → 0.42
General        → 0.80
Warning        → 0.77
```

The background row indicates missed detections:

``` text
Regulatory     → background = 0.07
Mandatory      → background = 0.02
Informatory    → background = 0.50
General        → background = 0.20
Warning        → background = 0.23
```

### Main observation

The model's main limitation is not severe class confusion. The larger
issue is **missed detections**, particularly for the Informatory class.

The Informatory class has approximately **0.50 background** in the
confusion matrix, indicating that many ground-truth Informatory signs
are not detected.

------------------------------------------------------------------------

## 5. Training Convergence

The training curves show consistent reduction in the training losses:

-   `train/box_loss` decreases to approximately **0.028**
-   `train/obj_loss` decreases to approximately **0.0057**
-   `train/cls_loss` decreases to approximately **0.0035**

Validation losses also decrease:

-   `val/box_loss` → approximately **0.0316**
-   `val/obj_loss` → approximately **0.0043**
-   `val/cls_loss` → approximately **0.0041**

The mAP curves increase throughout training and approach a plateau near
the final epochs.

There is no obvious indication of catastrophic overfitting from the
supplied training curves.

------------------------------------------------------------------------

## 6. Precision-Recall Characteristics

The PR curve reports:

  Class              AP@0.5
  ------------- -----------
  Regulatory      **0.935**
  Mandatory       **0.953**
  Informatory     **0.502**
  General         **0.849**
  Warning         **0.733**
  All classes     **0.795**

The high AP values for Regulatory and Mandatory demonstrate reliable
detection for these categories.

The lower Informatory AP indicates that additional data and/or targeted
optimization is required for this category.

------------------------------------------------------------------------

## 7. Detection Results

The qualitative prediction results demonstrate that the model can detect
traffic signs in:

-   Highway scenes
-   Hazy conditions
-   Distant traffic-sign regions
-   Small traffic signs
-   Multiple-sign scenes
-   Different traffic-sign categories

The qualitative results also reveal duplicate detections in some images,
especially around Mandatory signs.

Example observed behavior:

``` text
Same sign region
      ↓
Multiple overlapping detections
      ↓
Duplicate labels
```

This indicates that inference-time confidence and NMS settings should be
tuned before another full training run.

------------------------------------------------------------------------

## 8. Dataset Distribution

The supplied label-distribution visualization indicates an imbalance
between the five classes.

Approximate instance counts shown in the graph:

  Class           Approx. Instances
  ------------- -------------------
  Regulatory                  \~150
  Mandatory                   \~145
  Informatory                  \~55
  General                      \~95
  Warning                     \~160

The Informatory class has substantially fewer instances than the other
classes.

This imbalance is consistent with its weaker detection performance.

------------------------------------------------------------------------

## 9. Main Findings

### Strong points

-   **89.25% final precision**
-   **77.65% final recall**
-   **80.19% final mAP@0.5**
-   **45.86% final mAP@0.5:0.95**
-   Regulatory AP@0.5 = **93.5%**
-   Mandatory AP@0.5 = **95.3%**
-   Overall F1 ≈ **0.80**
-   Training and validation losses converge steadily
-   Works on real road scenes, including hazy highway conditions

### Weak points

-   Informatory AP@0.5 = **50.2%**
-   Informatory background/missed-detection value = **0.50**
-   Warning AP@0.5 = **73.3%**
-   Duplicate detections occur in some qualitative predictions
-   Dataset is imbalanced, particularly for Informatory

------------------------------------------------------------------------

## 10. Recommended Next Steps

### Step 1 --- Tune inference before retraining

Use the current `best.pt` and test different confidence thresholds:

``` text
0.30
0.35
0.40
0.43
0.45
0.50
0.55
```

Start with:

``` text
confidence = 0.43
```

because the F1-confidence curve reaches its overall maximum around this
threshold.

### Step 2 --- Tune NMS

The qualitative predictions contain overlapping duplicate detections.

Test appropriate NMS IoU thresholds together with confidence:

``` text
confidence ≈ 0.43
NMS IoU ≈ 0.45
```

Then compare duplicate detections, false positives, and missed signs.

### Step 3 --- Improve Informatory

Prioritize the Informatory class.

Recommended actions:

1.  Add more Informatory training images if available.
2.  Apply targeted augmentation.
3.  Improve representation of small Informatory signs.
4.  Re-evaluate the class after retraining.
5.  Compare class-wise recall and AP before and after the changes.

### Step 4 --- Re-evaluate Warning

After addressing Informatory, evaluate whether targeted improvements to
Warning are necessary.

------------------------------------------------------------------------

## 11. Current Project Status

  Component                       Status
  ------------------------------- ---------------------
  Enhanced YOLOv5s architecture   **Completed**
  1280 × 1280 training            **Completed**
  100-epoch training              **Completed**
  Validation                      **Completed**
  Confusion matrix                **Completed**
  PR/F1/P/R curves                **Completed**
  Qualitative detection testing   **Completed**
  Inference threshold tuning      **Next**
  Duplicate-detection reduction   **Next**
  Informatory improvement         **Next major task**

------------------------------------------------------------------------

## 12. Report-Ready Result

> The proposed Enhanced YOLOv5s traffic-sign detector trained at 1280 ×
> 1280 resolution for 100 epochs achieved a final precision of
> **89.25%**, recall of **77.65%**, mAP@0.5 of **80.19%**, and
> mAP@0.5:0.95 of **45.86%**. The model achieved its strongest
> class-wise performance for Mandatory and Regulatory signs, with AP@0.5
> values of **95.3%** and **93.5%**, respectively. The Informatory class
> remained the principal weakness with an AP@0.5 of **50.2%**, primarily
> associated with missed detections. The overall F1-confidence analysis
> indicated an optimal confidence threshold of approximately **0.431**.
> Qualitative evaluation showed successful detection under highway and
> hazy road conditions, although duplicate detections were observed in
> some scenes. Further optimization should therefore focus on
> inference-time NMS/confidence tuning and improving the representation
> of the underrepresented Informatory class.
