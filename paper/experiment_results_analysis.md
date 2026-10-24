# Experiment Results Analysis — Seonwoo

## 1. Purpose

This document summarizes the existing experimental results of the desk posture monitoring system and proposes additional model comparison experiments for the research paper.

The main objectives are:
- Organize the existing experimental results.
- Compare model performance and computational efficiency.
- Identify limitations and additional experiments.
- Provide evidence for the experiment and results sections of the paper.

## 2. Summary of Existing Experiments

### 2.1 Depth Measurement

The depth measurement experiment reported a maximum error of 1.63% over the tested distance range of approximately 37–92 cm.

This result provides preliminary evidence that depth measurements can be used as an input to the posture monitoring system under the tested conditions.

The result should not be generalized to all distances or environments without additional validation.

### 2.2 Rule-Based Posture Judgment

Across five logs containing 32,925 frames, the rule-based system reported no warning-level false positives for normal posture.

However, some logs showed caution-or-higher judgments for normal posture at rates of approximately 0.7–1.2%.

This indicates that the judgment thresholds should be evaluated for both sensitivity to incorrect posture and stability during normal posture.

### 2.3 Random Forest Feature Comparison

The nine-class Leave-One-Participant-Out (LOPO) experiment reported the following Macro-F1 scores:

| Feature configuration | Macro-F1 |
|---|---:|
| no_hip with missing-data treatment A | 0.667 |
| all | 0.659 |
| depth | 0.553 |

The `no_hip` configuration performed best among these tested configurations.

However, the experiment included only three participants. The result should therefore be interpreted as preliminary evidence rather than proof of generalization to a wider population.

### 2.4 Random Forest Efficiency Comparison

| Model | Macro-F1 | Model size | Median inference time | P95 inference time |
|---|---:|---:|---:|---:|
| Baseline RF | 0.667 | 23,806 KB | 13.54 ms | 20.29 ms |
| Lightweight RF | 0.664 | 2,249 KB | 3.01 ms | 4.38 ms |
| Tuned RF | 0.653 | 2,989 KB | 8.50 ms | 14.46 ms |

Compared with the baseline RF, the lightweight RF reduced model size by approximately 90.6% and median inference time by approximately 77.8%.

Its Macro-F1 decreased from 0.667 to 0.664, a difference of 0.003.

These results suggest that the lightweight RF is a promising candidate when inference efficiency and model size are important. The final choice should consider both classification performance and computational cost.

### 2.5 MoveNet ROI Experiment

The ROI experiment reported the following changes:

| Metric | Before ROI | With ROI |
|---|---:|---:|
| Hip keypoint missing rate | 0.487 | 0.106 |
| Wrist valid rate | 0.863 | 0.814 |
| Wrist jitter | 11.722 | 17.010 |
| Warning ratio | 0.598 | 0.720 |

ROI reduced the hip keypoint missing rate, but wrist validity decreased and wrist jitter increased.

Therefore, ROI should not be described as improving all keypoint measurements. Its effects depend on the body part and metric being evaluated.

### 2.6 Rule-Based Detection by Posture

| Posture | Reported result |
|---|---:|
| back_hard | 89.7% |
| back_light | 94.7% |
| neck_hard | 96.9% |
| neck_light | 66.2% |
| forward | 76.9% |
| forward, compound judgment | 93.2% |
| slouch | 88.6% |
| slouch, compound judgment | 88.7% |
| chin | 28.1% |
| chin, compound judgment | 49.1% |

The results vary considerably by posture. In particular, chin-related detection showed comparatively low performance.

The wrist-face signal was missing in approximately 40% of cases, which may limit the reliability of features that depend on this signal.

The exact metric definitions and evaluation conditions should be confirmed from the original experiment files before these values are used in the paper.

## 3. Important Reporting Precautions

### 3.1 Do Not Mix Different Classification Tasks

The following files contain results for different classification settings:

- `experiments/rf_lopo.md`: nine-class results.
- `experiments/rf_tuning.md`: nine-class results.
- `experiments/rf_lopo_merged.md`: seven-class results with merged intensity labels.
- `experiments/rf_tuning_merged.md`: seven-class results with merged intensity labels.

Results from the nine-class and seven-class tasks must not be compared as if they came from the same classification problem.

### 3.2 Distinguish Paper Replication from Project Results

The 98% accuracy reported during replication of the external WRF paper is not the same as the performance obtained on this project's own dataset.

The paper's reported results and the project's experimental results must be presented separately.

### 3.3 State the Dataset Limitations

The current LOPO experiment includes three participants. This limits the strength of conclusions about generalization to unseen users.

Future experiments should include more participants and, where possible, repeated recordings under consistent conditions.

## 4. Proposed Model Comparison Experiment

The following models are proposed for a controlled comparison:

1. Baseline Random Forest
2. Lightweight Random Forest
3. Tuned Random Forest
4. XGBoost
5. LightGBM
6. Lightweight MLP

These are proposed experiments, not completed results.

### 4.1 Experimental Controls

To make the comparison fair, use the same:

- Nine posture labels.
- `no_hip` feature configuration as the initial common feature set.
- Missing-data treatment A.
- LOPO evaluation protocol.
- Training and test participant separation.
- Metric calculation procedure.

Any deviation from these settings should be documented.

Preprocessing operations, including imputation and feature scaling where applicable, must be fitted using training data only within each fold. Test-participant data must not be used to fit preprocessing parameters.

### 4.2 Evaluation Metrics

Report the following metrics:

| Category | Metric | Purpose |
|---|---|---|
| Classification | Macro-F1 | Evaluates performance across classes |
| Classification | Accuracy | Measures overall correctness |
| Classification | Per-class recall | Identifies poorly detected postures |
| Classification | Confusion matrix | Shows patterns of misclassification |
| Efficiency | Model size | Measures storage requirements |
| Efficiency | Median inference time | Measures typical inference latency |
| Efficiency | P95 inference time | Measures slower inference cases |

Macro-F1 should be emphasized when class performance is uneven.

For reliable latency comparisons, document the hardware, software environment, measurement procedure, and whether preprocessing is included in the timing.

### 4.3 Additional Experiments

Recommended follow-up experiments include:

- Feature ablation to determine which features contribute most to classification.
- ROI evaluation using the same recordings and comparable conditions.
- Additional participant-level validation.
- Error analysis for difficult classes such as chin-related postures.
- Runtime evaluation under realistic camera operation.

## 5. Recommended Tables and Figures

The paper's experimental results section could include:

1. A table comparing the feature configurations.
2. A table comparing RF model performance and computational efficiency.
3. A confusion matrix for the selected model.
4. A per-class recall comparison.
5. A figure comparing keypoint missing rates before and after ROI.
6. A table summarizing limitations and proposed follow-up experiments.

Only figures and tables supported by completed experiments should be presented as actual results. Planned experiments should be clearly identified as future work.

## 6. Preliminary Conclusions

The existing experiments suggest three main findings.

First, the `no_hip` feature configuration achieved the highest Macro-F1 among the tested feature configurations in the reported nine-class LOPO experiment.

Second, the lightweight RF substantially reduced model size and inference latency while maintaining a Macro-F1 close to the baseline RF.

Third, ROI improved hip keypoint availability but introduced trade-offs in wrist-related measurements. Its effectiveness should therefore be evaluated using multiple metrics rather than a single measure.

These findings are preliminary. The limited participant count, differences between classification tasks, and class-specific detection errors must be considered when drawing conclusions.

The next step is to complete a controlled comparison of candidate models and report classification performance alongside computational efficiency.
