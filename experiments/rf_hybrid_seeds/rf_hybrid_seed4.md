# Tuned RF / Lightweight RF / Hybrid RF 비교

- 특징 세트: no_hip, 결측 처리 A, 9개 클래스, 참가자 3명 (p01, p02, p03), LOPO, seed=4.
- Tuned/Hybrid 탐색: 각각 RandomizedSearchCV 20회, GroupKFold(그룹=참가자×촬영회차), 탐색 시 5프레임마다 1개 사용.
- Hybrid 정의: Lightweight의 복잡도 제한 안에서 하이퍼파라미터를 탐색한 모델.
- 참가자가 3명으로 적으므로 평균 차이뿐 아니라 참가자별 결과와 폴드 간 편차를 함께 해석한다.

## 요약 (참가자별 결과의 평균 ± 표준편차)

| 모델 | Macro-F1 | Accuracy | 모델 크기(KB) | 추론 중앙값(ms) | 추론 p95(ms) | 학습 시간(s) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.647 ± 0.048 | 0.67 | 21421.37 | 13.47 | 16.24 | 1.90 |
| Lightweight | 0.661 ± 0.043 | 0.68 | 2287.13 | 2.28 | 3.13 | 0.34 |
| Tuned | 0.637 ± 0.039 | 0.65 | 1079.28 | 2.70 | 3.94 | 0.40 |
| Hybrid | 0.660 ± 0.050 | 0.67 | 1280.24 | 1.87 | 2.56 | 0.26 |

## 참가자별 Macro-F1

| 모델 | p01 | p02 | p03 |
|---|---|---|---|
| Baseline | 0.705 | 0.649 | 0.588 |
| Lightweight | 0.715 | 0.657 | 0.610 |
| Tuned | 0.688 | 0.594 | 0.631 |
| Hybrid | 0.729 | 0.636 | 0.614 |

## 클래스별 Recall(%, 참가자 평균)

| 자세 | Baseline | Lightweight | Tuned | Hybrid |
|---|---|---|---|---|
| back_hard | 46.6 | 45.1 | 46.8 | 45.6 |
| back_light | 46.8 | 56.3 | 43.1 | 49.8 |
| back_neck | 94.8 | 92.6 | 97.4 | 97.4 |
| chin | 84.9 | 83.5 | 70.9 | 76.4 |
| forward | 60.0 | 65.5 | 65.5 | 68.1 |
| neck_hard | 73.4 | 72.1 | 63.9 | 67.1 |
| neck_light | 59.5 | 60.3 | 64.6 | 66.6 |
| normal | 87.0 | 86.9 | 84.8 | 86.3 |
| slouch | 51.0 | 49.5 | 51.5 | 52.3 |

## 폴드별 선택 하이퍼파라미터

### Tuned
- p01 제외 학습: `{'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': 5, 'class_weight': 'balanced_subsample'}`
- p02 제외 학습: `{'n_estimators': 30, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'log2', 'max_depth': 5, 'class_weight': 'balanced_subsample'}`
- p03 제외 학습: `{'n_estimators': 50, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': 10, 'class_weight': 'balanced'}`

### Hybrid
- p01 제외 학습: `{'n_estimators': 40, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 10, 'class_weight': 'balanced'}`
- p02 제외 학습: `{'n_estimators': 50, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 5, 'class_weight': 'balanced'}`
- p03 제외 학습: `{'n_estimators': 30, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': 10, 'class_weight': None}`

