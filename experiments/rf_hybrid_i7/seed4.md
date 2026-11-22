# Tuned RF / Lightweight RF / Hybrid RF 비교

- 특징 세트: no_hip, 결측 처리 A, 9개 클래스, 참가자 3명 (p01, p02, p03), LOPO, seed=4.
- Tuned/Hybrid 탐색: 각각 RandomizedSearchCV 20회, GroupKFold(그룹=참가자×촬영회차), 탐색 시 5프레임마다 1개 사용.
- Hybrid 정의: Lightweight의 복잡도 제한 안에서 하이퍼파라미터를 탐색한 모델.
- 참가자가 3명으로 적으므로 평균 차이뿐 아니라 참가자별 결과와 폴드 간 편차를 함께 해석한다.

## 요약 (참가자별 결과의 평균 ± 표준편차)

| 모델 | Macro-F1 | Accuracy | 모델 크기(KB) | 추론 중앙값(ms) | 추론 p95(ms) | 학습 시간(s) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.658 ± 0.064 | 0.68 | 24175.09 | 23.38 | 29.76 | 1.13 |
| Lightweight | 0.652 ± 0.029 | 0.67 | 2226.12 | 3.47 | 6.24 | 0.28 |
| Tuned | 0.650 ± 0.058 | 0.67 | 1341.74 | 6.05 | 8.94 | 0.64 |
| Hybrid | 0.649 ± 0.051 | 0.66 | 531.91 | 2.73 | 4.35 | 0.22 |

## 참가자별 Macro-F1

| 모델 | p01 | p02 | p03 |
|---|---|---|---|
| Baseline | 0.740 | 0.652 | 0.582 |
| Lightweight | 0.680 | 0.662 | 0.613 |
| Tuned | 0.732 | 0.617 | 0.601 |
| Hybrid | 0.718 | 0.632 | 0.598 |

## 클래스별 Recall(%, 참가자 평균)

| 자세 | Baseline | Lightweight | Tuned | Hybrid |
|---|---|---|---|---|
| back_hard | 45.1 | 44.2 | 41.5 | 44.8 |
| back_light | 56.2 | 48.0 | 45.4 | 43.8 |
| back_neck | 98.4 | 86.0 | 96.9 | 97.6 |
| chin | 83.4 | 86.9 | 71.7 | 75.6 |
| forward | 59.1 | 64.8 | 67.1 | 67.7 |
| neck_hard | 76.8 | 76.8 | 67.9 | 67.4 |
| neck_light | 59.5 | 61.1 | 64.3 | 62.8 |
| normal | 87.2 | 86.1 | 86.5 | 86.4 |
| slouch | 49.7 | 51.1 | 60.3 | 53.3 |

## 폴드별 선택 하이퍼파라미터

### Tuned
- p01 제외 학습: `{'n_estimators': 50, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': 10, 'class_weight': 'balanced'}`
- p02 제외 학습: `{'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': 5, 'class_weight': 'balanced_subsample'}`
- p03 제외 학습: `{'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': 5, 'class_weight': 'balanced_subsample'}`

### Hybrid
- p01 제외 학습: `{'n_estimators': 30, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 8, 'class_weight': 'balanced'}`
- p02 제외 학습: `{'n_estimators': 50, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 5, 'class_weight': None}`
- p03 제외 학습: `{'n_estimators': 50, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 5, 'class_weight': 'balanced'}`

