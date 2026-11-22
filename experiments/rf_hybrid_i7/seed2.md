# Tuned RF / Lightweight RF / Hybrid RF 비교

- 특징 세트: no_hip, 결측 처리 A, 9개 클래스, 참가자 3명 (p01, p02, p03), LOPO, seed=2.
- Tuned/Hybrid 탐색: 각각 RandomizedSearchCV 20회, GroupKFold(그룹=참가자×촬영회차), 탐색 시 5프레임마다 1개 사용.
- Hybrid 정의: Lightweight의 복잡도 제한 안에서 하이퍼파라미터를 탐색한 모델.
- 참가자가 3명으로 적으므로 평균 차이뿐 아니라 참가자별 결과와 폴드 간 편차를 함께 해석한다.

## 요약 (참가자별 결과의 평균 ± 표준편차)

| 모델 | Macro-F1 | Accuracy | 모델 크기(KB) | 추론 중앙값(ms) | 추론 p95(ms) | 학습 시간(s) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.659 ± 0.056 | 0.68 | 24146.14 | 23.20 | 28.17 | 1.06 |
| Lightweight | 0.670 ± 0.034 | 0.69 | 2170.16 | 3.93 | 5.28 | 0.29 |
| Tuned | 0.669 ± 0.040 | 0.69 | 1097.92 | 11.31 | 13.68 | 0.89 |
| Hybrid | 0.671 ± 0.044 | 0.68 | 364.52 | 3.88 | 6.55 | 0.28 |

## 참가자별 Macro-F1

| 모델 | p01 | p02 | p03 |
|---|---|---|---|
| Baseline | 0.726 | 0.663 | 0.589 |
| Lightweight | 0.714 | 0.666 | 0.630 |
| Tuned | 0.719 | 0.667 | 0.621 |
| Hybrid | 0.733 | 0.646 | 0.634 |

## 클래스별 Recall(%, 참가자 평균)

| 자세 | Baseline | Lightweight | Tuned | Hybrid |
|---|---|---|---|---|
| back_hard | 47.4 | 49.3 | 43.7 | 57.3 |
| back_light | 50.8 | 49.8 | 45.8 | 42.0 |
| back_neck | 98.6 | 98.5 | 98.2 | 98.1 |
| chin | 83.3 | 83.0 | 78.3 | 74.6 |
| forward | 59.1 | 64.4 | 70.6 | 70.5 |
| neck_hard | 82.9 | 79.6 | 75.0 | 74.5 |
| neck_light | 59.4 | 62.5 | 69.4 | 69.1 |
| normal | 87.2 | 86.5 | 87.3 | 85.6 |
| slouch | 47.9 | 48.1 | 52.5 | 46.7 |

## 폴드별 선택 하이퍼파라미터

### Tuned
- p01 제외 학습: `{'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 5, 'class_weight': None}`
- p02 제외 학습: `{'n_estimators': 50, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 5, 'class_weight': 'balanced_subsample'}`
- p03 제외 학습: `{'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': 5, 'class_weight': 'balanced_subsample'}`

### Hybrid
- p01 제외 학습: `{'n_estimators': 50, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'log2', 'max_depth': 5, 'class_weight': None}`
- p02 제외 학습: `{'n_estimators': 50, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': 5, 'class_weight': 'balanced'}`
- p03 제외 학습: `{'n_estimators': 50, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 5, 'class_weight': None}`

