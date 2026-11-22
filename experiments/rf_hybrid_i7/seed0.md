# Tuned RF / Lightweight RF / Hybrid RF 비교

- 특징 세트: no_hip, 결측 처리 A, 9개 클래스, 참가자 3명 (p01, p02, p03), LOPO, seed=0.
- Tuned/Hybrid 탐색: 각각 RandomizedSearchCV 20회, GroupKFold(그룹=참가자×촬영회차), 탐색 시 5프레임마다 1개 사용.
- Hybrid 정의: Lightweight의 복잡도 제한 안에서 하이퍼파라미터를 탐색한 모델.
- 참가자가 3명으로 적으므로 평균 차이뿐 아니라 참가자별 결과와 폴드 간 편차를 함께 해석한다.

## 요약 (참가자별 결과의 평균 ± 표준편차)

| 모델 | Macro-F1 | Accuracy | 모델 크기(KB) | 추론 중앙값(ms) | 추론 p95(ms) | 학습 시간(s) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.667 ± 0.051 | 0.69 | 23805.79 | 23.39 | 31.18 | 1.12 |
| Lightweight | 0.664 ± 0.031 | 0.68 | 2248.69 | 3.31 | 5.04 | 0.30 |
| Tuned | 0.653 ± 0.047 | 0.67 | 2989.41 | 12.80 | 15.83 | 0.85 |
| Hybrid | 0.665 ± 0.058 | 0.68 | 623.09 | 2.85 | 4.38 | 0.21 |

## 참가자별 Macro-F1

| 모델 | p01 | p02 | p03 |
|---|---|---|---|
| Baseline | 0.729 | 0.669 | 0.603 |
| Lightweight | 0.699 | 0.667 | 0.624 |
| Tuned | 0.716 | 0.636 | 0.606 |
| Hybrid | 0.746 | 0.636 | 0.612 |

## 클래스별 Recall(%, 참가자 평균)

| 자세 | Baseline | Lightweight | Tuned | Hybrid |
|---|---|---|---|---|
| back_hard | 46.3 | 44.9 | 42.4 | 49.0 |
| back_light | 57.2 | 51.6 | 41.9 | 43.5 |
| back_neck | 98.5 | 96.1 | 98.3 | 94.4 |
| chin | 85.2 | 77.5 | 75.3 | 73.0 |
| forward | 59.5 | 62.9 | 68.5 | 69.2 |
| neck_hard | 80.6 | 83.6 | 69.6 | 67.5 |
| neck_light | 59.3 | 64.6 | 65.9 | 66.8 |
| normal | 87.3 | 86.4 | 86.6 | 86.6 |
| slouch | 49.6 | 50.6 | 56.6 | 62.2 |

## 폴드별 선택 하이퍼파라미터

### Tuned
- p01 제외 학습: `{'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': None, 'class_weight': None}`
- p02 제외 학습: `{'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'log2', 'max_depth': 5, 'class_weight': 'balanced_subsample'}`
- p03 제외 학습: `{'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 5, 'class_weight': None}`

### Hybrid
- p01 제외 학습: `{'n_estimators': 50, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 8, 'class_weight': None}`
- p02 제외 학습: `{'n_estimators': 40, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 5, 'class_weight': None}`
- p03 제외 학습: `{'n_estimators': 40, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 5, 'class_weight': None}`

