# Tuned RF / Lightweight RF / Hybrid RF 비교

- 특징 세트: no_hip, 결측 처리 A, 9개 클래스, 참가자 3명 (p01, p02, p03), LOPO, seed=3.
- Tuned/Hybrid 탐색: 각각 RandomizedSearchCV 20회, GroupKFold(그룹=참가자×촬영회차), 탐색 시 5프레임마다 1개 사용.
- Hybrid 정의: Lightweight의 복잡도 제한 안에서 하이퍼파라미터를 탐색한 모델.
- 참가자가 3명으로 적으므로 평균 차이뿐 아니라 참가자별 결과와 폴드 간 편차를 함께 해석한다.

## 요약 (참가자별 결과의 평균 ± 표준편차)

| 모델 | Macro-F1 | Accuracy | 모델 크기(KB) | 추론 중앙값(ms) | 추론 p95(ms) | 학습 시간(s) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.653 ± 0.040 | 0.68 | 24083.81 | 22.95 | 33.57 | 1.10 |
| Lightweight | 0.676 ± 0.045 | 0.69 | 2207.17 | 3.87 | 6.02 | 0.30 |
| Tuned | 0.647 ± 0.023 | 0.66 | 1477.00 | 15.29 | 18.59 | 0.72 |
| Hybrid | 0.673 ± 0.037 | 0.69 | 503.64 | 2.60 | 3.73 | 0.19 |

## 참가자별 Macro-F1

| 모델 | p01 | p02 | p03 |
|---|---|---|---|
| Baseline | 0.701 | 0.654 | 0.603 |
| Lightweight | 0.735 | 0.668 | 0.626 |
| Tuned | 0.678 | 0.639 | 0.623 |
| Hybrid | 0.725 | 0.652 | 0.643 |

## 클래스별 Recall(%, 참가자 평균)

| 자세 | Baseline | Lightweight | Tuned | Hybrid |
|---|---|---|---|---|
| back_hard | 44.8 | 49.1 | 44.4 | 49.7 |
| back_light | 50.2 | 49.0 | 42.5 | 47.1 |
| back_neck | 98.5 | 97.8 | 86.5 | 98.9 |
| chin | 83.7 | 84.7 | 75.6 | 73.6 |
| forward | 60.3 | 64.6 | 68.0 | 70.0 |
| neck_hard | 78.8 | 73.6 | 71.5 | 78.1 |
| neck_light | 57.6 | 62.3 | 69.1 | 64.9 |
| normal | 87.4 | 88.0 | 86.7 | 80.8 |
| slouch | 49.1 | 55.8 | 54.0 | 56.1 |

## 폴드별 선택 하이퍼파라미터

### Tuned
- p01 제외 학습: `{'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'log2', 'max_depth': 5, 'class_weight': 'balanced'}`
- p02 제외 학습: `{'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': 5, 'class_weight': 'balanced'}`
- p03 제외 학습: `{'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'log2', 'max_depth': 5, 'class_weight': 'balanced'}`

### Hybrid
- p01 제외 학습: `{'n_estimators': 40, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 8, 'class_weight': 'balanced'}`
- p02 제외 학습: `{'n_estimators': 30, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 5, 'class_weight': None}`
- p03 제외 학습: `{'n_estimators': 30, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': 5, 'class_weight': 'balanced'}`

