# Tuned RF / Lightweight RF / Hybrid RF 비교

- 특징 세트: no_hip, 결측 처리 A, 9개 클래스, 참가자 3명 (p01, p02, p03), LOPO, seed=1.
- Tuned/Hybrid 탐색: 각각 RandomizedSearchCV 20회, GroupKFold(그룹=참가자×촬영회차), 탐색 시 5프레임마다 1개 사용.
- Hybrid 정의: Lightweight의 복잡도 제한 안에서 하이퍼파라미터를 탐색한 모델.
- 참가자가 3명으로 적으므로 평균 차이뿐 아니라 참가자별 결과와 폴드 간 편차를 함께 해석한다.

## 요약 (참가자별 결과의 평균 ± 표준편차)

| 모델 | Macro-F1 | Accuracy | 모델 크기(KB) | 추론 중앙값(ms) | 추론 p95(ms) | 학습 시간(s) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.655 ± 0.050 | 0.68 | 21363.99 | 13.06 | 15.48 | 1.99 |
| Lightweight | 0.662 ± 0.023 | 0.68 | 2265.61 | 2.38 | 3.36 | 0.36 |
| Tuned | 0.655 ± 0.048 | 0.67 | 1511.14 | 4.98 | 6.53 | 0.68 |
| Hybrid | 0.658 ± 0.031 | 0.67 | 1069.96 | 2.32 | 3.43 | 0.31 |

## 참가자별 Macro-F1

| 모델 | p01 | p02 | p03 |
|---|---|---|---|
| Baseline | 0.716 | 0.654 | 0.593 |
| Lightweight | 0.672 | 0.683 | 0.629 |
| Tuned | 0.723 | 0.626 | 0.615 |
| Hybrid | 0.700 | 0.646 | 0.628 |

## 클래스별 Recall(%, 참가자 평균)

| 자세 | Baseline | Lightweight | Tuned | Hybrid |
|---|---|---|---|---|
| back_hard | 45.0 | 50.6 | 47.2 | 45.8 |
| back_light | 48.8 | 48.1 | 39.1 | 40.7 |
| back_neck | 98.5 | 83.2 | 98.2 | 98.5 |
| chin | 84.7 | 83.5 | 78.0 | 75.9 |
| forward | 60.8 | 64.8 | 65.7 | 67.6 |
| neck_hard | 80.3 | 78.9 | 70.7 | 74.7 |
| neck_light | 56.8 | 63.1 | 67.8 | 69.1 |
| normal | 86.7 | 86.3 | 87.0 | 87.0 |
| slouch | 50.1 | 52.1 | 51.7 | 49.6 |

## 폴드별 선택 하이퍼파라미터

### Tuned
- p01 제외 학습: `{'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': 5, 'class_weight': None}`
- p02 제외 학습: `{'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 5, 'class_weight': 'balanced_subsample'}`
- p03 제외 학습: `{'n_estimators': 30, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': 15, 'class_weight': 'balanced'}`

### Hybrid
- p01 제외 학습: `{'n_estimators': 50, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': 5, 'class_weight': None}`
- p02 제외 학습: `{'n_estimators': 50, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 5, 'class_weight': 'balanced'}`
- p03 제외 학습: `{'n_estimators': 50, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 10, 'class_weight': 'balanced'}`

