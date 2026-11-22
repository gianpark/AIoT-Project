# Tuned RF / Lightweight RF / Hybrid RF 비교

- 특징 세트: no_hip, 결측 처리 A, 9개 클래스, 참가자 3명 (p01, p02, p03), LOPO, seed=1.
- Tuned/Hybrid 탐색: 각각 RandomizedSearchCV 20회, GroupKFold(그룹=참가자×촬영회차), 탐색 시 5프레임마다 1개 사용.
- Hybrid 정의: Lightweight의 복잡도 제한 안에서 하이퍼파라미터를 탐색한 모델.
- 참가자가 3명으로 적으므로 평균 차이뿐 아니라 참가자별 결과와 폴드 간 편차를 함께 해석한다.

## 요약 (참가자별 결과의 평균 ± 표준편차)

| 모델 | Macro-F1 | Accuracy | 모델 크기(KB) | 추론 중앙값(ms) | 추론 p95(ms) | 학습 시간(s) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.662 ± 0.048 | 0.68 | 24075.49 | 22.88 | 34.01 | 1.10 |
| Lightweight | 0.667 ± 0.037 | 0.69 | 2247.01 | 3.83 | 5.06 | 0.29 |
| Tuned | 0.660 ± 0.044 | 0.67 | 978.13 | 10.09 | 12.09 | 0.64 |
| Hybrid | 0.663 ± 0.040 | 0.68 | 350.11 | 3.65 | 4.82 | 0.24 |

## 참가자별 Macro-F1

| 모델 | p01 | p02 | p03 |
|---|---|---|---|
| Baseline | 0.721 | 0.663 | 0.603 |
| Lightweight | 0.693 | 0.693 | 0.615 |
| Tuned | 0.723 | 0.630 | 0.627 |
| Hybrid | 0.700 | 0.680 | 0.607 |

## 클래스별 Recall(%, 참가자 평균)

| 자세 | Baseline | Lightweight | Tuned | Hybrid |
|---|---|---|---|---|
| back_hard | 46.8 | 45.6 | 47.0 | 43.7 |
| back_light | 47.7 | 48.8 | 40.5 | 41.2 |
| back_neck | 98.5 | 94.0 | 98.2 | 98.3 |
| chin | 86.3 | 86.9 | 77.3 | 84.8 |
| forward | 60.8 | 65.0 | 68.4 | 68.2 |
| neck_hard | 77.6 | 80.6 | 70.8 | 72.0 |
| neck_light | 60.1 | 64.7 | 68.5 | 66.9 |
| normal | 87.0 | 85.4 | 86.6 | 85.3 |
| slouch | 51.8 | 48.0 | 51.9 | 53.2 |

## 폴드별 선택 하이퍼파라미터

### Tuned
- p01 제외 학습: `{'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': 5, 'class_weight': None}`
- p02 제외 학습: `{'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 5, 'class_weight': 'balanced_subsample'}`
- p03 제외 학습: `{'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 5, 'class_weight': None}`

### Hybrid
- p01 제외 학습: `{'n_estimators': 50, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': 5, 'class_weight': None}`
- p02 제외 학습: `{'n_estimators': 50, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': 5, 'class_weight': None}`
- p03 제외 학습: `{'n_estimators': 40, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'log2', 'max_depth': 5, 'class_weight': 'balanced'}`

