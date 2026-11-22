# Tuned RF / Lightweight RF / Hybrid RF 비교

- 특징 세트: no_hip, 결측 처리 A, 9개 클래스, 참가자 3명 (p01, p02, p03), LOPO, seed=0.
- Tuned/Hybrid 탐색: 각각 RandomizedSearchCV 20회, GroupKFold(그룹=참가자×촬영회차), 탐색 시 5프레임마다 1개 사용.
- Hybrid 정의: Lightweight의 복잡도 제한 안에서 하이퍼파라미터를 탐색한 모델.
- 참가자가 3명으로 적으므로 평균 차이뿐 아니라 참가자별 결과와 폴드 간 편차를 함께 해석한다.

## 요약 (참가자별 결과의 평균 ± 표준편차)

| 모델 | Macro-F1 | Accuracy | 모델 크기(KB) | 추론 중앙값(ms) | 추론 p95(ms) | 학습 시간(s) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.672 ± 0.067 | 0.69 | 21310.75 | 8.56 | 8.99 | 0.68 |
| Lightweight | 0.658 ± 0.033 | 0.68 | 2274.96 | 1.53 | 1.67 | 0.18 |
| Tuned | 0.657 ± 0.043 | 0.67 | 4609.29 | 3.85 | 4.89 | 0.33 |
| Hybrid | 0.681 ± 0.047 | 0.69 | 921.97 | 1.23 | 1.32 | 0.12 |

## 참가자별 Macro-F1

| 모델 | p01 | p02 | p03 |
|---|---|---|---|
| Baseline | 0.758 | 0.663 | 0.594 |
| Lightweight | 0.698 | 0.657 | 0.618 |
| Tuned | 0.716 | 0.639 | 0.616 |
| Hybrid | 0.746 | 0.636 | 0.660 |

## 클래스별 Recall(%, 참가자 평균)

| 자세 | Baseline | Lightweight | Tuned | Hybrid |
|---|---|---|---|---|
| back_hard | 45.5 | 44.6 | 46.7 | 62.6 |
| back_light | 60.4 | 59.6 | 40.0 | 43.0 |
| back_neck | 98.0 | 94.7 | 98.3 | 94.4 |
| chin | 84.6 | 82.7 | 75.3 | 73.0 |
| forward | 60.3 | 64.4 | 67.6 | 67.5 |
| neck_hard | 81.2 | 74.9 | 70.2 | 67.4 |
| neck_light | 59.2 | 60.4 | 66.8 | 66.7 |
| normal | 87.5 | 85.6 | 86.6 | 86.4 |
| slouch | 50.5 | 44.7 | 55.8 | 61.8 |

## 폴드별 선택 하이퍼파라미터

### Tuned
- p01 제외 학습: `{'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': None, 'class_weight': None}`
- p02 제외 학습: `{'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'log2', 'max_depth': 5, 'class_weight': 'balanced_subsample'}`
- p03 제외 학습: `{'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 15, 'class_weight': 'balanced'}`

### Hybrid
- p01 제외 학습: `{'n_estimators': 50, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 8, 'class_weight': None}`
- p02 제외 학습: `{'n_estimators': 40, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 5, 'class_weight': None}`
- p03 제외 학습: `{'n_estimators': 30, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 10, 'class_weight': None}`

