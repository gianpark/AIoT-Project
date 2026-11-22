# Tuned RF / Lightweight RF / Hybrid RF 비교

- 특징 세트: no_hip, 결측 처리 A, 9개 클래스, 참가자 3명 (p01, p02, p03), LOPO, seed=3.
- Tuned/Hybrid 탐색: 각각 RandomizedSearchCV 20회, GroupKFold(그룹=참가자×촬영회차), 탐색 시 5프레임마다 1개 사용.
- Hybrid 정의: Lightweight의 복잡도 제한 안에서 하이퍼파라미터를 탐색한 모델.
- 참가자가 3명으로 적으므로 평균 차이뿐 아니라 참가자별 결과와 폴드 간 편차를 함께 해석한다.

## 요약 (참가자별 결과의 평균 ± 표준편차)

| 모델 | Macro-F1 | Accuracy | 모델 크기(KB) | 추론 중앙값(ms) | 추론 p95(ms) | 학습 시간(s) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.657 ± 0.048 | 0.68 | 21230.20 | 14.25 | 17.43 | 2.12 |
| Lightweight | 0.670 ± 0.043 | 0.69 | 2226.47 | 2.52 | 3.76 | 0.37 |
| Tuned | 0.673 ± 0.042 | 0.69 | 5778.73 | 6.85 | 8.52 | 1.21 |
| Hybrid | 0.667 ± 0.052 | 0.68 | 729.61 | 1.67 | 2.63 | 0.21 |

## 참가자별 Macro-F1

| 모델 | p01 | p02 | p03 |
|---|---|---|---|
| Baseline | 0.711 | 0.666 | 0.594 |
| Lightweight | 0.708 | 0.693 | 0.610 |
| Tuned | 0.724 | 0.674 | 0.622 |
| Hybrid | 0.736 | 0.652 | 0.612 |

## 클래스별 Recall(%, 참가자 평균)

| 자세 | Baseline | Lightweight | Tuned | Hybrid |
|---|---|---|---|---|
| back_hard | 46.1 | 48.5 | 48.3 | 44.6 |
| back_light | 48.0 | 52.1 | 45.3 | 51.4 |
| back_neck | 98.0 | 90.2 | 93.5 | 98.6 |
| chin | 83.4 | 86.2 | 77.9 | 72.9 |
| forward | 59.7 | 63.9 | 66.3 | 69.2 |
| neck_hard | 82.4 | 82.2 | 81.3 | 73.5 |
| neck_light | 58.7 | 67.4 | 65.4 | 65.4 |
| normal | 87.7 | 86.5 | 87.5 | 85.1 |
| slouch | 50.5 | 44.3 | 55.6 | 56.9 |

## 폴드별 선택 하이퍼파라미터

### Tuned
- p01 제외 학습: `{'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'log2', 'max_depth': 10, 'class_weight': 'balanced_subsample'}`
- p02 제외 학습: `{'n_estimators': 50, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': 5, 'class_weight': 'balanced_subsample'}`
- p03 제외 학습: `{'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 10, 'class_weight': 'balanced_subsample'}`

### Hybrid
- p01 제외 학습: `{'n_estimators': 40, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': 8, 'class_weight': None}`
- p02 제외 학습: `{'n_estimators': 30, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 5, 'class_weight': None}`
- p03 제외 학습: `{'n_estimators': 30, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 8, 'class_weight': 'balanced'}`

