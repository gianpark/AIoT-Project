# Tuned RF / Lightweight RF / Hybrid RF 비교

- 특징 세트: no_hip, 결측 처리 A, 9개 클래스, 참가자 3명 (p01, p02, p03), LOPO, seed=2.
- Tuned/Hybrid 탐색: 각각 RandomizedSearchCV 20회, GroupKFold(그룹=참가자×촬영회차), 탐색 시 5프레임마다 1개 사용.
- Hybrid 정의: Lightweight의 복잡도 제한 안에서 하이퍼파라미터를 탐색한 모델.
- 참가자가 3명으로 적으므로 평균 차이뿐 아니라 참가자별 결과와 폴드 간 편차를 함께 해석한다.

## 요약 (참가자별 결과의 평균 ± 표준편차)

| 모델 | Macro-F1 | Accuracy | 모델 크기(KB) | 추론 중앙값(ms) | 추론 p95(ms) | 학습 시간(s) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.653 ± 0.057 | 0.67 | 21305.64 | 14.75 | 28.50 | 1.94 |
| Lightweight | 0.663 ± 0.033 | 0.68 | 2231.26 | 4.13 | 6.10 | 0.48 |
| Tuned | 0.670 ± 0.038 | 0.68 | 843.70 | 4.93 | 6.59 | 0.59 |
| Hybrid | 0.660 ± 0.045 | 0.68 | 1259.36 | 2.32 | 3.56 | 0.28 |

## 참가자별 Macro-F1

| 모델 | p01 | p02 | p03 |
|---|---|---|---|
| Baseline | 0.729 | 0.638 | 0.591 |
| Lightweight | 0.692 | 0.680 | 0.616 |
| Tuned | 0.719 | 0.663 | 0.627 |
| Hybrid | 0.723 | 0.632 | 0.626 |

## 클래스별 Recall(%, 참가자 평균)

| 자세 | Baseline | Lightweight | Tuned | Hybrid |
|---|---|---|---|---|
| back_hard | 46.7 | 49.7 | 48.4 | 49.4 |
| back_light | 50.5 | 47.3 | 41.7 | 43.9 |
| back_neck | 98.6 | 84.6 | 98.1 | 98.4 |
| chin | 83.2 | 88.6 | 78.1 | 79.4 |
| forward | 59.1 | 65.4 | 67.9 | 71.4 |
| neck_hard | 74.7 | 75.9 | 75.2 | 68.2 |
| neck_light | 58.3 | 61.1 | 69.5 | 64.3 |
| normal | 87.6 | 85.1 | 87.2 | 85.4 |
| slouch | 50.1 | 53.5 | 51.9 | 49.6 |

## 폴드별 선택 하이퍼파라미터

### Tuned
- p01 제외 학습: `{'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 5, 'class_weight': None}`
- p02 제외 학습: `{'n_estimators': 50, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 5, 'class_weight': 'balanced_subsample'}`
- p03 제외 학습: `{'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'log2', 'max_depth': 5, 'class_weight': None}`

### Hybrid
- p01 제외 학습: `{'n_estimators': 50, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 10, 'class_weight': 'balanced'}`
- p02 제외 학습: `{'n_estimators': 50, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 5, 'class_weight': None}`
- p03 제외 학습: `{'n_estimators': 50, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 8, 'class_weight': None}`

