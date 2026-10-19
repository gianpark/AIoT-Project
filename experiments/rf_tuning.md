# RF 경량화·파라미터 검증 (참가자 단위 교차검증)

- 특징 세트 no_hip, 결측 처리 A, 9클래스, 참가자 3명(p01, p02, p03) leave-one-participant-out, 시드 0.
- Tuned: 랜덤 서치 20회, 안쪽 GroupKFold(그룹=참가자×촬영회차), 탐색은 5프레임마다 1개 사용.
- 참가자가 적어 모델 간 차이가 폴드 간 편차보다 작으면 '차이 없음'으로 해석한다.

## 요약 (평균, 폴드 간 표준편차)

| 모델 | macro-F1 | 정확도 | 모델 크기(KB) | 추론 중앙값(ms) | 추론 p95(ms) | 학습 시간(s) |
|---|---|---|---|---|---|---|
| Baseline | 0.667 ± 0.051 | 0.689 | 23806 | 13.54 | 20.29 | 0.9 |
| Lightweight | 0.664 ± 0.031 | 0.684 | 2249 | 3.01 | 4.38 | 0.3 |
| Tuned | 0.653 ± 0.047 | 0.670 | 2989 | 8.50 | 14.46 | 0.6 |

## 평가한 참가자별 macro-F1

| 모델 | p01 | p02 | p03 |
|---|---|---|---|
| Baseline | 0.729 | 0.669 | 0.603 |
| Lightweight | 0.699 | 0.667 | 0.624 |
| Tuned | 0.716 | 0.636 | 0.606 |

## 클래스별 recall(%, 평가 참가자 평균)

| 자세 | Baseline | Lightweight | Tuned |
|---|---|---|---|
| back_hard | 46 | 45 | 42 |
| back_light | 57 | 52 | 42 |
| back_neck | 98 | 96 | 98 |
| chin | 85 | 77 | 75 |
| forward | 59 | 63 | 69 |
| neck_hard | 81 | 84 | 70 |
| neck_light | 59 | 65 | 66 |
| normal | 87 | 86 | 87 |
| slouch | 50 | 51 | 57 |

## Tuned가 고른 파라미터(바깥 폴드별)

- p01 제외 학습: `{'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': None, 'class_weight': None}`
- p02 제외 학습: `{'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'log2', 'max_depth': 5, 'class_weight': 'balanced_subsample'}`
- p03 제외 학습: `{'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 5, 'class_weight': None}`
