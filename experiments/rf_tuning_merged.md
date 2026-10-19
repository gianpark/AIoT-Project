# RF 경량화·파라미터 검증 (참가자 단위 교차검증)

- 특징 세트 no_hip, 결측 처리 A, 7클래스(강도 합침), 참가자 3명(p01, p02, p03) leave-one-participant-out, 시드 0.
- Tuned: 랜덤 서치 20회, 안쪽 GroupKFold(그룹=참가자×촬영회차), 탐색은 5프레임마다 1개 사용.
- 참가자가 적어 모델 간 차이가 폴드 간 편차보다 작으면 '차이 없음'으로 해석한다.

## 요약 (평균, 폴드 간 표준편차)

| 모델 | macro-F1 | 정확도 | 모델 크기(KB) | 추론 중앙값(ms) | 추론 p95(ms) | 학습 시간(s) |
|---|---|---|---|---|---|---|
| Baseline | 0.731 ± 0.012 | 0.777 | 16711 | 13.27 | 16.28 | 0.8 |
| Lightweight | 0.738 ± 0.006 | 0.774 | 1526 | 1.57 | 1.76 | 0.2 |
| Tuned | 0.755 ± 0.038 | 0.793 | 3117 | 4.05 | 5.05 | 0.3 |

## 평가한 참가자별 macro-F1

| 모델 | p01 | p02 | p03 |
|---|---|---|---|
| Baseline | 0.745 | 0.730 | 0.717 |
| Lightweight | 0.735 | 0.747 | 0.734 |
| Tuned | 0.805 | 0.713 | 0.745 |

## 클래스별 recall(%, 평가 참가자 평균)

| 자세 | Baseline | Lightweight | Tuned |
|---|---|---|---|
| back | 86 | 85 | 86 |
| back_neck | 85 | 84 | 97 |
| chin | 84 | 81 | 78 |
| forward | 60 | 64 | 63 |
| neck | 92 | 88 | 91 |
| normal | 86 | 87 | 89 |
| slouch | 34 | 38 | 36 |

## Tuned가 고른 파라미터(바깥 폴드별)

- p01 제외 학습: `{'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': None, 'class_weight': None}`
- p02 제외 학습: `{'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': None, 'class_weight': None}`
- p03 제외 학습: `{'n_estimators': 30, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': None, 'class_weight': None}`
