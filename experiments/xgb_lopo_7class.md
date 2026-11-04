# 참가자 단위 교차검증 XGBoost (7개 자세, 참가자 3명)

- 데이터: 54070프레임 (p01 19760, p02 17967, p03 16343)
- 분류 클래스: 7개
- 분류 방식: 11개 기존 라벨을 7개 클래스로 통합
- 검증 방식: Leave-One-Participant-Out (LOPO)
- XGBoost: 300 estimators, max_depth=6
- learning_rate=0.05, subsample=0.9
- colsample_bytree=0.9, random_state=0
- 클래스 불균형 처리: balanced sample weights
- 참가자가 3명으로 제한되어 있어 일반화 성능 해석에 주의한다.

## 클래스 통합 기준

| 새 클래스 | 기존 라벨 |
|---|---|
| normal | normal |
| turtle_neck | neck_hard, neck_light |
| forward | forward |
| lean_back | back_hard, back_light |
| left_lean | left_lean_neck |
| right_lean | right_lean_neck |
| bad_posture | back_neck, chin, slouch |

## 특징 세트 × 결측 처리

| 특징 세트 | 결측 처리 | p01 Macro-F1 | p02 Macro-F1 | p03 Macro-F1 | 평균 | 평가 프레임 유지율 |
|---|---|---|---|---|---|---|
| depth | A | 0.469 | 0.402 | 0.453 | 0.441 | 100% / 100% / 100% |
| depth | B | 0.533 | 0.436 | 0.507 | 0.492 | 22% / 41% / 53% |
| no_hip | A | 0.513 | 0.444 | 0.523 | 0.493 | 100% / 100% / 100% |
| no_hip | B | 0.510 | 0.434 | 0.571 | 0.505 | 44% / 69% / 87% |
| all | A | 0.542 | 0.419 | 0.504 | 0.488 | 100% / 100% / 100% |
| all | B | 0.564 | 0.383 | 0.467 | 0.471 | 15% / 31% / 53% |
| relative | A | 0.515 | 0.333 | 0.458 | 0.435 | 100% / 100% / 100% |
| relative | B | 0.470 | 0.385 | 0.503 | 0.453 | 44% / 69% / 87% |

## 클래스별 Recall(%) — no_hip + A

| 자세 | p01 | p02 | p03 |
|---|---|---|---|
| bad_posture | 42 | 46 | 84 |
| forward | 100 | 55 | 5 |
| lean_back | 74 | 100 | 71 |
| left_lean | 0 | 0 | 32 |
| normal | 86 | 52 | 99 |
| right_lean | 0 | 0 | 0 |
| turtle_neck | 100 | 55 | 94 |

## 혼동 행렬 (행=실제, 열=예측)

| 실제＼예측 | bad_posture | forward | lean_back | left_lean | normal | right_lean | turtle_neck |
|---|---|---|---|---|---|---|---|
| bad_posture | 8416 | 1464 | 63 | 976 | 21 | 977 | 2959 |
| forward | 2361 | 2790 | 0 | 35 | 0 | 29 | 0 |
| lean_back | 33 | 0 | 7953 | 13 | 1125 | 608 | 0 |
| left_lean | 2843 | 303 | 213 | 435 | 0 | 615 | 350 |
| normal | 0 | 0 | 779 | 0 | 3486 | 13 | 251 |
| right_lean | 3241 | 151 | 506 | 183 | 12 | 2 | 780 |
| turtle_neck | 206 | 7 | 0 | 579 | 0 | 850 | 8442 |

## 특징 중요도 — no_hip + A

| 특징 | 중요도 |
|---|---|
| `neck_forward_offset_m` | 0.131 |
| `chest_depth_m` | 0.124 |
| `head_up_m` | 0.119 |
| `head_depth_m` | 0.116 |
| `head_forward_deg` | 0.089 |
| `torso_len_sw_na` | 0.064 |
| `elbow_left_deg` | 0.059 |
| `wrist_face_sw` | 0.046 |
| `elbow_right_deg_na` | 0.044 |
| `elbow_right_deg` | 0.043 |
