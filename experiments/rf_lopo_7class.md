# 참가자 단위 교차검증 RF (7개 자세, 참가자 3명)

- 데이터 54070프레임 (p01 19760, p02 17967, p03 16343)
- 분류 클래스: 7개
- 분류 방식: 11개 기존 라벨을 7개 클래스로 통합
- 검증 방식: Leave-One-Participant-Out (LOPO)
- RF: 300그루, class_weight=balanced, seed=0
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
| depth | A | 0.490 | 0.419 | 0.445 | 0.451 | 100% / 100% / 100% |
| depth | B | 0.521 | 0.434 | 0.478 | 0.477 | 22% / 41% / 53% |
| no_hip | A | 0.515 | 0.474 | 0.491 | 0.493 | 100% / 100% / 100% |
| no_hip | B | 0.512 | 0.462 | 0.571 | 0.515 | 44% / 69% / 87% |
| all | A | 0.514 | 0.479 | 0.551 | 0.515 | 100% / 100% / 100% |
| all | B | 0.499 | 0.393 | 0.486 | 0.459 | 15% / 31% / 53% |
| relative | A | 0.512 | 0.345 | 0.494 | 0.450 | 100% / 100% / 100% |
| relative | B | 0.501 | 0.389 | 0.508 | 0.466 | 44% / 69% / 87% |

## 클래스별 Recall(%) — no_hip + A

| 자세 | p01 | p02 | p03 |
|---|---|---|---|
| bad_posture | 43 | 60 | 89 |
| forward | 100 | 55 | 3 |
| lean_back | 76 | 100 | 64 |
| left_lean | 0 | 0 | 32 |
| normal | 84 | 71 | 96 |
| right_lean | 0 | 0 | 0 |
| turtle_neck | 100 | 45 | 77 |

## 혼동 행렬 (행=실제, 열=예측)

| 실제＼예측 | bad_posture | forward | lean_back | left_lean | normal | right_lean | turtle_neck |
|---|---|---|---|---|---|---|---|
| bad_posture | 9389 | 991 | 61 | 861 | 150 | 871 | 2553 |
| forward | 2325 | 2740 | 0 | 112 | 0 | 37 | 1 |
| lean_back | 464 | 0 | 7844 | 0 | 1051 | 373 | 0 |
| left_lean | 3274 | 356 | 187 | 435 | 0 | 166 | 341 |
| normal | 0 | 0 | 476 | 0 | 3722 | 42 | 289 |
| right_lean | 3341 | 166 | 506 | 207 | 1 | 6 | 648 |
| turtle_neck | 881 | 138 | 0 | 597 | 0 | 946 | 7522 |

## 특징 중요도 — no_hip + A

| 특징 | 중요도 |
|---|---|
| `head_depth_m` | 0.167 |
| `head_forward_deg` | 0.162 |
| `chest_depth_m` | 0.156 |
| `neck_forward_offset_m` | 0.126 |
| `head_up_m` | 0.105 |
| `elbow_left_deg` | 0.103 |
| `elbow_right_deg` | 0.088 |
| `torso_len_sw` | 0.031 |
| `wrist_face_sw` | 0.028 |
| `elbow_right_deg_na` | 0.009 |
