# 7-class 분할 분석

no_hip 9개 특징, 기존 3명 로그(좌표 불필요). 값은 Macro-F1 pooled, seed 평균(표준편차).

| 방식 | Macro-F1 |
|---|---|
| 7-class 전체 | 0.513 (0.002) |
| 7-class 결과에서 좌우 제외 평균 | 0.692 (0.003) |
| 앞뒤 5개만 다시 학습 | 0.789 (0.006) |

## bad_posture 세부 자세가 7-class 모델에서 어디로 예측되는가 (seed 0, 비율)

- back_neck: {'bad_posture': 0.785, 'left_lean': 0.126, 'right_lean': 0.069, 'lean_back': 0.012, 'normal': 0.008}
- chin: {'bad_posture': 0.83, 'forward': 0.141, 'right_lean': 0.025, 'left_lean': 0.003, 'turtle_neck': 0.0}
- slouch: {'turtle_neck': 0.487, 'bad_posture': 0.372, 'right_lean': 0.073, 'left_lean': 0.056, 'forward': 0.013}

## 라벨별 neck_forward_offset_m 중앙값 (좌우 라벨의 고개 내민 정도가 사람마다 다름)

| p | left_lean_neck | neck_hard | neck_light | normal | right_lean_neck |
|---|---|---|---|---|---|
| p01 | 0.216 | 0.220 | 0.182 | 0.143 | 0.214 |
| p02 | 0.130 | 0.213 | 0.178 | 0.124 | 0.120 |
| p03 | 0.172 | 0.209 | 0.187 | 0.135 | 0.140 |
