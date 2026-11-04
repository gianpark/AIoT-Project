# 환경 2 정상 자세 기준 보정 전후 비교

- 입력 `data/side_env2`, p01·p02, 10107프레임. 보정 이동량: p01 +0.044 m, p02 +0.063 m
- REF_NF = 0.143 (환경 1 정상 중앙값 p01 0.156, p02 0.130의 평균). normal은 보정 기준이라 구조적으로 개선됨

## 자세별 기대 kind 검출률(%) (주의 이상에서 1순위가 기대 kind) / 정상 판정률

| 자세 | 참가자 | 기대 kind | 검출 전 | 검출 후 | 정상 전 | 정상 후 |
|---|---|---|---|---|---|---|
| back_hard | p01 | slouch_back | 100 | 100 | 0 | 0 |
| back_hard | p02 | slouch_back | 100 | 100 | 0 | 0 |
| back_light | p01 | slouch_back | 100 | 87 | 0 | 13 |
| back_light | p02 | slouch_back | 100 | 100 | 0 | 0 |
| back_neck | p01 | - | - | - | 2 | 79 |
| back_neck | p02 | - | - | - | 1 | 79 |
| chin | p01 | chin_rest | 0 | 0 | 2 | 2 |
| chin | p02 | chin_rest | 10 | 8 | 3 | 3 |
| forward | p01 | neck_forward | 0 | 82 | 0 | 0 |
| forward | p02 | neck_forward | 0 | 66 | 2 | 2 |
| left_lean_neck | p01 | neck_forward | 0 | 39 | 0 | 0 |
| left_lean_neck | p02 | neck_forward | 0 | 0 | 11 | 97 |
| neck_hard | p01 | neck_forward | 0 | 99 | 100 | 1 |
| neck_hard | p02 | neck_forward | 84 | 97 | 16 | 3 |
| neck_light | p01 | neck_forward | 0 | 88 | 100 | 12 |
| neck_light | p02 | neck_forward | 0 | 0 | 12 | 100 |
| normal | p01 | - | - | - | 4 | 100 |
| normal | p02 | - | - | - | 2 | 100 |
| right_lean_neck | p01 | neck_forward | 30 | 44 | 22 | 22 |
| right_lean_neck | p02 | neck_forward | 0 | 10 | 5 | 22 |
| slouch | p01 | neck_forward | 0 | 0 | 99 | 99 |
| slouch | p02 | neck_forward | 0 | 59 | 100 | 41 |

## 정상 자세에서 나온 kind(주의 이상, %)

| 참가자 | 보정 | slouch_back | 기타 | 정상 |
|---|---|---|---|---|
| p01 | 보정 전 | 96 | 0 | 4 |
| p01 | 보정 후 | 0 | 0 | 100 |
| p02 | 보정 전 | 98 | 0 | 2 |
| p02 | 보정 후 | 0 | 0 | 100 |
