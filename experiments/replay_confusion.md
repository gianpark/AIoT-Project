# 로그 재생 판정 검증 (T2)

- 입력: `data/side_logs` 5묶음, 32925프레임 (프레임 단위; 참가자 2명 예비 결과)
- 어깨·엉덩이 합성 정렬 → tilt 판정 불가(해당 없음). 빈 칸은 None.

## 자세별 정상/주의/경고 비율(%)

| 자세 | 정상 | 주의 | 경고 | n |
|---|---|---|---|---|
| back_hard | 10.3 | 2.8 | 86.9 | 2998 |
| back_light | 5.3 | 14.4 | 80.4 | 2996 |
| back_neck | 63.9 | 4.4 | 31.7 | 2998 |
| chin | 13.7 | 0.5 | 85.8 | 2987 |
| forward | 4.6 | 8.9 | 86.5 | 2998 |
| left_lean_neck | 41.8 | 0.2 | 58.0 | 2997 |
| neck_hard | 3.0 | 0.2 | 96.9 | 2999 |
| neck_light | 33.1 | 10.2 | 56.7 | 2996 |
| normal | 99.6 | 0.4 | 0.0 | 2964 |
| right_lean_neck | 43.3 | 4.2 | 52.5 | 2997 |
| slouch | 11.3 | 18.3 | 70.5 | 2995 |

## 혼동 행렬: 자세(행) × judge kind(열), 프레임 %(주의 이상만 kind, 나머지 none)

| 자세 | chin_rest | neck_forward | none | slouch_back | slouch_forward |
|---|---|---|---|---|---|
| back_hard | 0.0 | 0.0 | 10.3 | 89.7 | 0.0 |
| back_light | 0.0 | 0.0 | 5.3 | 94.7 | 0.0 |
| back_neck | 0.0 | 0.0 | 63.9 | 36.1 | 0.0 |
| chin | 28.1 | 21.5 | 13.7 | 0.0 | 36.7 |
| forward | 0.0 | 76.9 | 4.6 | 0.0 | 18.5 |
| left_lean_neck | 0.0 | 37.7 | 41.8 | 18.7 | 1.9 |
| neck_hard | 0.0 | 96.9 | 3.0 | 0.0 | 0.2 |
| neck_light | 0.0 | 66.2 | 33.1 | 0.2 | 0.6 |
| normal | 0.0 | 0.0 | 99.6 | 0.4 | 0.0 |
| right_lean_neck | 0.0 | 38.5 | 43.3 | 18.2 | 0.0 |
| slouch | 0.0 | 88.6 | 11.3 | 0.0 | 0.1 |

## 참가자·회차별 정상 자세 경고 오탐률(%)

| 묶음 | 경고 | 주의 이상 |
|---|---|---|
| p01 r1 | 0.0 | 0.0 |
| p01 r2 | 0.0 | 0.0 |
| p01 r3 | 0.0 | 0.0 |
| p02 r1 | 0.0 | 0.7 |
| p02 r2 | 0.0 | 1.2 |

## 기대 kind 검출률(%) — 1순위 / issues(복합 포함)

| 자세 | 기대 kind | 1순위 | issues 포함 |
|---|---|---|---|
| back_hard | slouch_back | 89.7 | 89.7 |
| back_light | slouch_back | 94.7 | 94.7 |
| neck_hard | neck_forward | 96.9 | 96.9 |
| neck_light | neck_forward | 66.2 | 66.2 |
| forward | neck_forward | 76.9 | 93.2 |
| slouch | neck_forward | 88.6 | 88.7 |
| chin | chin_rest | 28.1 | 49.1 |
| left_lean_neck | neck_forward | 37.7 | 37.7 |
| right_lean_neck | neck_forward | 38.5 | 38.5 |
