# 주의 알림의 종류·방향 채점 (보정 있음)

열: 경고만 검출 / 경고만 종류 일치 / 경고+주의 검출 / 경고+주의 종류 일치 (%). back_neck은 기대 종류가 없어 종류 일치를 채점하지 않는다.

## 환경 1

| 자세 | 경고 검출 | 경고 종류 일치 | 경고+주의 검출 | 경고+주의 종류 일치 |
|---|---|---|---|---|
| back_hard | 79 | 79 | 88 | 88 |
| back_light | 59 | 59 | 71 | 71 |
| back_neck | 22 | 0 | 47 | 0 |
| chin | 66 | 14 | 67 | 16 |
| forward | 83 | 67 | 93 | 70 |
| left_lean_neck | 45 | 12 | 80 | 35 |
| neck_hard | 90 | 90 | 90 | 90 |
| neck_light | 66 | 66 | 77 | 77 |
| right_lean_neck | 20 | 8 | 90 | 60 |
| slouch | 61 | 61 | 80 | 80 |

주의 알림 종류 분포(위상·녹화 단위 횟수):

- left_lean_neck: tilt_left 14, neck_forward 12, slouch_back 6
- right_lean_neck: tilt_right 23, neck_forward 7, slouch_back 3, slouch_forward 3
- back_neck: slouch_back 33, neck_forward 9
- chin: slouch_forward 30, neck_forward 16, chin_rest 14

## 환경 2

| 자세 | 경고 검출 | 경고 종류 일치 | 경고+주의 검출 | 경고+주의 종류 일치 |
|---|---|---|---|---|
| back_hard | 100 | 100 | 100 | 100 |
| back_light | 80 | 80 | 80 | 80 |
| back_neck | 0 | 0 | 0 | 0 |
| chin | 95 | 5 | 95 | 5 |
| forward | 100 | 90 | 100 | 90 |
| left_lean_neck | 60 | 37 | 67 | 43 |
| neck_hard | 90 | 90 | 90 | 90 |
| neck_light | 0 | 0 | 35 | 35 |
| right_lean_neck | 27 | 0 | 57 | 27 |
| slouch | 5 | 5 | 5 | 5 |

주의 알림 종류 분포(위상·녹화 단위 횟수):

- left_lean_neck: tilt_left 13, neck_forward 7
- right_lean_neck: neck_forward 9, tilt_right 8
- back_neck: 없음
- chin: slouch_forward 16, neck_forward 2, chin_rest 1

