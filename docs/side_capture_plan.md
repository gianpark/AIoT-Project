# 측면 각도·복합 자세 로그 수집 계획 (10/8)

목적: 자세별 `head_forward_deg`, `torso_pitch_deg`, 팔꿈치 각도, 몸통 길이 등의 분포를 얻어 축별 기준값을 정한다.
복합 자세는 새 클래스를 만들지 않고 "축별 독립 판정(issues)"으로 처리하며, 아래 B는 그 검증용이다.

## 준비
```
cd C:\Users\gian\Downloads\aiot_project
.venv\Scripts\activate
git pull
```

## 촬영 명령 (파일명의 참가자·자세만 바꿔 반복, 예: data/side_p01_normal.csv)
처음 5초는 자세 잡는 시간(기록 안 함), 이어서 20초가 기록되고 자동 종료된다. 중간에 끝내려면 q.
```
python -m src.pose.movenet_keypoints --model models/movenet_lightning_int8.tflite --realsense --judge --log data/side_<참가자>_<자세>.csv --countdown 5 --duration 20
```

## A. 단일 자세 (기준값용)
| 파일(자세 부분) | 자세 |
|---|---|
| side_<참가자>_normal | 바른 자세, 팔은 책상 위(키보드 위치) |
| side_<참가자>_neck_light | 몸통 세우고 고개만 살짝 앞으로 |
| side_<참가자>_neck_hard | 몸통 세우고 고개를 많이 앞으로(거북목) |
| side_<참가자>_forward | 허리부터 몸 전체를 앞으로 숙임 |
| side_<참가자>_back_light | 평소 기대는 정도로 뒤로 기댐 |
| side_<참가자>_back_hard | 확실히 뒤로 젖힘 |
| side_<참가자>_slouch | 엉덩이는 두고 등을 말아 허리 구부정 |

## B. 복합 자세 (검증용)
| 파일(자세 부분) | 자세 |
|---|---|
| side_<참가자>_back_neck | 뒤로 기대면서 고개 내밈 |
| side_<참가자>_lean_neck | 옆으로 기대면서 고개 내밈 (좌/우 중 한쪽) |
| side_<참가자>_chin | 턱 괴기 (손이 얼굴 근처) |

## 로그에 새로 들어간 열 (CSV 맨 끝)
`head_forward_deg, torso_pitch_deg, elbow_left_deg, elbow_right_deg, torso_len_sw, head_up_m, wrist_face_sw`
- torso_len_sw: 몸통 길이/어깨너비 (구부정 후보), wrist_face_sw: 손목-코 거리/어깨너비 (턱 괴기 후보)

## 파일 이름 규칙
`side_<참가자>_<자세>.csv` (참가자는 기존과 같은 p01, p02 …). 참가자별로 따로 분석해 사람이 달라도 기준값이 유지되는지 본다.

## 확인할 것
- 화면에 복합 자세가 `slouch_back +tilt_left`처럼 여러 개 뜨는지(주의 이상인 문제만).
- 측면 뷰 창의 머리·팔·숫자가 정상 자세에서 자연스러운지.
- 로그 10개(참가자당)를 이 세션에 첨부하면 자세별 분포표와 기준값 제안을 만든다. (data/*.csv는 gitignore라 파일 첨부로)
