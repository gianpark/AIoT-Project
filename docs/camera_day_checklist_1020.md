# 10/20 카메라 재수령일 체크리스트

서론 마감(10/23 목표)이 있으므로 **촬영은 하루(약 2~3시간)에 끝낸다.** 우선순위 순서대로 하고, 모자라면 뒤에서부터 버린다.
모든 명령은 프로젝트 루트에서 가상환경을 켠 상태로 실행한다. 모델: `models/movenet_lightning_int8.tflite`.

## 0. 시작 전 (10분)

- [ ] `git pull`, `python -m pytest -q` (87개 통과가 정상. 이번에 `--log`에 열이 5개 늘었으니 실패가 나면 알려 주세요)
- [ ] 카메라를 10/15~18 촬영과 같은 자리·높이·각도로 고정(다르면 이전 로그와 비교가 안 된다)
- [ ] **새 로그 열 확인** (오늘 밤 변경, 카메라로 돌려 본 적 없음):
  ```
  python -m src.pose.movenet_keypoints --model models/movenet_lightning_int8.tflite --realsense --judge --log data/_check.csv --duration 5
  python -c "import pandas as pd; d=pd.read_csv('data/_check.csv'); print(d.shape, list(d.columns[-5:]))"
  ```
  기대: 마지막 5열이 `chest_mid_m, chest_c25_m, chest_c50_m, chest_lsh_m, chest_rsh_m`이고 값이 0.3~1.0 근처.
- [ ] 촬영 동의 확인. 사진과 `data/snaps/`는 git에 올리지 않는다.

## 1. CPU/RAM/fps 측정 (4.7) — 20~30분

- 조건: 기본(옵션 없음) / `--roi-crop` / `--depth-refine` / 둘 다, 각 60초.
- 실행 중 작업 관리자(또는 `psutil`)로 python 프로세스의 CPU %, 메모리(MB)를 30초 시점에 기록. 로그 타임스탬프로 실효 fps 계산.
- 노트북 전원 연결 상태, 다른 프로그램 종료. CPU 모델·코어 수·RAM도 기록(논문 하드웨어 표에 필요).
- 원하면 측정 스크립트(`psutil` 기반)를 만들 수 있다. 말해 주세요.

## 2. 새 참가자 촬영 (p04, 가능하면 p05) — 사람당 10~15분

x/y가 기록되는 로그를 늘려 2D 기준선을 3명 이상으로 다시 돌리기 위함. 자세 11종 × 20초, 조건은 `base`.
```
python -m src.pose.movenet_keypoints --model models/movenet_lightning_int8.tflite --realsense --judge --log data/side_logs/side_p04_<자세>.csv --countdown 5 --duration 20 --snap-every 2
```
- 자세 목록: `docs/side_capture_plan.md`. 같은 `$p`·`<자세>` 변수가 남아 파일을 덮어쓰지 않도록 매번 확인.
- **normal은 앉는 거리를 바꿔 2번**(`side_p04_normal.csv`, `side_p04_normal_r2.csv`). 지난번 일반화에서 normal recall이 거의 0이었던 원인을 보려는 것이라 normal을 충분히 모은다.
- 촬영 후: 새 로그 `rf_2d_baseline`을 다시 돌린다(참가자 수가 늘면 LOPO로 확장).

## 2-1. Thunder 동시 기록 (학습형 보정기 데이터, 신규)

촬영 명령에 아래 두 옵션만 붙이면 Lightning과 Thunder 키포인트를 같은 프레임에서 숫자로 함께 기록한다(얼굴 이미지는 저장 안 함).
```
--teacher-model models/movenet_thunder_int8.tflite --teacher-log data/teacher_logs/th_<참가자>_<자세>.csv
```
- Thunder 파일은 아직 `models/`에 없다. 입력 256px, 출력 [1,1,17,3]로 Lightning과 같다. 공식 배포 주소(TF Hub, 갱신 2024-03): `https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4?lite-format=tflite`
- 두 모델을 같이 돌리면 fps가 떨어질 수 있다. 시작 때 5초 시험으로 fps를 보고, 너무 낮으면 `--judge` 없이 찍는다.
- 열: timestamp, Lightning 17점(x,y,score), Thunder 17점, Lightning 점 위치 depth 17개, 깊이 특징 5개(총 125열). 같은 파일명이 있으면 거부(`--overwrite`로만 덮어씀).

## 3. 측면 정답 비교 (checklist_1015 §2) — 30분

- 휴대폰 삼각대를 사용자 정확히 옆, 어깨 높이. 어깨 끝과 코끝에 스티커.
- 자세 5개(`normal`, `neck_light`, `neck_hard`, `forward`, `back_hard`)를 10초 유지, 사진 1장씩: `gt_<참가자>_<자세>.jpg` (git 제외).
- 같은 자세의 카메라 로그를 같은 시점에 찍는다. 각도 숫자는 `data/side_gt/gt_angles.csv`(참가자, 자세, 각도).

## 4. 처리 방식 ablation용 촬영 (4.5 A0~A6) — 15분, 3번과 겸할 수 있음

오늘 밤 로깅을 바꿔서, 이제 로그 하나로 **가슴 depth를 한 점으로 읽은 경우 vs 4점 중앙값**을 오프라인으로 비교할 수 있다(`chest_mid_m`, `chest_c25_m`, `chest_c50_m`, `chest_lsh_m`, `chest_rsh_m`).
- 따라서 처리 방식을 바꿔 다시 찍지 않는다. **얼굴이 가슴을 가리는 자세**(`neck_hard`, `forward`, `back_neck`)를 참가자 2명 이상으로 20초씩 찍는 것이면 충분하다. 3번 촬영과 같은 로그를 쓴다.
- 홀 필링(nearest)은 깊이 보정 전 값을 남기지 않아 비교 불가. 이전 D2 결과를 부분 근거로만 쓴다.
- 2D 각도 A0는 x/y 열이 있는 새 로그(p02 r3, p03, p04)에서 재계산한다.

## 5. 시간이 남으면

- [ ] 좌우 기울임 재촬영(`left_lean_neck`, `right_lean_neck`), p01~p04
- [ ] 약한 강도 재촬영(`neck_light`, `back_light`): "코가 어깨선 기준 어디까지"를 현장에서 정해 기록
- [ ] 근접 40cm 이하 판정 검증

## 마무리 (10분)

- [ ] 숫자 CSV만 커밋(`data/side_logs`, `data/side_gt`). 사진·snaps는 제외.
- [ ] 로그를 이 세션에 알려 주기 → 2D 기준선 재실행, 4.5·4.7 표 채우기
- [ ] 반납 전 위 항목 확인(depth는 나중에 복구할 수 없다)

## 기록란

- 카메라 높이·각도:
- 조명·시간대:
- CPU/RAM/하드웨어:
- 참가자와 촬영 순서:
- 이상 현상·메모:

## 다 못 했을 때 자르는 순서

5 → 4 → 3 → 2의 p05. 1(CPU/RAM)과 2의 p04는 반드시.
