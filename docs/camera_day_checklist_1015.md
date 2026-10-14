# 10/15 카메라 수령일 체크리스트 (논문 근거 확보용)

목표: 논문 서론의 기여 2번(MoveNet 수정·측면 복원)과 4번(예비 분석)의 근거를 **10/16~18** 안에 모은다.
먼저 한 일(10/6의 필터 튜닝·판정 검증)은 `camera_day_checklist.md`에 있다. 이 문서는 이번 라운드에서 새로 할 일만 담았다.

모든 명령은 프로젝트 루트에서 가상환경을 켠 상태(`.venv\\Scripts\\activate`)로 실행한다. 모델 경로는 `models/movenet_lightning_int8.tflite`.
**우선순위: 0 → 1 → 2 → 3 → 4.** 시간이 모자라면 뒤에서부터 버린다. 0·1이 안 나와도 논문은 "구현했고 평가 진행 중"으로 쓸 수 있다.

## 0. 시작 전 (10분)

- [ ] `git pull`, `python -m pytest -q` (현재 86개 통과가 정상)
- [ ] 카메라를 **10/8~9 수집 때와 같은 자리·높이·각도**로 고정. 달라지면 이전 로그와 비교가 안 된다. 높이·각도는 메모(`--note`는 없으므로 이 문서 아래 기록란에 적을 것)
- [ ] `python -m src.capture.realsense_capture gate --seconds 10` 으로 프레임·fps 확인
- [ ] **새 로그 열 확인** (이번 변경은 카메라로 돌려 본 적 없음):
  ```
  python -m src.pose.movenet_keypoints --model models/movenet_lightning_int8.tflite --realsense --judge --log data/_check.csv --duration 5
  python -c "import pandas as pd; d=pd.read_csv('data/_check.csv'); print(d.shape, list(d.columns[-4:]))"
  ```
  기대: 열 수가 기존보다 34개 많고 마지막 열이 `right_ankle_y`. `nose_x`, `nose_y` 값이 0~1 사이.
- [ ] `--roi-crop`, `--depth-refine`, 둘 다 켜고 5초씩 에러 없이 도는지 확인(위 명령에 옵션만 추가). `--depth-refine`은 종료 시 "옮겨진 점 누적 N개"를 출력한다.
- [ ] 참가자에게 촬영 동의(논문 데이터 사용)를 받는다. 얼굴이 나오는 `data/snaps/`와 휴대폰 사진은 **git에 올리지 않는다**.

## 1. MoveNet 수정 전후 비교 (가장 중요)

같은 사람이 **같은 자세를 4가지 조건으로 연속 촬영**한다(자세가 달라지는 것을 막기 위해 자세 하나를 4조건 모두 끝낸 뒤 다음 자세로).

| 조건 | 옵션 |
|---|---|
| base | (없음) |
| roi | `--roi-crop` |
| refine | `--depth-refine` |
| both | `--roi-crop --depth-refine` |

자세(4개): `normal`, `chin`(턱 괴기), `neck_hard`, `back_neck`. 참가자는 p01 먼저(시간 되면 p02).

```
python -m src.pose.movenet_keypoints --model models/movenet_lightning_int8.tflite --realsense --judge --countdown 5 --duration 20 [옵션] --log data/mod_logs/mod_<참가자>_<자세>_<조건>.csv
```
예: `data/mod_logs/mod_p01_chin_roi.csv`. 폴더가 없으면 먼저 `mkdir data\\mod_logs`.
- 한 자세당 20초 × 4조건 + 자세 잡기 = 약 3분, 4자세 약 12분.
- 조건 순서는 자세마다 바꿔도 된다(피로 편향 방지). 순서를 아래 기록란에 적는다.
- 나중에 뽑을 지표: 손목·코 keypoint 점수, `wrist_face_sw` 결측률, 엉덩이 depth 유효율, `judge` 검출률, **로그 타임스탬프로 계산한 실효 프레임률**(조건별 속도 비교).
- 종료 시 `--depth-refine`의 "옮겨진 점 누적 N개"를 기록한다(0에 가까우면 효과가 없다는 뜻이므로 그 자체가 결과).

## 2. 측면 정답 비교 (기여 3번의 근거)

카메라 한 대로 만든 측면 값(`head_forward_deg`)이 **실제 옆모습**과 맞는지 본다.

- 휴대폰을 삼각대로 사용자 **정확히 옆**(카메라 광축과 직각)에 세운다. 높이는 어깨 높이쯤.
- 사용자의 **어깨 끝(견봉)** 과 **코끝**에 작은 스티커를 붙이면 사진에서 위치 찾기가 쉽다.
- 자세 5개를 각 10초 유지, 중간에 휴대폰으로 사진 1장: `normal`, `neck_light`, `neck_hard`, `forward`, `back_hard`. 같은 시점의 카메라 로그는 1번의 `base` 로그(`data/mod_logs/mod_<참가자>_<자세>_base.csv`)를 쓴다. 1번에 없는 자세(`neck_light`, `forward`, `back_hard`)는 같은 명령으로 `base`만 추가 촬영한다.
- 사진 파일명은 `gt_<참가자>_<자세>.jpg`. **사진은 git 제외**, 분석에는 "어깨→코 선의 수직선 기준 각도"만 숫자로 뽑아 `data/side_gt/gt_angles.csv`(참가자, 자세, 각도)에 기록한다.
- 비교는 절대 일치가 아니라 **자세 간 순서와 상관**(두 값이 같은 방향으로 움직이는지)으로 본다. 정의가 완전히 같지 않기 때문이다.
- 각도 추출은 사진에 MoveNet을 돌려 코·어깨 점으로 자동 계산하는 스크립트를 이후 작성할 수 있다.

## 3. 추가 참가자 (약점 보완)

새 참가자 p03, p04(가능하면). 10/8의 계획대로 자세 11종 × 20초, **이번에는 x/y 열이 같이 기록**된다. 조건은 `base`(옵션 없음)로 통일한다.

```
python -m src.pose.movenet_keypoints --model models/movenet_lightning_int8.tflite --realsense --judge --log data/side_logs/side_<참가자>_<자세>.csv --countdown 5 --duration 20 --snap-every 2
```
자세 목록과 설명은 `docs/side_capture_plan.md`. 사람당 약 10~15분. 같은 사람을 이틀에 나눠 찍으면 회차 간 변동도 얻는다(`_r2`).

## 4. 시간이 남으면

- [ ] 좌우 기울임 재촬영: p01, p02의 `left_lean_neck`, `right_lean_neck` (x/y가 있어야 판정 검증 가능)
- [ ] 약한 강도 재촬영: `neck_light`, `back_light`를 "어깨선 기준으로 코가 어디까지 나오는지" 같은 가이드를 현장에서 정해 기록
- [ ] 추론시간·CPU·RAM(실험 4.7): 조건별 60초 실행 중 작업 관리자 값 기록. 이후 `psutil` 기반 측정 스크립트를 작성할 수 있다.

## 마무리 (10분)

- [ ] 숫자 CSV만 커밋: `data/mod_logs/*.csv`, `data/side_logs/*.csv`, `data/side_gt/gt_angles.csv` (`.gitignore`에 허용 추가됨). 사진과 `data/snaps/`는 올리지 않는다.
- [ ] 이 세션에 로그 파일 첨부 → 분석 스크립트와 결과표 작성
- [ ] 카메라 반납 전 위 항목이 끝났는지 확인(depth 값은 나중에 복구할 수 없다)

## 기록란 (촬영 중 채우기)

- 카메라 높이·각도:
- 조명·시간대:
- 참가자와 촬영 순서(조건 순서):
- 이상 현상·메모:

## 다 못 했을 때 자르는 순서

4 → 3의 두 번째 참가자 → 2 → 1의 p02. 0과 1(p01 한 명)은 반드시.
