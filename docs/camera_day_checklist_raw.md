# 카메라 당일 체크리스트 — IRB 승인 후 RGB-D 원본 촬영

목적: IRB 승인 후 서면동의를 받은 참가자(p01, p02)의 11자세 RGB-D 원본을 찍고, 공통 6자세(P0~P5)를 A·B팀에 제공한다.
기준: IRB 신청서(C팀, 11/11 제출)와 A팀 요청 7가지(RGB 원본, 16bit Depth, 연속 프레임, 시간정보, 보정정보, 라벨·구간, 가명·무결성).
승인 전 촬영분과 파생자료는 이번 분석·제공에 쓰지 않는다.

## 0. 전날 준비

- [ ] IRB 승인 통지서의 **승인일·승인번호** 확인 (승인일 전 촬영은 무효)
- [ ] 동의서 [서식 9] 출력: 참가자 수 x 2부(참가자 사본 포함). 마지막 두 체크(촬영 동의 / 원본 타 팀 제공 동의)를 따로 받는다. 연구자 본인(기안)도 서명한다.
- [ ] 가명 코드 정하기: p01 → C01, p02 → C02. 성명 대응표는 종이나 저장소 밖 파일에만 둔다.
- [ ] 노트북: 충전기, **성능 모드**, 다른 프로그램 종료, 디스크 여유 **30GB 이상** (사람당 약 8~10GB 예상, PNG 원본)
- [ ] 가상환경에서 `pip install pyzipper` (암호화 zip용)
- [ ] 코드 최신 상태 확인: `scripts/capture_session.py`, `scripts/capture_raw.py`, `scripts/package_share.py`가 있는지

## 1. 카메라 설치 (약 10분)

- [ ] D455를 **모니터 위 가운데, 참가자 정면**에 둔다. 촬영이 끝날 때까지 옮기지 않는다.
- [ ] USB-C 포트에 연결. USB 3로 잡히는지 확인(예전에 USB 2.1로 잡혀 "No Frames Received"가 난 적 있음 → 다시 꽂기)
- [ ] fps 확인: `python -m src.capture.realsense_capture gate --seconds 10` → 평균 fps 29 이상
- [ ] 구도 확인: `python -m src.capture.realsense_capture view` → 머리부터 엉덩이까지 화면 안, 참가자까지 약 0.5 m 이상
- [ ] 아래 기록란에 장소, 카메라 높이, 기울기, 거리 적기

## 2. 5초 샘플 (A팀 확인용, 약 5분)

샘플도 사람 얼굴이 찍히므로 **동의서에 서명한 사람(기안)** 으로 찍는다.
- [ ] `python -m scripts.capture_session --participant C01 --sample`
- [ ] 출력 확인: **실제 저장 fps 28 이상, 누락 0~2개**. 많이 낮으면 다른 프로그램을 끄고 다시 찍는다.
- [ ] 샘플 패키지:
```
python scripts\package_share.py --map share\sample_map.csv --init-map data\raw_rgbd\sessions.csv --sample-only
```
  `share\sample_map.csv`를 열어 consent_signed·consent_share를 Y로 바꾼 뒤
```
python scripts\package_share.py --map share\sample_map.csv --approval-date <승인일> --recipient A --tag sample --encrypt
```
- [ ] `share\out\..._enc.zip`을 정우 님께 보내고 **비밀번호는 다른 경로(전화·대면)** 로 전달. A팀 확인을 기다리는 동안 본 촬영을 진행한다(카메라가 그날만 있으면).

## 3. 참가자별 본 촬영 (사람당 약 15분, 회당 30분 이하)

- [ ] 동의서 서명 받기. **원본 제공 동의 여부**를 기록란에 적는다(비동의여도 촬영은 가능, 제공만 안 함).
- [ ] 위험 안내: 불편하면 언제든 중단, 자세는 무리 없는 범위.
- [ ] `python -m scripts.capture_session --participant C01` (C02도 같은 방식)
  - 순서(자동): P0 정상(기준값) → P1 거북목 약·강 → P2 상체 후방 약·강 → P5 전방 → P3 좌측 기울임 → P4 우측 기울임 → 턱 괴기 → 허리 구부정 → 뒤로 기대며 고개 내밈 → P0 정상(마무리)
  - 화면의 안내 문장을 참가자에게 읽어 준다. 자세마다 5초 준비 후 20초 녹화.
  - Enter = 촬영, r = 방금 자세 다시(회차 +1), s = 건너뛰기, q = 종료
  - P3·P4는 **목을 내밀지 않고 상체를 본인 기준 왼쪽·오른쪽으로** 기울인다.
- [ ] 한 사람이 끝나면 `data\raw_rgbd\sessions.csv`에서 saved_fps(28 이상), missing_frames(수십 개 이상이면 r로 재촬영) 확인

## 4. 촬영 후 (자리에서, 약 10분)

- [ ] 공유 대상 목록 만들기:
```
python scripts\package_share.py --map share\share_map.csv --init-map data\raw_rgbd\sessions.csv
```
- [ ] `share\share_map.csv`에서 **동의서로 확인한 사람만** consent_signed·consent_share를 Y로. 보내지 않을 회차 행은 지운다. (턱 괴기 등 BAD 자세는 자동으로 빠져 있음)
- [ ] 미리보기: `python scripts\package_share.py --map share\share_map.csv --approval-date <승인일> --recipient A --dry-run`
- [ ] 암호화 패키지: 위 명령에서 `--dry-run` 대신 `--encrypt` (B팀용은 `--recipient B`로 한 번 더)
- [ ] 전송: 학교 계정 드라이브에서 받을 사람만 지정해 공유. 비밀번호는 따로 전달.
- [ ] 공유 기록표(Claude 시트 "바른자세 C팀 데이터 공유 기록표")에 한 줄: 패키지 ID, 전달일, 받는 팀·사람, 코드, 자세, 세션·파일 수, zip sha256, 만료일 2027-12-31
- [ ] 암호화 전 패키지 폴더(`share\out\C_to_A_...`)는 보관만 하고 보내지 않는다.

## 5. 카메라 반납 전

- [ ] P0~P5가 사람마다 빠짐없이 찍혔는지 sessions.csv로 확인 (원본은 나중에 다시 만들 수 없다)
- [ ] `data\raw_rgbd`와 `share`는 git에 올라가지 않는다(.gitignore 처리됨). 노트북 밖으로 옮길 때는 암호화 저장.

## 기록란

| 항목 | 값 |
|---|---|
| 촬영일 / 장소 | |
| IRB 승인일 / 번호 | |
| 카메라 높이(책상 기준) / 아래 기울기 | |
| 참가자까지 거리(대략) | |
| 전원 모드 / 충전기 | |
| gate 결과(평균 fps) | |
| 샘플 fps / 누락 | |
| C01 동의(촬영 / 제공) | |
| C02 동의(촬영 / 제공) | |
| 특이사항(재촬영, 중단 등) | |
