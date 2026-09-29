# 바른자세 (Bareunjase)

듀얼캠(스테레오 카메라) 기반 책상 자세 · 디지털 디톡스 모니터링 앱 — 임베디드소프트웨어학과 캡스톤 프로젝트

전체 계획·일정·의사결정 로그는 `project.md` 참고.

## 개발 환경 세팅 (팀원 모두 동일하게)

```bash
# 1. 저장소 클론
git clone <이 레포 URL> bareunjase
cd bareunjase

# 2. 가상환경 생성 (Python 3.10~3.11 권장)
python3 -m venv .venv
source .venv/bin/activate      # Windows는 .venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt
```

`tflite-runtime` 설치가 실패하면(Python 3.12 이상 등) `requirements.txt` 안내대로
`tensorflow`로 대체 설치하면 된다 — 코드가 둘 다 자동으로 시도하도록 짜여 있음.

## 스테레오 카메라 (Intel RealSense D455)

> 계획 단계에는 위드로봇 oCamS-1CGN-U 기준이었으나, 4주차(9/29)에 실제로는 **RealSense D455**를 수령해서 SDK가 바뀌었다. `requirements.txt`의 `pyrealsense2`가 이 SDK다.

- `pip install -r requirements.txt`로 `pyrealsense2`까지 같이 설치됨 (Windows/Linux/Intel Mac은 pip 설치로 충분, Apple Silicon Mac은 별도 빌드가 필요할 수 있음 — 안 되면 팀 채널에 공유)
- RealSense는 출고 시 스테레오 캘리브레이션이 끝나 있어서, oCamS 때 계획했던 체스보드 수동 캘리브레이션은 필요 없다. 대신 카메라 연결 후 depth 스트림이 정상적으로 나오는지, 줄자로 잰 실제 거리와 SDK가 보고하는 값이 맞는지 검증하는 절차로 대체한다.
- 연결 확인: `realsense-viewer`(SDK 설치 시 같이 깔리는 GUI 도구)로 먼저 depth/RGB 스트림이 뜨는지 확인하는 걸 추천 — 파이썬 코드 짜기 전에 하드웨어 자체가 인식되는지부터 눈으로 보는 게 디버깅이 빠름
- 캡처 코드는 `src/capture/realsense_capture.py`에 있음:

```bash
# 화면 중앙 십자선까지의 거리 + fps 실시간 표시 (줄자로 거리 정확도 확인, q로 종료)
python -m src.capture.realsense_capture view

# 30초 동안 fps·프레임타임 스파이크 자동 측정 (4주차 게이트)
python -m src.capture.realsense_capture gate --seconds 30
```

## MoveNet 모델 파일

용량 때문에 git에 올리지 않는다. 아래에서 각자 받아서 `models/` 폴더에 넣을 것:

- https://www.kaggle.com/models/google/movenet/tfLite/singlepose-lightning-tflite-int8
- Kaggle에서 받으면 `.tflite` 단일 파일이 아니라 `.tar.gz`로 받아진다(4주차 실측 확인). 압축을 풀면 `4.tflite` 같은 이름이 나오는데, 이걸 `models/movenet_lightning_int8.tflite`로 이름을 바꿔서 넣으면 된다.

## 실행

프로젝트 루트에서, 모듈(`-m`)로 실행한다 (내부에서 `src.features...`를 절대경로로
import하기 때문에 `src/pose/movenet_keypoints.py`를 직접 실행하면 안 됨):

```bash
# 웹캠으로 실시간 확인 (q로 종료)
python -m src.pose.movenet_keypoints --model models/movenet_lightning_int8.tflite

# RealSense D455의 color 스트림으로 실행 (실제 자세 데이터 수집은 이 방식 — 최종 제품과 화각을 맞추기 위함)
python -m src.pose.movenet_keypoints --model models/movenet_lightning_int8.tflite --realsense
```

## RF 기준모델 스켈레톤 (4주차, 합성 데이터)

`src/logic/rf_baseline.py` — 실제 자세 데이터가 모이기 전(5주차 수집 예정)까지, 합성
keypoint 데이터로 "증강(좌우반전+jitter) → 특징추출 → RF 학습 → 평가" 파이프라인이
동작하는지만 확인하는 스켈레톤이다. **여기 나오는 정확도는 합성 데이터 기준이라 의미
없음** — 6주차에 실제 데이터로 데이터 로딩 부분만 교체해서 재사용할 것.

```bash
python -m src.logic.rf_baseline
```

## 테스트

카메라·모델 없이 검증 가능한 로직(정규화, 각도 계산, RF 스켈레톤 등)은 `tests/`에 있다:

```bash
python -m pytest tests/
```

## 폴더 구조

```
src/
  capture/   # 카메라 입력 (pyrealsense2), depth 정확도 검증 (센싱 담당)
  pose/      # MoveNet 키포인트 추출 (로직 담당)
  features/  # 정규화 + 각도 특징 계산
  logic/     # 판정 상태머신, RF 기준모델 스켈레톤(rf_baseline.py, 4주차 - 합성 데이터)
models/      # .tflite 모델 파일 (git 미포함, 위에서 직접 다운로드)
data/        # 수집한 자세 데이터 (git 미포함)
tests/
```

## 역할 분담

- **센싱 담당**: `src/capture/` — RealSense 카메라 입력(`pyrealsense2`), depth 정확도 검증, 필터 파라미터 튜닝
- **로직 담당**: `src/pose/`, `src/features/`, `src/logic/` — 포즈 추정, 특징 추출, 판정 로직
