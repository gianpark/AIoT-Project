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

## MoveNet 모델 파일

용량 때문에 git에 올리지 않는다. 아래에서 각자 받아서 `models/` 폴더에 넣을 것:

- https://www.kaggle.com/models/google/movenet/tfLite/singlepose-lightning-tflite-int8

## 폴더 구조

```
src/
  capture/   # 카메라 입력, 스테레오 캘리브레이션 (센싱 담당)
  pose/      # MoveNet 키포인트 추출 (로직 담당)
  features/  # 정규화 + 각도 특징 계산
  logic/     # 판정 상태머신
models/      # .tflite 모델 파일 (git 미포함, 위에서 직접 다운로드)
data/        # 수집한 자세 데이터 (git 미포함)
tests/
```

## 역할 분담

- **센싱 담당**: `src/capture/` — 카메라 입력, 스테레오 캘리브레이션, 디스패리티 튜닝
- **로직 담당**: `src/pose/`, `src/features/`, `src/logic/` — 포즈 추정, 특징 추출, 판정 로직
