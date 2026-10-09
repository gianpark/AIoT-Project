# 바른자세 — 6주차 카메라 없이 할 수 있는 일 (팀 공유용)

*작성일: 2026-10-09 (6주차). 지금 카메라가 없어서, 이미 수집한 로그와 코드만으로 할 수 있는 일을 정리했다.*

각 작업은 **입력(이미 있는 것) → 결과물(파일) → 확인 방법** 순서로 적었다. 담당자는 아래 표의 `담당` 칸에 이름을 적고 시작한다.

---

## 0. 먼저 읽을 것 (15분)

1. `project.md` 5장 "5주차 구현 진행 현황" 안의 **7번(10/6 실측)** 과 **8번(10/8~10/9 2인 수집)** — 지금까지의 발견·기준값·한계가 여기 있다.
2. `docs/side_capture_plan.md` — 어떤 자세를 어떻게 찍었는지.
3. `src/logic/decision.py` — 판정 로직(`judge`, `AlertStateMachine`). 기준값 상수가 맨 위에 있다.

### 환경 준비

```
git pull
.venv\Scripts\activate          # 없으면: python -m venv .venv 후 pip install -r requirements.txt
python -m pytest -q tests       # 64개 통과해야 정상
```

카메라·RealSense가 없어도 위 테스트와 아래 작업은 모두 돌아간다. 그림을 그리려면 `pip install matplotlib`가 추가로 필요하다(requirements.txt에는 아직 없음).

---

## 1. 수집한 로그 (`data/side_logs/`)

10/8~10/9에 2명(p01, p02)이 11가지 자세를 20초씩(30Hz, 약 600줄) 찍은 CSV다. **숫자만 들어 있고 얼굴 사진은 없다.** 확인용 사진(`data/snaps/`)은 개인정보라 git에 올리지 않았다.

| 항목 | 내용 |
|---|---|
| 파일명 | `side_<참가자>_<자세>[_r2|_r3].csv` — 접미사 없음=1회차, `_r2`, `_r3`=반복 촬영 |
| 참가자 | p01 (자세당 3회), p02 (자세당 2회) |
| 자세 11종 | normal, neck_light, neck_hard, forward, back_light, back_hard, slouch, back_neck, left_lean_neck, right_lean_neck, chin |
| 열 | `timestamp`, 17개 keypoint의 `*_score`, `head_depth_m`, `chest_depth_m`, `hip_depth_m`, `neck_forward_offset_m`, `torso_recline_offset_m`, `head_forward_deg`, `torso_pitch_deg`, `elbow_left_deg`, `elbow_right_deg`, `torso_len_sw`, `head_up_m`, `wrist_face_sw` |
| 빈 칸 | depth가 무효(근접 사각지대·가림)이거나 신뢰도 미달인 프레임. **0이 아니라 결측**이다. 비율을 같이 보고할 것 |

### 자세 한 줄 설명

| 이름 | 자세 |
|---|---|
| normal | 바른 자세, 팔은 책상 위 |
| neck_light / neck_hard | 몸통은 세우고 고개만 앞으로 (약 / 강, 거북목) |
| forward | 허리부터 몸 전체를 앞으로 숙임 |
| back_light / back_hard | 뒤로 기댐 (평소 정도 / 확실히 젖힘) |
| slouch | 엉덩이는 두고 등을 말아 구부정 |
| back_neck | 뒤로 기대면서 고개 내밈 |
| left_lean_neck / right_lean_neck | 옆으로 기대면서 고개 내밈 |
| chin | 턱 괴기 |

### 알려진 한계 (분석할 때 꼭 고려)

- p01은 책상에 가려 엉덩이 depth가 거의 안 읽혀(0~17%) `torso_recline_offset_m`, `torso_pitch_deg`를 쓸 수 없다. p02는 대체로 유효하다.
- "살짝/많이" 강도가 사람·회차마다 달랐다(neck_light 중앙값이 p02에서 0.21 대 0.16).
- 가슴 depth 정의를 10/8에 바꿨다(어깨 중점 1픽셀 → 가슴 4점 중앙값, 커밋 `c4599e6`). 1회차와 반복 회차의 값이 일관한지 먼저 확인할 것.
- 로그에 keypoint 위치(x, y)가 없어서 **좌우 기울임은 이 로그로 분석할 수 없다.**

### 지금 판정 기준값 (잠정, `src/logic/decision.py`)

| 항목 | 기준 |
|---|---|
| 뒤로 기댐 | `neck_forward_offset_m` ≤ 0.10 경고 (0.13에서 선형, 약 0.109부터 주의) |
| 고개 내밈 | `neck_forward_offset_m` ≥ 0.18 경고 (0.15에서 선형, 약 0.171부터 주의) |
| 턱 괴기 | `wrist_face_sw` ≤ 0.65 경고 (1.2에서 선형, 약 0.815부터 주의) |
| 화면 근접 | 머리·가슴 depth < 0.40m |

---

## 2. 작업 목록

| # | 작업 | 담당 | 예상 | 우선순위 |
|---|---|---|---|---|
| T1 | 재현성 분석 (회차 간 vs 사람 간 변동) | | 반나절 | 높음 |
| T2 | 로그 재생기로 판정 검증 (혼동 행렬) | | 반나절 | 높음 |
| T3 | 알림 쿨다운 오프라인 검증 | | 2~3시간 | 높음 |
| T4 | 기준값 최적화(ROC)와 새 신호 탐색 | | 하루 | 중간 |
| T5 | 참가자 단위 교차검증 RF 파이프라인 | | 하루 | 중간 |
| T6 | 논문 방법·실험 섹션 초안과 그림 | | 이틀 | 중간 |

T1~T3은 서로 독립이라 나눠서 병렬로 해도 된다. T4는 T1의 결과를 쓰면 좋다.

### T1. 재현성 분석

- **목적**: "같은 자세를 다시 취했을 때 값이 얼마나 비슷한가"와 "사람이 다르면 얼마나 다른가"를 숫자로 보여 준다. 기준값이 이 변동보다 충분히 벌어져 있어야 믿을 수 있다.
- **입력**: `data/side_logs/*.csv`
- **할 일**:
  1. 파일 이름에서 참가자·자세·회차를 파싱해 한 표로 합친다 (`pandas`).
  2. 자세별 회차 중앙값을 구한다 (`neck_forward_offset_m`, `head_forward_deg`, `head_up_m`, `wrist_face_sw` 우선).
  3. 같은 사람 내 변동(회차 간 표준편차)과 사람 간 차이를 비교한다. 자세 간 거리 대비 변동 비율도 같이 낸다.
  4. 결측 비율을 자세별로 표시한다.
- **결과물**: `scripts/analyze_repeatability.py`, `experiments/repeatability_results.md`(표 + 해석 몇 줄)
- **확인 방법**: `project.md` 8번의 "normal 0.12~0.16, 고개 내밈 0.16~0.23" 같은 수치와 크게 어긋나지 않는지 대조.

### T2. 로그 재생기로 판정 검증

- **목적**: 수집한 로그를 `judge`에 넣었을 때 자세별로 얼마나 맞게 나오는지(정상 오탐, 검출률)를 숫자로 낸다.
- **입력**: `data/side_logs/*.csv`, `src/logic/decision.py`의 `judge`
- **할 일**:
  1. 로그 한 줄을 `depth_features` 딕셔너리(`head_depth_m`, `chest_depth_m`, `hip_depth_m`, `neck_forward_offset_m`, `torso_recline_offset_m`)로 바꾼다. 빈 칸은 `None`.
  2. keypoint는 로그에 없으므로 `judge`가 keypoint를 쓰는 부분(좌우 기울임, 턱 괴기)은 우회하거나 `wrist_face_sw` 값으로 대체하도록 얇은 어댑터를 만든다. (`wrist_face_distance` 함수가 keypoint를 받으므로 어댑터에서 직접 비율만 계산해도 된다.)
  3. 자세별로 프레임 중 경고/주의/정상 비율과 `posture_kind` 분포를 낸다.
- **결과물**: `scripts/replay_judge_on_logs.py`, `experiments/replay_confusion.md`
- **확인 방법**: normal에서 경고 오탐이 0%에 가까워야 한다 (10/9 분석 기준). 안 그렇다면 어댑터 버그부터 의심.
- **주의**: 좌우 기울임은 이 방식으로 검증할 수 없다. 표에 "해당 없음"으로 적을 것.

### T3. 알림 쿨다운 오프라인 검증

- **목적**: 보류했던 "근접 알림 1회 후 60초 쿨다운, 자세 경고 연속 3회 후 300초 쿨다운"이 맞는지 카메라 없이 확인한다.
- **입력**: `AlertStateMachine` (`src/logic/decision.py`), 로그의 `timestamp`
- **할 일**:
  1. 로그의 `timestamp`를 5초 간격(`--sample-interval` 기본값)으로 샘플링해 `AlertStateMachine.update(now, judgement)`에 넣는다.
  2. 시나리오를 만든다 (예: 근접 상태 3분 지속, 자세 경고 7분 지속, 경고가 끊겼다 이어지는 경우). 합성 `Judgement`를 쓰면 로그 없이도 된다.
  3. 알림이 나온 시각(`+N.Ns`)이 기대한 쿨다운 간격과 맞는지 확인한다.
- **결과물**: `tests/test_alert_scenarios.py`(재현 가능한 시나리오 테스트), 필요하면 `experiments/alert_cooldown_check.md`
- **확인 방법**: `python -m pytest -q tests`가 통과하고, 근접 알림 간격이 ≥60초, 자세 알림 간격이 ≥300초임을 단언문으로 남긴다.

### T4. 기준값 최적화와 새 신호 탐색

- **목적**: 지금 기준값이 최선인지 검증하고, 아직 구분이 안 되는 자세를 가를 신호를 찾는다.
- **입력**: T1 결과, 로그
- **할 일**:
  1. 신호별(`neck_forward_offset_m` 등)로 자세 쌍(normal 대 back, normal 대 neck)의 ROC 곡선과 최적 임계값을 구한다. 참가자별로 따로, 합쳐서도.
  2. 구분이 안 되는 쌍을 가르는 특징을 찾는다. 후보는 다음과 같다.
     - slouch 대 neck_hard: `chest_depth_m`의 상대값, `head_up_m`, `torso_len_sw`, 어깨·엉덩이 score
     - back_neck 대 normal: `head_forward_deg`, `chest_depth_m`, 엉덩이 score
  3. 찾은 신호는 우선 분포 그림으로만 남기고, 판정에 넣는 것은 팀 합의 후에 한다.
- **결과물**: `experiments/threshold_roc.md`, 그림 파일(`experiments/figs/`)
- **주의**: 표본이 2인이라 "가능성 확인"으로만 쓸 것. 과적합 위험을 문서에 한 줄 적는다.

### T5. 참가자 단위 교차검증 RF

- **목적**: 로그 특징으로 자세를 분류했을 때 새 사람에게도 통하는지 본다. 논문 실험 섹션 후보.
- **입력**: `data/side_logs/*.csv`, 기존 `src/logic/rf_baseline.py`(참고)
- **할 일**:
  1. 프레임을 학습 샘플로 쓰되, **p01로 학습해 p02로 평가, 반대로도 평가**한다(참가자 단위 분할). 같은 회차의 연속 프레임을 학습·평가에 섞지 않는다.
  2. 특징 중요도와 자세별 혼동 행렬을 낸다.
  3. 결측은 대치하지 말고 "결측 여부" 특징을 두 가지 방식(결측 플래그 / 결측 프레임 제외)으로 비교한다.
- **결과물**: `src/logic/rf_side_logs.py`(또는 `scripts/`), `experiments/rf_logo_results.md`
- **주의**: 라벨은 **파일 이름의 자세**다. 자세를 잡는 처음 몇 초는 이미 제외돼 있다(`--countdown 5`).

### T6. 논문 방법·실험 섹션 초안

- **목적**: 12/2 드래프트 마감 대비. 이미 있는 결과로 쓸 수 있는 부분을 먼저 쓴다.
- **입력**: `project.md`, `docs/absolute_posture_design.md`, T1·T4·T5 결과
- **할 일**: 방법(스테레오 깊이 + MoveNet 융합, 가슴 영역 중앙값 depth, 근접 보완), 실험 설정(수집 프로토콜, 2인, 자세 11종), 결과 그림 초안.
- **결과물**: 위치는 팀에서 쓰는 논문 문서 폴더. 파일이 없으면 `paper/` 아래 마크다운으로 시작.

---

## 3. 카메라가 있어야 하는 일 (복귀 후)

- 참가자·거리를 늘린 재보정, 40cm 이하 근접 검증
- 좌우 기울임 자세 재촬영 (로그에 keypoint 위치를 추가한 뒤)
- 강도 기준을 맞춘 neck_light, back_light 재촬영
- 수정한 판정의 실시간 검증
- IMU 중력 보정 연결

## 4. 작업 규칙

- 코드를 바꾸면 `python -m pytest -q tests`가 통과해야 한다. 테스트는 합성 입력으로 만든다 (카메라 불필요).
- 기준값 상수(`src/logic/decision.py`)를 바꾸려면 근거(분석 결과)를 `project.md`에 같이 적는다. 바꾸기 전에 서로 알린다.
- 커밋 메시지는 한국어로, 무엇을 왜 바꿨는지 적는다.
- `data/*.csv`는 `.gitignore` 대상이다. 새 로그를 팀과 공유할 때는 `data/side_logs/`에만 넣고 얼굴이 나오는 사진은 올리지 않는다.
- 막히면 결과를 억지로 맞추지 말고 "이 쌍은 구분 안 됨"으로 기록한다. 한계도 논문의 결과다.
