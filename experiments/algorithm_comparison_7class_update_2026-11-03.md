# 7개 자세 분류 알고리즘 비교 실험 변경 및 결과 보고서

-   작성일: 2026-11-03
-   대상 알고리즘: Random Forest, XGBoost, LightGBM
-   검증 방식: Leave-One-Participant-Out (LOPO)
-   목적: 공동작업자가 최근 코드 변경 사항, 실험 구성, 결과 파일 및
    현재까지의 결론을 이해하고 후속 분석을 이어갈 수 있도록 기록

------------------------------------------------------------------------

## 1. 이번 작업의 목적과 진행 상황

기존 7개 자세 분류 알고리즘 비교 실험에 대해, 각 알고리즘의 Markdown
요약 결과뿐 아니라 후속 분석에 필요한 CSV 결과도 동일한 형식으로
저장하도록 코드를 보완하였다.

이번 변경의 핵심은 다음과 같다.

1.  세 알고리즘의 기존 학습 및 평가 설정은 유지한다.
2.  참가자 단위 LOPO 평가 결과를 모은다.
3.  전체 OOF 예측 기준 클래스별 성능과 혼동행렬을 CSV로 저장한다.
4.  참가자별 성능과 혼동행렬도 별도 CSV로 저장한다.
5.  세 알고리즘을 같은 특징 세트와 결측 처리 조건에서 비교한다.

현재 Random Forest, XGBoost, LightGBM 모두 수정된 스크립트 실행을
완료했고, 알고리즘별 CSV 5개가 생성된 상태다.

## 2. 데이터 및 분류 설정

### 2.1 데이터

-   로그 디렉터리: `data/side_logs/`
-   총 로그 파일: 99개
-   참가자: 3명 (`p01`, `p02`, `p03`)
-   전체 프레임: 54,070개
-   원래 자세 라벨: 11개
-   통합 분류 클래스: 7개

### 2.2 11개 라벨을 7개 클래스로 통합

  통합 클래스     기존 라벨
  --------------- -------------------------------
  `normal`        `normal`
  `turtle_neck`   `neck_hard`, `neck_light`
  `forward`       `forward`
  `lean_back`     `back_hard`, `back_light`
  `left_lean`     `left_lean_neck`
  `right_lean`    `right_lean_neck`
  `bad_posture`   `back_neck`, `chin`, `slouch`

클래스별 데이터 수는 다음과 같다.

  클래스             프레임 수
  --------------- ------------
  `bad_posture`         14,876
  `turtle_neck`         10,084
  `lean_back`            9,732
  `forward`              5,215
  `right_lean`           4,875
  `left_lean`            4,759
  `normal`               4,529
  **합계**          **54,070**

클래스별 데이터 수에 차이가 있으므로 전체 Accuracy만으로 모델을 판단하지
않고 Macro-F1을 주요 비교 지표로 사용한다.

## 3. 특징 세트 및 결측 처리 조건

### 3.1 특징 세트

  특징 세트    구성
  ------------ ----------------------------------------
  `depth`      DEPTH 특징 5개
  `no_hip`     DEPTH + EXTRA 중 고관절 의존 특징 제외
  `all`        DEPTH + EXTRA 전체
  `relative`   `no_hip`에서 절대 깊이 특징 제외

DEPTH 특징: - `head_depth_m` - `chest_depth_m` - `hip_depth_m` -
`neck_forward_offset_m` - `torso_recline_offset_m`

EXTRA 특징: - `head_forward_deg` - `torso_pitch_deg` -
`elbow_left_deg` - `elbow_right_deg` - `torso_len_sw` - `head_up_m` -
`wrist_face_sw`

`no_hip`에서는 `hip_depth_m`, `torso_recline_offset_m`,
`torso_pitch_deg`를 제외한다. `relative`에서는 추가로 `head_depth_m`,
`chest_depth_m`를 제외한다.

### 3.2 결측 처리

  -----------------------------------------------------------------------
  모드                    처리 방법               특징
  ----------------------- ----------------------- -----------------------
  A                       학습 데이터 중앙값으로  평가 프레임을 제거하지
                          결측 대체 + 결측 여부   않음
                          플래그 추가             

  B                       선택된 특징에 결측치가  평가 프레임 유지율이
                          있는 행 제거            낮아질 수 있음
  -----------------------------------------------------------------------

따라서 Mode B의 성능 수치는 평가 데이터 일부를 제외한 결과다. Mode A와
B의 점수를 단순히 같은 조건의 성능처럼 비교하면 안 된다. 결과를 해석할
때 Macro-F1과 함께 `keep_ratio`를 반드시 확인한다.

## 4. 검증 방식 및 알고리즘 설정

### 4.1 검증 방식

Leave-One-Participant-Out (LOPO)를 사용한다.

-   한 참가자의 데이터를 테스트 세트로 둔다.
-   나머지 참가자들의 데이터를 학습 세트로 사용한다.
-   테스트 참가자를 바꾸어 총 3회 평가한다.

참가자가 3명뿐이므로 새로운 사람에 대한 일반화 성능을 확정적으로 주장할
수 없다. 참가자별 결과를 함께 살펴보고 제한점을 명시한다.

### 4.2 모델 설정 요약

  --------------------------------------------------------------------------------
  알고리즘                주요 설정                    클래스 불균형 처리
  ----------------------- ---------------------------- ---------------------------
  Random Forest           300 trees,                   클래스 가중치
                          `class_weight="balanced"`,   
                          `random_state=0`             

  XGBoost                 300 estimators,              balanced sample weights
                          `max_depth=6`,               
                          `learning_rate=0.05`,        
                          `subsample=0.9`,             
                          `colsample_bytree=0.9`,      
                          `tree_method="hist"`,        
                          `random_state=0`             

  LightGBM                300 estimators,              `class_weight="balanced"`
                          `num_leaves=31`,             
                          `learning_rate=0.05`,        
                          `random_state=0`             
  --------------------------------------------------------------------------------

이번 CSV 출력 기능 추가는 성능 분석 자료를 더 세분화하기 위한 변경이다.
위 모델의 주요 학습 설정과 7개 클래스 매핑, 특징 세트 및 A/B 조건은
유지하는 것을 목표로 했다.

## 5. 전체 알고리즘 비교 결과

아래 값은 2026-11-03에 각 스크립트를 실행해 생성한 `*_summary.csv`에서
가져왔다.

### 5.1 Macro-F1

  특징 세트   결측 모드     Random Forest      XGBoost     LightGBM
  ----------- ----------- --------------- ------------ ------------
  depth       A                    0.4566       0.4460       0.4431
  depth       B                    0.4659   **0.4856**       0.4831
  no_hip      A                **0.5091**       0.5055       0.4937
  no_hip      B                **0.5164**       0.4997       0.4877
  all         A                **0.5192**       0.4954       0.5083
  all         B                    0.4595       0.4499   **0.4832**
  relative    A                **0.4577**       0.4457       0.4496
  relative    B                **0.4834**       0.4745       0.4739

### 5.2 Accuracy 및 평가 프레임 유지율

  ------------------------------------------------------------------------
  특징 세트  모드       RF Accuracy XGB Accuracy         LGBM  평가 프레임
                                                     Accuracy       유지율
  ---------- --------- ------------ ------------ ------------ ------------
  depth      A               0.5231       0.5122       0.5113       100.0%

  depth      B               0.5897       0.6257       0.6137        38.9%

  no_hip     A               0.5855       0.5830       0.5661       100.0%

  no_hip     B               0.6372       0.6348       0.6588       RF/XGB
                                                               66.5%, LGBM
                                                                     33.3%

  all        A               0.6080       0.5685       0.5802       100.0%

  all        B               0.6260       0.5968       0.6483        33.3%

  relative   A               0.5357       0.5156       0.5204       100.0%

  relative   B               0.6221       0.6055       0.6229        66.5%
  ------------------------------------------------------------------------

참고: 표의 유지율은 각 알고리즘 summary CSV에 저장된 값이다. 동일한 특징
세트와 모드에서는 대체로 동일하지만, `no_hip + B`의 LightGBM 값은 다른
두 알고리즘과 다르므로 모델별 CSV 값을 그대로 보고해야 한다.

### 5.3 전체 프레임을 유지하는 Mode A 기준 최고 조합

  알고리즘        최고 특징 조합       Macro-F1     Accuracy   유지율
  --------------- ---------------- ------------ ------------ --------
  Random Forest   `all + A`          **0.5192**   **0.6080**     100%
  XGBoost         `no_hip + A`           0.5055       0.5830     100%
  LightGBM        `all + A`              0.5083       0.5802     100%

현재 summary 결과에서 가장 높은 Macro-F1은 Random Forest의 `all + A`
조합(0.5192)이다. 따라서 후속 분석을 위한 우선 기준 모델로 이 조합을
검토할 수 있다. 다만 0.5192와 다른 알고리즘의 차이가 크지 않으므로,
참가자별 성능과 클래스별 성능을 확인하기 전에는 최종 우수 모델로
단정하지 않는다.

### 5.4 Mode B 해석 시 주의

Mode B는 결측치가 포함된 행을 제거하므로 평가 프레임이 줄어든다. 예를
들어 `depth + B`는 유지율이 약 38.9%이고, `all + B`는 약 33.3%다. 이
조건에서 Accuracy가 높게 나오더라도 전체 입력 상황에서의 성능이 더
좋다는 뜻은 아니다.

따라서 현재 보고서에서는 Mode A를 주된 알고리즘 비교 기준으로 삼고, Mode
B는 결측 행 제거의 영향을 살펴보는 보조 조건으로 해석한다.

## 6. 코드 변경 및 생성 파일

### 6.1 수정된 실행 스크립트

-   `scripts/rf_lopo_7class.py`
-   `scripts/xgb_lopo_7class.py`
-   `scripts/lgbm_lopo_7class.py`

RF와 XGBoost 스크립트는 기존 `.py.bak` 백업 파일도 로컬에 생성했다. 백업
파일은 복구용이며, 팀 공유가 꼭 필요하지 않으면 저장소에 추가하지
않는다.

### 6.2 알고리즘별 CSV 출력

각 알고리즘은 아래 5개 CSV를 `experiments/`에 저장한다.

  파일 접두사              내용
  ------------------------ --------------------
  `rf_lopo_7class_...`     Random Forest 결과
  `xgb_lopo_7class_...`    XGBoost 결과
  `lgbm_lopo_7class_...`   LightGBM 결과

각 접두사별 파일: 1. `*_summary.csv`: 특징 세트 × 결측 모드별 Macro-F1,
Accuracy, 유지율 2. `*_metrics.csv`: 전체 OOF 예측 기준 클래스별
Precision, Recall, F1, Support 3. `*_fold_metrics.csv`: 참가자별 클래스
Precision, Recall, F1, Support 4. `*_confusion.csv`: 전체 OOF 혼동행렬
5. `*_fold_confusion.csv`: 참가자별 혼동행렬

따라서 세 알고리즘에 대해 총 15개 CSV가 생성되었다.

### 6.3 주요 파일 경로

``` text
scripts/
├── rf_lopo_7class.py
├── xgb_lopo_7class.py
└── lgbm_lopo_7class.py

experiments/
├── rf_lopo_7class.md
├── xgb_lopo_7class.md
├── lgbm_lopo_7class_summary.csv
├── rf_lopo_7class_summary.csv
├── rf_lopo_7class_metrics.csv
├── rf_lopo_7class_fold_metrics.csv
├── rf_lopo_7class_confusion.csv
├── rf_lopo_7class_fold_confusion.csv
├── xgb_lopo_7class_summary.csv
├── xgb_lopo_7class_metrics.csv
├── xgb_lopo_7class_fold_metrics.csv
├── xgb_lopo_7class_confusion.csv
├── xgb_lopo_7class_fold_confusion.csv
└── lgbm_lopo_7class_*.csv
```

`lgbm_lopo_7class_*.csv`는 동일한 다섯 종류의 파일을 의미한다.

## 7. 현재까지의 잠정 결론

1.  전체 평가 프레임을 유지하는 Mode A에서 Random Forest `all + A`가
    가장 높은 Macro-F1 0.5192를 기록했다.
2.  XGBoost는 `no_hip + A`에서 Macro-F1 0.5055를 기록했다.
3.  LightGBM은 `all + A`에서 Macro-F1 0.5083을 기록했다.
4.  Mode B는 평가 프레임 제거가 동반되므로 Macro-F1과 Accuracy뿐 아니라
    유지율을 함께 해석해야 한다.
5.  세 알고리즘 모두 참가자 3명으로 평가했기 때문에 결과의 일반화에는
    제한이 있다.

이 결과는 현재까지의 실험에서 나온 잠정 결론이다. 클래스별 Recall,
참가자별 Macro-F1 및 혼동행렬을 확인한 후 최종 모델 선정과 연구 결론을
확정한다.

## 8. 다음 작업

후속 작업은 다음 순서로 진행한다.

1.  **클래스별 성능 비교**: 세 알고리즘의 `*_metrics.csv`를 비교해 어떤
    자세의 Recall/F1이 낮은지 확인한다.
2.  **참가자별 성능 비교**: `*_fold_metrics.csv`에서 특정 참가자에 대한
    성능 편차를 확인한다.
3.  **혼동행렬 분석**: `*_confusion.csv`, `*_fold_confusion.csv`로 자세
    간 오분류 패턴을 분석한다.
4.  **어블레이션 실험**: A0\~A6 조건의 특징 제거/추가 효과를 비교한다.
5.  **2D 기준선 재실행**: 2D 기반 기준 모델과 현재 스테레오/깊이 특징
    기반 결과를 비교한다.
6.  위 분석을 반영해 발표 자료와 논문 결과 서술을 갱신한다.

------------------------------------------------------------------------

## 9. 공동작업자 실행 방법

프로젝트 루트에서 가상환경을 활성화한 뒤 다음과 같이 실행한다.

``` bash
cd ~/AIoT-Project
source venv/bin/activate

PYTHONPATH=. python scripts/rf_lopo_7class.py
PYTHONPATH=. python scripts/xgb_lopo_7class.py
PYTHONPATH=. python scripts/lgbm_lopo_7class.py
```

실행 후 CSV 결과 확인:

``` bash
ls -lh experiments/*lopo_7class*.csv
```

각 알고리즘의 전체 요약만 빠르게 확인:

``` bash
cat experiments/rf_lopo_7class_summary.csv
cat experiments/xgb_lopo_7class_summary.csv
cat experiments/lgbm_lopo_7class_summary.csv
```

실험을 다시 실행하면 해당 이름의 결과 CSV가 덮어써질 수 있으므로, 이전
결과를 보존해야 한다면 실행 전에 별도 복사본을 만든다.

## 10. Git 커밋 전 확인 사항

이번 변경과 직접 관련된 파일만 선별해 커밋한다. 저장소에 다른 미완성
작업이나 개인 파일이 있을 수 있으므로 `git add .`는 사용하지 않는다.

``` bash
cd ~/AIoT-Project
git status --short
```

변경 파일을 확인한 다음, 필요한 스크립트·보고서·CSV만 개별 지정해
스테이징한다. 예:

``` bash
git add scripts/rf_lopo_7class.py
git add scripts/xgb_lopo_7class.py
git add scripts/lgbm_lopo_7class.py
git add experiments/algorithm_comparison_7class_update_2026-11-03.md
git add experiments/rf_lopo_7class_summary.csv
git add experiments/rf_lopo_7class_metrics.csv
git add experiments/rf_lopo_7class_fold_metrics.csv
git add experiments/rf_lopo_7class_confusion.csv
git add experiments/rf_lopo_7class_fold_confusion.csv
git add experiments/xgb_lopo_7class_summary.csv
git add experiments/xgb_lopo_7class_metrics.csv
git add experiments/xgb_lopo_7class_fold_metrics.csv
git add experiments/xgb_lopo_7class_confusion.csv
git add experiments/xgb_lopo_7class_fold_confusion.csv
git add experiments/lgbm_lopo_7class_summary.csv
git add experiments/lgbm_lopo_7class_metrics.csv
git add experiments/lgbm_lopo_7class_fold_metrics.csv
git add experiments/lgbm_lopo_7class_confusion.csv
git add experiments/lgbm_lopo_7class_fold_confusion.csv
```

그 후 반드시 스테이징 목록을 확인한다.

``` bash
git status --short
git diff --cached --stat
```

문제가 없으면 커밋 및 푸시한다.

``` bash
git commit -m "Add detailed 7-class LOPO metrics for RF XGBoost and LightGBM"
git push origin master
```

푸시 후 원격 반영 여부를 확인한다.

``` bash
git status --short
git log -1 --oneline
```

**주의:** 위 명령은 예시다. `git add` 전에 실제 `git status --short`
결과를 확인하고, 파일이 존재하며 이번 실험 변경과 관련된 경우에만
추가한다.
