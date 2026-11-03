# 7-Class LOPO 알고리즘 비교 후속 분석 보고서
작성일: 2026-11-03

## 1. 분석 목적

RF(Random Forest), XGBoost, LightGBM 세 알고리즘의 7-class 자세 분류 결과를 클래스별·참가자별로 비교했다. 본 보고서는 기존 알고리즘 비교 실험을 보완하는 후속 분석 기록이다.

## 2. 분석 조건

- 데이터 분류: 7개 자세 클래스
- 평가 방식: LOPO(Leave-One-Participant-Out)
- 본문 비교 조건: `feature_set=all`, `mode=A`
- 특징 구성: DEPTH 5개 + EXTRA 7개, 총 12개 특징
- Mode A: 전체 프레임을 유지하고 학습 데이터 기준 결측값 대체 및 결측 여부 표시
- 참가자: p01, p02, p03
- 주요 지표: Macro-F1, 클래스별 Precision·Recall·F1

세 알고리즘을 같은 특징 구성과 Mode A로 비교했다. Mode B는 프레임 유지율이 달라질 수 있으므로 이번 후속 비교표의 주 비교 조건에서는 제외했다.

## 3. 클래스별 성능 비교 — pooled OOF 결과

아래 값은 각 알고리즘의 `*_metrics.csv`에서 `all + A` 행을 선택한 결과다. 전체 OOF 예측을 클래스별로 집계한 Precision, Recall, F1이다.

| 클래스 | RF Precision | RF Recall | RF F1 | XGBoost Precision | XGBoost Recall | XGBoost F1 | LightGBM Precision | LightGBM Recall | LightGBM F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bad_posture | 0.5105 | 0.6581 | **0.5750** | 0.4676 | 0.5614 | 0.5102 | 0.4724 | 0.5978 | 0.5277 |
| forward | 0.5978 | 0.6385 | **0.6175** | 0.5981 | 0.6176 | 0.6077 | 0.6359 | 0.5937 | 0.6140 |
| lean_back | 0.8635 | 0.8119 | 0.8369 | 0.8463 | 0.8436 | 0.8450 | 0.8412 | 0.8564 | **0.8487** |
| left_lean | 0.1649 | 0.0574 | **0.0851** | 0.0737 | 0.0317 | 0.0444 | 0.1249 | 0.0496 | 0.0710 |
| normal | 0.7759 | 0.8097 | 0.7924 | 0.8268 | 0.7693 | 0.7970 | 0.8561 | 0.7673 | **0.8093** |
| right_lean | 0.0010 | 0.0004 | 0.0006 | 0.0025 | 0.0014 | **0.0018** | 0.0012 | 0.0008 | 0.0010 |
| turtle_neck | 0.6768 | 0.7848 | **0.7268** | 0.6087 | 0.7254 | 0.6619 | 0.6497 | 0.7274 | 0.6864 |

### 클래스별 관찰

- RF는 bad_posture, forward, left_lean, turtle_neck에서 가장 높은 pooled F1을 기록했다.
- LightGBM은 lean_back과 normal에서 가장 높은 pooled F1을 기록했다.
- right_lean은 세 모델 모두 F1이 0.002 이하로 사실상 분류하지 못했다.
- left_lean 역시 F1이 0.1 미만으로 낮다. 특히 Recall이 낮아 실제 좌측 기울기 샘플을 놓치는 문제가 크다.
- lean_back과 normal은 상대적으로 높은 분류 성능을 보였다.

## 4. 참가자별 Macro-F1 비교 — fold 결과

각 참가자를 테스트 대상으로 둔 fold에서 7개 클래스 F1의 산술평균을 계산했다. 아래 값은 `*_fold_metrics.csv`에서 `all + A`를 선택해 계산한 값이다.

| 테스트 참가자 | RF Macro-F1 | XGBoost Macro-F1 | LightGBM Macro-F1 |
|---|---:|---:|---:|
| p01 | 0.5143 | 0.5421 | **0.5597** |
| p02 | **0.4794** | 0.4186 | 0.4101 |
| p03 | **0.5512** | 0.5044 | 0.5224 |
| 세 참가자 평균 | **0.5149** | 0.4884 | 0.4974 |

### 참가자별 관찰

- p01에서는 LightGBM이 가장 높은 Macro-F1을 보였다.
- p02와 p03에서는 RF가 가장 높았다.
- 세 참가자 평균은 RF 0.5149, LightGBM 0.4974, XGBoost 0.4884 순이다.
- 참가자별 최고 알고리즘이 동일하지 않으므로 개인별 데이터 특성에 따른 성능 차이가 존재한다.
- 참가자가 3명뿐이므로 이를 근거로 일반화 성능을 확정적으로 주장해서는 안 된다.

## 5. 참가자별 클래스 F1 평균

아래는 각 참가자 fold의 클래스별 F1을 세 참가자에 대해 산술평균한 값이다. 이는 pooled OOF F1과 계산 방식이 다르므로 별도로 해석해야 한다.

| 클래스 | RF | XGBoost | LightGBM |
|---|---:|---:|---:|
| bad_posture | **0.5648** | 0.5044 | 0.5176 |
| forward | **0.5910** | 0.5788 | 0.5653 |
| lean_back | 0.8357 | 0.8413 | **0.8441** |
| left_lean | **0.0887** | 0.0583 | 0.0885 |
| normal | **0.7964** | 0.7911 | 0.7952 |
| right_lean | 0.0009 | **0.0021** | 0.0015 |
| turtle_neck | **0.7271** | 0.6424 | 0.6694 |

## 6. 종합 해석 및 한계

1. 현재 `all + A` 조건에서 참가자 평균 Macro-F1은 RF가 가장 높다.
2. RF가 모든 참가자에서 1위인 것은 아니다. p01에서는 LightGBM이 더 높다.
3. 좌우 기울기(left_lean, right_lean)는 세 알고리즘 공통의 취약 클래스다.
4. 좌우 기울기 성능 저하 원인은 현재 지표만으로 확정할 수 없다. 특징값의 구분력, 라벨 품질, 자세 수집 조건, 키포인트 추정 품질과 클래스 간 혼동을 추가 점검해야 한다.
5. 참가자는 3명뿐이므로 참가자 평균과 편차를 함께 제시하고, 더 많은 참가자에 대한 검증이 필요함을 명시한다.
6. pooled OOF 지표와 참가자별 fold 지표의 평균은 서로 다른 집계 방식이다. 보고서나 논문에서 두 값을 혼용하지 않는다.

## 7. 다음 작업

- A0~A6 Ablation 실험의 특징 구성과 가설을 기존 코드·연구 설계에 맞춰 확정한다.
- RF를 우선 기준 모델로 활용하되, LightGBM과 XGBoost 비교 결과도 보존한다.
- 좌우 기울기 클래스에 대해 confusion matrix, 원본 로그, 특징 분포 및 라벨링 기준을 점검한다.
- Ablation 실험 후 2D baseline을 재실행하고 2D와 stereo/depth 기반 결과를 비교한다.

## 8. 재현에 사용한 결과 파일

- `experiments/rf_lopo_7class_metrics.csv`
- `experiments/xgb_lopo_7class_metrics.csv`
- `experiments/lgbm_lopo_7class_metrics.csv`
- `experiments/rf_lopo_7class_fold_metrics.csv`
- `experiments/xgb_lopo_7class_fold_metrics.csv`
- `experiments/lgbm_lopo_7class_fold_metrics.csv`

본 보고서는 위 결과 파일의 `all + A` 행을 기준으로 한 후속 분석 기록이며, 새로운 모델 학습이나 재실험 결과를 의미하지 않는다.
