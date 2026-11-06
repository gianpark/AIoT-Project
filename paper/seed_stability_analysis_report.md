# Seed 안정성 실험 결과 및 분석 보고서

## 1. 실험 목적

데스크 자세 분류 모델의 성능이 학습 난수 시드에 따라 얼마나 달라지는지 확인하기 위해 RF, LightGBM, XGBoost의 A0~A6 특징 조합을 동일한 LOPO 분할에서 Seed 0~4로 평가하였다. 핵심 목적은 특정 Seed에 의존하지 않고 성능이 비교적 일관되게 유지되는 모델과 특징 조합을 확인하는 것이다.

## 2. 실험 설정

| 항목 | 설정 |
|---|---|
| 전체 프레임 | 54,070 |
| 참가자 | p01, p02, p03 |
| 클래스 | 7개 |
| 모델 | RF, LightGBM, XGBoost |
| Ablation | A0~A6 |
| Seed | 0, 1, 2, 3, 4 |
| 검증 | LOPO 3-Fold |
| 전처리 | 학습 폴드 중앙값 대치 + 결측 플래그 |
| Seed 변경 | 각 모델의 random_state |
| 총 학습 횟수 | 3 × 7 × 5 × 3 = 315회 |

### 클래스별 프레임 수

| 클래스 | 프레임 수 |
|---|---:|
| normal | 4,529 |
| turtle_neck | 10,084 |
| forward | 5,215 |
| lean_back | 9,732 |
| left_lean | 4,759 |
| right_lean | 4,875 |
| bad_posture | 14,876 |
| **합계** | **54,070** |

클래스별 데이터 수에 차이가 있으므로 Accuracy뿐 아니라 Macro-F1을 주요 평가 지표로 함께 확인하였다.

## 3. Ablation 구성

A0 기본 특징 9개: `head_depth_m`, `chest_depth_m`, `neck_forward_offset_m`, `head_forward_deg`, `elbow_left_deg`, `elbow_right_deg`, `torso_len_sw`, `head_up_m`, `wrist_face_sw`.

| Ablation | 제거 특징 | 사용 특징 수 |
|---|---|---:|
| A0 | 제거 없음 | 9 |
| A1 | `head_depth_m`, `chest_depth_m` | 7 |
| A2 | `head_forward_deg`, `head_up_m` | 7 |
| A3 | `elbow_left_deg`, `elbow_right_deg`, `wrist_face_sw` | 6 |
| A4 | `torso_len_sw` | 8 |
| A5 | `neck_forward_offset_m` | 8 |
| A6 | `head_depth_m`, `chest_depth_m`, `neck_forward_offset_m`, `head_up_m` | 5 |

## 4. 실험 결과

아래 값은 Seed 0~4 평균 ± Seed 간 표준편차(ddof=0)이다. Fold mean은 각 LOPO Fold 점수의 평균이며, Pooled OOF는 모든 Fold의 테스트 예측을 합쳐 계산한 지표이다.

### 4.1 Pooled OOF Macro-F1

| 모델 | Ablation | 평균 | 표준편차 |
|---|---|---:|---:|
| LightGBM | A0 | 0.5013 | 0.0000 |
| LightGBM | A1 | 0.4496 | 0.0000 |
| LightGBM | A2 | 0.5006 | 0.0000 |
| LightGBM | A3 | 0.5122 | 0.0000 |
| LightGBM | A4 | 0.4896 | 0.0000 |
| LightGBM | A5 | 0.5033 | 0.0000 |
| LightGBM | A6 | 0.4030 | 0.0000 |
| RF | A0 | 0.5088 | 0.0049 |
| RF | A1 | 0.4579 | 0.0020 |
| RF | A2 | 0.5120 | 0.0010 |
| RF | A3 | **0.5191** | 0.0027 |
| RF | A4 | 0.5033 | 0.0047 |
| RF | A5 | 0.5029 | 0.0036 |
| RF | A6 | 0.4153 | 0.0011 |
| XGBoost | A0 | 0.5043 | 0.0017 |
| XGBoost | A1 | 0.4464 | 0.0013 |
| XGBoost | A2 | 0.4953 | 0.0028 |
| XGBoost | A3 | 0.4996 | 0.0007 |
| XGBoost | A4 | 0.4946 | 0.0005 |
| XGBoost | A5 | 0.4964 | 0.0015 |
| XGBoost | A6 | 0.4118 | 0.0017 |

### 4.2 Fold mean Macro-F1

| 모델 | Ablation | 평균 | 표준편차 |
|---|---|---:|---:|
| LightGBM | A0 | 0.4853 | 0.0000 |
| LightGBM | A1 | 0.4364 | 0.0000 |
| LightGBM | A2 | 0.4964 | 0.0000 |
| LightGBM | A3 | 0.5056 | 0.0000 |
| LightGBM | A4 | 0.4752 | 0.0000 |
| LightGBM | A5 | 0.4868 | 0.0000 |
| LightGBM | A6 | 0.3976 | 0.0000 |
| RF | A0 | 0.4940 | 0.0047 |
| RF | A1 | 0.4505 | 0.0023 |
| RF | A2 | 0.5085 | 0.0023 |
| RF | A3 | **0.5193** | 0.0026 |
| RF | A4 | 0.4861 | 0.0041 |
| RF | A5 | 0.4892 | 0.0035 |
| RF | A6 | 0.4140 | 0.0009 |
| XGBoost | A0 | 0.4917 | 0.0018 |
| XGBoost | A1 | 0.4353 | 0.0011 |
| XGBoost | A2 | 0.4905 | 0.0035 |
| XGBoost | A3 | 0.4955 | 0.0009 |
| XGBoost | A4 | 0.4817 | 0.0005 |
| XGBoost | A5 | 0.4840 | 0.0019 |
| XGBoost | A6 | 0.4076 | 0.0017 |

### 4.3 Pooled OOF Accuracy

| 모델 | Ablation | 평균 | 표준편차 |
|---|---|---:|---:|
| LightGBM | A0 | 0.5698 | 0.0000 |
| LightGBM | A1 | 0.5204 | 0.0000 |
| LightGBM | A2 | 0.5667 | 0.0000 |
| LightGBM | A3 | 0.5653 | 0.0000 |
| LightGBM | A4 | 0.5536 | 0.0000 |
| LightGBM | A5 | 0.5659 | 0.0000 |
| LightGBM | A6 | 0.4691 | 0.0000 |
| RF | A0 | 0.5846 | 0.0056 |
| RF | A1 | 0.5363 | 0.0025 |
| RF | A2 | **0.5855** | 0.0022 |
| RF | A3 | 0.5772 | 0.0020 |
| RF | A4 | 0.5753 | 0.0051 |
| RF | A5 | 0.5741 | 0.0051 |
| RF | A6 | 0.4855 | 0.0007 |
| XGBoost | A0 | 0.5799 | 0.0028 |
| XGBoost | A1 | 0.5177 | 0.0019 |
| XGBoost | A2 | 0.5620 | 0.0017 |
| XGBoost | A3 | 0.5449 | 0.0012 |
| XGBoost | A4 | 0.5613 | 0.0012 |
| XGBoost | A5 | 0.5652 | 0.0029 |
| XGBoost | A6 | 0.4724 | 0.0020 |

## 5. 주요 결과 및 해석

### 5.1 RF A3의 Macro-F1이 가장 높음

Pooled OOF Macro-F1 기준 RF A3가 0.5191 ± 0.0027로 전체 21개 모델-특징 조합 중 가장 높은 평균을 보였다. Fold mean Macro-F1도 0.5193 ± 0.0026으로 가장 높았다. A3는 팔꿈치 각도(`elbow_left_deg`, `elbow_right_deg`)와 손목-얼굴 거리(`wrist_face_sw`)를 제외한 조합이다. 이 특징들을 제외했을 때 성능이 향상된 것은 현재 데이터와 설정에서 해당 특징들이 추가 판별 정보를 충분히 제공하지 못하거나 다른 특징과 함께 사용될 때 잡음으로 작용했을 가능성을 시사한다. 다만 해당 특징들이 일반적으로 불필요하다고 단정할 수는 없다.

### 5.2 A1과 A6에서 공통적인 성능 저하

세 알고리즘 모두 A1에서 A0보다 Pooled OOF Macro-F1이 낮았다. A1은 `head_depth_m`, `chest_depth_m`을 제거한 조합이다. A6은 깊이 정보와 머리·목 관련 특징을 함께 제거했으며, 세 알고리즘 모두에서 가장 낮은 Macro-F1을 보였다. 이는 깊이 및 머리·목 관련 특징이 본 데이터에서 유용한 정보를 포함할 가능성을 보여준다. 단, 특징 제거 결과는 현재 실험 조건에서의 성능 차이를 보여주는 것이며 특징의 인과적 중요성을 증명하지 않는다.

### 5.3 지표에 따른 최적 조합 차이

- Pooled OOF Macro-F1 최고: RF A3, 0.5191.
- Pooled OOF Accuracy 최고: RF A2, 0.5855.
- LightGBM의 Macro-F1 최고: A3, 0.5122.
- XGBoost의 Macro-F1 최고: A0, 0.5043.

따라서 Macro-F1을 주요 지표로 선택하면 RF A3가 가장 유망하다. Accuracy만 기준으로 하면 최적 조합이 RF A2로 달라진다.

### 5.4 Seed 안정성 및 LightGBM 점검 사항

RF와 XGBoost는 여러 조합에서 Seed 간 표준편차가 작게 나타났다. RF A3의 Pooled OOF Macro-F1 표준편차는 0.0027이고, XGBoost A3는 0.0007이다. XGBoost A3의 변동성은 더 작지만 평균 Macro-F1은 RF A3보다 낮으므로 평균 성능과 안정성을 함께 고려해야 한다.

LightGBM은 모든 Ablation에서 Seed 간 표준편차가 0.0000으로 집계되었다. 이는 이번 실험에서 Seed별 결과가 동일하게 집계되었음을 의미한다. 현재 설정에서 난수 시드가 학습 결과에 영향을 주지 않았을 가능성이 있지만, 확정하기 전에 `runs` 파일에서 Seed별·참가자별 점수가 실제로 동일한지 확인하고 샘플링 관련 설정을 점검해야 한다. 표준편차가 0이라는 사실만으로 구현 오류나 보편적인 결정성을 단정할 수 없다.

## 6. 한계 및 주의사항

1. 참가자가 3명이므로 새로운 사용자 집단에 대한 일반화 성능을 확정하기 어렵다.
2. Seed 간 표준편차는 동일한 분할과 설정에서 난수 시드에 따른 변동을 요약하며, 참가자 간 변동성이나 모집단 수준의 불확실성을 나타내지 않는다.
3. Fold mean Macro-F1과 Pooled OOF Macro-F1은 계산 방식이 다르므로 서로 다른 요약값이다.
4. 작은 Seed 표준편차가 통계적 유의성이나 외부 데이터 일반화 성능을 입증하지는 않는다.
5. LightGBM의 Seed별 결과가 모두 동일하게 집계된 이유는 원시 결과와 모델 설정을 추가 점검해야 한다.
6. 결론은 현재 데이터, 특징, 전처리, LOPO 분할 및 모델 설정에 한정된다.

## 7. 결론 및 후속 작업

동일한 LOPO 분할에서 3개 알고리즘의 A0~A6 조합을 Seed 0~4로 반복 평가한 결과, RF A3가 Pooled OOF Macro-F1 0.5191 ± 0.0027로 가장 높은 평균 성능을 보였다. A1과 A6은 세 알고리즘 모두에서 성능 저하가 관찰되어 깊이 및 머리·목 관련 특징의 활용 가능성을 확인하였다. 현재 실험 범위에서는 RF A3를 후속 검증의 우선 후보로 선정할 수 있다. 다만 참가자 수가 3명이라는 한계가 있으므로 추가 참가자 또는 독립 테스트 데이터에서 재검증해야 한다. 또한 LightGBM의 Seed별 결과가 모두 동일한 이유를 원시 결과와 모델 설정으로 확인해야 한다.

## 8. 실험 산출물

- `experiments/seed_stability_a0_a6_runs.csv`: Seed별·Fold별 개별 실행 결과
- `experiments/seed_stability_a0_a6_seed_summary.csv`: Seed별 결과 요약
- `experiments/seed_stability_a0_a6_oof_summary.csv`: Pooled OOF 요약
- `experiments/seed_stability_a0_a6_summary.csv`: 평균 및 표준편차 종합 결과
- `experiments/seed_stability_a0_a6_report.md`: 자동 생성 보고서

본 분석 보고서는 위 실험 산출물의 요약 결과를 바탕으로 작성하였다.
