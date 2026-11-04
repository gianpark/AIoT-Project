# 7-Class Algorithm Comparison

## 1. 실험 목적

동일한 7개 자세 분류 데이터와 LOPO(Leave-One-Participant-Out) 검증 조건에서 Random Forest, XGBoost, LightGBM의 성능을 비교한다.

비교 지표는 Macro-F1, Accuracy, 데이터 유지율이다.

## 2. 알고리즘별 성능 비교

| Feature Set | Mode | Random Forest | XGBoost | LightGBM |
|---|---|---:|---:|---:|
| depth | A | 0.451 | 0.441 | 0.443 |
| depth | B | 0.477 | 0.492 | 0.483 |
| no_hip | A | 0.493 | 0.493 | 0.494 |
| no_hip | B | **0.515** | 0.505 | 0.488 |
| all | A | **0.515** | 0.488 | 0.508 |
| all | B | 0.459 | 0.471 | 0.483 |
| relative | A | 0.450 | 0.435 | 0.450 |
| relative | B | 0.466 | 0.453 | 0.474 |

※ A: 학습 데이터의 중앙값으로 결측값을 대체하고 결측 여부를 나타내는 특성을 추가한다. 데이터 유지율은 100%이다.  
※ B: 선택한 특성 중 결측값이 있는 행을 제거한다. 따라서 데이터 유지율이 달라진다.

## 3. 주요 결과

- Random Forest: all + A 조건에서 Macro-F1 0.515
- XGBoost: no_hip + B 조건에서 Macro-F1 0.505
- LightGBM: all + A 조건에서 Macro-F1 0.508

A 조건에서는 Random Forest가 가장 높은 Macro-F1을 기록했다.

B 조건은 결측 행을 제거하므로 A 조건과 평가 데이터 범위가 다르다. 따라서 점수만으로 우열을 판단하지 않고 데이터 유지율도 함께 고려해야 한다.

## 4. 중간 분석

세 알고리즘 모두 좌우 기울임 자세(left_lean, right_lean)의 분류 성능이 낮게 나타나는 문제가 확인되었다.

이는 특정 알고리즘만의 문제가 아닐 가능성이 있으므로 자세별 데이터 특성, 사람 간 차이, 특징값의 분포 등을 추가로 분석할 필요가 있다.

## 5. 다음 분석 항목

1. 알고리즘별 클래스 Precision, Recall, F1 비교
2. 알고리즘별 혼동행렬 비교
3. 참가자별 LOPO 성능 비교
4. 모델 크기 및 추론 시간 측정
5. 최종 알고리즘 선정