# Hybrid RF 5-Seed Stability Experiment

## 1. 실험 목적

Baseline RF, Lightweight RF, Tuned RF, Hybrid RF의 분류 성능과 모델 자원 효율성을 5개 시드로 비교하였다.

## 2. 실험 구성

- 비교 모델: Baseline, Lightweight, Tuned, Hybrid
- Random seed: 0~4
- 검증 방식: LOPO
- 평가 참가자: p01, p02, p03
- 전체 결과: 60행
- 평가 지표: Macro-F1, Accuracy, 모델 크기, 추론 시간, 학습 시간

## 3. 5개 시드 종합 결과

각 시드에서 참가자 3명의 결과를 평균한 후, 5개 시드의 평균과 표본 표준편차(ddof=1)를 계산하였다.

| Model | Macro-F1 (Mean ± SD) | Accuracy | Model Size (KB) | Median (ms) | P95 (ms) | Training (s) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.6565 ± 0.0092 | 0.6782 | 21326.40 | 13.642 | 18.693 | 1.951 |
| Lightweight | 0.6627 ± 0.0047 | 0.6791 | 2257.09 | 2.709 | 3.980 | 0.375 |
| Tuned | 0.6584 ± 0.0142 | 0.6730 | 2764.44 | 5.014 | 6.714 | 0.732 |
| Hybrid | 0.6650 ± 0.0093 | 0.6792 | 1052.24 | 2.005 | 3.045 | 0.255 |

## 4. 참가자별 Macro-F1

각 참가자를 테스트 대상으로 제외했을 때의 Macro-F1을 5개 시드에 걸쳐 평균하였다.

| Model | p01 | p02 | p03 |
|---|---:|---:|---:|
| Baseline | 0.7237 | 0.6539 | 0.5921 |
| Lightweight | 0.6971 | 0.6739 | 0.6169 |
| Tuned | 0.7139 | 0.6391 | 0.6222 |
| Hybrid | 0.7268 | 0.6402 | 0.6280 |

## 5. 결과 해석

- Hybrid RF 평균 Macro-F1: 0.6650
- Lightweight RF 평균 Macro-F1: 0.6627
- 두 모델의 Macro-F1 차이: +0.0024
- Hybrid RF의 Baseline 대비 모델 크기 감소율: 95.1%
- Hybrid RF의 Baseline 대비 추론 중앙값 감소율: 85.3%

Hybrid RF는 평균 Macro-F1과 자원 효율성 측면에서 경쟁력 있는 결과를 보였다. 다만 Lightweight RF와의 Macro-F1 차이가 작으므로 분류 성능에서 확실히 우수하다고 단정할 수 없다.

## 6. 결론 및 한계

5개 시드 실험을 통해 모델별 성능과 자원 효율성을 비교하였다. Hybrid RF는 경량화 측면에서 유망한 결과를 보였으나, 참가자가 3명이라는 한계가 있으므로 추가 참가자 검증이 필요하다.

※ 본 보고서의 수치는 통합 CSV의 실제 데이터로 계산하였다.
