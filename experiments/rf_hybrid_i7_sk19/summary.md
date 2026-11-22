# Hybrid RF 재실행: 4.8절 노트북(i7-14700HX)에서 추론 시간 측정 (11/22)

목적: 선우의 Hybrid RF 5-seed 실험(experiments/rf_hybrid_seed_stability.md, i5 노트북·WSL Ubuntu 24.04)의 추론 시간을
논문 4.8절과 같은 노트북·조건에서 다시 잰다. 스크립트·데이터·옵션은 같다(scripts/rf_hybrid.py --n-iter 20, seed 0~4).

## 실행 조건
- 노트북: Intel Core i7-14700HX, Windows 11, 프로젝트 .venv(Python 3.14) 네이티브 실행
- 성능 모드, 충전기 연결, 다른 프로그램 종료(Claude 앱 포함)
- scikit-learn을 1.9.1로 잠시 올려 실행하고 끝난 뒤 1.8.0으로 되돌림(선우 결과와 같은 버전 계열로 맞추기 위함)
- 결과: experiments/rf_hybrid_i7_sk19/seed0~4.{md,csv}

## 결과 (3 fold × 5 seed = 15개 평균)

| 모델 | Macro-F1 | 선우 Macro-F1 | 크기(KB, joblib 비압축) | 추론 중앙값(ms) | 추론 p95(ms) | 선우 추론 중앙값(ms) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.6565 | 0.6565 | 21,326 | 8.47 | 8.92 | 13.64 |
| Lightweight | 0.6626 | 0.6627 | 2,257 | 1.53 | 1.63 | 2.71 |
| Tuned | 0.6584 | 0.6584 | 2,764 | 3.31 | 4.07 | 5.01 |
| Hybrid | 0.6650 | 0.6650 | 1,052 | 1.31 | 1.56 | 2.00 |

- Macro-F1과 모델 크기는 선우 결과와 같다(Lightweight 1개 fold만 5e-5 차이).
- 추론 시간: 프레임 1개 predict() 1회, 단일 스레드, 평가 참가자 프레임 300개 무작위, 워밍업 5프레임 제외, fold별 중앙값의 평균.

## 참고: 첫 실행(experiments/rf_hybrid_i7)
- scikit-learn 1.8.0, 다른 프로그램을 모두 닫지 않은 상태. Macro-F1이 선우와 조금 다르고(예: Baseline 0.6599),
  추론 시간도 길었다(Baseline 23.2 ms, Hybrid 3.1 ms). 같은 데이터로 제 쪽 환경(1.9.1)에서는 선우 결과가 그대로 재현되어,
  차이는 scikit-learn 버전 때문으로 확인. 논문에는 쓰지 않는다.
