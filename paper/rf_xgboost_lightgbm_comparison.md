# RF vs XGBoost vs LightGBM 비교 실험 보고서

## 1. 실험 목적
자세 분류 모델 Random Forest(RF), XGBoost, LightGBM을 같은 데이터 분할 및 특징 조건에서 비교했다.

## 2. 데이터 및 평가 조건
- 입력 파일: `side_logs.zip` 내 CSV 99개
- 분석 대상: 9개 자세 클래스, 44,436 프레임
- 참가자: p01 16,163프레임, p02 14,722프레임, p03 13,551프레임
- 제외: `left_lean_neck`, `right_lean_neck` 2개 클래스. 기존 프로젝트 실험 코드와 동일하게 좌우 기울임 클래스를 제외했다.
- 특징 세트: `no_hip` 9개 특징
  - `head_depth_m`, `chest_depth_m`, `neck_forward_offset_m`
  - `head_forward_deg`, `elbow_left_deg`, `elbow_right_deg`
  - `torso_len_sw`, `head_up_m`, `wrist_face_sw`
- 결측 처리: 학습 데이터 중앙값 대치 + 결측 여부 표시(SimpleImputer, add_indicator=True)
- 평가: 참가자 단위 Leave-One-Participant-Out(LOPO) 3-fold
  - 학습 p02+p03 → 평가 p01
  - 학습 p01+p03 → 평가 p02
  - 학습 p01+p02 → 평가 p03
- 각 모델은 300개 트리 기반 설정을 사용했고, seed=42를 고정했다.
- RF: 300 trees, `class_weight='balanced'`
- XGBoost: 300 estimators, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, balanced sample weights
- LightGBM: 300 estimators, num_leaves=31, learning_rate=0.05, `class_weight='balanced'`

## 3. 전체 결과 요약
| 모델 | Accuracy 평균 ± 표준편차 | Macro-F1 평균 ± 표준편차 | 학습 시간 평균(s) | 단일 샘플 추론 중앙값(ms) | 단일 샘플 추론 p95(ms) | 모델 크기 평균(KB) |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.6931 ± 0.0553 | **0.6788 ± 0.0613** | 2.47 | 1.03 | 1.36 | 3,529.8 |
| RF | 0.6747 ± 0.0469 | 0.6522 ± 0.0568 | 2.09 | 45.99 | 52.77 | 23,015.5 |
| LightGBM | 0.6702 ± 0.0546 | 0.6499 ± 0.0673 | 2.42 | 1.32 | 1.84 | 5,569.8 |

## 4. 참가자별 결과
| 모델 | 평가 참가자 | Accuracy | Macro-F1 | 학습 프레임 | 평가 프레임 |
|---|---|---:|---:|---:|---:|
| RF | p01 | 0.7256 | 0.7113 | 28,273 | 16,163 |
| XGBoost | p01 | 0.7485 | 0.7332 | 28,273 | 16,163 |
| LightGBM | p01 | 0.7133 | 0.6930 | 28,273 | 16,163 |
| RF | p02 | 0.6653 | 0.6475 | 29,714 | 14,722 |
| XGBoost | p02 | 0.6929 | 0.6910 | 29,714 | 14,722 |
| LightGBM | p02 | 0.6884 | 0.6843 | 29,714 | 14,722 |
| RF | p03 | 0.6332 | 0.5980 | 30,885 | 13,551 |
| XGBoost | p03 | 0.6378 | 0.6124 | 30,885 | 13,551 |
| LightGBM | p03 | 0.6088 | 0.5723 | 30,885 | 13,551 |

## 5. 자세별 Recall
아래 값은 세 참가자의 out-of-fold 예측을 합산해 계산한 Recall이다.

| 자세 | RF | XGBoost | LightGBM |
|---|---:|---:|---:|
| back_hard | 0.466 | 0.507 | 0.484 |
| back_light | 0.496 | 0.556 | 0.440 |
| back_neck | 0.984 | 0.983 | 0.982 |
| chin | 0.833 | 0.752 | 0.816 |
| forward | 0.597 | 0.653 | 0.579 |
| neck_hard | 0.800 | 0.791 | 0.730 |
| neck_light | 0.577 | 0.656 | 0.635 |
| normal | 0.851 | 0.879 | 0.884 |
| slouch | 0.511 | 0.513 | 0.531 |

## 6. 해석
1. 현재 설정에서 XGBoost가 평균 Macro-F1 0.6788로 가장 높았다. RF보다 약 0.0266, LightGBM보다 약 0.0289 높다.
2. XGBoost는 세 평가 참가자 모두에서 RF보다 Macro-F1이 높았다.
3. `back_neck`와 `normal`은 Recall이 비교적 높지만, `back_hard`, `back_light`, `slouch`는 세 모델 모두 Recall이 낮다. 유사 자세 간 특징 분포가 겹치거나 라벨 경계가 모호한지 추가 확인이 필요하다.
4. 저장된 모델 크기는 RF가 약 23 MB로 가장 컸고, XGBoost 약 3.5 MB, LightGBM 약 5.6 MB였다.
5. RF 추론 지연시간이 다른 두 모델보다 크게 측정됐다. 이 측정은 현재 실행 환경에서 병렬 추론 설정을 사용한 결과이므로, 배포 성능 결론을 내리기 전에 모든 모델을 단일 스레드로 맞춰 다시 측정하는 것이 좋다. 모델 파일 크기와 추론 시간은 저장 형식, 라이브러리 버전, CPU, 스레드 설정에 따라 달라질 수 있다.

## 7. 한계 및 다음 단계
- 참가자는 3명뿐이므로 결과는 예비 실험이다. 프레임 수가 많더라도 독립적인 참가자 수가 적어 새로운 사람에 대한 일반화 성능을 강하게 주장할 수 없다.
- 한 CSV 안의 인접 프레임은 시간적으로 유사할 수 있다. 이번 외부 평가는 참가자 단위로 분리했지만, 프레임별 점수를 독립 표본처럼 해석하지 않도록 주의해야 한다.
- 모델별 하이퍼파라미터를 완전히 최적화한 비교는 아니다. 현재는 합리적인 고정 설정의 1차 비교다.
- 다음 권장 실험:
  1. 단일 스레드 조건에서 추론시간 재측정
  2. 세 모델에 대해 학습 참가자 내부의 CSV 단위 GroupKFold를 사용한 하이퍼파라미터 탐색
  3. 혼동행렬을 보고 `back_hard`/`back_light`/`slouch` 오분류 원인 확인
  4. 참가자 추가 수집 후 동일 LOPO 프로토콜 재평가
