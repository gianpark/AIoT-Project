# 자세 추정 알고리즘 비교 조사 — 3주차 발표용

## 왜 이 조사가 필요한가

듀얼캠 기반 책상 자세 모니터링이라는 주제가 이번 학기 조별 프로젝트들의 공통 주제가 되면서, 하드웨어 구성만으로는 차별화가 어려워졌다. 그래서 차별화 지점을 **자세 추정 알고리즘 자체**로 옮기기로 했다. 구체적으로는 (1) 어떤 포즈 추정 모델을 쓸지, (2) 그 위에서 어떤 분류기(랜덤포레스트 등)를 쓸지를 "실행 사양 대비 성능"이라는 기준으로 논문 근거를 갖고 정당화하는 것이 이번 조사의 목표다. 바른자세는 이미 MoveNet Lightning(포즈 추정) + 각도 특징 + 랜덤포레스트(분류) 조합을 잠정 채택한 상태이므로, 이 조합이 "저사양 노트북에서 백그라운드로 상시 구동"이라는 제약 조건에 얼마나 잘 맞는지를 최근 논문들로 뒷받침하는 방향으로 정리했다.

---

## 0. 테스트 환경 (3주차 실측 기준)

3주차 게이트(포즈 추정 성능 실측)에 사용할 노트북 사양이 확정됐다 — **기안 소유 노트북**: Intel Core i7-14700HX(2.10GHz), RAM 32GB, NVIDIA GeForce RTX 5060 Laptop GPU(8GB VRAM, CUDA 지원), 내장그래픽 Intel UHD Graphics.

⚠️ **주의할 점**: 이 사양은 전용 GPU가 탑재된 게이밍 노트북급 고사양이라, 이 문서 전반에서 "저사양 노트북 대응"이라는 표현으로 정당화해온 설계 목표와는 결이 다르다. 원래 "저사양"은 팀이 상정한 배포 대상(별도 GPU 없는 보급형 노트북에서도 매끄럽게 돌아가야 한다는 제약)을 뜻했던 것이므로, 발표에서는 아래 두 가지를 구분해서 말하는 게 안전하다.

- **설계 목표(유지)**: 전용 GPU 없이(CPU만으로도) 5초 주기로 실시간 구동 가능해야 한다는 제약은 그대로 유효.
- **실측 환경(신규 확정)**: 이번 3주차 실측은 GPU 탑재 고사양 노트북(i7-14700HX + RTX5060)에서 진행됨 → 여기서 나온 FPS는 "GPU 활용 시 상한선"에 가깝고, 그 자체로 "저사양에서도 이만큼 나온다"는 근거로 쓰면 안 됨. 가능하면 **CPU 전용 실행 모드로도 한 번 더 측정**해서 두 수치(CPU-only / GPU 활용)를 함께 제시하면 "저사양에서도 충분하다"는 주장이 설득력을 갖는다. RTX 5060이 있으므로 CUDA 가속 옵션도 열려 있다.

---

## 1. 포즈 추정 모델 비교 (정확도 vs 속도)

| 논문/자료 | 비교 대상 | 핵심 수치 | 시사점 |
|---|---|---|---|
| [Comparative Analysis of OpenPose, PoseNet, and MoveNet Models for Pose Estimation in Mobile Devices](https://www.iieta.org/journals/ts/paper/10.18280/ts.390111) (IIETA, 2022) | OpenPose vs PoseNet vs MoveNet Lightning/Thunder, COCO·MPII 1,000장 기준 | 정확도: PoseNet 97.6% > OpenPose 86.2% > MoveNet Thunder 80.6% > MoveNet Lightning 75.1%. 속도(1,000장 처리시간): MoveNet Lightning 53.5초(최속) < PoseNet 75.2초 < MoveNet Thunder 140.0초 < OpenPose 643.5초(MoveNet 대비 12배 느림) | **정확도 최고는 PoseNet이지만, 속도는 MoveNet Lightning이 압도적**. 저사양 노트북에서 5초 주기로 상시 구동해야 하는 바른자세 조건에서는 속도 우위가 정확도 손실(75%대)보다 중요하다는 근거로 인용 가능 |
| [Comparative Evaluation of MediaPipe and YOLOv8 for Real-Time Pose Estimation](https://link.springer.com/chapter/10.1007/978-3-032-00232-7_13) (Springer, 2026) | MediaPipe(lite/full/heavy) vs YOLOv8-pose(n/s/m/l/x), 보행·격투·군중 시나리오 | ⚠️ **원문 미확인**: 웹 요약본 기준 "MediaPipe lite가 FPS·지연시간에서 실시간 애플리케이션에 가장 적합, YOLOv8은 군중·가림 상황에서 더 강함"이라는 내용만 확보. 직접 원문 접근을 두 번 시도했으나 접속 제한(rate limit)으로 실패 | 바른자세는 **책상 앞 1인 고정 시나리오**라 군중·가림 대응력이 중요하지 않음 → 경량 모델이 유리하다는 논리로 쓸 순 있지만, 수치 자체가 미검증이므로 정량적 근거로는 아직 쓰면 안 됨. **원문 PDF 필요** |

**요약**: 포즈 추정 단계에서는 정확도 1등 모델(PoseNet, YOLOv8-x급)이 아니라, 속도·경량성이 확보된 모델(MoveNet Lightning, MediaPipe lite)을 선택하는 것이 바른자세처럼 "다른 작업과 백그라운드로 동시 구동"해야 하는 제약 조건에서는 더 합리적인 선택이라는 근거가 확보됐다.

---

## 2. 분류기 비교 (랜덤포레스트가 합리적인 선택인 이유)

| 논문 | 비교 분류기 | 핵심 결과 | 시사점 |
|---|---|---|---|
| [Toward Real-Time Posture Classification: Reality Check](https://www.mdpi.com/2079-9292/14/9/1876) (Electronics/MDPI, 2025, Zhang, Gračanin, Zhou, Dudash, Rushton) | SVM(linear), Gaussian NB, **Random Forest**(트리 10개, min_samples_split=2), Neural Network(3층, hidden 9), AdaBoost(estimator 50, lr=1) / LSTM(5층, hidden_dim=256) | **원문 PDF 직접 확인 완료.** 목적: 낙상 방지용 실시간 자세 분류(앉기-서기/다리올리기/구부리기/돌기/점프 5동작, 자세가 아니라 "동작" 분류에 가까움). Kinect V2, 참가자 5명(18~30세)·10,000개 샘플, 60/40 split(5-fold CV와 유사한 결과 확인). **정확도**: SVM·Gaussian NB·RF·NN·LSTM 모두 특정 시나리오에서 99% 도달(Table 4), AdaBoost만 일관되게 최저(특히 다리올리기 동작). 노이즈 30%에서도 대부분 80%대 유지, SVM·LSTM만 "가장 노이즈에 강함"으로 표 요약(본문에는 "RF·SVM이 노이즈에 가장 강함"이라고도 적혀 있어 본문-표 간 약간의 불일치 있음). **속도(RTX 4070 12GB + i5-13400F 2.5GHz)**: Gaussian NB가 최속(테스트 데이터 40% 추론에 0.0024초). CPU 단독 비교에서 LSTM이 최저속(17.6초/40%). "GPU 없이(CPU만) LSTM은 가장 느린 고전 분류기 대비 약 20배 느림"이 **본문에 직접 명시된 문장**(17.6초 ÷ 0.77초(AdaBoost, CPU) ≈ 22.9배로 계산이 맞음) — **20배라는 숫자, 원문으로 확정됨.** ⚠️ 다만 **이 속도 비교(Figure 10, Table 4)에 Random Forest는 포함되지 않음** — 속도 그래프의 5개 막대는 SVM/AdaBoost/NN/Gaussian NB/LSTM뿐이고 RF는 빠짐. 그래서 "RF가 LSTM보다 20배 빠르다"라고 딱 집어 말할 순 없고, "고전 ML 계열(가장 느린 AdaBoost 기준)이 LSTM보다 CPU에서 약 20배 빠르다"는 표현이 정확함. 학습 데이터량은 Methods 섹션에 인용 기반 참고치로 SVM 300개·Gaussian NB 30개·RF 500개·AdaBoost 400개·**LSTM 8000개** 필요(선행연구 인용, 이 논문 자체 실험 결과 아님) — "1000배"라는 표현은 서론의 딥러닝 일반론이고 RF 대비 LSTM 배수는 8000/500=16배가 이 논문 기준 수치 | **가장 강력한 핵심 근거, 이제 확정 인용 가능**: "고전 ML로도 최대 99% 정확도 가능 + CPU에서 가장 느린 고전 분류기조차 LSTM보다 약 20배 빠름"이 우리가 RF를 선택한 이유를 뒷받침. 다만 발표에서는 "RF가 20배 빠르다"가 아니라 "고전 ML 계열이 20배 빠르다(RF는 이 논문 속도 비교엔 없었지만 같은 계열)"로 정확하게 인용할 것 |
| [Comparing the Performance of Different Classifiers for Posture Detection](https://www.academia.edu/129113537/Comparing_the_Performance_of_Different_Classifiers_for_Posture_Detection) (EAI, 2021) | SVM(RBF), Naive Bayes, Logistic Regression, kNN, **Random Forest** / MLP, 1D-CNN, 2D-CNN, LSTM, BiLSTM (10-fold 교차검증) | **원문 확인 완료**. 고전 ML: SVM(RBF) 83.42% > Logistic Regression 80.91% > **Random Forest 77.08%** > kNN 76.98% > Naive Bayes 74.56%. 딥러닝: 1D-CNN 93.45%(전체 최고) > 2D-CNN 91.59% > BiLSTM 90.87% > LSTM 88.19% > MLP 82.56% | 특징이 관절 각도라는 점이 바른자세의 "각도 특징 추출" 설계와 동일. 다만 **이 논문에서는 RF 단독 정확도가 77.08%로 80% 미만** — "고전 ML이 80%대"라는 주장의 근거로는 SVM/LogReg을 들어야지 RF 자체를 들면 안 됨. RF가 여기선 중위권이라는 점은 발표에서 과장하지 않도록 주의 |
| [Classification Algorithm for Sitting Postures Using Weighted Random Forest](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ipr2.70126) (IET Image Processing, 2025) | 가중 랜덤포레스트 단독 | 착석 자세 분류에 랜덤포레스트를 직접 적용한 2025년 최신 논문 (원문 접근 제한으로 세부 수치는 미확인) | 바른자세가 채택하려는 알고리즘과 대상(착석 자세)이 가장 정확히 일치하는 논문 — 발표에서 "우리와 가장 가까운 선행 연구"로 직접 인용 |
| [A Proposal of Implementation of Sitting Posture Monitoring System for Wheelchair Utilizing Machine Learning Methods](https://doi.org/10.3390/s21196349) (Sensors/MDPI, 2021, Ahmad, Sidén, Andersson) | k-NN, SVM, **Random Forest**, Decision Tree, **LightGBM** — Desktop(i7-4790K) / Raspberry Pi 3B / Raspberry Pi 4B 3개 하드웨어에서 각각 학습·평가 | **원문 PDF 이중 교차 확인 완료(Table 4).** 압력센서 16개, 32명, 4,096데이터포인트, 좌/우/전/후 기울임 4자세. 정확도: Desktop 기준 LightGBM 99.03% > Decision Tree 98.85% > **RF 98.65%** > kNN 98.07% > SVM 95.96%. 임베디드(RPi 3B/4B)에서도 LightGBM이 99.03%로 단독 1위 유지, RF는 98.07~98.46%로 2위권. 연산시간(Desktop): Decision Tree 0.049초 < SVM 0.051초 < kNN 0.054초 < LightGBM 0.131초 < **RF 0.167초(최장)**. 저사양 RPi 3B에서는 격차가 더 벌어져 **LightGBM 1.576초 vs RF 2.181초로 LightGBM이 더 빠름** | ⚠️ **RF에 불리한 근거**: 이 논문 기준으로는 정확도·속도 모두 LightGBM이 RF를 근소하게 앞서고, 특히 저사양(RPi) 환경일수록 격차가 커짐. "RF가 가장 빠르고 정확하다"고 단정하면 이 논문과 모순되므로, 발표에서는 LightGBM도 정직하게 비교 후보로 언급하고 Feature Importance(해석 가능성)·구현 단순성 등 정확도·속도 외의 선택 근거를 함께 제시하는 게 안전함 |

**요약**: 랜덤포레스트가 "딥러닝 대비 CPU에서 압도적으로 빠르다"는 방향성은 이제 Reality Check 논문 원문으로 확정됐다 — CPU 단독 비교에서 고전 ML 계열(가장 느린 AdaBoost 기준)이 LSTM보다 약 20배 빠르고, 정확도는 SVM·Gaussian NB·RF·NN·LSTM 모두 99%까지 도달 가능. 다만 이 논문의 속도 비교 자체에 RF가 포함되지 않았다는 점은 발표에서 정확히 짚어야 함("RF가 20배"가 아니라 "고전 ML 계열이 20배"). 정확도 면에서는 논문마다 결과가 갈린다 — Weighted RF 논문에서는 RF가 최상위권(WRF 98%)이지만, EAI 논문에서는 RF 단독이 77.08%로 중위권이고, 휠체어 압력센서 논문(Sensors 2021)에서는 **정확도·속도 둘 다 LightGBM이 RF보다 근소 우위**(특히 임베디드 보드일수록 격차 확대)라, "RF가 항상 최고"라고 과장하면 안 된다. **"딥러닝만큼의 정확도를 훨씬 적은 연산으로 낼 수 있다"는 속도·효율 중심 논리**로 발표하되, 동급 후보인 LightGBM·XGBoost 대비로는 "속도·정확도 1위"가 아니라 "충분히 실용적 + 해석 가능성(Feature Importance)·구현 단순성"이라는 논리로 방어하는 게 가장 정확하다.

---

## 3. 종합 포지셔닝 (발표 스토리라인 제안)

1. **문제 제기**: 이번 학기 다수 팀이 유사한 듀얼캠·책상 자세 모니터링 주제를 채택 → 하드웨어만으로는 차별화 어려움.
2. **차별화 지점**: "어떤 포즈 추정 모델 + 어떤 분류기를 쓰느냐"가 실제 사용성(백그라운드 상시 구동 가능 여부)을 가른다는 점을 논문으로 입증.
3. **근거 1 (포즈 추정)**: MoveNet Lightning은 정확도는 최고는 아니지만(75%대), OpenPose 대비 12배 빠르고 저사양 환경에 적합 (IIETA 2022, 원문 검증 완료). 우리 시나리오(1인 고정, 군중·가림 없음)에서는 YOLOv8 같은 무거운 모델의 강점(occlusion 대응)이 불필요할 것으로 보이나, 이 부분(Springer 2026)은 아직 원문 미확인.
4. **근거 2 (분류기)**: 고전 ML(SVM/Gaussian NB/RF/NN)이 CPU에서 LSTM보다 약 20배 빠르면서도 최대 99% 정확도를 낼 수 있다는 것이 2025년 논문(Reality Check, 원문 검증 완료)으로 확정됨. 단, 이 논문의 속도 비교엔 RF가 직접 포함되지 않았으므로 "고전 ML 계열이 20배 빠르다"로 정확히 인용(=RF가 20배라고 단정하지 않기). 관절 각도 기반 특징으로도 실용적인 정확도가 가능하다는 것은 EAI 2021 논문(원문 검증 완료)으로 확인되나, 그 논문에서 RF 단독 정확도는 77.08%(중위권)이므로 "RF가 최고"라는 식으로 과장하지 않는다.
5. **근거 3 (RF vs XGBoost/LightGBM 직접 비교, 팀원 추가 조사)**: 팀원이 조사한 Weighted RF 논문 내 8개 분류기 비교(WRF 98% > XGBoost 96% > ... > LightGBM 89%)에서는 RF 계열이 우위지만, 휠체어 압력센서 논문(Sensors 2021, 원문 이중 검증 완료)에서는 반대로 **LightGBM이 정확도·연산속도 모두 RF를 근소하게 앞섬**(특히 저사양 임베디드 보드일수록 격차 확대). 두 논문의 결과가 엇갈리므로 "RF가 XGBoost·LightGBM보다 항상 우월하다"고 주장하지 않고, RF·XGBoost·LightGBM 모두 실용적 후보군이며 **RF는 Feature Importance 기반 해석 가능성과 구현 단순성이 강점**이라는 논리로 방어하는 것이 정확함.
6. **결론**: 바른자세의 MoveNet Lightning + 각도 특징 + 랜덤포레스트 조합은 "정확도 극대화"가 아니라 "저사양 노트북에서 다른 작업과 동시에 상시 구동 가능한 실용적 실시간 시스템"을 목표로 한 의도적 선택이며, 이것이 다른 팀들과의 차별점이다. RF가 XGBoost·LightGBM 대비 절대적으로 가장 빠르거나 정확하다는 주장은 하지 않고, "충분히 실용적인 성능 + 해석 가능성"이라는 균형 잡힌 논리를 유지한다.
7. **다음 단계 연결**: project.md에 이미 잡혀있는 3주차 게이트(카메라 캘리브레이션 + 노트북 CPU 기준 포즈 추정 성능 실측)에서, 이번 조사로 얻은 문헌상의 속도·정확도 수치를 우리 실제 환경에서 검증하는 것으로 자연스럽게 이어감.

---

## 4-1. 추가로 찾은 논문 (팀원 문서의 후보 분류기별 실제 사례)

팀원이 정리한 후보 알고리즘(XGBoost, LightGBM, GCN/ST-GCN 등)이 실제 자세 분류에 쓰인 사례를 찾아봤다. 이후 사용자가 원문 PDF를 직접 구해 업로드해줘서, 아래 XGBoost 논문과 MoveNet 논문은 **원문 기준으로 수치를 재검증**했다(이전엔 웹 요약본 기준이었음).

| 논문 | 방법 | 핵심 결과 | 시사점 |
|---|---|---|---|
| [Automated ergonomic sitting postures detection for office workstation using XGBoost method](https://doi.org/10.11591/ijai.v15.i1.pp506-514) (Pawitra et al., *IAES Int. J. Artificial Intelligence*, Vol.15 No.1, 2026.02, pp.506-514) | **MoveNet Thunder**로 17개 키포인트 추출 → 머리~엉덩이(상반신)만 남기고 정규화·중심이동 → AdaBoost/XGBoost/MLP 비교, SNI 9011:2021 기준 정상/비정상 자세 2분류. 참가자 30명, 정면+측면 촬영, 유효 인스턴스 1,550개(정상 873/비정상 677) | 10-fold 교차검증 결과 — AdaBoost: Acc 0.894±0.021; MLP: Acc 0.918±0.221(**폴드 간 편차가 매우 큼 — 불안정**); **XGBoost: Acc 0.930±0.019, Precision 0.946, Recall 0.914, F1 0.929, ROC-AUC 0.974±0.010(최고·가장 안정적)**. 논문 자체 비교표(Table 2)에서 Deep recurrent hierarchical network 91.47%, MediaPipe+decision tree 91.5%/97.05%, **OpenPose+random forest 90.64%**(다른 과업인 인간 행동 인식이지만 RF 계열 참고치)보다 우수하다고 주장 | **우리와 같은 MoveNet 계열(Thunder)을 실제 자세 분류에 쓴 2026년 2월 최신 논문, 원문 확보 완료.** 랜덤포레스트는 비교군에 없음(AdaBoost/XGBoost/MLP만 비교) — 우리가 여기에 RF/WRF를 추가로 비교하면 이 논문의 갭을 메우는 것. 저자들도 한계로 "상반신 키포인트만 사용, 시점·가림·조명에 민감, 임베디드 자원 소모 미평가"를 명시 — 우리 설계(전신 17키포인트+스테레오 깊이)가 이 한계를 어떻게 보완할 수 있는지 가설로 제시 가능 |
| [Posture Detection Using MoveNet](https://ijirt.org/publishedpaper/IJIRT171529_PAPER.pdf) (V. Dhana Sri Hima et al., *IJIRT* Vol.11 No.8, 2025.01) | 제목과 서론은 **MoveNet Lightning**(우리와 동일 variant)을 내세우지만, 본문 의사코드·구현 설명은 실제로는 **MediaPipe**를 부른다고 적혀 있어 내부적으로 용어가 섞여 있음. 추출된 키포인트를 CSV로 저장 → 신경망으로 분류 | **자체 정확도는 전혀 보고하지 않음** — 다른 연구의 요가자세 인식 정확도(99.5%, 99.88% 등)만 인용하고 자기 시스템 성능은 수치 없이 "실시간 가능"이라고만 서술. 문장 품질도 낮음("the use of Tensorflow MoveNet... to come across the important thing points" 등 어색한 표현 다수) | **인용 근거로는 약한 논문** — IJIRT는 임팩트 팩터가 없는 인도 소재 저널로 보이고, 자체 실험 결과가 없어 신뢰도가 낮음. "MoveNet Lightning + 다운스트림 분류기" 파이프라인 구조 자체를 언급하는 용도로만 가볍게 인용하고, 정량적 근거로는 쓰지 않는 걸 권장 |
| [An attention-based adaptive spatial–temporal graph convolutional network for long-video ergonomic risk assessment](https://www.sciencedirect.com/science/article/pii/S0952197623019644) (AAST-GCN, 2024) | 학습 가능한 인접행렬 + 시공간 어텐션 + TCN 결합한 고급 ST-GCN, 장시간 영상의 인체공학 리스크 평가 | GCN 계열 방법들보다 우수(정확한 수치는 본문 접근 제한으로 미확인), TCN이 RNN보다 연산비용 낮다고 언급 | 팀원 문서의 "GCN/ST-GCN 고급 연구 후보"가 실제로 어디까지 발전했는지 보여주는 참고 사례. 다만 구현 난도가 높아 한 학기 스코프에는 부담 — "고려했지만 실현 가능성 때문에 RF 계열을 택했다"는 근거로 활용 |
| [A Proposal of Implementation of Sitting Posture Monitoring System for Wheelchair Utilizing Machine Learning Methods](https://doi.org/10.3390/s21196349) (Ahmad, Sidén, Andersson, *Sensors*, Vol.21 No.19, Article 6349, 2021) | 압력센서 16개, k-NN/SVM/**RF**/Decision Tree/**LightGBM**을 Desktop·Raspberry Pi 3B·4B 3개 하드웨어에서 각각 비교 | **원문 PDF 이중 교차 확인 완료.** Desktop 기준 정확도 LightGBM 99.03% > Decision Tree 98.85% > RF 98.65% > kNN 98.07% > SVM 95.96%. 연산시간은 Decision Tree(0.049초)가 가장 빠르고 **RF(0.167초)가 가장 느림**. 저사양 Raspberry Pi 3B에서는 **LightGBM(1.576초)이 RF(2.181초)보다 더 빠름** | 팀원이 새로 조사한 자료(2번째 docx, 8~14절)에서 발견한 논문. 팀원 문서의 XGBoost/LightGBM 후보 비교 시나리오와 직접 연결됨. ⚠️ 이 논문 기준으로는 **LightGBM이 RF보다 정확도·속도 모두 근소 우위** — RF를 "가장 빠르다"고 과장하지 않는 근거로 활용, "실용적 성능 + 해석 가능성"으로 방어 논리 구성 |

### 참고: OpenPose/PoseNet/MoveNet 원본 비교 논문도 원문 확보

[Comparative Analysis of OpenPose, PoseNet, and MoveNet Models for Pose Estimation in Mobile Devices](https://doi.org/10.18280/ts.390111) (Jo & Kim, *Traitement du Signal*, Vol.39 No.1, 2022, pp.119-124) — 이 문서 1절 표에서 이미 인용 중인 논문의 원문을 확보해 수치를 재확인했고, 기존에 정리한 수치(정확도 OpenPose 86.2%/PoseNet 97.6%/MoveNet Lightning 75.1%/MoveNet Thunder 80.6%, 속도 MoveNet Lightning이 OpenPose 대비 약 12배 빠름)가 전부 정확히 일치함을 확인했다. 추가로 확인된 세부사항: 키포인트 수는 OpenPose 137개(얼굴·손 포함) vs PoseNet/MoveNet 17개(동일)이고, MoveNet만 유일하게 "사람이 없는 이미지"에서도 안정적으로 낮은 오탐률을 보였다(3번째 이미지 그룹 기준 MoveNet Lightning 71.5%로 오히려 하락 — 즉 사람이 없을 때 오탐 발생 여지가 있다는 뜻이라 우리도 참고할 부분).

### 4-1-1. 팀원 정리: RF·XGBoost·LightGBM 세 선행연구 비교와 우리 프로젝트 차별화

팀원이 추가로 정리한 자료(2번째 docx, 8~14절)는 RF·XGBoost·LightGBM 각각을 실제로 자세 분류에 적용한 선행연구 3편(위 표의 Weighted RF, XGBoost, 휠체어 압력센서 논문)을 나란히 놓고, 우리 프로젝트와 항목별로 어디가 다른지 정리한 내용이다. 3편 모두 이제 원문 기준으로 검증됐으므로 아래 비교는 그대로 신뢰할 수 있다.

| 구분 | RF 선행연구(Weighted RF, IET 2025) | XGBoost 선행연구(Pawitra, IAES 2026) | LightGBM 선행연구(휠체어, Sensors 2021) | 바른자세 |
|---|---|---|---|---|
| 입력 데이터 | 얼굴·어깨 중심 좌표/각도(MediaPipe, 2D) | MoveNet Thunder 17 Keypoint(상반신만) | 압력센서 16개(카메라 아님) | MoveNet Lightning 17 Keypoint 전신 + 스테레오 깊이 |
| 분류 대상 | 5가지 앉은 자세 | 정상/비정상 2분류 | 좌우전후 기울임 4가지 | 정상/주의/경고 3단계 |
| 깊이 정보 | 미사용 | 미사용(2D 중심) | 미사용(압력 기반) | **스테레오 카메라로 깊이 정보 활용** — 세 선행연구 모두 없는 요소 |
| 평가 지표 | Accuracy/Precision/Recall/F1 | Accuracy/Precision/Recall/F1/ROC-AUC | Accuracy + 연산시간(하드웨어별) | Accuracy/Macro F1/추론시간/CPU·RAM까지 종합 평가 예정 |

**우리 프로젝트의 차별화 포인트**(팀원 자료 원안 유지, 검증된 근거로 재확인): (1) 세 선행연구 모두 깊이 정보를 쓰지 않는데 바른자세는 스테레오 카메라로 깊이 정보를 추가해 2D 한계(옷·배경색 등)를 보완할 수 있다는 가설을 세울 수 있음(단, 실측 전 가설임을 명시). (2) 세 선행연구는 각자 하나의 알고리즘만 밀지만, 바른자세는 RF를 기준 모델로 두고 XGBoost·LightGBM과 동일 데이터·동일 특징으로 공정 비교한 뒤 개선 모델(Feature Selection/Weighting)을 설계하는 구조 — 단, 위에서 확인했듯 **RF가 XGBoost·LightGBM보다 항상 우월하다고 예단하지 않고 실측으로 검증**하는 것이 정직한 태도.

---

## 3-1. MLP(딥러닝) 비교 후보 추가 — 팀원 조사, 원문 검증 완료

지금까지는 RF·XGBoost·LightGBM 등 트리 기반 머신러닝만 비교했는데, 팀원이 "머신러닝만 비교했으니 딥러닝도 넣어보자"는 취지로 MLP(Multilayer Perceptron) 관련 선행논문 4편을 추가로 조사했다. 팀원 문서 자체에는 구체적 수치가 없어서, 4편 모두 원문을 직접 찾아 수치까지 검증했다(KeypointNet은 2회 교차 확인).

| 논문 | 방법 | 핵심 결과(원문에서 새로 확인한 수치) | 시사점 |
|---|---|---|---|
| [Yoga Pose Estimation and Feedback Generation Using Deep Learning](https://onlinelibrary.wiley.com/doi/10.1155/2022/4311350) (Thoutam et al., *Computational Intelligence and Neuroscience*, 2022) | 관절 각도 12개(Keypoint 사이 각도)를 MLP 입력으로 사용. 구조: 입력 12 → 은닉 10 → 은닉 8 → 출력 6(요가 자세 6종) | **원문 확인 완료.** 총 350개 인스턴스(학습 320/검증 30/테스트 30), 6개 요가자세(Cobra, Lotus, Corpse, Mountain, Triangle, Tree) 분류. **테스트 정확도 0.9958** | 우리 프로젝트와 가장 구조가 비슷한 선행연구 — "Keypoint 각도 특징 → MLP" 파이프라인이 실제로 매우 높은 정확도를 낸 직접적 근거. 단, 데이터가 350개로 적어 과적합 가능성 있음(저자가 직접 언급하진 않았으나 우리가 주의할 점) |
| [KeypointNet: An Efficient Deep Learning Model with Multi-View Recognition Capability for Sitting Posture Recognition](https://www.mdpi.com/2079-9292/14/4/718) (Cao et al., *Electronics*, 2025, 14(4), 718) | 지역 특징 추출(6개 신체부위 Keypoint, 1D conv+MLP) + 전역 특징 추출(Keypoint 간 관계) 결합 후 MLP로 최종 분류 | **원문 2회 교차 확인 완료.** 정확도: 자체 제안 MVSP 데이터셋 98.46%, HASP 96.52%, MSRA 99.48% — "기존 SOTA 모델들보다 우수하면서 경량"이라고 주장 | 앉은 자세 인식에 MLP를 실제로 쓴 2025년 최신 사례. 다만 이 논문은 MLP를 "네트워크 내부 구성요소"로만 쓰고(KeypointNet이라는 전체 아키텍처가 핵심), 우리처럼 MLP 자체를 독립 분류기로 비교하는 구조는 아님 — 참고만 할 것 |
| [Human posture recognition model integrating YOLOv8 pose and graph multilayer perceptron](https://link.springer.com/article/10.1007/s44163-026-01933-6) (Tian & Liu, *Discover Artificial Intelligence*, 2026, 6, 1043) | YOLOv8-Pose로 Keypoint 검출 후 Graph MLP로 골격 구조+시간적 일관성 학습 | **원문 확인 완료.** Precision 0.95/Recall 0.94/F1 0.95, 키포인트 위치오류(KLE) 3.10. 기존 YOLOv8-Pose 단독 대비: 추론시간 0.23초→0.08초, FPS 58→125로 대폭 개선. COCO Keypoints 데이터셋 기준 | 일반 MLP가 아니라 Graph MLP + 시간적 일관성까지 결합한 고급 구조 — 우리가 향후 확장 연구로 고려할 수 있는 방향(1차는 일반 MLP, 확장 시 Graph MLP)이라는 팀원 의견의 근거로 활용 가능 |
| [ConMLP: MLP-Based Self-Supervised Contrastive Learning for Skeleton Data Analysis and Action Recognition](https://doi.org/10.3390/s23052452) (Dai et al., *Sensors*, 2023, 23(5), 2452) | Skeleton sequence를 MLP 기반 자기지도 대조학습(contrastive learning)으로 표현 학습 | **원문 확인 완료.** NTU RGB+D에서 96.9%(자기지도 학습 SOTA 대비 +0.2%p, 감독학습 Shift-GCN 95.1% 상회). 연산량 46.54M FLOPs로 Shift-GCN(2.5G FLOPs) 등 GCN 계열보다 훨씬 경량 | MLP가 무겁고 느리다는 통념과 달리, 작게 설계하면 GCN 계열보다 오히려 가벼울 수 있다는 근거. 다만 자세 "분류"가 아니라 행동 인식용 표현학습이라 우리 과업과는 거리가 있음 — 배경 자료로만 활용 |

**요약**: 4편 모두 실제 존재하는 논문이며(전부 원문에서 검증), 그 중 ①Thoutam(요가) 논문이 우리 구조(Keypoint 각도→MLP)와 가장 유사하고 정확도도 매우 높다(99.58%, 단 데이터 350개로 작음). 팀원 문서의 결론(MLP를 최종 모델로 미리 정하지 않고 RF·XGBoost·LightGBM과 동일 데이터·특징으로 공정 비교하는 딥러닝 기준 모델로 추가)은 지금까지의 조사 기조와 일관되며, 발표에서 "우리는 트리 기반 ML뿐 아니라 딥러닝(MLP)까지 비교 후보에 넣어 공정하게 검증했다"는 근거로 쓸 수 있다.

---

## 4. 아직 확인 못한 것 / 추가 조사가 필요할 수 있는 부분

- ~~Weighted Random Forest 논문(IET, 2025)의 정확한 정확도 수치~~ → **해결**: 원문 PDF 확보, Accuracy·F1 0.983 (가중치 없는 RF 대비 96%→98%), 8개 분류기 중 최고 성능 확인. 상세 내용은 `weighted_rf_paper_extract.md` 참고. 특히 이 논문이 결론에서 직접 제안한 "후속 연구 방향(깊이 카메라로 3D 데이터 수집)"이 바른자세의 스테레오 카메라 설계와 정확히 일치함 — 발표 스토리라인에 강력하게 활용 가능.
- MoveNet Lightning + 랜덤포레스트를 **동일 논문에서 함께 사용한 선행 사례**는 아직 못 찾음 — 각각 따로 검증된 조합이라, 발표에서는 "두 선택을 각각 문헌으로 정당화"하는 구조로 가는 게 안전함(직접 결합 사례가 있다고 주장하지 않기).
- 우리 팀의 실제 oCamS-1CGN-U + 노트북 조합에서의 실측 FPS·정확도는 아직 없음 — 3주차 게이트 실측치로 이 문서를 업데이트할 예정. (테스트 노트북 사양은 0절에 확정 기재됨: i7-14700HX + RTX5060 8GB + RAM 32GB. CPU-only 측정도 함께 하는 것을 권장.)

### 4-2. 원문 PDF가 필요한 논문 목록 (내가 직접 검증할 수 없었던 것들)

발표에서 정량적 근거로 쓰기 전에 원문으로 재검증이 필요한 순서대로 정리:

1. **[Comparative Evaluation of MediaPipe and YOLOv8 for Real-Time Pose Estimation](https://link.springer.com/chapter/10.1007/978-3-032-00232-7_13)** (Springer, 2026) — 두 번 시도 중 한 번은 접속 제한(rate limit)으로 실패, 페이월일 가능성도 있음. 군산대 도서관의 Springer/Wiley 계열 접근 권한으로 원문 확보 가능한지 확인해보면 좋을 듯.
2. **[An attention-based adaptive spatial–temporal graph convolutional network...](https://www.sciencedirect.com/science/article/pii/S0952197623019644)** (AAST-GCN, ScienceDirect, 2024) — ScienceDirect 페이월, 아직 본문 미확인. GCN 계열 후보를 "고려는 했지만 실현 가능성 때문에 제외했다"는 근거로만 가볍게 쓸 거라 우선순위는 낮지만, 정확한 수치를 인용하려면 원문 필요.

**직접 원문으로 검증 완료된 논문** (더 이상 PDF 불필요): Weighted RF(IET), OpenPose/PoseNet/MoveNet 비교(IIETA/Traitement du Signal), XGBoost 자세논문(IAES IJ-AI), MoveNet 자세감지(IJIRT), EAI 분류기 비교(academia.edu 경유), **Reality Check(MDPI) — 사용자가 PDF를 직접 올려줘서 원문 전체(Figure 포함) 확인 완료.**, **휠체어 압력센서 LightGBM/RF 비교(Sensors, MDPI) — 원문 PDF를 두 번 독립적으로 질의해 Table 4 수치가 정확히 일치함을 확인.**, **MLP 관련 4편(Yoga/Thoutam, KeypointNet/Electronics, Graph MLP/Discover AI, ConMLP/Sensors) — 전부 원문에서 정확도 수치까지 확인 완료(KeypointNet은 2회 교차 확인).**

**검토 후 제외한 논문**: MovePose(arXiv, Yu et al.) — 원문 PDF 확인 완료. GFLOPs 기준으로 보면 MoveNet-Lightning의 "경량화 트레이드오프" 근거로 쓰기 어렵고(같은 논문 제안 모델 대비 GFLOPs는 0.17만 적은데 AP는 17.5점 낮음), 공개 코드·사전학습 모델도 없어(논문엔 "공개 예정"만 명시, 실제 저장소 미확인) 학기 내 구현도 불가능하다고 판단해 인용에서 제외함.
