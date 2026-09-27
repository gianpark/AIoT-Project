# 참고 논문 10편 — 원문 검증 후 필요한 내용만 정리

3주차 발표에서 실제로 원문을 확보해 직접 읽고 검증한 10편의 논문을 정리한다(1~6번은 PDF/웹 원문 직접 확인, 7-1~7-4는 MLP 관련 추가 조사분). 모두 웹 요약이 아니라 원문(Figure·Table 포함)을 직접 읽고 확인한 수치만 담았다.

---

## 1. Toward Real-Time Posture Classification: Reality Check

**서지정보**: Hongbo Zhang, Denis Gračanin, Wenjing Zhou, Drew Dudash, Gregory Rushton, *Electronics* (MDPI), Vol.14, No.9, Article 1876, 2025.05.05. DOI: 10.3390/electronics14091876. Middle Tennessee State University 외. 오픈액세스.

**목적**: 낙상 방지를 위한 실시간 자세 분류기 개발 — 고전 머신러닝과 딥러닝(LSTM)의 정확도·견고성·속도를 비교.

**방법**: Kinect V2 RGBD 카메라로 관절 (x,y,z) 좌표 수집. 참가자 5명(18~30세, BMI 21~26), 총 10,000개 자세 샘플. 5개 동작(구부리기·앉기-서기·다리올리기/내리기·돌기·점프) — "앉은 자세"가 아니라 낙상 방지용 전신 "동작" 분류. 분류기: SVM(linear), Gaussian NB, Random Forest(트리 10개, min_samples_split=2), Neural Network(3층, hidden 9), AdaBoost(estimator 50, lr=1) vs LSTM(5층, hidden_dim=256). 60/40 train-test split(5-fold CV와 유사한 결과 확인).

**필요한 수치만**
- 정확도: SVM·Gaussian NB·RF·NN·LSTM 모두 특정 시나리오에서 **99%** 도달(Table 4). AdaBoost만 일관되게 최저(특히 다리올리기 동작).
- 노이즈 강건성: 30% 노이즈에서도 SVM·LSTM은 99%→86%까지만 하락(Table 4 요약). 본문에는 "RF·SVM이 노이즈에 가장 강함"이라는 서술도 있어 Table 4 요약과 약간의 불일치가 있음(참고만 할 것).
- 속도(하드웨어: RTX 4070 12GB + i5-13400F 2.5GHz): Gaussian NB가 최속(테스트 데이터 40% 추론에 0.0024초). CPU 단독 비교에서 LSTM이 최저속(17.6초/40%). GPU로 LSTM 돌릴 때 나머지는 CPU 유지 시 AdaBoost가 가장 느림(0.77초/40%).
- **본문에 직접 명시된 문장**: "Without hardware acceleration, the LSTM method was about 20 times slower than the slowest classical machine learning classification methods while running on a CPU." (17.6초 ÷ 0.77초 ≈ 22.9배로 계산도 일치)
- 학습 데이터 필요량(Methods 섹션, 선행연구 인용 기반 — 이 논문 자체 실험 결과 아님): SVM 300개, Gaussian NB 30개, RF 500개, AdaBoost 400개, **LSTM 8000개**.

**⚠️ 주의**: 속도 비교(Figure 10, Table 4)의 5개 막대는 SVM·AdaBoost·NN·Gaussian NB·LSTM뿐이고 **Random Forest는 포함되지 않음**. "RF가 20배 빠르다"가 아니라 "고전 ML 계열(가장 느린 AdaBoost 기준)이 20배 빠르다"로 인용할 것.

**우리 프로젝트에 필요한 부분**: 랜덤포레스트 선정의 가장 강력한 근거. "실시간성이 최우선인 응용에서는 고전 ML로 충분하고 딥러닝은 CPU에서 비효율적"이라는 결론이 백그라운드 상시 구동 조건과 정확히 맞아떨어짐.

---

## 2. Classification Algorithm for Sitting Postures Using Weighted Random Forest

**서지정보**: Jaeeun Lee, Hongseok Choi, Jongnam Kim, *IET Image Processing*, 19:e70126, 2025. Pukyong National University. 오픈액세스.

**목적**: 정면 카메라 1대로 얼굴·어깨 좌표만 추출해 5가지 앉은 자세를 분류 — 변수 중요도 기반 가중치를 적용한 Weighted Random Forest(WRF) 제안.

**방법**: MediaPipe로 얼굴·양 어깨 좌표만 검출(전신 키포인트 아님). 6개 특징 변수(A=얼굴면적비 0.30, xc 0.20, θ3 0.15, θ2 0.15, θ1 0.15, yc 0.05 — 괄호는 Gini 중요도 기반 가중치)를 RF로 중요도 산출 → 가중치 적용 → 재학습. 20명(남12·여8), 100장 이미지, 10-fold 교차검증.

**필요한 수치만**
- 가중치 없는 RF 96% → WRF 98% (10-fold 최종 정확도)
- 8개 분류기 비교(전체): **WRF 0.98(1위)** > XGBoost 0.96 > Neural Network 0.95 > SVM 0.89 ≈ LightGBM 0.89 ≈ AdaBoost 0.89 > Logistic Regression 0.87 > KNN 0.81
- 최종 성능(Table 8 기준): Recall 0.988, Precision 0.978, F1 0.983, **Accuracy 0.983**, 처리시간 1.51초(정지 이미지 1장 기준, 프레임 스트림 아님)
- 통계 검정: Friedman test(χ²=75.98, p<0.001) + Nemenyi 사후검정으로 WRF가 KNN(p=0.0012)·LR(p=0.0153) 대비 유의하게 우수

**오분류 원인(저자가 명시)**: (1) 참가자 옷 색깔이 배경색과 유사해 어깨 검출 실패, (2) 얼굴이 아래로 기울어져 얼굴 면적비 계산 부정확 — 즉 의류색·배경색·조명 조건에 성능이 흔들리는 게 이 방법의 명확한 약점.

**저자가 결론에서 직접 제시한 후속 연구**: "depth-sensing camera를 이용해 3D 데이터를 수집하고 다양한 앉은 자세를 분류하는 후속 연구를 계획" — 원문 명시.

**우리 프로젝트에 필요한 부분**: 가장 가까운 선행 연구. (1) 저자가 남긴 "3D 깊이 카메라" 과제를 바른자세의 스테레오 카메라 설계가 이미 구현하고 있음. (2) 이 논문의 약점(2D 단안, 얼굴·어깨만 사용 → 옷/배경색에 취약)을 우리는 전신 17키포인트+깊이 정보로 원천적으로 완화할 수 있다는 가설로 제시 가능(단, 실측 전 가설임을 명시해야 함). (3) "기존 RF → 중요도 분석 → 가중치 적용 → 개선 RF" 구조가 96%→98% 향상으로 실증된 선행 사례.

---

## 3. Comparative Analysis of OpenPose, PoseNet, and MoveNet Models for Pose Estimation in Mobile Devices

**서지정보**: Jo, J.; Kim, Y., *Traitement du Signal*, Vol.39, No.1, 2022, pp.119-124. DOI: 10.18280/ts.390111 (IIETA).

**목적**: 모바일 기기 환경에서 OpenPose·PoseNet·MoveNet(Lightning/Thunder)의 정확도와 처리 속도를 비교.

**방법**: COCO·MPII 기준 이미지 1,000장으로 각 모델의 정확도와 처리시간 측정. 3번째 이미지 그룹은 "사람이 없는 이미지"로 오탐률 테스트.

**필요한 수치만**
- 정확도: PoseNet **97.6%**(최고) > OpenPose 86.2% > MoveNet Thunder 80.6% > MoveNet Lightning **75.1%**
- 속도(1,000장 처리시간): MoveNet Lightning **53.5초**(최속) < PoseNet 75.2초 < MoveNet Thunder 140.0초 < OpenPose 643.5초(최장, MoveNet 대비 약 12배 느림)
- 키포인트 수: OpenPose 137개(얼굴·손 포함) vs PoseNet/MoveNet 17개(동일)
- "사람이 없는 이미지" 그룹에서 MoveNet Lightning 정확도가 71.5%로 하락 — 사람이 없을 때 오탐 발생 여지가 있다는 뜻

**우리 프로젝트에 필요한 부분**: MoveNet Lightning 선택 근거. 정확도 1등은 아니지만(75%대) 속도가 압도적이라 "저사양 노트북에서 5초 주기 백그라운드 상시 구동" 조건에 부합. 빈 자리(사람 없음) 상황에서의 오탐 가능성은 우리 시스템 설계 시 참고할 리스크 포인트.

---

## 4. Automated ergonomic sitting postures detection for office workstation using XGBoost method

**서지정보**: Pawitra et al., *IAES International Journal of Artificial Intelligence*, Vol.15, No.1, 2026.02, pp.506-514. DOI: 10.11591/ijai.v15.i1.pp506-514.

**목적**: MoveNet Thunder로 추출한 키포인트 기반 사무실 착석 자세(정상/비정상 2분류)를 AdaBoost·XGBoost·MLP로 비교.

**방법**: MoveNet Thunder로 17개 키포인트 추출 → 머리~엉덩이(상반신)만 남기고 정규화·중심이동 → SNI 9011:2021 기준 정상/비정상 2분류. 참가자 30명, 정면+측면 촬영, 유효 인스턴스 1,550개(정상 873/비정상 677), 10-fold 교차검증.

**필요한 수치만**
- AdaBoost: Accuracy 0.894±0.021, Precision 0.936±0.031, Recall 0.847±0.313, F1 0.889±0.022, ROC-AUC 0.950±0.016
- MLP: Accuracy 0.918±0.221 (**폴드 간 편차가 매우 큼 — 불안정**, 수치만 보면 나쁘지 않지만 신뢰하기 어려움), Precision 0.946±0.024, Recall 0.886±0.035, F1 0.915±0.024, ROC-AUC 0.960±0.011
- **XGBoost(최고·가장 안정적)**: Accuracy 0.930±0.019, Precision 0.946±0.032, Recall 0.914±0.019, F1 0.929±0.019, ROC-AUC 0.974±0.010
- 논문 자체 비교표(Table 2, 선행연구 대조): Deep recurrent hierarchical network 91.47%, MediaPipe+decision tree 91.5%/97.05%, **OpenPose+random forest 90.64%**(다른 과업인 인간 행동 인식이지만 RF 계열 참고치)

**저자가 명시한 한계**: 상반신 키포인트만 사용, 시점·가림·조명에 민감, 임베디드 자원 소모(CPU/RAM/추론시간) 미평가.

**우리 프로젝트에 필요한 부분**: 우리와 같은 MoveNet 계열(Thunder)을 실제 착석 자세 분류에 쓴 2026년 최신 사례지만 **비교군에 Random Forest가 없음**(AdaBoost/XGBoost/MLP만). 우리가 RF/WRF를 추가 비교하면 이 논문의 공백을 메우는 기여가 됨. 저자가 명시한 한계(상반신만, 임베디드 자원 미평가)를 우리 설계(전신 17키포인트+스테레오 깊이, 저사양 CPU 실측)가 어떻게 보완하는지 가설로 제시 가능.

---

## 5. Comparing the Performance of Different Classifiers for Posture Detection

**서지정보**: EAI, 2021. (academia.edu 경유 원문 확인)

**목적**: Kinect 관절 각도 특징 기반으로 고전 ML과 딥러닝 분류기의 자세 인식 성능 비교.

**방법**: 관절 각도 7개 특징, 10-fold 교차검증.

**필요한 수치만**

| 계열 | 분류기 | Accuracy |
|---|---|---:|
| 고전 ML | SVM(RBF) | 83.42% |
| 고전 ML | Logistic Regression | 80.91% |
| 고전 ML | **Random Forest** | **77.08%** |
| 고전 ML | kNN | 76.98% |
| 고전 ML | Naive Bayes | 74.56% |
| 딥러닝 | **1D-CNN(전체 최고)** | 93.45% |
| 딥러닝 | 2D-CNN | 91.59% |
| 딥러닝 | BiLSTM | 90.87% |
| 딥러닝 | LSTM | 88.19% |
| 딥러닝 | MLP | 82.56% |

**우리 프로젝트에 필요한 부분**: 특징이 "관절 각도"라는 점이 바른자세의 각도 기반 특징 추출 설계와 동일해 참고 가치가 높음. 다만 **이 논문에서는 RF 단독 정확도가 77.08%로 80% 미만, 중위권** — "고전 ML이 80%대"라는 주장의 근거로는 SVM/LogReg을 들어야지 RF 자체를 들면 안 됨. RF가 항상 최고 정확도라고 과장하지 않도록 발표에서 주의할 것(정확도보다 속도·효율 중심으로 RF를 정당화하는 게 안전).

---

## 6. A Proposal of Implementation of Sitting Posture Monitoring System for Wheelchair Utilizing Machine Learning Methods

**서지정보**: Jawad Ahmad, Johan Sidén, Henrik Andersson, *Sensors* (MDPI), Vol.21, No.19, Article 6349, 2021.09.23. DOI: 10.3390/s21196349. 오픈액세스.

**목적**: 휠체어 착석 자세를 압력센서로 실시간 모니터링 — 임베디드 보드(라즈베리파이)에서도 독립 구동 가능한 시스템 제안, 여러 ML 분류기의 정확도·연산비용을 하드웨어별로 비교.

**방법**: 좌석에 16개 스크린프린팅 압력센서 배열 설치(카메라 아님, 압력 기반). 참가자 32명(남25·여7), 256회 관측×16센서 = 4,096개 데이터 포인트. 4가지 기울임 자세(좌/우/전방/후방) 분류. k-NN·SVM(linear)·Random Forest·Decision Tree·LightGBM 5개 분류기를 **3개 하드웨어(Desktop i7-4790K, Raspberry Pi 3B, Raspberry Pi 4B)** 각각에서 학습·평가(Spyder IDE, Python 3.7.3).

**필요한 수치만 (원문 Table 4, 2회 교차 확인 완료)**

| 하드웨어 | k-NN | Random Forest | SVM(linear) | Decision Tree | LightGBM |
|---|---:|---:|---:|---:|---:|
| Desktop i7-4790K(4.0GHz) | 98.07% | 98.65% | 95.96% | **98.85%** | **99.03%**(공동 최고) |
| Raspberry Pi 3B(1.2GHz) | 97.88% | 98.46% | 95.00% | 97.11% | **99.03%**(최고) |
| Raspberry Pi 4B(1.5GHz) | 98.27% | 98.07% | 95.00% | 97.81% | **99.03%**(최고) |

컴파일(연산) 시간도 함께 보고됨 — Desktop 기준 Decision Tree 0.049초(최속) < SVM 0.051초 < k-NN 0.054초 < LightGBM 0.131초 < **Random Forest 0.167초(최장)**. **Raspberry Pi 3B(저사양) 기준으로는 LightGBM이 1.576초로 Random Forest(2.181초)보다 오히려 더 빠름.**

**⚠️ 우리 프로젝트에 중요한 주의점**: 이 논문 기준으로는 **정확도·연산속도 둘 다 LightGBM이 Random Forest를 근소하게 앞섬** — 특히 임베디드(저사양) 보드일수록 그 격차가 더 벌어짐. RF를 "가장 빠르고 정확한 선택"이라고 단정하면 이 논문과 배치되므로, "RF도 임베디드급에서 충분히 실용적인 속도·정확도를 낸다"는 근거로는 쓰되 "RF가 최고"라는 과장은 피할 것. 저자가 명시한 한계: Precision/Recall/F1 등 세부지표 미제시, 소규모 데이터셋에서 LightGBM 과적합 위험 언급.

**우리 프로젝트에 필요한 부분**: 팀원이 조사한 XGBoost·LightGBM 후보 비교 시나리오와 직접 연결되는 논문. 카메라가 아니라 압력센서 기반이라는 차이는 있지만, "동일 데이터로 RF/LightGBM 등을 저사양 임베디드 보드에서까지 실측 비교"한 구조 자체가 우리가 3주차 이후 하려는 실험(노트북 CPU-only 실측)의 좋은 선례. 특히 **RF보다 LightGBM이 근소 우위**라는 이 결과는, 우리가 "왜 RF를 최종 선택했는가"를 발표할 때 LightGBM도 정직하게 비교 후보로 언급하고 "해석 가능성(Feature Importance)·구현 단순성" 등 정확도·속도 외의 이유를 함께 제시하는 게 더 설득력 있다는 시사점을 준다.

---

## 7. MLP(딥러닝) 관련 선행연구 4편 — 팀원 추가 조사, 원문 검증 완료

지금까지 RF·XGBoost·LightGBM 등 트리 기반 ML만 비교했는데, 팀원이 "딥러닝도 비교 후보에 넣자"며 MLP 관련 선행논문 4편을 조사했다. 팀원 문서엔 수치가 없어서 4편 모두 원문에서 직접 수치를 찾았다(KeypointNet은 2회 교차 확인).

**7-1. Yoga Pose Estimation and Feedback Generation Using Deep Learning**
서지정보: Thoutam et al., *Computational Intelligence and Neuroscience*, 2022. DOI: 10.1155/2022/4311350.
방법: 관절 각도 12개(Keypoint 사이 각도) → MLP(입력 12 → 은닉 10 → 은닉 8 → 출력 6) → 요가자세 6종 분류(Cobra/Lotus/Corpse/Mountain/Triangle/Tree).
수치: 총 350개 인스턴스(학습 320/검증 30/테스트 30). **테스트 정확도 0.9958**.
필요한 부분: 우리와 가장 구조가 비슷한 선행연구("Keypoint 각도→MLP"). 다만 데이터가 350개로 적어 과적합 위험을 우리가 주의해야 함.

**7-2. KeypointNet: An Efficient Deep Learning Model with Multi-View Recognition Capability for Sitting Posture Recognition**
서지정보: Cao et al., *Electronics* (MDPI), 2025, 14(4), 718.
방법: 지역 특징(6개 신체부위 Keypoint, 1D conv+MLP) + 전역 특징(Keypoint 간 관계) 결합 후 MLP로 최종 분류.
수치: MVSP 98.46%, HASP 96.52%, MSRA 99.48% (2회 교차 확인 일치).
필요한 부분: 앉은 자세 인식에 MLP를 쓴 2025년 최신 사례. 단, MLP가 독립 분류기가 아니라 네트워크 내부 구성요소로만 쓰여 우리 비교 실험 구조와는 다름 — 참고용.

**7-3. Human posture recognition model integrating YOLOv8 pose and graph multilayer perceptron**
서지정보: Tian & Liu, *Discover Artificial Intelligence*, 2026, 6, 1043.
방법: YOLOv8-Pose로 Keypoint 검출 → Graph MLP로 골격 구조+시간적 일관성 학습.
수치: Precision 0.95/Recall 0.94/F1 0.95, KLE 3.10. 기존 YOLOv8-Pose 단독 대비 추론시간 0.23초→0.08초, FPS 58→125.
필요한 부분: 일반 MLP의 확장형(Graph MLP). 1차는 일반 MLP로 비교하고, 향후 확장연구 방향으로 Graph MLP를 제시할 근거.

**7-4. ConMLP: MLP-Based Self-Supervised Contrastive Learning for Skeleton Data Analysis and Action Recognition**
서지정보: Dai et al., *Sensors* (MDPI), 2023, 23(5), 2452.
방법: Skeleton sequence를 MLP 기반 자기지도 대조학습으로 표현 학습.
수치: NTU RGB+D 96.9%(자기지도 SOTA 대비 +0.2%p, 감독학습 Shift-GCN 95.1% 상회). 연산량 46.54M FLOPs로 Shift-GCN(2.5G FLOPs)보다 훨씬 경량.
필요한 부분: "MLP는 가볍게 설계하면 GCN 계열보다 오히려 연산량이 작을 수 있다"는 배경 근거. 단 행동인식용 표현학습이라 우리 자세분류 과업과는 거리가 있어 배경자료로만 활용.

**우리 프로젝트에 필요한 부분(종합)**: 4편 모두 실제 논문이고, 그중 7-1(Thoutam)이 우리 구조와 가장 가깝고 정확도도 매우 높다. 팀원 결론대로 MLP를 최종 모델로 미리 정하지 않고 RF·XGBoost·LightGBM과 동일 데이터·특징으로 공정 비교하는 딥러닝 기준 모델로 추가하는 것이 맞는 방향.

---

## 요약: 논문별 핵심 한 줄

| 논문 | 핵심 한 줄 |
|---|---|
| Reality Check (Electronics 2025) | 고전 ML이 CPU에서 LSTM보다 약 20배 빠르면서 99% 정확도 가능 |
| Weighted RF (IET 2025) | 특징 가중치로 RF 96%→98%, 우리와 가장 가까운 선행 연구 |
| OpenPose/PoseNet/MoveNet (Traitement du Signal 2022) | MoveNet Lightning은 정확도 최고는 아니나(75%) OpenPose 대비 12배 빠름 |
| XGBoost 착석자세 (IAES IJAI 2026) | 우리와 같은 MoveNet 계열 최신 사례, RF 비교군 없음 — 우리가 메울 공백 |
| EAI 분류기 비교 (2021) | 관절 각도 특징으로 RF 77%(중위권) — RF를 정확도로 과장하지 않을 근거 |
| 휠체어 압력센서 (Sensors 2021) | 임베디드 보드 실측에서 LightGBM이 RF보다 정확도·속도 모두 근소 우위 — RF 과장 금지 근거 |
| Yoga MLP (CIN 2022) | Keypoint 각도→MLP 구조로 정확도 99.58% — 우리와 가장 유사한 구조 |
| KeypointNet (Electronics 2025) | 앉은자세 인식에 MLP 활용, 98~99%대 정확도 (참고용, 독립분류기 아님) |
| Graph MLP+YOLOv8 (Discover AI 2026) | Graph MLP로 추론시간 0.23s→0.08s 단축 — 향후 확장연구 방향 |
| ConMLP (Sensors 2023) | 경량 MLP(46.54M FLOPs)로 GCN 계열(2.5G FLOPs)보다 가벼움 (배경자료) |
