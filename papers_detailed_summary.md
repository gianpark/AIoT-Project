# 참고 논문 10편 — 섹션별 상세 요약

원문 PDF를 직접 읽고(①②③④) 웹에서 원문을 확인한(⑤⑥) 내용을 바탕으로, 각 논문의 흐름(배경·방법·결과·논의·결론)을 따라가며 요약했다. 전문 번역이 아니라 내용을 우리말로 정리·재구성한 것이며, 원문 그대로의 문장은 최소한의 직접 인용(따옴표 표시)만 포함했다. ⑥은 원문 PDF를 두 번 교차 확인해 수치 신뢰도를 검증했다.

---

## ① Toward Real-Time Posture Classification: Reality Check
*Zhang, Gračanin, Zhou, Dudash, Rushton — Electronics(MDPI), 2025*

**배경**: 낙상은 노인 응급실 방문의 주요 원인 중 하나이며(전체 응급실 방문의 21.3%), 실시간 자세 분류는 낙상 방지의 핵심 기술이다. 기존 연구는 점점 더 딥러닝(AlphaPose, ViTPose++, WideHRNet 등)으로 쏠리고 있지만, 저자들은 이들 모델이 GPU에서는 39fps 이상을 내도 CPU에서는 1~5fps로 급락한다는 점을 지적한다. 딥러닝은 학습 데이터도 고전 ML보다 최대 1000배 더 필요하다는 일반적 한계도 서론에서 언급한다. 이런 배경에서 "정말로 실시간 자세 분류에 딥러닝이 필요한가"를 검증하는 것이 이 논문의 문제의식이다.

**방법**: Kinect V2로 실내에서 5명(18~30세, BMI 21~26)의 참가자를 촬영해 5가지 동작(점프, 앉기-서기, 다리 올리기/내리기, 구부리기, 돌기)에 대해 총 10,000개 트라이얼을 수집했다. 관절 (x,y,z) 좌표를 라벨링 GUI로 라벨링했다. 세 단계 실험을 설계했다: (1) 노이즈·결손 없는 기준선에서 5개 고전 분류기(SVM, Gaussian NB, Random Forest, Neural Network, AdaBoost) 비교, (2) 동작별로 관련성 높은 관절만 사용했을 때의 견고성 검증, (3) 무작위 노이즈(1.5%~30%)를 추가했을 때의 정확도 변화 측정. 이후 5층 LSTM(hidden_dim=256)을 구현해 같은 조건에서 고전 ML과 비교했다. 학습은 cyclic learning rate + RMSprop, 5000 epoch로 진행했다.

**결과**: 노이즈·결손이 없을 때 Gaussian NB와 Random Forest가 모든 동작에서 안정적으로 높은 정확도를, AdaBoost가 상대적으로 낮은 정확도를 보였다. 일부 관절만 사용해도(예: 다리 올리기 동작에 발 관절만) 정확도 저하가 크지 않았고, 노이즈가 20%를 넘기 전까지는 성능이 크게 떨어지지 않았다. LSTM도 유사한 패턴을 보이며, 고전 ML(75~99%)과 LSTM(86~99%)의 정확도 범위가 겹친다. 반면 속도 측면에서는 뚜렷한 차이가 났다: Gaussian NB가 테스트 데이터 40% 추론에 0.0024초로 가장 빨랐고, CPU 전용 LSTM은 같은 조건에서 17.6초로 가장 느렸다. "GPU 가속 없이 CPU만 사용할 경우 LSTM은 가장 느린 고전 분류기보다 약 20배 느리다"는 것이 저자들이 직접 명시한 핵심 결론이다. GPU를 쓰면 LSTM 성능이 고전 ML과 비슷해진다.

**논의·한계**: 저자들은 표본 크기(참가자 5명)의 한계, 성별 균형 미고려, 단일 카메라로 인한 가림 문제, 딥러닝 비교군이 LSTM 하나뿐이라는 점을 스스로 한계로 언급했다. 향후 연구로 Halpe-FullBody(40,000장) 같은 대규모 데이터셋 활용, Transformer 등 추가 딥러닝 비교, 강화학습과의 결합, 다중 카메라를 통한 가림 해결 등을 제시했다.

**결론**: "실시간 성능이 최우선인 응용에서는 값비싼 딥러닝 구현 없이도 비용 효율적인 고전 분류기가 충분하다"는 것이 최종 메시지다. 건강·안전 모니터링처럼 CPU 기반 상시 구동이 필요한 응용에 특히 적합하다고 강조한다.

---

## ② Classification Algorithm for Sitting Postures Using Weighted Random Forest
*Lee, Choi, Kim — IET Image Processing, 2025*

**배경**: 장시간 앉아있는 생활 습관이 목·허리 통증 등 근골격계 문제를 유발한다는 문제의식에서 출발한다. 기존 자세 분류 연구들은 웨어러블 센서(착용 불편)나 다중 카메라(비용·설치 부담)를 쓰는 경우가 많았고, 단일 카메라 기반 연구도 있었지만 저자들은 "특징의 중요도를 고려하지 않고 균등하게 다루는 기존 Random Forest의 한계"를 지적하며 개선 여지를 제시한다.

**방법**: 모니터 위에 CCD 카메라 1대를 정면으로 설치하고, MediaPipe로 얼굴과 양 어깨 좌표만 검출한다(전신이 아님). 여기서 6개 특징을 만든다: 얼굴 검출 면적비(A, 화면과의 거리 반영), 얼굴 중심 x좌표(xc), 얼굴 중심-좌측 어깨 각도(θ3), 얼굴 중심-우측 어깨 각도(θ2), 두 각도의 합(θ1), 얼굴 중심 y좌표(yc). 먼저 일반 Random Forest로 이 6개 변수의 중요도(Gini 불순도 감소량, MDI)를 계산한 뒤, 그 중요도를 가중치로 변수에 곱해 데이터를 재구성하고, 가중된 데이터로 Random Forest를 다시 학습시키는 것이 Weighted Random Forest(WRF)다. 20명(남12·여8)을 대상으로 정자세부터 시작해 자연스럽게 5가지 자세(정자세·거북목·뒤로 기댐·좌/우 기울임)를 취하게 해 100장을 촬영했고, 10-fold 교차검증(9:1)으로 평가했다.

**결과**: 가중치 적용 전 RF의 최종 정확도는 96%였는데, WRF 적용 후 98%로 향상됐다. 8개 분류기(WRF, XGBoost, Neural Network, SVM, LightGBM, AdaBoost, Logistic Regression, KNN)를 가중 변수로 비교했을 때 WRF가 0.98로 1위, XGBoost 0.96, Neural Network 0.95 순이었고 KNN이 0.81로 가장 낮았다. Friedman 검정(χ²=75.98, p<0.001)과 Nemenyi 사후검정으로 WRF가 KNN(p=0.0012), 로지스틱회귀(p=0.0153) 대비 통계적으로 유의하게 우수함을 확인했다. 최종 성능은 Recall 0.988, Precision 0.978, F1 0.983, Accuracy 0.983, 처리시간 1.51초(정지 이미지 1장 기준)였다. 선행 방법들(각도 추정법, 스태킹 앙상블, CVA, 거리 추정법)과 비교했을 때도 WRF가 정확도·속도 모두에서 우위를 보였다. 특히 5가지 자세를 전부 분류하는 유일한 비교군인 스태킹 앙상블은 정확도 82.5%에 처리시간 14.28초로, WRF(1.51초) 대비 훨씬 느렸다.

**논의·한계**: 오분류 원인을 두 가지로 분석했다: (1) 참가자의 옷 색깔이 배경색과 비슷해 어깨 검출에 실패한 경우, (2) 얼굴이 아래로 많이 기울어져 얼굴 면적비 계산이 부정확해진 경우. 즉 의류·배경색과 조명 조건에 성능이 민감하다는 게 이 방법의 명확한 약점으로 지적된다.

**결론**: 저자들은 결론부에서 "depth-sensing camera를 이용해 3D 데이터를 수집하고 다양한 앉은 자세를 분류하는 후속 연구를 계획하고 있다"고 명시적으로 밝힌다. 이는 단안 2D 카메라의 한계(옷·배경색·조명에 취약)를 깊이 정보로 보완하려는 의도로 읽힌다.

---

## ③ Comparative Analysis of OpenPose, PoseNet, and MoveNet Models for Pose Estimation in Mobile Devices
*Jo, Kim — Traitement du Signal, 2022*

**배경**: 모바일 기기에서 포즈 추정을 활용하는 애플리케이션(피트니스, 재활, 모션 캡처 등)이 늘면서, 제한된 연산 자원에서도 실시간으로 동작할 수 있는 경량 모델 선택이 중요해졌다. OpenPose(2017)는 높은 정확도로 널리 쓰이지만 무겁고, Google이 내놓은 PoseNet과 MoveNet(Lightning/Thunder 두 버전)은 모바일 환경을 겨냥해 설계됐다. 이 논문은 세 계열 모델을 같은 조건에서 직접 비교해 실무적인 선택 기준을 제시하는 것을 목표로 한다.

**방법**: COCO·MPII에서 추출한 1,000장의 이미지를 세 그룹(사람이 명확히 있는 이미지, 부분적으로 가려진 이미지, 사람이 없는 이미지)으로 나누어 각 모델에 입력하고, 키포인트 검출 정확도와 1,000장 전체 처리 시간을 측정했다. OpenPose는 137개(얼굴·손 포함) 키포인트를, PoseNet과 MoveNet은 동일하게 17개 키포인트를 출력한다.

**결과**: 정확도는 PoseNet이 97.6%로 가장 높았고, OpenPose 86.2%, MoveNet Thunder 80.6%, MoveNet Lightning 75.1% 순이었다. 반면 처리 속도(1,000장 기준)는 정반대 순서에 가까웠다: MoveNet Lightning이 53.5초로 가장 빨랐고, PoseNet 75.2초, MoveNet Thunder 140.0초, OpenPose가 643.5초로 가장 느렸다(MoveNet Lightning 대비 약 12배). 즉 정확도가 가장 높은 모델(PoseNet)과 속도가 가장 빠른 모델(MoveNet Lightning)이 다르다는, 뚜렷한 정확도-속도 트레이드오프가 확인됐다. "사람이 없는 이미지" 그룹에서는 MoveNet Lightning의 정확도가 71.5%로 오히려 하락했는데, 이는 사람이 없는 상황에서 오탐(false positive)이 발생할 여지가 있음을 시사한다.

**논의**: 저자들은 모바일 기기처럼 연산 자원이 제한적인 환경에서는 정확도보다 속도가 실용적으로 더 중요한 경우가 많다고 논의하며, 애플리케이션의 목적(정밀도 중심 vs 실시간성 중심)에 따라 모델을 다르게 선택해야 한다고 제안한다.

**결론**: 세 모델 계열 각각의 강점이 다르므로 "가장 좋은 모델"은 존재하지 않고, 용도에 맞는 선택이 중요하다는 것이 최종 메시지다. 실시간·저사양 환경에는 MoveNet Lightning이, 정확도가 중요한 환경에는 PoseNet이 적합하다고 결론짓는다.

---

## ④ Automated ergonomic sitting postures detection for office workstation using XGBoost method
*Pawitra et al. — IAES International Journal of Artificial Intelligence, 2026*

**배경**: 사무직 근로자의 장시간 착석이 근골격계 질환(MSD)의 주요 위험 요인이라는 인체공학적 문제의식에서 출발한다. 인도네시아 산업표준 SNI 9011:2021의 착석 자세 기준을 활용해, 카메라 기반으로 정상/비정상 자세를 자동 판별하는 시스템을 구축하는 것이 목표다.

**방법**: MoveNet Thunder로 17개 키포인트를 추출한 뒤, 머리부터 엉덩이까지의 상반신 키포인트만 남기고 정규화·중심이동 전처리를 했다. 참가자 30명을 정면과 측면에서 촬영해 유효 인스턴스 1,550개(정상 873개, 비정상 677개)를 확보했다. AdaBoost, XGBoost, MLP 세 분류기를 10-fold 교차검증으로 비교했다.

**결과**: AdaBoost는 Accuracy 0.894±0.021, Precision 0.936±0.031, Recall 0.847±0.313, F1 0.889±0.022, ROC-AUC 0.950±0.016을 기록했다. MLP는 평균 Accuracy 0.918로 수치상으로는 나쁘지 않았지만 표준편차가 ±0.221로 폴드마다 성능 편차가 매우 커서 불안정한 것으로 평가됐다(Precision 0.946±0.024, Recall 0.886±0.035, F1 0.915±0.024, ROC-AUC 0.960±0.011). XGBoost가 Accuracy 0.930±0.019, Precision 0.946±0.032, Recall 0.914±0.019, F1 0.929±0.019, ROC-AUC 0.974±0.010으로 가장 높고 가장 안정적인 성능을 보였다. 논문 자체의 선행연구 비교표(Table 2)에서는 Deep recurrent hierarchical network(91.47%), MediaPipe+decision tree(91.5%/97.05%), OpenPose+random forest(90.64%, 다른 과업인 인간 행동 인식 기준) 등과 비교해도 이 연구의 XGBoost가 우수하다고 주장한다.

**논의·한계**: 저자들은 스스로 세 가지 한계를 명시했다: (1) 상반신 키포인트만 사용해 하반신 자세 정보를 반영하지 못한다는 점, (2) 카메라 시점·가림·조명 조건에 민감하다는 점, (3) 임베디드/저사양 환경에서의 실제 자원 소모(CPU, RAM, 추론시간)를 평가하지 않았다는 점이다.

**결론**: MoveNet 기반 키포인트와 XGBoost 조합이 사무실 착석 자세 자동 감지에 실용적으로 활용 가능하다고 결론짓는다. 다만 비교 분류기 목록에 Random Forest가 포함되지 않았다는 점은 이 논문 자체의 공백으로 남아 있다.

---

## ⑤ Comparing the Performance of Different Classifiers for Posture Detection
*EAI, 2021*

**배경**: Kinect 같은 깊이 카메라로 얻은 관절 각도 데이터를 특징으로 삼아, 고전 머신러닝과 딥러닝 계열 분류기 중 어떤 방식이 자세 인식에 더 적합한지 비교하는 것이 목적이다.

**방법**: 관절 각도 7개를 특징으로 사용해 10-fold 교차검증으로 두 그룹, 총 10개 분류기를 비교했다. 고전 ML: SVM(RBF 커널), Naive Bayes, Logistic Regression, kNN, Random Forest. 딥러닝: MLP, 1D-CNN, 2D-CNN, LSTM, BiLSTM.

**결과**: 고전 ML 중에서는 SVM(RBF)이 83.42%로 가장 높았고, Logistic Regression 80.91%, Random Forest 77.08%, kNN 76.98%, Naive Bayes 74.56% 순이었다. 딥러닝 중에서는 1D-CNN이 93.45%로 전체 10개 분류기 중 최고 성능을 기록했고, 2D-CNN 91.59%, BiLSTM 90.87%, LSTM 88.19%, MLP 82.56% 순이었다.

**논의·결론**: 저자들은 1D-CNN이 관절 각도라는 저차원 시계열 유사 데이터의 패턴을 학습하는 데 특히 효과적이었다고 논의하며, 딥러닝 계열이 전반적으로 고전 ML보다 우수한 정확도를 보였다고 결론짓는다. (이 논문 역시 웹에서 확인 가능한 범위의 원문 내용을 바탕으로 정리했으며, PDF 전문을 직접 확보하지는 못했다.)

---

## ⑥ A Proposal of Implementation of Sitting Posture Monitoring System for Wheelchair Utilizing Machine Learning Methods
*Ahmad, Sidén, Andersson — Sensors(MDPI), 2021*

**배경**: 휠체어 사용자는 장시간 같은 자세로 앉아 있는 경우가 많아 욕창 등 이차적 건강 문제가 생기기 쉽다는 문제의식에서 출발한다. 기존 자세 모니터링 연구는 카메라 기반이 많은데, 저자들은 휠체어 환경에서는 프라이버시·설치 위치 제약 때문에 카메라보다 좌석에 내장하는 압력센서가 더 실용적이라고 보고, 이를 독립적인 임베디드 시스템(라즈베리파이)으로 구현할 수 있는지까지 검증하는 것을 목표로 삼는다.

**방법**: 휠체어 좌석에 16개의 스크린프린팅 압력센서를 배열로 설치했다. 32명(남25·여7)의 참가자로부터 256회 관측(관측당 16개 센서값) = 총 4,096개 데이터 포인트를 수집했다. 좌측 기울임·우측 기울임·전방 기울임·후방 기울임 4가지 자세를 분류 대상으로 삼았다. k-NN, SVM(linear), Random Forest, Decision Tree, LightGBM 5개 분류기를 학습시키고, 이를 **3개의 서로 다른 하드웨어**(데스크톱 PC의 Core i7-4790K, 임베디드 보드인 Raspberry Pi 3B, Raspberry Pi 4B)에 각각 배포해 정확도와 연산(컴파일) 시간을 비교했다.

**결과**: Table 4에 하드웨어별·분류기별 정확도가 정리돼 있다. Desktop(i7-4790K) 기준으로는 LightGBM 99.03%로 Decision Tree(98.85%), Random Forest(98.65%), k-NN(98.07%), SVM(95.96%) 순이었다. 임베디드 보드로 가면 순위가 조금 바뀌는데, Raspberry Pi 3B·4B 모두에서 LightGBM이 99.03%로 단독 1위를 유지했고, Random Forest는 98.07~98.46%로 2위권을 지켰다. 연산 시간(컴파일 시간)은 Desktop 기준 Decision Tree(0.049초)·SVM(0.051초)·k-NN(0.054초)이 가장 빠르고, LightGBM 0.131초, **Random Forest가 0.167초로 가장 느렸다.** 저사양인 Raspberry Pi 3B에서는 이 격차가 더 벌어져 LightGBM 1.576초 vs Random Forest 2.181초로, **LightGBM이 Random Forest보다 정확도·속도 둘 다 앞섰다.**

**논의·한계**: 저자들은 Precision·Recall·F1 같은 세부 성능 지표는 이 논문에서 보고하지 않았다는 점, 그리고 데이터셋 규모가 작아 LightGBM 같은 부스팅 계열 모델이 과적합됐을 가능성이 있다는 점을 한계로 언급한다.

**결론**: 저비용 압력센서와 경량 ML 모델을 라즈베리파이 같은 소형 임베디드 보드에 올려도 최대 99.03%의 정확도로 독립 구동 가능한 휠체어 자세 모니터링 시스템을 만들 수 있다는 것이 결론이다. 카메라 없이도 실시간 자세 분류가 가능함을 보여준 사례다.

---

## ⑦ Yoga Pose Estimation and Feedback Generation Using Deep Learning
*Thoutam et al. — Computational Intelligence and Neuroscience, 2022*

**배경**: 요가 자세를 스스로 교정하기 어렵다는 문제의식에서, 카메라만으로 사용자의 요가 자세를 인식하고 피드백을 주는 딥러닝 시스템을 제안한다. 관절 좌표 자체보다 관절 사이의 각도가 자세의 본질적 특징이라고 보고, 각도 기반 특징을 신경망 입력으로 사용하는 접근을 택했다.

**방법**: 사람의 Keypoint에서 12개의 관절 각도(angle features)를 계산해 MLP에 입력했다. MLP 구조는 입력층 12개 → 첫 번째 은닉층 10개 → 두 번째 은닉층 8개 → 출력층 6개(요가 자세 6종: Cobra, Lotus, Corpse, Mountain, Triangle, Tree)다. 총 70개 비디오에서 350개 인스턴스를 추출해 학습 320개·검증 30개·테스트 30개로 나눴다.

**결과**: MLP는 테스트 데이터셋에서 정확도 0.9958(99.58%)을 기록했다. "MLP...obtained an accuracy of 0.9958"이라고 원문에 직접 명시돼 있다.

**논의·한계**: 데이터셋이 350개로 비교적 작다는 점, 그리고 요가처럼 정적이고 뚜렷하게 구분되는 자세일수록 각도 특징만으로도 분류가 쉬워질 수 있다는 점이 스스로 명시하진 않았지만 해석상 주의할 부분이다.

**결론**: Keypoint 사이의 각도를 MLP에 입력하는 비교적 단순한 파이프라인으로도 매우 높은 정확도를 얻을 수 있음을 보여준다. 우리 프로젝트의 "관절 각도 특징 → 분류기" 구조와 가장 유사한 선행 사례다.

---

## ⑧ KeypointNet: An Efficient Deep Learning Model with Multi-View Recognition Capability for Sitting Posture Recognition
*Cao, Wu, Wu, Jiao, Xiao, Zhang, Zhou — Electronics(MDPI), 2025*

**배경**: 앉은 자세 인식은 시점(viewpoint)이 바뀌면 성능이 크게 떨어지는 문제가 있다. 저자들은 여러 시점에서도 강건하게 동작하는 경량 모델을 목표로 KeypointNet을 제안한다.

**방법**: 신체를 6개 부위로 나눠 각 부위의 Keypoint를 1D 합성곱과 MLP로 처리하는 "지역 특징 추출(Local Feature Extraction)" 모듈과, Keypoint 전체 간의 관계를 학습하는 "전역 특징 추출(Global Feature Extraction)" 모듈을 결합했다. 최종적으로 지역·전역 특징을 합쳐 MLP로 자세를 분류한다. 자체 구축한 다중시점 데이터셋 MVSP(7개 고정 시점으로 학습, 1개 동적 시점으로 테스트)와 공개 데이터셋 HASP·MSRA에서 평가했다.

**결과**: MVSP에서 98.46%, HASP에서 96.52%, MSRA에서 99.48%의 정확도를 기록했다(2회 독립 질의로 교차 확인, 두 번 다 일치). 초록에서 "KeypointNet outperforms the other state-of-the-art methods on both the proposed MVSP dataset and the other public datasets, while maintaining a lightweight and efficient design"라고 주장한다.

**논의·한계**: 구체적인 비교 모델별 수치 표는 확보한 텍스트 범위에서 찾지 못했다.

**결론**: 앉은 자세 인식에 MLP를 네트워크 구성요소로 활용해 고정확도·경량 설계를 동시에 달성한 2025년 최신 사례다. 다만 MLP가 독립 분류기가 아니라 KeypointNet이라는 전체 아키텍처의 일부로 쓰였다는 점에서, 우리가 계획하는 "MLP를 RF·XGBoost·LightGBM과 나란히 놓고 비교"하는 실험 구조와는 성격이 다르다.

---

## ⑨ Human posture recognition model integrating YOLOv8 pose and graph multilayer perceptron
*Tian, Liu — Discover Artificial Intelligence, 2026*

**배경**: 복잡한 장면이나 가림(occlusion)이 있는 환경에서 자세 인식 정확도와 실시간성을 동시에 확보하기 어렵다는 문제의식에서, YOLOv8-Pose와 Graph MLP를 결합한 모델을 제안한다.

**방법**: YOLOv8-Pose로 Keypoint를 검출한 뒤, Graph MLP로 인체 골격의 공간적 관계(관절 간 연결 구조)와 채널 의존성을 함께 모델링한다. 시간적 일관성을 높이기 위한 temporal smoothing도 적용했다. COCO Keypoints 데이터셋으로 평가했다.

**결과**: Precision 0.95, Recall 0.94, F1 0.95, 키포인트 위치오류(KLE) 3.10을 기록했다. 기존 YOLOv8-Pose 단독 대비(Precision 0.90, 추론시간 0.23초, FPS 58, KLE 4.20) 추론시간이 0.08초로, FPS는 125로 대폭 개선됐다(KLE도 약 26% 감소). Graph MLP 모듈만 단독 적용해도 F1이 0.93±0.01로 상승하고 KLE가 3.40±0.19로 감소하는 것으로 나타나, Graph MLP 자체의 기여도도 확인된다.

**논의·한계**: 확보한 텍스트 범위에서는 세부 한계 서술을 찾지 못했다.

**결론**: 일반 MLP보다 발전된 형태인 Graph MLP가 정확도와 속도를 동시에 크게 개선할 수 있음을 보여준다. 우리 프로젝트에서는 1차로 일반 MLP를 비교 후보에 넣고, 향후 확장 연구에서 Graph MLP나 GCN 계열로 발전시킬 수 있다는 팀원의 로드맵을 뒷받침하는 근거로 쓸 수 있다.

---

## ⑩ ConMLP: MLP-Based Self-Supervised Contrastive Learning for Skeleton Data Analysis and Action Recognition
*Dai et al. — Sensors(MDPI), 2023*

**배경**: 골격(skeleton) 데이터 기반 행동 인식 연구는 대개 GCN처럼 무거운 모델을 쓰는데, 저자들은 MLP만으로도 자기지도 학습(self-supervised learning) 방식으로 충분한 성능을 낼 수 있는지 검증한다.

**방법**: Skeleton sequence를 입력으로 MLP 기반 대조학습(contrastive learning)을 통해 표현을 학습한다. NTU RGB+D(60개 클래스, 56,880개 액션 클립, X-View/X-Sub 프로토콜)와 NTU RGB+D 120(120개 클래스, 114,480개 클립)에서 평가했다.

**결과**: NTU RGB+D에서 96.9%의 정확도를 기록했으며, 이는 기존 자기지도 학습 SOTA 대비 0.2%p 높고, 감독학습 방식인 Shift-GCN(95.1%)보다도 높은 수치다. 연산량은 MLP(256 계층 기준) 46.54M FLOPs로, Shift-GCN(2.5G FLOPs) 등 GCN 계열보다 훨씬 가볍다.

**논의·한계**: 파라미터 수는 논문 Table 9에 제시돼 있다고 언급되나 구체적 수치는 확보한 범위에서 확인하지 못했고, 추론시간도 명시되지 않았다.

**결론**: MLP가 무겁고 느리다는 일반적 인식과 달리, 작게 설계하면 GCN 계열보다 오히려 가벼우면서 높은 정확도를 낼 수 있음을 보여준다. 다만 이 논문은 자세 "분류"가 아니라 행동 "인식"을 위한 표현학습이라 우리 과업과는 거리가 있어, 배경 자료(MLP가 가벼울 수 있다는 근거)로만 활용한다.

---

## 참고: 검증 수준 안내

①②③④는 사용자가 직접 업로드한 PDF 원문을 전체(그림·표 포함) 읽고 정리한 내용이다. ⑤는 arXiv/웹 페이지에서 확인 가능한 범위의 원문 텍스트를 근거로 정리했으며, 그림(Figure)으로만 제시된 세부 수치까지는 확인하지 못했을 수 있다. ⑥은 원문 PDF를 두 번의 독립적인 질의로 교차 확인해 Table 4 수치가 두 번 다 정확히 일치함을 확인했다(Reality Check 논문 때 웹 요약이 한 번은 틀렸던 전례가 있어, 숫자가 중요한 경우 이렇게 이중 확인하는 절차를 이번에도 적용함). ⑦~⑩(MLP 관련 4편)은 모두 원문에서 수치를 확인했고, 그중 ⑧(KeypointNet)만 2회 독립 질의로 교차 확인했다(⑦은 Wiley 원문 접속이 막혀 academia.edu 경유로 1회만 확인). 더 엄밀한 검증이 필요하면 ⑤의 PDF도 구해서 올려주면 같은 방식으로 재검증할 수 있어.
