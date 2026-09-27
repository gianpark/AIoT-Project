# 기존 자세 알림·모니터링 제품·연구 조사

바른자세(책상 자세·디지털 디톡스 모니터링) 프로젝트의 관련 연구 근거 확보를 위해, 시중 자세 알림 앱·기기와 학술 선행 연구를 조사했다. 크게 (1) 국내 모바일 앱, (2) 해외 웹캠/PC AI 앱, (3) 웨어러블 기기, (4) 학술 선행 연구 4개 그룹으로 나눴다.

---

## 1. 국내 모바일 앱 (스마트폰 카메라·센서 기반)

| 앱 | 방식 | 특징 |
|---|---|---|
| [목 스트레칭 알리미 — 거북목 자세 교정 습관 앱](https://apps.apple.com/us/app/%EB%AA%A9-%EC%8A%A4%ED%8A%B8%EB%A0%88%EC%B9%AD-%EC%95%8C%EB%A6%AC%EB%AF%B8-%EA%B1%B0%EB%B6%81%EB%AA%A9-%EC%9E%90%EC%84%B8-%EA%B5%90%EC%A0%95-%EC%8A%B5%EA%B4%80/id6760984260) | 정해진 주기로 스트레칭 알림 | 실시간 자세 감지가 아니라 시간 기반 리마인더 |
| [포즈프로 — 자세분석&스트레칭](https://apps.apple.com/kr/app/%ED%8F%AC%EC%A6%88%ED%94%84%EB%A1%9C-%EC%9E%90%EC%84%B8%EB%B6%84%EC%84%9D-%EC%8A%A4%ED%8A%B8%EB%A0%88%EC%B9%AD/id1523871489) | 스마트폰 카메라로 한 번 촬영해 자세 분석 | 상시 모니터링이 아니라 스팟 체크형 |
| [LookUp — 거북목 자세 교정 스마트폰 목통증 앱](https://apps.apple.com/kr/app/lookup-%EA%B1%B0%EB%B6%81%EB%AA%A9-%EC%9E%90%EC%84%B8-%EA%B5%90%EC%A0%95-%EC%8A%A4%EB%A7%88%ED%8A%B8%ED%8F%B0-%EB%AA%A9%ED%86%B5%EC%A6%9D/id6758551795) | 스마트폰을 보는 각도 기반 알림 | 책상 앞이 아니라 스마트폰 사용 자세에 초점 |
| [자세 알림 — 허리 통증](https://apps.apple.com/kr/app/%EC%9E%90%EC%84%B8-%EC%95%8C%EB%A6%BC-%ED%97%88%EB%A6%AC-%ED%86%B5%EC%A6%9D/id1493102391) | 타이머 기반 알림 | 카메라 감지 없음 |
| 에어팟 기반 자세 알림 ([관련 기사](https://www.earlyadopter.co.kr/156734)) | 에어팟 내장 자이로센서로 고개 숙임 감지 | 카메라 없이도 목 각도 추정 가능하지만 상체 전체 자세는 파악 못함 |

**공통 한계**: 대부분 스마트폰을 직접 들거나 보는 상황을 전제로 하며, 책상 앞에 앉아 작업 중인 상시 자세를 카메라로 지속 모니터링하는 방식이 아니다. 화면 사용시간과 자세 악화의 상관관계를 분석하는 기능도 없다.

---

## 2. 해외 웹캠/PC 기반 AI 자세 앱

이 그룹이 바른자세와 센싱 방식(카메라 기반 실시간 자세 추정)이 가장 유사하다.

| 앱 | 감지 방식 | 알림 방식 | 개인정보 처리 |
|---|---|---|---|
| [SitApp](https://sitapp.app/) (Mac/Win/Linux) | 웹캠 + 온디바이스 AI, 2분 캘리브레이션으로 개인화 | 음성(11종)·소리(7종) 알림 | "웹캠 영상이 기기 밖으로 나가지 않으며 이미지가 저장·업로드되지 않음"이라고 명시 |
| [Posture Reminder AI](https://posturereminderapp.com/) (Mac) | 웹캠 + 온디바이스 ML, iPhone 카메라(Continuity Camera) 연동 지원 | 실시간 슬라우치 감지 알림 | 온디바이스 처리 명시 |
| [BLiiNK](https://posturereminderapp.com/blog/posture-monitoring-apps/) | 웹캠 AI + 화면과의 거리·눈 깜빡임 횟수까지 함께 추적 | — | 온디바이스 처리 언급 |
| [Zen](https://posturereminderapp.com/blog/posture-monitoring-apps/) | 웹캠 또는 에어팟 센서 | 스트레칭 코칭 콘텐츠 포함 | — |
| [SitWit](https://sitapp.app/blog/best-posture-app) | 웹캠 | 메뉴바에 색상으로 자세 점수 상시 표시 | — |
| [Posturr](https://sitapp.app/blog/best-posture-app) | 애플 Vision 프레임워크(웹캠) | 자세가 나빠질수록 화면이 점점 흐려지는 시각적 피드백 | 완전 오픈소스 |
| [PostureCorrector](https://chromewebstore.google.com/detail/posturecorrector-ai-postu/glbckpboobaemcfljiijgndjlkcokppi) (Chrome 확장) | 웹캠 AI | 브라우저 내 알림 | — |
| [PostureMinder](https://sitapp.app/blog/best-posture-app) | 카메라 없음, 순수 타이머 | 주기적 텍스트 알림 | 카메라 미사용이라 해당 없음 |

**시사점**: SitApp·Posturr 등 최신 웹캠 앱들은 "온디바이스 처리·영상 미저장"을 이미 강하게 내세우고 있어, 이 부분만으로는 바른자세의 차별점이 크지 않다. 다만 이들 앱은 (1) 대부분 카메라 한 대의 휴리스틱 방식으로만 거리를 추정해 정밀한 삼각측량 기반 거리값을 얻지 못하고, (2) 화면 사용시간과 자세 악화 빈도의 상관관계 분석까지는 제공하지 않는다. 바른자세는 이미 소프트웨어(PC 상주 앱) 형태라는 점에서 이들과 같은 카테고리이며, 듀얼캠 스테레오 매칭으로 더 정밀한 거리 추정을 하고 디지털 디톡스 리포트로 상관관계 분석까지 다룬다는 점에서 차별화를 노린다.

---

## 3. 웨어러블 기기

| 기기 | 감지 방식 | 알림 방식 |
|---|---|---|
| [Upright GO / GO2 / GO S](https://www.uprightpose.com/how-it-works/) | 등이나 목에 부착하는 멀티센서 웨어러블 | 슬라우치 시 진동. "트레이닝 모드"(즉시 진동)와 "트래킹 모드"(통계만 기록) 구분 |

**한계**: 신체에 직접 부착해야 해서 착용 부담이 있고, 피부 접촉·탈부착 편의성 이슈가 꾸준히 제기된다. 카메라 기반 방식과 달리 셋업이 매번 필요하다.

---

## 4. 학술 선행 연구

가장 직접적으로 비교할 만한 선행 연구를 찾았다.

**[PoseTrack: An Intelligent Mobile Application to Monitor and Correct Sitting Posture Using Raspberry Pi and MediaPipe Pose Detection](https://arxiv.org/html/2508.11683) (2025)**

- **하드웨어**: 라즈베리파이5 + 카메라 모듈 + 스피커, 부팅 시 systemd로 자동 실행
- **소프트웨어**: MediaPipe Pose로 관절 좌표 추출 → 벡터 내적으로 관절 각도 계산 → 전방 기울임·구부정한 자세·다리 꼬임·발 위치 등을 규칙 기반으로 판정. 라즈베리파이가 Flask 서버로 동작하고, Flutter 앱이 결과를 받아 표시
- **데이터 처리**: 영상 프레임 자체는 저장하지 않고 base64로 인코딩해 HTTP로 전송, 분석은 라즈베리파이에서 로컬로 수행 — 이 점은 바른자세와 동일한 설계 방향
- **저장소**: 그러나 사용자 데이터·이력은 Firebase(Firestore) 클라우드에 저장 — **바른자세는 이 부분에서 명확히 차별화된다**: 별도 서버·클라우드 전송 없이 사용자 노트북 한 대 안에서 판정부터 리포트까지 전부 완결
- **정확도(실험 결과)**: 좋은 자세 판정 70%, 전방 기울임(구부정) 판정 100%, 다리 꼬임 판정 0%(책상에 가려 관절이 안 보여 실패), 다리 올림 판정 50%. 조명 조건별로는 70~100% 범위. 즉 **몸의 일부가 가려지는 상황에서 정확도가 크게 떨어진다**는 것이 확인됨 — 바른자세도 카메라 각도·고정 위치 검증(리스크 대응 항목)이 왜 중요한지 뒷받침하는 근거로 쓸 수 있다.

---

## 5. 바른자세의 위치 정리 (관련 연구 슬라이드용 결론)

1. **국내 모바일 앱**은 대부분 스마트폰 자체를 보는 자세나 타이머 알림에 그쳐, 책상 앞 상시 자세 모니터링과는 목적이 다르다.
2. **해외 웹캠 AI 앱**(SitApp, Posturr 등)은 온디바이스 처리·영상 미저장이라는 점에서 바른자세와 방향이 같지만, 대부분 카메라 한 대의 휴리스틱 방식으로만 거리를 추정하고 사용시간-자세 상관관계 분석 같은 "디지털 디톡스 리포트" 기능은 없다.
3. **웨어러블**(Upright GO)은 착용 부담이라는 근본적 한계가 있다.
4. **가장 유사한 선행 연구인 PoseTrack**(라즈베리파이+MediaPipe)은 아키텍처는 비슷하지만 결국 Firebase 클라우드에 데이터를 올린다는 점, 그리고 신체 일부가 가려지면 정확도가 급락한다는 실증 결과를 남겼다. 바른자세는 **별도 서버·클라우드 전송 없이 노트북 한 대 안에서 완결되는 구조**와 **듀얼캠 스테레오 매칭 기반의 정밀한 거리 추정**으로 이 지점을 보완한다.

이 조사 결과는 발표 자료(`barunjase_week1.pptx`) 6번 슬라이드 "관련 연구" 섹션과 참고문헌 목록에 반영해뒀다.

---

## 6. 최근 3년 이내(2023~2026) 관련 학술 논문

### 6-1. 카메라·키포인트 기반 자세 분류 (방법론 직접 참고)

| 논문 | 저자/venue/연도 | 관련성 |
|---|---|---|
| [PoseTrack: An Intelligent Mobile Application to Monitor and Correct Sitting Posture Using Raspberry Pi and MediaPipe Pose Detection](https://arxiv.org/html/2508.11683) | Hsieh & Sun, arXiv, 2025.08 | 라즈베리파이+카메라+MediaPipe 구성이 바른자세와 거의 동일. 앞으로 숙임 100%, 좋은 자세 70%, 다리 꼬임 0%(테이블에 가려짐) — 카메라 각도·가림 문제의 실증 근거 |
| [Classification Algorithm for Sitting Postures Using Weighted Random Forest](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ipr2.70126) | Lee et al., IET Image Processing, 2025 | 바른자세가 채택하려는 알고리즘(랜덤포레스트)을 그대로 사용해 착석 자세를 분류한 최신 논문 — 분류기 선택 근거로 직접 인용 가능 |
| [KeypointNet: An Efficient Deep Learning Model with Multi-View Recognition Capability for Sitting Posture Recognition](https://www.mdpi.com/2079-9292/14/4/718) | Cao et al., Electronics(MDPI), 2025 | 원본 영상이 아니라 **COCO 17개 키포인트 좌표만으로** 7가지 착석 자세를 96~99% 정확도로 분류 — "영상을 저장하지 않고 좌표만 처리해도 고정확도가 가능하다"는 바른자세의 설계 전제를 뒷받침하는 가장 강력한 근거 |
| [Sitting Posture Recognition Based on the Computer's Camera](https://dl.acm.org/doi/10.1145/3663976.3664014) | ACM ICCVIPPR, 2024 | 웹캠 하나로 착석 자세를 인식 — 바른자세와 센싱 구성이 가장 유사한 선행 연구 |

### 6-2. 인체공학 리스크 평가 자동화 (RULA 연계)

| 논문 | 저자/venue/연도 | 관련성 |
|---|---|---|
| [Ergonomic Risk Assessment Using Human Pose Estimation with MediaPipe Pose](https://dl.acm.org/doi/10.1145/3719384.3719453) | ACM AICCC, 2024 | MediaPipe 키포인트로 RULA류 인체공학 점수를 자동 산출 — 바른자세의 "판정 로직" 설계와 직접 연결되는 근거 |
| [Validation of computer vision-based ergonomic risk assessment tools for real manufacturing environments](https://www.nature.com/articles/s41598-024-79373-4) | Scientific Reports, 2024 | 실제 산업 현장에서 컴퓨터비전 기반 인체공학 평가 도구의 신뢰도를 검증 — RULA 자동화의 타당성 근거로 인용 가능 |
| [A review of machine learning techniques for ergonomic risk assessment based on human pose estimation](https://link.springer.com/article/10.1007/s44163-025-00566-5) | Discover Artificial Intelligence(Springer), 2025 | 포즈 추정 기반 인체공학 리스크 평가 연구 전체를 정리한 최신 리뷰 논문 — 서론·관련연구 작성 시 개관용으로 유용 |

### 6-3. 배경 지식용 서베이

- [A survey on deep learning for 2D and 3D human pose estimation](https://link.springer.com/article/10.1007/s10462-025-11430-4) — Artificial Intelligence Review(Springer), 2025. 포즈 추정 모델 전반(MoveNet·MediaPipe·YOLO Pose 계열 포함)을 정리한 최신 서베이.

**추천 우선순위**: 논문 관련연구 섹션에는 KeypointNet(2025, 좌표만으로 고정확도 — 프라이버시 설계 근거), Weighted Random Forest(2025, 분류기 선택 근거), PoseTrack(2025, 가장 유사한 하드웨어 구성 + 한계점)을 핵심으로 인용하고, 인체공학 자동화 3편은 RULA를 판정 기준으로 쓰는 이유를 뒷받침하는 근거로 함께 배치하는 것을 추천한다.
