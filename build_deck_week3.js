const pptxgen = require("pptxgenjs");
const fs = require("fs");

// ---------- palette (same as week1/week2 deck — dark navy + mint teal) ----------
const NAVY      = "0B1A26";
const TEAL      = "2FE0C4";
const TEAL_DARK = "12655A";
const CARD      = "FFFFFF";
const TXT_LIGHT = "EAF3F1";
const TXT_MUTED = "9FB6BC";
const TXT_DARK  = "16262E";
const WARN      = "E8734A";
const WARN_SOFT = "3A2A22";
const GOOD      = "3FD48C";

const FONT_HEAD = "Arial";
const FONT_BODY = "Calibri";

function img(path) {
  return "data:image/png;base64," + fs.readFileSync(path).toString("base64");
}
const ICON = {
  brain: img("icons/brain.png"), chip: img("icons/chip.png"), ruler: img("icons/ruler.png"),
  volume: img("icons/volume.png"), calendar: img("icons/calendar.png"), wallet: img("icons/wallet.png"),
  alert: img("icons/alert.png"), book: img("icons/book.png"), camera: img("icons/camera.png"),
  target: img("icons/target.png"), diagram: img("icons/diagram.png"), person: img("icons/person.png"),
  fan: img("icons/fan.png"),
};
const BG = "bg_dark.png";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
const PW = 13.33, PH = 7.5;

function newSlide() {
  const s = pres.addSlide();
  s.background = { path: BG };
  return s;
}
function iconCircle(slide, iconKey, x, y, d, circleColor) {
  slide.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: circleColor }, line: { type: "none" } });
  const pad = d * 0.26;
  slide.addImage({ data: ICON[iconKey], x: x + pad, y: y + pad, w: d - pad * 2, h: d - pad * 2 });
}
function kicker(slide, text) {
  slide.addText(text.toUpperCase(), {
    x: 0.7, y: 0.42, w: 9, h: 0.35, fontFace: FONT_BODY, fontSize: 12, bold: true,
    color: TEAL, charSpacing: 2, isTextBox: true, margin: 0,
  });
}
function pageTitle(slide, text, kickerText) {
  if (kickerText) kicker(slide, kickerText);
  slide.addText(text, {
    x: 0.7, y: kickerText ? 0.75 : 0.55, w: 11.9, h: 0.9, fontFace: FONT_HEAD, fontSize: 28, bold: true,
    color: TXT_LIGHT, isTextBox: true, margin: 0,
  });
}
function pageNum(slide, n) {
  slide.addText(String(n).padStart(2, "0"), {
    x: PW - 0.9, y: PH - 0.55, w: 0.6, h: 0.35, fontFace: FONT_BODY, fontSize: 11,
    color: TXT_MUTED, align: "right", isTextBox: true, margin: 0,
  });
}
function verifiedTag(slide, x, y, verified) {
  const w = verified ? 1.55 : 1.9;
  slide.addShape(pres.ShapeType.roundRect, {
    x, y, w, h: 0.34, rectRadius: 0.17,
    fill: { color: verified ? "12432F" : WARN_SOFT }, line: { type: "none" },
  });
  slide.addText(verified ? "✓ 원문 검증" : "참고용", {
    x, y, w, h: 0.34, align: "center", valign: "middle",
    fontFace: FONT_BODY, fontSize: 10.5, bold: true, color: verified ? GOOD : WARN, isTextBox: true, margin: 0,
  });
}

// =====================================================================================
// 1. TITLE
// =====================================================================================
{
  const s = newSlide();
  iconCircle(s, "brain", PW / 2 - 0.55, 0.68, 1.1, TEAL_DARK);
  s.addText("임베디드소프트웨어학과 · 캡스톤 프로젝트", {
    x: 1, y: 2.05, w: 11.33, h: 0.4, align: "center", fontFace: FONT_BODY, fontSize: 14,
    color: TEAL, bold: true, charSpacing: 1.5, isTextBox: true, margin: 0,
  });
  s.addText("자세 분류 알고리즘 비교 및 선정 근거", {
    x: 1, y: 2.5, w: 11.33, h: 1.05, align: "center", fontFace: FONT_HEAD, fontSize: 42,
    bold: true, color: TXT_LIGHT, isTextBox: true, margin: 0,
  });
  s.addText("바른자세 · 3주차 발표 — 공통주제화에 따른 알고리즘 차별화 전략", {
    x: 1, y: 3.6, w: 11.33, h: 0.55, align: "center", fontFace: FONT_BODY, fontSize: 17,
    color: TXT_MUTED, isTextBox: true, margin: 0,
  });
  s.addText("실행 사양 대비 성능(execution spec vs. performance) 기준으로 Random Forest 선정 근거를 문헌으로 검증한다", {
    x: 1.5, y: 4.25, w: 10.33, h: 0.4, align: "center", fontFace: FONT_BODY, fontSize: 12.5,
    italic: true, color: TEAL, isTextBox: true, margin: 0,
  });
  s.addShape(pres.ShapeType.roundRect, {
    x: PW / 2 - 1.9, y: 4.85, w: 3.8, h: 0.55, rectRadius: 0.28,
    fill: { color: TEAL_DARK }, line: { type: "none" },
  });
  s.addText("3주차 발표 · 관련 논문 조사", {
    x: PW / 2 - 1.9, y: 4.85, w: 3.8, h: 0.55, align: "center", valign: "middle",
    fontFace: FONT_BODY, fontSize: 13, bold: true, color: TXT_LIGHT, isTextBox: true, margin: 0,
  });
  s.addText("발표자: 기안 외 1인", {
    x: 1, y: PH - 0.9, w: 11.33, h: 0.4, align: "center", fontFace: FONT_BODY, fontSize: 12,
    color: TXT_MUTED, isTextBox: true, margin: 0,
  });
}

// =====================================================================================
// 2. CONTENTS
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "목차", "Contents");
  const items = [
    "01   문제 제기 · 차별화 전략", "02   프로젝트 구조 · 연구 목표", "03   포즈 추정 모델 비교",
    "04   분류기 비교: 핵심 근거", "05   가장 가까운 선행 연구", "06   최신 관련 사례",
    "07   RF·XGBoost·LightGBM 교차비교", "08   MLP(딥러닝) 비교 후보", "09   최종 연구 방향 · 결론", "10   참고문헌",
  ];
  const colW = 5.55, gapX = 0.5, startX = 0.7, startY = 1.9, rowH = 0.78;
  const perCol = Math.ceil(items.length / 2);
  items.forEach((t, i) => {
    const col = Math.floor(i / perCol);
    const row = i % perCol;
    const x = startX + col * (colW + gapX);
    const y = startY + row * rowH;
    s.addShape(pres.ShapeType.roundRect, {
      x, y, w: colW, h: 0.56, rectRadius: 0.1, fill: { color: TEAL_DARK }, line: { type: "none" },
    });
    s.addText(t, {
      x: x + 0.25, y, w: colW - 0.4, h: 0.56, valign: "middle", fontFace: FONT_BODY,
      fontSize: 14, bold: true, color: TXT_LIGHT, isTextBox: true, margin: 0,
    });
  });
  pageNum(s, 2);
}

// =====================================================================================
// 3. PROBLEM STATEMENT / DIFFERENTIATION STRATEGY
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "문제 제기: 왜 알고리즘으로 차별화해야 하는가", "Problem Statement");
  s.addText(
    "이번 학기 다수 팀이 듀얼캠 기반 책상 자세 모니터링이라는 유사한 주제를 채택하면서, 하드웨어 구성만으로는 더 이상 차별화가 어려워졌다. " +
    "그래서 차별화 지점을 하드웨어에서 자세 추정 알고리즘 자체로 옮긴다.",
    { x: 0.7, y: 1.8, w: 11.93, h: 1.0, fontFace: FONT_BODY, fontSize: 15.5, color: TXT_LIGHT, isTextBox: true, margin: 0, lineSpacingMultiple: 1.25 }
  );

  const cards = [
    { icon: "alert", title: "주제 공통화", desc: "듀얼캠·책상 자세 모니터링이 다수 조의 공통 주제가 됨 — 하드웨어 구성만으로는 차별화 불가" },
    { icon: "target", title: "차별화 지점 재설정", desc: "\"어떤 포즈 추정 모델 + 어떤 분류기를 쓰느냐\"가 실사용성(백그라운드 상시 구동)을 가른다" },
    { icon: "ruler", title: "평가 기준", desc: "정확도 극대화가 아니라 \"실행 사양 대비 성능\" — 저사양 노트북에서 다른 작업과 동시 구동 가능한가" },
    { icon: "book", title: "이번 발표 목표", desc: "관련 논문을 조사해 Random Forest 선정 근거와 개선 방향 아이디어를 문헌으로 뒷받침" },
  ];
  const cardW = 2.75, gap = 0.3, startX = 0.7, y = 3.15, h = 3.05;
  cards.forEach((c, i) => {
    const x = startX + i * (cardW + gap);
    s.addShape(pres.ShapeType.roundRect, { x, y, w: cardW, h, rectRadius: 0.12, fill: { color: CARD }, line: { type: "none" } });
    iconCircle(s, c.icon, x + 0.3, y + 0.3, 0.7, TEAL_DARK);
    s.addText(c.title, { x: x + 0.3, y: y + 1.15, w: cardW - 0.6, h: 0.6, fontFace: FONT_HEAD, fontSize: 14.5, bold: true, color: TXT_DARK, isTextBox: true, margin: 0 });
    s.addText(c.desc, { x: x + 0.3, y: y + 1.75, w: cardW - 0.6, h: h - 1.95, fontFace: FONT_BODY, fontSize: 11.5, color: "3C4A50", isTextBox: true, margin: 0, lineSpacingMultiple: 1.15 });
  });
  pageNum(s, 3);
}

// =====================================================================================
// 4. PROJECT STRUCTURE + RESEARCH GOAL
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "프로젝트 구조와 연구 목표", "Structure & Goal");

  // pipeline diagram
  const steps = ["스테레오\n카메라", "캘리브레이션\n· 깊이 정보", "MoveNet\nLightning", "17개\nKeypoint", "특징 추출", "자세 분류", "정상/주의\n/경고"];
  const n = steps.length, gap = 0.14;
  const totalGap = gap * (n - 1);
  const boxW = (11.93 - totalGap) / n, boxY = 1.9, boxH = 1.15;
  steps.forEach((t, i) => {
    const x = 0.7 + i * (boxW + gap);
    const isLast = i === n - 1;
    s.addShape(pres.ShapeType.roundRect, { x, y: boxY, w: boxW, h: boxH, rectRadius: 0.08, fill: { color: isLast ? TEAL_DARK : CARD }, line: { type: "none" } });
    s.addText(t, { x: x + 0.05, y: boxY, w: boxW - 0.1, h: boxH, align: "center", valign: "middle", fontFace: FONT_BODY, fontSize: 10.5, bold: true, color: isLast ? TXT_LIGHT : TXT_DARK, isTextBox: true, margin: 0, lineSpacingMultiple: 1.05 });
    if (i < n - 1) {
      s.addShape(pres.ShapeType.line, { x: x + boxW, y: boxY + boxH / 2, w: gap, h: 0.001, line: { color: TEAL, width: 2, endArrowType: "triangle" } });
    }
  });

  s.addText("연구 목표", {
    x: 0.7, y: 3.5, w: 6, h: 0.4, fontFace: FONT_HEAD, fontSize: 16, bold: true, color: TEAL, isTextBox: true, margin: 0,
  });
  s.addText(
    "단순히 가장 정확한 알고리즘을 선택하는 것이 아니라, 기존 자세 분류 알고리즘을 적용한 뒤 성능과 연산 효율을 분석하고, " +
    "적절한 알고리즘을 개선하여 기존 모델과 개선 모델을 비교하는 것을 목표로 한다.",
    { x: 0.7, y: 3.95, w: 11.93, h: 0.95, fontFace: FONT_BODY, fontSize: 15, color: TXT_LIGHT, isTextBox: true, margin: 0, lineSpacingMultiple: 1.25 }
  );

  const notes = [
    { icon: "chip", t: "MoveNet Lightning", d: "이미 잠정 채택 — 17개 keypoint 기반 포즈 추정" },
    { icon: "diagram", t: "Random Forest", d: "관절 좌표·각도·거리 특징 분류에 잠정 채택" },
    { icon: "target", t: "이번 조사의 역할", d: "이 조합이 \"저사양 노트북 백그라운드 상시 구동\" 조건에 맞는지 문헌으로 검증" },
  ];
  const cardW2 = 3.85, gap2 = 0.19, startX2 = 0.7, y2 = 5.15, h2 = 1.55;
  notes.forEach((nn, i) => {
    const x = startX2 + i * (cardW2 + gap2);
    s.addShape(pres.ShapeType.roundRect, { x, y: y2, w: cardW2, h: h2, rectRadius: 0.1, fill: { color: CARD }, line: { type: "none" } });
    iconCircle(s, nn.icon, x + 0.22, y2 + 0.25, 0.6, TEAL_DARK);
    s.addText(nn.t, { x: x + 0.98, y: y2 + 0.18, w: cardW2 - 1.15, h: 0.4, fontFace: FONT_HEAD, fontSize: 13, bold: true, color: TXT_DARK, isTextBox: true, margin: 0 });
    s.addText(nn.d, { x: x + 0.98, y: y2 + 0.58, w: cardW2 - 1.15, h: h2 - 0.7, fontFace: FONT_BODY, fontSize: 10.5, color: "3C4A50", isTextBox: true, margin: 0, lineSpacingMultiple: 1.1 });
  });
  pageNum(s, 4);
}

// =====================================================================================
// 5. POSE ESTIMATION MODEL COMPARISON
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "포즈 추정 모델 비교: 정확도 vs 속도", "Pose Estimation Models");

  const headerOpts = { fill: { color: TEAL_DARK }, color: TXT_LIGHT, bold: true, fontFace: FONT_BODY, fontSize: 12, valign: "middle" };
  const cellOpts = { fill: { color: CARD }, color: TXT_DARK, fontFace: FONT_BODY, fontSize: 11.5, valign: "middle" };
  const mnOpts = { ...cellOpts, fill: { color: "E9FBF7" }, bold: true, color: TEAL_DARK };
  const rows = [
    [{ text: "모델", options: headerOpts }, { text: "정확도", options: headerOpts }, { text: "1,000장 처리시간", options: headerOpts }, { text: "Keypoint 수", options: headerOpts }],
    [{ text: "PoseNet", options: cellOpts }, { text: "97.6% (최고)", options: cellOpts }, { text: "75.2초", options: cellOpts }, { text: "17개", options: cellOpts }],
    [{ text: "OpenPose", options: cellOpts }, { text: "86.2%", options: cellOpts }, { text: "643.5초 (최장)", options: cellOpts }, { text: "137개", options: cellOpts }],
    [{ text: "MoveNet Thunder", options: cellOpts }, { text: "80.6%", options: cellOpts }, { text: "140.0초", options: cellOpts }, { text: "17개", options: cellOpts }],
    [{ text: "MoveNet Lightning ★", options: mnOpts }, { text: "75.1%", options: mnOpts }, { text: "53.5초 (최속)", options: mnOpts }, { text: "17개", options: mnOpts }],
  ];
  s.addTable(rows, { x: 0.7, y: 1.9, w: 7.3, h: 2.55, colW: [2.55, 1.65, 2.1, 1.0], border: { type: "solid", color: "0B1A26", pt: 1.5 }, autoPage: false });
  s.addText("③ Jo & Kim, Traitement du Signal, 2022 (IIETA) — 원문 PDF 직접 확인 완료", {
    x: 0.7, y: 4.55, w: 7.3, h: 0.35, fontFace: FONT_BODY, fontSize: 10.5, italic: true, color: TXT_MUTED, isTextBox: true, margin: 0,
  });
  s.addText("MoveNet Lightning은 OpenPose 대비 약 12배 빠르며, \"사람이 없는 이미지\"에서도 낮은 오탐률 유지 (3그룹 기준 71.5%로 하락하는 구간은 참고 필요)", {
    x: 0.7, y: 5.45, w: 7.3, h: 0.9, fontFace: FONT_BODY, fontSize: 11.5, color: TXT_LIGHT, isTextBox: true, margin: 0, lineSpacingMultiple: 1.2,
  });

  // right column: selection rationale summary
  s.addShape(pres.ShapeType.roundRect, { x: 8.35, y: 1.9, w: 4.28, h: 4.45, rectRadius: 0.12, fill: { color: TEAL_DARK }, line: { type: "none" } });
  s.addText("MoveNet Lightning을 고른 이유", { x: 8.6, y: 2.15, w: 3.8, h: 0.4, fontFace: FONT_HEAD, fontSize: 14, bold: true, color: TXT_LIGHT, isTextBox: true, margin: 0 });

  s.addText("12×", { x: 8.6, y: 2.65, w: 3.8, h: 1.0, align: "center", fontFace: FONT_HEAD, fontSize: 60, bold: true, color: TEAL, isTextBox: true, margin: 0 });
  s.addText("OpenPose 대비 처리 속도 (IIETA 2022)", { x: 8.6, y: 3.65, w: 3.8, h: 0.35, align: "center", fontFace: FONT_BODY, fontSize: 11, color: TXT_MUTED, isTextBox: true, margin: 0 });

  const rationale = [
    { icon: "chip", t: "저사양 CPU에서 5초 주기 백그라운드 상시 구동에 적합한 속도" },
    { icon: "book", t: "TFLite 등 성숙한 온디바이스 배포 생태계 — 사전학습 모델 바로 활용 가능" },
  ];
  rationale.forEach((r, i) => {
    const ry = 4.35 + i * 1.05;
    iconCircle(s, r.icon, 8.6, ry, 0.55, "0E4A42");
    s.addText(r.t, { x: 9.3, y: ry, w: 3.1, h: 0.9, fontFace: FONT_BODY, fontSize: 11.5, color: TXT_LIGHT, isTextBox: true, margin: 0, lineSpacingMultiple: 1.2, valign: "middle" });
  });
  pageNum(s, 5);
}

// =====================================================================================
// 6. CLASSIFIER COMPARISON — CORE EVIDENCE
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "분류기 비교: 핵심 근거", "Classifier Evidence");

  // Reality Check big callout
  s.addShape(pres.ShapeType.roundRect, { x: 0.7, y: 1.75, w: 11.93, h: 2.75, rectRadius: 0.12, fill: { color: TEAL_DARK }, line: { type: "none" } });
  s.addText("① Toward Real-Time Posture Classification: Reality Check", { x: 1.0, y: 1.93, w: 9, h: 0.4, fontFace: FONT_HEAD, fontSize: 15.5, bold: true, color: TXT_LIGHT, isTextBox: true, margin: 0 });
  s.addText("Zhang et al., Electronics(MDPI), 2025", { x: 1.0, y: 2.33, w: 9, h: 0.3, fontFace: FONT_BODY, fontSize: 11.5, color: TXT_MUTED, isTextBox: true, margin: 0 });
  s.addText("목적: 실시간 자세 분류에 값비싼 딥러닝이 정말 필요한지 CPU 단독 환경에서 검증하기 위해 고전 ML과 LSTM(딥러닝)의 속도·정확도를 비교한 논문 — 그 결과가 아래 수치다.", {
    x: 1.0, y: 2.65, w: 11.3, h: 0.45, fontFace: FONT_BODY, fontSize: 11, italic: true, color: "C9F4EA", isTextBox: true, margin: 0, lineSpacingMultiple: 1.15,
  });
  s.addText([
    { text: "고전 ML(SVM·Gaussian NB·Random Forest·Neural Network)이 CPU 단독 비교에서 ", options: { fontSize: 13 } },
    { text: "가장 느린 고전 분류기(AdaBoost)조차 LSTM보다 약 20배 빠르면서", options: { fontSize: 13, bold: true, color: TEAL } },
    { text: "도, 최대 99% 정확도를 낼 수 있음을 원문(Table 4)으로 확인함.", options: { fontSize: 13 } },
  ], { x: 1.0, y: 3.15, w: 11.3, h: 0.75, fontFace: FONT_BODY, color: TXT_LIGHT, isTextBox: true, margin: 0, lineSpacingMultiple: 1.25 });
  s.addText("주의: 이 논문의 속도 비교(Fig.10)에 RF는 직접 포함되지 않음 — \"RF가 20배\"가 아니라 \"고전 ML 계열이 20배\"로 정확히 인용", {
    x: 1.0, y: 3.95, w: 11.3, h: 0.4, fontFace: FONT_BODY, fontSize: 11, italic: true, color: "C9F4EA", isTextBox: true, margin: 0,
  });

  // two supplementary cards — RF vs SVM은 데이터·모달리티마다 순위가 다르다는 것을 보여주는 대조 사례
  const cards = [
    {
      title: "⑤ EAI (2021) — 관절 각도 특징 비교 (Kinect 관절각도 기반)",
      body: "10-fold 교차검증: SVM(RBF) 83.42% > Logistic Reg. 80.91% > RF 77.08% > kNN 76.98% > Naive Bayes 74.56%. RF는 중위권 — \"RF가 항상 최고\"라고 과장하지 않는다.",
    },
    {
      title: "⑥ 휠체어 압력센서 (Sensors, 2021) — 반대 양상",
      body: "압력센서 데이터 기준 RF 98.65% > SVM 95.96%로 역전되지만 속도는 SVM 0.051초 < RF 0.167초로 SVM이 더 빠르다 — 정확도·속도 우위는 데이터·모달리티마다 달라진다.",
    },
  ];
  const cw = 5.85, gap = 0.23, y2 = 4.7, h2 = 2.15;
  cards.forEach((c, i) => {
    const x = 0.7 + i * (cw + gap);
    s.addShape(pres.ShapeType.roundRect, { x, y: y2, w: cw, h: h2, rectRadius: 0.12, fill: { color: CARD }, line: { type: "none" } });
    s.addText(c.title, { x: x + 0.3, y: y2 + 0.2, w: cw - 0.6, h: 0.6, fontFace: FONT_HEAD, fontSize: 12.5, bold: true, color: TXT_DARK, isTextBox: true, margin: 0, lineSpacingMultiple: 1.1 });
    s.addText(c.body, { x: x + 0.3, y: y2 + 0.85, w: cw - 0.6, h: h2 - 1.05, fontFace: FONT_BODY, fontSize: 11, color: "3C4A50", isTextBox: true, margin: 0, lineSpacingMultiple: 1.2 });
  });
  pageNum(s, 6);
}

// =====================================================================================
// 7. CLOSEST PRIOR WORK — WEIGHTED RF DEEP DIVE
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "가장 가까운 선행 연구: Weighted Random Forest", "Closest Prior Work");
  s.addText("② Lee, Choi, Kim, \"Classification Algorithm for Sitting Postures Using Weighted Random Forest\", IET Image Processing, 2025", {
    x: 0.7, y: 1.65, w: 10, h: 0.35, fontFace: FONT_BODY, fontSize: 12, color: TXT_MUTED, isTextBox: true, margin: 0,
  });

  // left: process + result
  s.addText("프로세스", { x: 0.7, y: 2.15, w: 5.6, h: 0.35, fontFace: FONT_HEAD, fontSize: 14, bold: true, color: TEAL, isTextBox: true, margin: 0 });
  s.addText("카메라 입력 → 얼굴·어깨 좌표/각도 추출 → 특징 중요도 분석 → 특징별 가중치 적용 → Weighted Random Forest → 5가지 자세 분류(정상/숙임/기대기/좌우 기울임)", {
    x: 0.7, y: 2.55, w: 5.6, h: 1.15, fontFace: FONT_BODY, fontSize: 12, color: TXT_LIGHT, isTextBox: true, margin: 0, lineSpacingMultiple: 1.2,
  });
  s.addShape(pres.ShapeType.roundRect, { x: 0.7, y: 3.85, w: 5.6, h: 1.3, rectRadius: 0.1, fill: { color: TEAL_DARK }, line: { type: "none" } });
  s.addText("5가지 착석 자세(정상/숙임/기대기/좌우 기울임) 분류 정확도·F1", { x: 0.9, y: 3.97, w: 5.2, h: 0.3, fontFace: FONT_BODY, fontSize: 10.5, color: "C9F4EA", isTextBox: true, margin: 0 });
  s.addText("0.983", { x: 0.9, y: 4.27, w: 5.2, h: 0.5, fontFace: FONT_HEAD, fontSize: 26, bold: true, color: TXT_LIGHT, isTextBox: true, margin: 0 });
  s.addText("가중치 없는 RF 96%→WRF 98% · 8개 분류기 중 최고, 최저 KNN(0.81) 대비 +0.17", { x: 0.9, y: 4.77, w: 5.2, h: 0.35, fontFace: FONT_BODY, fontSize: 10, color: "C9F4EA", isTextBox: true, margin: 0 });

  s.addShape(pres.ShapeType.roundRect, { x: 0.7, y: 5.35, w: 5.6, h: 1.5, rectRadius: 0.1, fill: { color: CARD }, line: { type: "none" } });
  s.addText("발표 포인트: 저자가 결론에서 직접 제안한 \"후속 연구 방향 — depth-sensing camera로 3D 데이터 수집\"이 바른자세의 스테레오 카메라 설계와 정확히 일치한다. \"선행 연구가 미래 과제로 남긴 것을 우리는 구현했다\"는 스토리로 활용.", {
    x: 0.9, y: 5.5, w: 5.2, h: 1.2, fontFace: FONT_BODY, fontSize: 11, italic: true, color: "3C4A50", isTextBox: true, margin: 0, lineSpacingMultiple: 1.2,
  });

  // right: comparison table
  s.addText("선행 논문 vs 우리 프로젝트", { x: 6.65, y: 2.15, w: 6, h: 0.35, fontFace: FONT_HEAD, fontSize: 14, bold: true, color: TEAL, isTextBox: true, margin: 0 });
  const headerOpts = { fill: { color: TEAL_DARK }, color: TXT_LIGHT, bold: true, fontFace: FONT_BODY, fontSize: 11, valign: "middle" };
  const cellOpts = { fill: { color: CARD }, color: TXT_DARK, fontFace: FONT_BODY, fontSize: 10.5, valign: "middle" };
  const rows = [
    [{ text: "구분", options: headerOpts }, { text: "선행 논문", options: headerOpts }, { text: "바른자세", options: headerOpts }],
    [{ text: "카메라", options: cellOpts }, { text: "단일 전면 2D", options: cellOpts }, { text: "스테레오 Depth", options: cellOpts }],
    [{ text: "특징 추출", options: cellOpts }, { text: "얼굴·어깨 중심 좌표", options: cellOpts }, { text: "전신 17 Keypoint", options: cellOpts }],
    [{ text: "데이터 차원", options: cellOpts }, { text: "2D 픽셀 좌표", options: cellOpts }, { text: "3D 좌표+Depth", options: cellOpts }],
    [{ text: "개선 방식", options: cellOpts }, { text: "Feature Weighting", options: cellOpts }, { text: "Selection+Weighting\n+HP 최적화", options: cellOpts }],
    [{ text: "오분류 원인", options: cellOpts }, { text: "옷·배경색 유사, 얼굴 기울임", options: cellOpts }, { text: "전신 특징+깊이로\n원천적 완화 가설", options: cellOpts }],
  ];
  s.addTable(rows, { x: 6.65, y: 2.55, w: 5.98, h: 4.1, colW: [1.5, 2.24, 2.24], border: { type: "solid", color: "0B1A26", pt: 1 }, autoPage: false });
  pageNum(s, 7);
}

// =====================================================================================
// 8. RECENT RELATED CASE — XGBOOST PAPER
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "최신 관련 사례: MoveNet 기반 자세 분류", "Recent Related Case");
  s.addText("④ Pawitra et al., \"Automated ergonomic sitting postures detection for office workstation using XGBoost method\", IAES Int. J. Artificial Intelligence, Vol.15 No.1, 2026.02", {
    x: 0.7, y: 1.7, w: 10.8, h: 0.55, fontFace: FONT_BODY, fontSize: 12, color: TXT_MUTED, isTextBox: true, margin: 0, lineSpacingMultiple: 1.15,
  });

  s.addText(
    "MoveNet Thunder로 17개 keypoint 추출 → 상반신만 정규화 → AdaBoost / XGBoost / MLP 비교, 정상/비정상 2분류 (참가자 30명, 유효 인스턴스 1,550개).",
    { x: 0.7, y: 2.4, w: 11.93, h: 0.6, fontFace: FONT_BODY, fontSize: 13.5, color: TXT_LIGHT, isTextBox: true, margin: 0, lineSpacingMultiple: 1.2 }
  );
  s.addText("↓ 아래 표: 정상/비정상 착석 자세 2분류 정확도·F1·ROC-AUC (참가자 30명 · 유효 인스턴스 1,550개, 10-fold 교차검증 평균)", {
    x: 0.7, y: 3.05, w: 11.93, h: 0.3, fontFace: FONT_BODY, fontSize: 10.5, italic: true, color: TXT_MUTED, isTextBox: true, margin: 0,
  });

  const headerOpts = { fill: { color: TEAL_DARK }, color: TXT_LIGHT, bold: true, fontFace: FONT_BODY, fontSize: 12, valign: "middle", align: "center" };
  const cellOpts = { fill: { color: CARD }, color: TXT_DARK, fontFace: FONT_BODY, fontSize: 11.5, valign: "middle", align: "center" };
  const bestOpts = { ...cellOpts, fill: { color: "E9FBF7" }, bold: true, color: TEAL_DARK };
  const rows = [
    [{ text: "분류기", options: headerOpts }, { text: "Accuracy", options: headerOpts }, { text: "F1", options: headerOpts }, { text: "ROC-AUC", options: headerOpts }, { text: "비고", options: headerOpts }],
    [{ text: "AdaBoost", options: cellOpts }, { text: "0.894 ± 0.021", options: cellOpts }, { text: "0.889 ± 0.022", options: cellOpts }, { text: "0.950 ± 0.016", options: cellOpts }, { text: "", options: cellOpts }],
    [{ text: "MLP", options: cellOpts }, { text: "0.918 ± 0.221", options: cellOpts }, { text: "0.915 ± 0.024", options: cellOpts }, { text: "0.960 ± 0.011", options: cellOpts }, { text: "폴드 간 편차 매우 큼 — 불안정", options: { ...cellOpts, color: WARN, align: "left" } }],
    [{ text: "XGBoost", options: bestOpts }, { text: "0.930 ± 0.019", options: bestOpts }, { text: "0.929 ± 0.019", options: bestOpts }, { text: "0.974 ± 0.010", options: bestOpts }, { text: "최고 · 가장 안정적", options: { ...bestOpts, align: "left" } }],
  ];
  s.addTable(rows, { x: 0.7, y: 3.4, w: 11.93, h: 1.7, colW: [2.3, 2.6, 1.7, 2.2, 3.13], border: { type: "solid", color: "0B1A26", pt: 1 }, autoPage: false });

  s.addShape(pres.ShapeType.roundRect, { x: 0.7, y: 5.3, w: 11.93, h: 1.5, rectRadius: 0.1, fill: { color: TEAL_DARK }, line: { type: "none" } });
  s.addText([
    { text: "우리에게 남겨진 갭:  ", options: { bold: true, fontSize: 13.5, color: TEAL } },
    { text: "이 논문은 우리와 같은 MoveNet 계열(Thunder)을 실제 자세 분류에 쓴 2026년 최신 사례지만, 비교군에 Random Forest가 없다(AdaBoost/XGBoost/MLP만 비교). ", options: { fontSize: 13, color: TXT_LIGHT } },
    { text: "우리가 여기에 RF/WRF를 추가로 비교하면 이 논문의 공백을 메우는 기여가 된다. 저자도 \"상반신 키포인트만 사용, 시점·가림·조명에 민감\"을 한계로 명시 — 전신 17키포인트+스테레오 깊이로 보완 가능하다는 가설 제시.", options: { fontSize: 12.5, color: TXT_LIGHT } },
  ], { x: 0.95, y: 5.45, w: 11.4, h: 1.2, fontFace: FONT_BODY, isTextBox: true, margin: 0, lineSpacingMultiple: 1.22 });
  pageNum(s, 8);
}

// =====================================================================================
// 8-1. RF vs XGBOOST vs LIGHTGBM — CROSS-PAPER HONEST COMPARISON
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "RF·XGBoost·LightGBM: 선행연구 교차비교", "Cross-Paper Comparison");
  s.addText("팀원이 조사한 3편을 표로 나란히 비교한다 — RF가 항상 1위는 아니므로 정직하게 비교한다.", {
    x: 0.7, y: 1.6, w: 11.93, h: 0.4, fontFace: FONT_BODY, fontSize: 13, italic: true, color: TXT_MUTED, isTextBox: true, margin: 0,
  });

  const headerOpts9 = { fill: { color: TEAL_DARK }, color: TXT_LIGHT, bold: true, fontFace: FONT_BODY, fontSize: 11, valign: "middle" };
  const cellOpts9 = { fill: { color: CARD }, color: TXT_DARK, fontFace: FONT_BODY, fontSize: 10, valign: "middle" };
  const algoOpts9 = { ...cellOpts9, bold: true, color: TEAL_DARK };
  const rows9 = [
    [{ text: "알고리즘", options: headerOpts9 }, { text: "선행연구 입력", options: headerOpts9 }, { text: "주요 방법", options: headerOpts9 }, { text: "주요 결과/의의", options: headerOpts9 }, { text: "본 프로젝트 적용", options: headerOpts9 }],
    [{ text: "② Random Forest", options: algoOpts9 }, { text: "얼굴·어깨 좌표/각도/비율", options: cellOpts9 }, { text: "변수중요도 기반 Weighted RF", options: cellOpts9 }, { text: "5개 자세 분류 RF 98%", options: cellOpts9 }, { text: "기준모델 + Feature Selection·Weighting 개선", options: cellOpts9 }],
    [{ text: "④ XGBoost", options: algoOpts9 }, { text: "MoveNet 17 Keypoint", options: cellOpts9 }, { text: "정규화·중심화 + XGBoost", options: cellOpts9 }, { text: "Accuracy 93.0%±1.9%, F1 92.9%", options: cellOpts9 }, { text: "MoveNet 기반 구조 유사", options: cellOpts9 }],
    [{ text: "⑥ LightGBM", options: algoOpts9 }, { text: "16개 압력센서 수치", options: cellOpts9 }, { text: "KNN·SVM·RF·DT·LightGBM 비교", options: cellOpts9 }, { text: "실시간 모니터링 가능성 및 경량성 비교", options: cellOpts9 }, { text: "정확도+추론시간+CPU·RAM 비교", options: cellOpts9 }],
  ];
  s.addTable(rows9, { x: 0.7, y: 2.15, w: 11.93, h: 2.65, colW: [1.9, 2.3, 2.53, 2.6, 2.6], border: { type: "solid", color: "0B1A26", pt: 1 }, autoPage: false });

  s.addText("핵심: WRF는 RF 계열이 가장 높고(0.98·1위), XGBoost 논문엔 애초 RF 비교군이 없으며, LightGBM 압력센서 사례는 LightGBM이 근소 우위(99.03%>98.65%) — 논문마다 RF의 순위가 다르다.", {
    x: 0.7, y: 4.95, w: 11.93, h: 0.5, fontFace: FONT_BODY, fontSize: 11.5, italic: true, color: TXT_MUTED, isTextBox: true, margin: 0, lineSpacingMultiple: 1.2,
  });

  s.addShape(pres.ShapeType.roundRect, { x: 0.7, y: 5.6, w: 11.93, h: 1.3, rectRadius: 0.12, fill: { color: TEAL_DARK }, line: { type: "none" } });
  s.addText([
    { text: "결론:  ", options: { bold: true, fontSize: 13.5, color: TEAL } },
    { text: "\"RF가 절대적으로 가장 빠르고 정확하다\"고 주장하지 않는다. 대신 RF 성능을 객관적으로 검증한 뒤, ", options: { fontSize: 12.5, color: TXT_LIGHT } },
    { text: "동일 데이터(MoveNet Keypoint+깊이)로 XGBoost·LightGBM과 직접 실측 비교해 최적화하는 것을 다음 단계 목표로 삼는다.", options: { fontSize: 12.5, bold: true, color: TXT_LIGHT } },
  ], { x: 0.95, y: 5.75, w: 11.4, h: 1.05, fontFace: FONT_BODY, isTextBox: true, margin: 0, lineSpacingMultiple: 1.22 });
  pageNum(s, 9);
}

// =====================================================================================
// 8-2. MLP (DEEP LEARNING) COMPARISON CANDIDATE
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "MLP(딥러닝) 비교 후보 추가", "Deep Learning Candidate");
  s.addText("RF·XGBoost·LightGBM은 모두 트리 기반 ML — 팀원이 딥러닝 비교 축(MLP)을 추가 조사, 원문에서 수치까지 검증", {
    x: 0.7, y: 1.6, w: 11.93, h: 0.35, fontFace: FONT_BODY, fontSize: 12.5, italic: true, color: TXT_MUTED, isTextBox: true, margin: 0,
  });

  s.addShape(pres.ShapeType.roundRect, { x: 0.7, y: 2.0, w: 11.93, h: 1.2, rectRadius: 0.12, fill: { color: TEAL_DARK }, line: { type: "none" } });
  s.addText([
    { text: "Reality Check와 배치되지 않는 이유:  ", options: { bold: true, fontSize: 12.5, color: TEAL } },
    { text: "MLP는 LSTM과 달리 순환 구조가 없는 경량 피드포워드 신경망이다 (ConMLP 46.54M FLOPs vs GCN 2.5G FLOPs, Yoga MLP는 12→10→8→6의 초소형 구조) — \"비싼 딥러닝은 불필요\"라는 결론과 배치되지 않는다. ", options: { fontSize: 11.5, color: TXT_LIGHT } },
    { text: "다만 RF와 CPU 상에서 직접 비교한 검증은 아직 없어, 최종 모델로 미리 정하지 않고 RF·XGBoost·LightGBM과 동일 조건에서 공정 비교할 딥러닝 기준 모델로 추가한다.", options: { fontSize: 11.5, color: TXT_LIGHT } },
  ], { x: 0.9, y: 2.13, w: 11.5, h: 0.95, fontFace: FONT_BODY, isTextBox: true, margin: 0, lineSpacingMultiple: 1.2 });

  const mlpCards = [
    { code: "⑦", title: "Yoga Pose MLP (CIN, 2022)", body: "관절각도 12개 → MLP(은닉 10·8) → 요가자세 6종. 테스트 정확도 0.9958 (350개 인스턴스). 우리와 구조가 가장 유사" },
    { code: "⑧", title: "KeypointNet (Electronics, 2025)", body: "지역+전역 Keypoint 특징을 MLP로 결합해 분류. MVSP 98.46%/HASP 96.52%/MSRA 99.48%. MLP는 독립분류기가 아닌 구성요소" },
    { code: "⑨", title: "YOLOv8+Graph MLP (Discover AI, 2026)", body: "Graph MLP 도입으로 추론시간 0.23초→0.08초, FPS 58→125로 개선. 향후 확장연구(Graph MLP/GCN) 방향의 근거" },
    { code: "⑩", title: "ConMLP (Sensors, 2023)", body: "경량 MLP 자기지도학습으로 NTU RGB+D 96.9%, 46.54M FLOPs — GCN 계열(2.5G FLOPs)보다 훨씬 가벼움. 배경자료용" },
  ];
  const rowH = 0.78, rowGap = 0.1, rowY0 = 3.35;
  mlpCards.forEach((c, i) => {
    const y = rowY0 + i * (rowH + rowGap);
    s.addShape(pres.ShapeType.roundRect, { x: 0.7, y, w: 11.93, h: rowH, rectRadius: 0.1, fill: { color: CARD }, line: { type: "none" } });
    s.addShape(pres.ShapeType.roundRect, { x: 0.9, y: y + (rowH - 0.4) / 2, w: 0.42, h: 0.4, rectRadius: 0.07, fill: { color: "E9FBF7" }, line: { type: "none" } });
    s.addText(c.code, { x: 0.9, y: y + (rowH - 0.4) / 2, w: 0.42, h: 0.4, align: "center", valign: "middle", fontFace: FONT_BODY, fontSize: 12, bold: true, color: TEAL_DARK, isTextBox: true, margin: 0 });
    s.addText(c.title, { x: 1.5, y: y + 0.08, w: 3.6, h: rowH - 0.16, valign: "middle", fontFace: FONT_HEAD, fontSize: 11.5, bold: true, color: TXT_DARK, isTextBox: true, margin: 0, lineSpacingMultiple: 1.1 });
    s.addText(c.body, { x: 5.2, y: y + 0.08, w: 7.2, h: rowH - 0.16, valign: "middle", fontFace: FONT_BODY, fontSize: 10.3, color: "3C4A50", isTextBox: true, margin: 0, lineSpacingMultiple: 1.15 });
  });
  pageNum(s, 10);
}

// =====================================================================================
// 9. CONCLUSION
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "결론: 우리의 차별화 지점", "Conclusion");

  const concl = [
    { code: "③", text: "MoveNet Lightning은 정확도 1등은 아니지만(75%대) OpenPose 대비 12배 빠르고 저사양 환경에 적합 — 우리 시나리오(1인 고정)엔 속도가 더 중요" },
    { code: "①", text: "고전 ML(RF 포함 계열)은 LSTM 대비 CPU에서 약 20배 빠르면서 최대 99% 정확도 가능 — 딥러닝 없이도 충분한 실시간 성능 확보" },
    { code: "②", text: "Weighted RF 선행 연구가 남긴 \"3D 깊이 카메라\" 과제를 스테레오 카메라로 이미 구현 중 — 가장 가까운 선행 연구의 다음 단계를 우리가 실현" },
    { code: "④", text: "MoveNet Thunder+XGBoost 최신 사례(2026)에는 Random Forest 비교군이 없음 — 우리가 RF/WRF를 추가 비교하면 이 공백을 메우는 기여가 됨" },
    { code: "종합", text: "바른자세의 조합은 \"정확도 극대화\"가 아니라 \"저사양 노트북에서 다른 작업과 동시 구동 가능한 실용적 실시간 시스템\"을 목표로 한 의도적 선택" },
  ];
  concl.forEach((item, i) => {
    const isLast = i === concl.length - 1;
    s.addShape(pres.ShapeType.roundRect, { x: 0.7, y: 1.9 + i * 0.92, w: 11.93, h: 0.8, rectRadius: 0.1, fill: { color: isLast ? TEAL_DARK : CARD }, line: { type: "none" } });
    s.addShape(pres.ShapeType.roundRect, { x: 0.9, y: 1.9 + i * 0.92 + 0.2, w: 0.5, h: 0.4, rectRadius: 0.08, fill: { color: isLast ? "0E5348" : "E9FBF7" }, line: { type: "none" } });
    s.addText(item.code, { x: 0.9, y: 1.9 + i * 0.92 + 0.2, w: 0.5, h: 0.4, align: "center", valign: "middle", fontFace: FONT_BODY, fontSize: item.code === "종합" ? 9 : 13, bold: true, color: isLast ? TEAL : TEAL_DARK, isTextBox: true, margin: 0 });
    s.addText(item.text, {
      x: 1.55, y: 1.9 + i * 0.92, w: 10.8, h: 0.8, valign: "middle", fontFace: FONT_BODY, fontSize: 13, color: isLast ? TXT_LIGHT : TXT_DARK, isTextBox: true, margin: 0, lineSpacingMultiple: 1.15,
    });
  });
  s.addText("다음 단계: 3주차 게이트(카메라 캘리브레이션 + 노트북 CPU 기준 포즈 추정 성능 실측)에서 문헌상의 속도·정확도 수치를 우리 실제 환경에서 검증한다.", {
    x: 0.7, y: 1.9 + concl.length * 0.92 + 0.15, w: 11.93, h: 0.5, fontFace: FONT_BODY, fontSize: 12, italic: true, color: TXT_MUTED, isTextBox: true, margin: 0,
  });
  pageNum(s, 11);
}

// =====================================================================================
// 10a. REFERENCES (1/2)
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "참고문헌 (1/2)", "References");
  const refs = [
    { code: "①", text: "Zhang, H.; Gračanin, D.; Zhou, W.; Dudash, D.; Rushton, G. \"Toward Real-Time Posture Classification: Reality Check.\" Electronics 2025, 14, 1876.", used: "사용: p.6, p.11" },
    { code: "②", text: "Lee, J.; Choi, H.; Kim, J. \"Classification Algorithm for Sitting Postures Using Weighted Random Forest.\" IET Image Processing 2025, 19, e70126.", used: "사용: p.6, p.7, p.9, p.11" },
    { code: "③", text: "Jo, J.; Kim, Y. \"Comparative Analysis of OpenPose, PoseNet, and MoveNet Models for Pose Estimation in Mobile Devices.\" Traitement du Signal 2022, 39(1), 119-124.", used: "사용: p.5, p.11" },
    { code: "④", text: "Pawitra et al. \"Automated ergonomic sitting postures detection for office workstation using XGBoost method.\" IAES Int. J. Artificial Intelligence 2026, 15(1), 506-514.", used: "사용: p.8, p.9, p.11" },
    { code: "⑤", text: "\"Comparing the Performance of Different Classifiers for Posture Detection.\" EAI, 2021.", used: "사용: p.6" },
  ];
  refs.forEach((r, i) => {
    const y = 1.7 + i * 0.85;
    s.addShape(pres.ShapeType.roundRect, { x: 0.7, y, w: 11.93, h: 0.72, rectRadius: 0.08, fill: { color: CARD }, line: { type: "none" } });
    s.addShape(pres.ShapeType.roundRect, { x: 0.85, y: y + 0.16, w: 0.42, h: 0.4, rectRadius: 0.07, fill: { color: "E9FBF7" }, line: { type: "none" } });
    s.addText(r.code, { x: 0.85, y: y + 0.16, w: 0.42, h: 0.4, align: "center", valign: "middle", fontFace: FONT_BODY, fontSize: 12, bold: true, color: TEAL_DARK, isTextBox: true, margin: 0 });
    s.addText(r.text, { x: 1.45, y: y + 0.06, w: 9.55, h: 0.6, valign: "middle", fontFace: FONT_BODY, fontSize: 10.8, color: TXT_DARK, isTextBox: true, margin: 0, lineSpacingMultiple: 1.05 });
    s.addText(r.used, { x: 11.05, y, w: 1.5, h: 0.72, align: "right", valign: "middle", fontFace: FONT_BODY, fontSize: 9.5, italic: true, color: "6B7A80", isTextBox: true, margin: 0 });
  });
  pageNum(s, 12);
}

// =====================================================================================
// 10b. REFERENCES (2/2)
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "참고문헌 (2/2)", "References");
  const refs = [
    { code: "⑥", text: "Ahmad, J.; Sidén, J.; Andersson, H. \"A Proposal of Implementation of Sitting Posture Monitoring System for Wheelchair Utilizing Machine Learning Methods.\" Sensors 2021, 21(19), 6349.", used: "사용: p.9" },
    { code: "⑦", text: "Thoutam, A. et al. \"Yoga Pose Estimation and Feedback Generation Using Deep Learning.\" Computational Intelligence and Neuroscience 2022.", used: "사용: p.10" },
    { code: "⑧", text: "Cao, Z. et al. \"KeypointNet: An Efficient Deep Learning Model with Multi-View Recognition Capability for Sitting Posture Recognition.\" Electronics 2025, 14(4), 718.", used: "사용: p.10" },
    { code: "⑨", text: "Tian, D.; Liu, B. \"Human posture recognition model integrating YOLOv8 pose and graph multilayer perceptron.\" Discover Artificial Intelligence 2026, 6, 1043.", used: "사용: p.10" },
    { code: "⑩", text: "Dai, C. et al. \"ConMLP: MLP-Based Self-Supervised Contrastive Learning for Skeleton Data Analysis and Action Recognition.\" Sensors 2023, 23(5), 2452.", used: "사용: p.10" },
  ];
  refs.forEach((r, i) => {
    const y = 1.7 + i * 0.98;
    s.addShape(pres.ShapeType.roundRect, { x: 0.7, y, w: 11.93, h: 0.85, rectRadius: 0.08, fill: { color: CARD }, line: { type: "none" } });
    s.addShape(pres.ShapeType.roundRect, { x: 0.85, y: y + 0.22, w: 0.42, h: 0.4, rectRadius: 0.07, fill: { color: "E9FBF7" }, line: { type: "none" } });
    s.addText(r.code, { x: 0.85, y: y + 0.22, w: 0.42, h: 0.4, align: "center", valign: "middle", fontFace: FONT_BODY, fontSize: 12, bold: true, color: TEAL_DARK, isTextBox: true, margin: 0 });
    s.addText(r.text, { x: 1.45, y: y + 0.08, w: 9.55, h: 0.7, valign: "middle", fontFace: FONT_BODY, fontSize: 10.8, color: TXT_DARK, isTextBox: true, margin: 0, lineSpacingMultiple: 1.1 });
    s.addText(r.used, { x: 11.05, y, w: 1.5, h: 0.85, align: "right", valign: "middle", fontFace: FONT_BODY, fontSize: 9.5, italic: true, color: "6B7A80", isTextBox: true, margin: 0 });
  });
  pageNum(s, 13);
}

pres.writeFile({ fileName: "barunjase_week3.pptx" }).then(() => {
  console.log("done");
});
