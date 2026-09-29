const pptxgen = require("pptxgenjs");
const fs = require("fs");

// ---------- palette (same as week1/week2/week3 deck — dark navy + mint teal) ----------
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
    x: 0.7, y: kickerText ? 0.75 : 0.55, w: 11.9, h: 0.9, fontFace: FONT_HEAD, fontSize: 27, bold: true,
    color: TXT_LIGHT, isTextBox: true, margin: 0,
  });
}
function pageNum(slide, n) {
  slide.addText(String(n).padStart(2, "0"), {
    x: PW - 0.9, y: PH - 0.55, w: 0.6, h: 0.35, fontFace: FONT_BODY, fontSize: 11,
    color: TXT_MUTED, align: "right", isTextBox: true, margin: 0,
  });
}
function noteTag(slide, x, y, text, kind) {
  // kind: "warn" | "good"
  const w = 0.35 + text.length * 0.115;
  slide.addShape(pres.ShapeType.roundRect, {
    x, y, w, h: 0.34, rectRadius: 0.17,
    fill: { color: kind === "good" ? "12432F" : WARN_SOFT }, line: { type: "none" },
  });
  slide.addText(text, {
    x, y, w, h: 0.34, align: "center", valign: "middle",
    fontFace: FONT_BODY, fontSize: 10.5, bold: true, color: kind === "good" ? GOOD : WARN, isTextBox: true, margin: 0,
  });
}

// =====================================================================================
// 1. TITLE
// =====================================================================================
{
  const s = newSlide();
  iconCircle(s, "camera", PW / 2 - 0.55, 0.68, 1.1, TEAL_DARK);
  s.addText("임베디드소프트웨어학과 · 캡스톤 프로젝트", {
    x: 1, y: 2.05, w: 11.33, h: 0.4, align: "center", fontFace: FONT_BODY, fontSize: 14,
    color: TEAL, bold: true, charSpacing: 1.5, isTextBox: true, margin: 0,
  });
  s.addText("카메라 확정, 첫 실측 게이트 통과", {
    x: 1, y: 2.5, w: 11.33, h: 1.05, align: "center", fontFace: FONT_HEAD, fontSize: 40,
    bold: true, color: TXT_LIGHT, isTextBox: true, margin: 0,
  });
  s.addText("바른자세 · 4주차 발표 — 계획과 실제의 간극을 발견 즉시 바로잡은 한 주", {
    x: 1, y: 3.6, w: 11.33, h: 0.55, align: "center", fontFace: FONT_BODY, fontSize: 16.5,
    color: TXT_MUTED, isTextBox: true, margin: 0,
  });
  s.addText("실측 없이는 몰랐을 것들 — 카메라 기종, USB 인식 속도, depth 근접 사각지대", {
    x: 1.5, y: 4.25, w: 10.33, h: 0.4, align: "center", fontFace: FONT_BODY, fontSize: 12.5,
    italic: true, color: TEAL, isTextBox: true, margin: 0,
  });
  s.addShape(pres.ShapeType.roundRect, {
    x: PW / 2 - 2.1, y: 4.85, w: 4.2, h: 0.55, rectRadius: 0.28,
    fill: { color: TEAL_DARK }, line: { type: "none" },
  });
  s.addText("4주차 발표 · 하드웨어 검증 + 리스크 발견", {
    x: PW / 2 - 2.1, y: 4.85, w: 4.2, h: 0.55, align: "center", valign: "middle",
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
    "01   개발 환경 정비 · 데이터 수집 프로토콜", "02   카메라 기종 확인과 환경 트러블슈팅",
    "03   4주차 게이트 측정 결과", "04   발견한 설계 리스크: depth 근접 사각지대",
    "05   선행 논문 데이터로 파이프라인 사전 검증", "06   다음 주(5주차) 계획",
  ];
  const colW = 5.55, gapX = 0.5, startX = 0.7, startY = 2.1, rowH = 0.9;
  const perCol = Math.ceil(items.length / 2);
  items.forEach((t, i) => {
    const col = Math.floor(i / perCol);
    const row = i % perCol;
    const x = startX + col * (colW + gapX);
    const y = startY + row * rowH;
    s.addShape(pres.ShapeType.roundRect, {
      x, y, w: colW, h: 0.66, rectRadius: 0.1, fill: { color: TEAL_DARK }, line: { type: "none" },
    });
    s.addText(t, {
      x: x + 0.25, y, w: colW - 0.4, h: 0.66, valign: "middle", fontFace: FONT_BODY,
      fontSize: 13.5, bold: true, color: TXT_LIGHT, isTextBox: true, margin: 0,
    });
  });
  s.addText("지난주(3주차)는 알고리즘 비교 연구조사로 대체 진행 — 그만큼 밀린 캘리브레이션·게이트가 이번 주로 이월됐습니다.", {
    x: 0.7, y: startY + perCol * rowH + 0.25, w: 11.93, h: 0.5, fontFace: FONT_BODY, fontSize: 12,
    italic: true, color: TXT_MUTED, isTextBox: true, margin: 0,
  });
  pageNum(s, 2);
}

// =====================================================================================
// 3. DEV ENV + DATA COLLECTION PROTOCOL
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "개발 환경 정비 · 데이터 수집 프로토콜", "Setup & Protocol");
  s.addText(
    "카메라 도착 전(9/27~28)까지 카메라 없이 할 수 있는 소프트웨어 작업을 먼저 끝내뒀습니다.",
    { x: 0.7, y: 1.75, w: 11.93, h: 0.5, fontFace: FONT_BODY, fontSize: 14.5, color: TXT_LIGHT, isTextBox: true, margin: 0 }
  );

  const cards = [
    {
      icon: "chip", title: "개발 환경 정비",
      items: [
        "git 저장소 초기화, GitHub 원격 연결",
        "단일 파일이던 MoveNet 코드를 pose/features/capture/logic 4개 패키지로 분리",
        "합성 좌표 데이터 기반 단위테스트 도입 — 카메라·모델 없이도 로직 검증 가능",
      ],
    },
    {
      icon: "book", title: "데이터 수집 프로토콜 문서화",
      items: [
        "자세 5종 정의: 정상·숙임·기대기·좌측기울임·우측기울임",
        "참가자/샷수 목표, 촬영조건(날짜·조명·복장 분산) 확정",
        "팀원 2인 교차 라벨링 절차, labels.csv 스키마 정의",
      ],
    },
  ];
  const cardW = 5.85, gap = 0.23, y0 = 2.5, h0 = 4.1;
  cards.forEach((c, i) => {
    const x = 0.7 + i * (cardW + gap);
    s.addShape(pres.ShapeType.roundRect, { x, y: y0, w: cardW, h: h0, rectRadius: 0.12, fill: { color: CARD }, line: { type: "none" } });
    iconCircle(s, c.icon, x + 0.3, y0 + 0.3, 0.65, TEAL_DARK);
    s.addText(c.title, { x: x + 1.1, y: y0 + 0.35, w: cardW - 1.4, h: 0.55, valign: "middle", fontFace: FONT_HEAD, fontSize: 15.5, bold: true, color: TXT_DARK, isTextBox: true, margin: 0 });
    const bodyText = c.items.map((it) => ({ text: "•  " + it, options: { breakLine: true, paraSpaceAfter: 10 } }));
    s.addText(bodyText, { x: x + 0.35, y: y0 + 1.25, w: cardW - 0.7, h: h0 - 1.5, fontFace: FONT_BODY, fontSize: 12.5, color: "3C4A50", isTextBox: true, margin: 0, lineSpacingMultiple: 1.2 });
  });
  pageNum(s, 3);
}

// =====================================================================================
// 4. CAMERA CONFIRMATION + TROUBLESHOOTING
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "카메라 기종 확인과 환경 트러블슈팅", "Camera & Troubleshooting");

  // camera swap highlight
  s.addShape(pres.ShapeType.roundRect, { x: 0.7, y: 1.75, w: 11.93, h: 1.15, rectRadius: 0.12, fill: { color: TEAL_DARK }, line: { type: "none" } });
  s.addText([
    { text: "계획:  ", options: { fontSize: 14, color: TXT_MUTED } },
    { text: "위드로봇 oCamS-1CGN-U", options: { fontSize: 14, color: TXT_MUTED, strike: true } },
    { text: "      →      실제 수령:  ", options: { fontSize: 14, color: TXT_MUTED } },
    { text: "Intel RealSense D455", options: { fontSize: 18, bold: true, color: TEAL } },
  ], { x: 1.0, y: 1.75, w: 11.4, h: 1.15, valign: "middle", fontFace: FONT_BODY, isTextBox: true, margin: 0 });
  s.addText("9/29 수령 당일 확인 → 시스템 구성·BOM·SDK(pyrealsense2) 전면 수정. RealSense는 출고 시 캘리브레이션 완료 — 체스보드 캘리브레이션 불필요해짐", {
    x: 0.7, y: 2.95, w: 11.93, h: 0.4, fontFace: FONT_BODY, fontSize: 11.5, italic: true, color: TXT_MUTED, isTextBox: true, margin: 0,
  });

  s.addText("이어서 실제 연동 과정에서 순서대로 나온 문제들", {
    x: 0.7, y: 3.55, w: 11.93, h: 0.4, fontFace: FONT_HEAD, fontSize: 14, bold: true, color: TEAL, isTextBox: true, margin: 0,
  });

  const issues = [
    { icon: "chip", t: "Python 3.14 패키지 호환성", d: "tflite-runtime wheel 부재 → tensorflow로 대체" },
    { icon: "alert", t: "Windows 스마트 앱 제어", d: "tensorflow DLL 차단 → 기능 비활성화로 해결" },
    { icon: "target", t: "USB 인식 속도", d: "2.1(USB2)로 인식 → 포트 재연결로 3.2 정상화" },
    { icon: "diagram", t: "프레임 미수신", d: "RealSense Viewer가 장치 점유 중 → 종료로 해결" },
  ];
  const cw = 2.83, gap = 0.2, y2 = 4.15, h2 = 2.35;
  issues.forEach((it, i) => {
    const x = 0.7 + i * (cw + gap);
    s.addShape(pres.ShapeType.roundRect, { x, y: y2, w: cw, h: h2, rectRadius: 0.12, fill: { color: CARD }, line: { type: "none" } });
    iconCircle(s, it.icon, x + 0.24, y2 + 0.24, 0.58, TEAL_DARK);
    s.addText(it.t, { x: x + 0.24, y: y2 + 1.0, w: cw - 0.48, h: 0.6, fontFace: FONT_HEAD, fontSize: 12, bold: true, color: TXT_DARK, isTextBox: true, margin: 0, lineSpacingMultiple: 1.1 });
    s.addText(it.d, { x: x + 0.24, y: y2 + 1.6, w: cw - 0.48, h: h2 - 1.8, fontFace: FONT_BODY, fontSize: 10, color: "3C4A50", isTextBox: true, margin: 0, lineSpacingMultiple: 1.15 });
  });
  pageNum(s, 4);
}

// =====================================================================================
// 5. GATE MEASUREMENT RESULTS
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "4주차 게이트 측정 결과", "Gate Results");

  const headerOpts = { fill: { color: TEAL_DARK }, color: TXT_LIGHT, bold: true, fontFace: FONT_BODY, fontSize: 12, valign: "middle" };
  const cellOpts = { fill: { color: CARD }, color: TXT_DARK, fontFace: FONT_BODY, fontSize: 11.5, valign: "middle" };
  const goodOpts = { ...cellOpts, fill: { color: "E9FBF7" }, bold: true, color: TEAL_DARK };
  const rows = [
    [{ text: "항목", options: headerOpts }, { text: "측정값", options: headerOpts }, { text: "목표치", options: headerOpts }],
    [{ text: "평균 fps", options: goodOpts }, { text: "29.43 fps", options: goodOpts }, { text: "최소 5 (목표 8~10)", options: cellOpts }],
    [{ text: "평균 프레임 간격", options: cellOpts }, { text: "33.4 ms", options: cellOpts }, { text: "—", options: cellOpts }],
    [{ text: "최대 프레임 간격", options: cellOpts }, { text: "34.2 ms", options: cellOpts }, { text: "—", options: cellOpts }],
    [{ text: "스파이크(67ms 초과)", options: goodOpts }, { text: "0회", options: goodOpts }, { text: "—", options: cellOpts }],
    [{ text: "거리 유효 측정 시작점", options: cellOpts }, { text: "약 40cm~ (정성 확인)", options: cellOpts }, { text: "—", options: cellOpts }],
  ];
  s.addTable(rows, { x: 0.7, y: 1.85, w: 7.6, h: 3.0, colW: [3.0, 2.6, 2.0], border: { type: "solid", color: "0B1A26", pt: 1.5 }, autoPage: false });
  s.addText("측정 도구: src/capture/realsense_capture.py — gate 모드(30초 자동 측정), view 모드(줄자 대조 거리 확인)", {
    x: 0.7, y: 5.0, w: 7.6, h: 0.35, fontFace: FONT_BODY, fontSize: 10.5, italic: true, color: TXT_MUTED, isTextBox: true, margin: 0,
  });
  s.addText("주의: 카메라 캡처 단독 수치 — MoveNet 추론 통합 후 재측정 필요. 거리 정밀 오차값은 줄자 미확보로 다음 주로 이월", {
    x: 0.7, y: 5.4, w: 7.6, h: 0.8, fontFace: FONT_BODY, fontSize: 11, color: TXT_LIGHT, isTextBox: true, margin: 0, lineSpacingMultiple: 1.2,
  });

  s.addShape(pres.ShapeType.roundRect, { x: 8.65, y: 1.85, w: 3.98, h: 4.35, rectRadius: 0.12, fill: { color: TEAL_DARK }, line: { type: "none" } });
  s.addText("29.43", { x: 8.65, y: 2.5, w: 3.98, h: 1.2, align: "center", fontFace: FONT_HEAD, fontSize: 56, bold: true, color: TEAL, isTextBox: true, margin: 0 });
  s.addText("fps (평균)", { x: 8.65, y: 3.6, w: 3.98, h: 0.4, align: "center", fontFace: FONT_BODY, fontSize: 13, color: TXT_MUTED, isTextBox: true, margin: 0 });
  s.addText("목표(최소 5fps) 대비 약 5.9배", { x: 8.9, y: 4.35, w: 3.48, h: 0.4, align: "center", fontFace: FONT_BODY, fontSize: 11.5, italic: true, color: "C9F4EA", isTextBox: true, margin: 0 });
  noteTag(s, PW / 2 + 0.9, 5.4, "게이트 통과", "good");
  pageNum(s, 5);
}

// =====================================================================================
// 6. RISK: DEPTH NEAR-RANGE BLIND SPOT
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "발견한 설계 리스크: depth 근접 사각지대", "Design Risk");

  s.addShape(pres.ShapeType.roundRect, { x: 0.7, y: 1.75, w: 11.93, h: 1.7, rectRadius: 0.12, fill: { color: WARN_SOFT }, line: { type: "none" } });
  iconCircle(s, "alert", 1.0, 2.0, 0.7, "5A2E1F");
  s.addText(
    "D455 depth는 약 40cm 미만에서 무효값을 반환한다 — 그런데 저희가 감지하려는 상황 " +
    "\"화면에 너무 가까워짐\"이 하필 이 무효 구간과 겹친다. 정작 감지하고 싶은 순간에 센서가 먹통이 되는 셈.",
    { x: 1.9, y: 1.95, w: 10.5, h: 1.3, valign: "middle", fontFace: FONT_BODY, fontSize: 13.5, color: TXT_LIGHT, isTextBox: true, margin: 0, lineSpacingMultiple: 1.25 }
  );

  const cols = [
    {
      title: "완화 방안: 이중화 판정",
      icon: "diagram",
      body: "depth 유효 구간(40cm~)은 정확한 거리값을 그대로 사용. 그 이하 근접 구간은 RGB 화면 속 얼굴·어깨 면적 비율로 \"더 가까워짐\"만 이분류 판정.",
    },
    {
      title: "선행 연구와의 접점",
      icon: "book",
      body: "이 방법은 WRF 논문(3주차 조사)이 깊이 카메라 없이 쓰던 방식과 동일한 아이디어 — 선행 연구의 한계를 우리 설계의 보완책으로 재활용.",
    },
  ];
  const cw = 5.85, gap = 0.23, y2 = 3.75, h2 = 2.85;
  cols.forEach((c, i) => {
    const x = 0.7 + i * (cw + gap);
    s.addShape(pres.ShapeType.roundRect, { x, y: y2, w: cw, h: h2, rectRadius: 0.12, fill: { color: CARD }, line: { type: "none" } });
    iconCircle(s, c.icon, x + 0.3, y2 + 0.3, 0.6, TEAL_DARK);
    s.addText(c.title, { x: x + 1.05, y: y2 + 0.35, w: cw - 1.35, h: 0.55, valign: "middle", fontFace: FONT_HEAD, fontSize: 14, bold: true, color: TXT_DARK, isTextBox: true, margin: 0 });
    s.addText(c.body, { x: x + 0.3, y: y2 + 1.15, w: cw - 0.6, h: h2 - 1.35, fontFace: FONT_BODY, fontSize: 12, color: "3C4A50", isTextBox: true, margin: 0, lineSpacingMultiple: 1.25 });
  });
  noteTag(s, 0.7, y2 + h2 + 0.15, "구현 전 — 5주차 필터 튜닝과 함께 진행 예정", "warn");
  pageNum(s, 6);
}

// =====================================================================================
// 7. PRIOR-WORK DATA PIPELINE VALIDATION
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "선행 논문 데이터로 파이프라인 사전 검증", "Pipeline Pre-Validation");
  s.addText("아직 우리 자세 데이터가 없어서, 6주차부터 쓸 학습 파이프라인 코드가 실제로 동작하는지 먼저 확인했습니다.", {
    x: 0.7, y: 1.7, w: 11.93, h: 0.45, fontFace: FONT_BODY, fontSize: 13.5, color: TXT_LIGHT, isTextBox: true, margin: 0,
  });

  // left card: WRF paper data
  s.addShape(pres.ShapeType.roundRect, { x: 0.7, y: 2.35, w: 5.85, h: 4.25, rectRadius: 0.12, fill: { color: CARD }, line: { type: "none" } });
  iconCircle(s, "book", 1.0, 2.65, 0.6, TEAL_DARK);
  s.addText("WRF 논문 공개 데이터 검증", { x: 1.75, y: 2.7, w: 4.6, h: 0.5, valign: "middle", fontFace: FONT_HEAD, fontSize: 14, bold: true, color: TXT_DARK, isTextBox: true, margin: 0 });
  const headerOpts = { fill: { color: TEAL_DARK }, color: TXT_LIGHT, bold: true, fontFace: FONT_BODY, fontSize: 10.5, valign: "middle" };
  const cellOpts = { fill: { color: "F4F8F7" }, color: TXT_DARK, fontFace: FONT_BODY, fontSize: 10.5, valign: "middle" };
  const rows1 = [
    [{ text: "", options: headerOpts }, { text: "우리 재현", options: headerOpts }, { text: "논문 보고치", options: headerOpts }],
    [{ text: "RF (가중치 없음)", options: cellOpts }, { text: "0.980", options: cellOpts }, { text: "0.96", options: cellOpts }],
    [{ text: "WRF (가중치 적용)", options: cellOpts }, { text: "0.980", options: cellOpts }, { text: "0.98", options: cellOpts }],
  ];
  s.addTable(rows1, { x: 1.0, y: 3.35, w: 5.25, h: 1.15, colW: [2.25, 1.5, 1.5], border: { type: "solid", color: "0B1A26", pt: 1 }, autoPage: false });
  s.addText("변수 중요도 순서도 논문 Table과 일치 — 파이프라인이 논문 방식대로 정상 동작", {
    x: 1.0, y: 4.65, w: 5.25, h: 0.5, fontFace: FONT_BODY, fontSize: 11, color: "3C4A50", isTextBox: true, margin: 0, lineSpacingMultiple: 1.2,
  });
  noteTag(s, 1.0, 5.3, "정확도 수치는 다른 특징공간 — 우리 성능 예측 아님", "warn");

  // right card: RF skeleton
  s.addShape(pres.ShapeType.roundRect, { x: 6.78, y: 2.35, w: 5.85, h: 4.25, rectRadius: 0.12, fill: { color: CARD }, line: { type: "none" } });
  iconCircle(s, "diagram", 7.08, 2.65, 0.6, TEAL_DARK);
  s.addText("RF 기준모델 학습 스켈레톤", { x: 7.83, y: 2.7, w: 4.6, h: 0.5, valign: "middle", fontFace: FONT_HEAD, fontSize: 14, bold: true, color: TXT_DARK, isTextBox: true, margin: 0 });
  const bodyText2 = [
    "우리 프로젝트 특징(neck_tilt_deg 등) 기준으로 재구성",
    "합성 keypoint 5클래스 생성 → 좌우반전+jitter 증강 → RF 학습·평가",
    "단위테스트 5종 추가, 로컬 환경에서도 재현 확인 (전체 테스트 10개 통과)",
  ].map((it) => ({ text: "•  " + it, options: { breakLine: true, paraSpaceAfter: 8 } }));
  s.addText(bodyText2, { x: 7.08, y: 3.35, w: 5.25, h: 1.9, fontFace: FONT_BODY, fontSize: 11.5, color: "3C4A50", isTextBox: true, margin: 0, lineSpacingMultiple: 1.2 });
  s.addText("5주차 실데이터 수집 후 데이터 로딩 부분만 교체해 재사용 예정", {
    x: 7.08, y: 5.05, w: 5.25, h: 0.4, fontFace: FONT_BODY, fontSize: 11, italic: true, color: "3C4A50", isTextBox: true, margin: 0,
  });
  noteTag(s, 7.08, 5.55, "합성 데이터 — 정확도 수치 의미 없음", "warn");
  pageNum(s, 7);
}

// =====================================================================================
// 8. NEXT WEEK PLAN
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "다음 주(5주차) 계획", "Next Week");
  const plan = [
    { n: "1", t: "RealSense depth 프리셋·필터 튜닝, 줄자 확보해 거리 정확도 정밀 재검증" },
    { n: "2", t: "근접 사각지대 보완 — 얼굴·어깨 면적비 기반 근접 판정 실제 구현" },
    { n: "3", t: "판정 로직과 스테레오+포즈 파이프라인 1차 통합" },
    { n: "4", t: "데이터 수집 프로토콜에 따라 실제 촬영 착수" },
    { n: "5", t: "논문 Introduction · Related Work 초안 (이번 주 이월분)" },
  ];
  plan.forEach((item, i) => {
    const y = 1.9 + i * 0.92;
    s.addShape(pres.ShapeType.roundRect, { x: 0.7, y, w: 11.93, h: 0.8, rectRadius: 0.1, fill: { color: CARD }, line: { type: "none" } });
    s.addShape(pres.ShapeType.roundRect, { x: 0.9, y: y + 0.2, w: 0.5, h: 0.4, rectRadius: 0.08, fill: { color: "E9FBF7" }, line: { type: "none" } });
    s.addText(item.n, { x: 0.9, y: y + 0.2, w: 0.5, h: 0.4, align: "center", valign: "middle", fontFace: FONT_BODY, fontSize: 13, bold: true, color: TEAL_DARK, isTextBox: true, margin: 0 });
    s.addText(item.t, { x: 1.55, y, w: 10.8, h: 0.8, valign: "middle", fontFace: FONT_BODY, fontSize: 13.5, color: TXT_DARK, isTextBox: true, margin: 0, lineSpacingMultiple: 1.15 });
  });
  pageNum(s, 8);
}

// =====================================================================================
// 9. CONCLUSION
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "마무리", "Conclusion");
  const concl = [
    { code: "✓", t: "4주차 게이트 통과 — 평균 29.43fps, 스파이크 0회, 목표치(최소 5fps) 대폭 상회" },
    { code: "✓", t: "계획(oCamS)과 실제(D455)가 다르다는 걸 발견 즉시 문서 전체를 바로잡음" },
    { code: "✓", t: "측정 중 depth 근접 사각지대 리스크를 발견해 해결책 없이 넘기지 않고 문서화" },
    { code: "✓", t: "실데이터 없이도 RF/WRF 학습 파이프라인이 정상 동작함을 미리 검증" },
    { code: "남음", t: "거리 정밀 오차 측정, 근접 사각지대 실제 구현, 논문 Introduction/Related Work 초안" },
  ];
  concl.forEach((item, i) => {
    const isLast = i === concl.length - 1;
    const y = 1.9 + i * 0.92;
    s.addShape(pres.ShapeType.roundRect, { x: 0.7, y, w: 11.93, h: 0.8, rectRadius: 0.1, fill: { color: isLast ? TEAL_DARK : CARD }, line: { type: "none" } });
    s.addShape(pres.ShapeType.roundRect, { x: 0.9, y: y + 0.2, w: 0.5, h: 0.4, rectRadius: 0.08, fill: { color: isLast ? "0E5348" : "E9FBF7" }, line: { type: "none" } });
    s.addText(item.code, { x: 0.9, y: y + 0.2, w: 0.5, h: 0.4, align: "center", valign: "middle", fontFace: FONT_BODY, fontSize: item.code === "남음" ? 9 : 15, bold: true, color: isLast ? TEAL : TEAL_DARK, isTextBox: true, margin: 0 });
    s.addText(item.t, { x: 1.55, y, w: 10.8, h: 0.8, valign: "middle", fontFace: FONT_BODY, fontSize: 13, color: isLast ? TXT_LIGHT : TXT_DARK, isTextBox: true, margin: 0, lineSpacingMultiple: 1.15 });
  });
  pageNum(s, 9);
}

// =====================================================================================
// 10. REFERENCES / SOURCES
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "참고 자료", "Sources");
  const refs = [
    { code: "01", text: "project.md — 전체 프로젝트 계획 (이번 주 업데이트 반영 최신본)" },
    { code: "02", text: "data_collection_protocol.md — 자세 5종 정의·촬영/라벨링 절차" },
    { code: "03", text: "src/capture/realsense_capture.py — RealSense 캡처·게이트 측정 코드" },
    { code: "04", text: "src/logic/rf_baseline.py, tests/test_rf_baseline.py — RF 기준모델 스켈레톤" },
    { code: "05", text: "experiments/wrf_paper_replication.py — WRF 논문 데이터 파이프라인 검증" },
    { code: "06", text: "weekly_report_week4.md — 4주차 상세 진행보고서" },
  ];
  refs.forEach((r, i) => {
    const y = 1.75 + i * 0.78;
    s.addShape(pres.ShapeType.roundRect, { x: 0.7, y, w: 11.93, h: 0.65, rectRadius: 0.08, fill: { color: CARD }, line: { type: "none" } });
    s.addShape(pres.ShapeType.roundRect, { x: 0.85, y: y + 0.13, w: 0.42, h: 0.4, rectRadius: 0.07, fill: { color: "E9FBF7" }, line: { type: "none" } });
    s.addText(r.code, { x: 0.85, y: y + 0.13, w: 0.42, h: 0.4, align: "center", valign: "middle", fontFace: FONT_BODY, fontSize: 11.5, bold: true, color: TEAL_DARK, isTextBox: true, margin: 0 });
    s.addText(r.text, { x: 1.45, y, w: 10.9, h: 0.65, valign: "middle", fontFace: FONT_BODY, fontSize: 12, color: TXT_DARK, isTextBox: true, margin: 0 });
  });
  s.addText("질문 받겠습니다", {
    x: 0.7, y: 1.75 + refs.length * 0.78 + 0.25, w: 11.93, h: 0.45, align: "center",
    fontFace: FONT_BODY, fontSize: 13.5, italic: true, color: TXT_MUTED, isTextBox: true, margin: 0,
  });
  pageNum(s, 10);
}

pres.writeFile({ fileName: "barunjase_week4.pptx" }).then(() => {
  console.log("done");
});
