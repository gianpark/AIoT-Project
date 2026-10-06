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


// ---------- helpers for week5 ----------
function card(slide, x, y, w, h, fill) {
  slide.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.12, fill: { color: fill || CARD }, line: { type: "none" } });
}
function cardTitle(slide, icon, x, y, w, text) {
  iconCircle(slide, icon, x + 0.28, y + 0.25, 0.6, TEAL_DARK);
  slide.addText(text, { x: x + 1.05, y: y + 0.25, w: w - 1.3, h: 0.6, valign: "middle", fontFace: FONT_HEAD, fontSize: 14.5, bold: true, color: TXT_DARK, isTextBox: true, margin: 0 });
}
function bullets(slide, items, x, y, w, h, size, color) {
  const runs = items.map((it) => ({ text: "•  " + it, options: { breakLine: true, paraSpaceAfter: 7 } }));
  slide.addText(runs, { x, y, w, h, fontFace: FONT_BODY, fontSize: size || 12, color: color || "3C4A50", isTextBox: true, margin: 0, valign: "top", lineSpacingMultiple: 1.15 });
}
function listRow(slide, i, n, text, y0, rowH, gap, size) {
  const y = y0 + i * (rowH + gap);
  card(slide, 0.7, y, 11.93, rowH);
  slide.addShape(pres.ShapeType.roundRect, { x: 0.9, y: y + (rowH - 0.4) / 2, w: 0.55, h: 0.4, rectRadius: 0.08, fill: { color: "E9FBF7" }, line: { type: "none" } });
  slide.addText(n, { x: 0.9, y: y + (rowH - 0.4) / 2, w: 0.55, h: 0.4, align: "center", valign: "middle", fontFace: FONT_BODY, fontSize: 11, bold: true, color: TEAL_DARK, isTextBox: true, margin: 0 });
  slide.addText(text, { x: 1.65, y, w: 10.8, h: rowH, valign: "middle", fontFace: FONT_BODY, fontSize: size || 13, color: TXT_DARK, isTextBox: true, margin: 0, lineSpacingMultiple: 1.1 });
}
const H_OPT = { fill: { color: TEAL_DARK }, color: TXT_LIGHT, bold: true, fontFace: FONT_BODY, fontSize: 11.5, valign: "middle" };
const C_OPT = { fill: { color: CARD }, color: TXT_DARK, fontFace: FONT_BODY, fontSize: 11.5, valign: "middle" };
const G_OPT = { ...C_OPT, fill: { color: "E9FBF7" }, bold: true, color: TEAL_DARK };
const W_OPT = { ...C_OPT, fill: { color: "FDEFE8" }, color: "9A3F1F", bold: true };

// =====================================================================================
// 1. TITLE
// =====================================================================================
{
  const s = newSlide();
  iconCircle(s, "person", PW / 2 - 0.55, 0.68, 1.1, TEAL_DARK);
  s.addText("임베디드소프트웨어학과 · 바른자세", {
    x: 1, y: 2.05, w: 11.33, h: 0.4, align: "center", fontFace: FONT_BODY, fontSize: 14, color: TEAL, bold: true, charSpacing: 1.5, isTextBox: true, margin: 0,
  });
  s.addText("데이터 수집 착수, 판정 로직 1차 통합", {
    x: 1, y: 2.5, w: 11.33, h: 1.05, align: "center", fontFace: FONT_HEAD, fontSize: 40, bold: true, color: TXT_LIGHT, isTextBox: true, margin: 0,
  });
  s.addText("바른자세 · 5주차 발표 — 화면에 사람이 둘일 때도 동작하게, 매번 기준을 잡지 않아도 되게", {
    x: 1, y: 3.6, w: 11.33, h: 0.55, align: "center", fontFace: FONT_BODY, fontSize: 16.5, color: TXT_MUTED, isTextBox: true, margin: 0,
  });
  s.addText("촬영하면서 부딪힌 문제 → 대응 → 교수님 피드백으로 방향 재정리", {
    x: 1.5, y: 4.25, w: 10.33, h: 0.4, align: "center", fontFace: FONT_BODY, fontSize: 12.5, italic: true, color: TEAL, isTextBox: true, margin: 0,
  });
  s.addShape(pres.ShapeType.roundRect, { x: PW / 2 - 2.3, y: 4.85, w: 4.6, h: 0.55, rectRadius: 0.28, fill: { color: TEAL_DARK }, line: { type: "none" } });
  s.addText("5주차 발표 · 구현 + 설계 재정리", {
    x: PW / 2 - 2.3, y: 4.85, w: 4.6, h: 0.55, align: "center", valign: "middle", fontFace: FONT_BODY, fontSize: 13, bold: true, color: TXT_LIGHT, isTextBox: true, margin: 0,
  });
  s.addText("발표자: 기안 외 1인", { x: 1, y: PH - 0.9, w: 11.33, h: 0.4, align: "center", fontFace: FONT_BODY, fontSize: 12, color: TXT_MUTED, isTextBox: true, margin: 0 });
}

// =====================================================================================
// 2. CONTENTS
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "목차", "Contents");
  const items = [
    "01   5주차 계획 대비 진행 상황", "02   데이터 수집 현황 · 라벨링 교차검증",
    "03   촬영 중 발견한 문제와 대응", "04   판정 로직 · 근접 보완 · 카메라 실측",
    "05   교수님 피드백과 방향 재정리", "06   절대값 판정 설계",
    "07   한계 · 카메라 확보 후 할 일 · 6주차 계획",
  ];
  const colW = 5.55, gapX = 0.5, startX = 0.7, startY = 2.1, rowH = 0.9;
  const perCol = Math.ceil(items.length / 2);
  items.forEach((t, i) => {
    const col = Math.floor(i / perCol), row = i % perCol;
    const x = startX + col * (colW + gapX), y = startY + row * rowH;
    s.addShape(pres.ShapeType.roundRect, { x, y, w: colW, h: 0.66, rectRadius: 0.1, fill: { color: TEAL_DARK }, line: { type: "none" } });
    s.addText(t, { x: x + 0.25, y, w: colW - 0.4, h: 0.66, valign: "middle", fontFace: FONT_BODY, fontSize: 13.5, bold: true, color: TXT_LIGHT, isTextBox: true, margin: 0 });
  });
  pageNum(s, 2);
}

// =====================================================================================
// 3. PLAN VS ACTUAL
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "5주차 계획 대비 진행 상황", "Plan vs Actual");
  const rows = [
    [{ text: "계획 항목", options: H_OPT }, { text: "상태", options: H_OPT }, { text: "비고", options: H_OPT }],
    [{ text: "자체 데이터 수집 착수", options: C_OPT }, { text: "진행", options: G_OPT }, { text: "참가자 2명 235장 (6주차 몫을 앞당김)", options: C_OPT }],
    [{ text: "판정 로직 + 스테레오·포즈 파이프라인 1차 통합", options: C_OPT }, { text: "구현", options: G_OPT }, { text: "자세·근접 판정, 알림 상태머신 (--judge) — 실시간 실측으로 기준 재보정", options: C_OPT }],
    [{ text: "근접 사각지대 보완", options: C_OPT }, { text: "구현", options: G_OPT }, { text: "어깨너비 역산 + 얼굴 depth 무효 시 근접 판정(실측 확인). 40cm 미만 정확도는 미검증", options: C_OPT }],
    [{ text: "depth 필터 튜닝", options: C_OPT }, { text: "완료", options: G_OPT }, { text: "45/60/80cm 실측 → SDK 기본 + 구멍 메우기 ON 확정", options: C_OPT }],
    [{ text: "거리 정확도 재검증", options: C_OPT }, { text: "완료", options: G_OPT }, { text: "37~92cm 줄자 실측 (정렬 유무 차이 뚜렷하지 않음)", options: C_OPT }],
    [{ text: "카메라 수령 후 --judge 실측", options: C_OPT }, { text: "1인 완료", options: W_OPT }, { text: "5자세 판정·알림 확인, 한 명 기준이라 참가자·거리 확대 필요", options: C_OPT }],
  ];
  s.addTable(rows, { x: 0.7, y: 1.85, w: 11.93, colW: [5.0, 1.6, 5.33], rowH: 0.62, border: { type: "solid", color: "0B1A26", pt: 1.5 }, autoPage: false });
  s.addText("계획에 없던 것: 다인 환경 대응(자동 사용자 인식·추적), 라벨링 교차검증 도구, 교수님 피드백 반영 설계, 실측에서 나온 문제 4건 수정", {
    x: 0.7, y: 6.4, w: 11.93, h: 0.4, fontFace: FONT_BODY, fontSize: 12, italic: true, color: TEAL, isTextBox: true, margin: 0,
  });
  pageNum(s, 3);
}

// =====================================================================================
// 4. DATA COLLECTION STATUS
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "데이터 수집 현황 · 라벨링 교차검증", "Data Collection");
  card(s, 0.7, 1.85, 6.4, 4.75);
  s.addText("클래스별 수집 장수 (총 235장)", { x: 1.0, y: 2.0, w: 5.8, h: 0.4, fontFace: FONT_HEAD, fontSize: 14, bold: true, color: TXT_DARK, isTextBox: true, margin: 0 });
  s.addChart(pres.charts.BAR, [{
    name: "장수", labels: ["normal", "slouch_back", "slouch_forward", "tilt_left", "tilt_right"], values: [53, 52, 50, 41, 39],
  }], {
    x: 0.9, y: 2.45, w: 6.0, h: 3.5, barDir: "col", chartColors: [TEAL_DARK], showValue: true, dataLabelPosition: "outEnd", dataLabelColor: TXT_DARK, dataLabelFontSize: 12,
    catAxisLabelColor: "3C4A50", catAxisLabelFontSize: 10.5, valAxisHidden: true, valGridLine: { style: "none" }, showLegend: false, valAxisMinVal: 0, valAxisMaxVal: 65,
  });
  s.addText("참가자 p01 166장 + p02 69장, 전부 2026-10-01 촬영", { x: 1.0, y: 6.0, w: 5.8, h: 0.4, fontFace: FONT_BODY, fontSize: 11, italic: true, color: "3C4A50", isTextBox: true, margin: 0 });

  card(s, 7.3, 1.85, 5.33, 2.75);
  cardTitle(s, "target", 7.3, 1.85, 5.33, "라벨링 교차검증 도구");
  bullets(s, [
    "전체의 15%(36장)를 클래스 비율 유지로 추출, 파일명 숨겨 블라인드 폴더 생성",
    "촬영 안 한 팀원이 독립 라벨링 → 일치율·혼동 쌍 자동 계산",
    "결과: 불일치 2건 — 고개 기울임이 원인 → \"tilt는 몸통 기울임만\"으로 정의 확정, 기존 tilt 80장 전수 확인(수정 없음)",
  ], 7.6, 2.8, 4.8, 1.75, 11.5);
  noteTag(s, 7.6, 4.2, "일치율 94.4% (34/36) — 기준 통과", "good");

  card(s, 7.3, 4.8, 5.33, 1.8, WARN_SOFT);
  s.addText("한계", { x: 7.6, y: 4.92, w: 4.8, h: 0.3, fontFace: FONT_BODY, fontSize: 11, bold: true, color: WARN, isTextBox: true, margin: 0 });
  s.addText("프로토콜 권장은 3~5명, 날짜·조명·복장 분산. 지금은 2명·하루·한 환경이라 일반화 검증은 아직 못 합니다.", {
    x: 7.6, y: 5.25, w: 4.8, h: 1.2, fontFace: FONT_BODY, fontSize: 12, color: TXT_LIGHT, isTextBox: true, margin: 0, valign: "top", lineSpacingMultiple: 1.2,
  });
  pageNum(s, 4);
}

// =====================================================================================
// 5. PROBLEMS FOUND DURING CAPTURE
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "촬영 중 발견한 문제와 대응", "Problems & Fixes");
  const cols = [
    { icon: "alert", title: "자기 가림 (self-occlusion)",
      items: ["위에서 내려다보는 각도라 고개를 내밀면 머리가 어깨·엉덩이를 가리고, 턱을 괴면 팔이 엉덩이를 가림",
        "대응: 짧은 가림은 마지막 확실한 위치 유지, 긴 가림은 엉덩이-어깨 오프셋을 학습해 어깨를 따라 엉덩이 위치 추정",
        "단위 테스트 포함"] },
    { icon: "person", title: "화면에 사람이 둘",
      items: ["MoveNet SinglePose는 여러 명 중 누구를 잡을지 보장하지 않음",
        "대응: 시작 직후 확실히 잡힌 사람을 사용자로 자동 인식 → 관심영역(ROI) + 깊이 0.3~1.0m 밖 제거 → Kalman 필터로 추적",
        "다음 슬라이드에서 구조 설명"] },
    { icon: "ruler", title: "depth 이상치",
      items: ["일부 사진에서 약 4.68m의 무효 수준 값이 기록됨 (홀 필링 영향으로 추정)",
        "대응: 필터 비교 도구로 45/60/80cm 실측 — 재현되지 않음(일회성). 단일 픽셀 조회가 배경을 집을 가능성은 남아 keypoint 주변 패치 조회를 검토",
        "특징 확정(6주차) 때 이상치 필터링"] },
  ];
  const cw = 3.85, gap = 0.19, y0 = 1.85, h0 = 4.85;
  cols.forEach((c, i) => {
    const x = 0.7 + i * (cw + gap);
    card(s, x, y0, cw, h0);
    cardTitle(s, c.icon, x, y0, cw, c.title);
    bullets(s, c.items, x + 0.3, y0 + 1.1, cw - 0.6, h0 - 1.3, 13.5);
  });
  pageNum(s, 5);
}

// =====================================================================================
// 6. MULTI-PERSON PIPELINE
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "다인 환경 대응: 자동 사용자 인식 + 추적", "Multi-person Handling");
  const steps = [
    { t: "① 자동 인식", d: "처음 확실히 잡힌 사람(코·어깨·엉덩이 중 2개 이상, confidence 0.15 이상)을 사용자로 고정", icon: "target" },
    { t: "② ROI 생성", d: "어깨너비 ×2.2로 가로폭, 세로는 최초 값 고정. depth 범위는 사용자 거리 ±0.35m", icon: "diagram" },
    { t: "③ 마스킹", d: "ROI·깊이 범위 밖(다른 사람·배경)을 지운 뒤 MoveNet 입력. 저장 사진에도 동일 적용", icon: "camera" },
    { t: "④ Kalman 추적", d: "위치+속도 예측으로 사용자가 움직이거나 잠깐 가려져도 ROI가 따라감. 'r'로 재인식", icon: "chip" },
  ];
  const cw = 2.85, gap = 0.18, y0 = 1.9, h0 = 3.2;
  steps.forEach((st, i) => {
    const x = 0.7 + i * (cw + gap);
    card(s, x, y0, cw, h0);
    iconCircle(s, st.icon, x + 0.25, y0 + 0.25, 0.6, TEAL_DARK);
    s.addText(st.t, { x: x + 0.95, y: y0 + 0.25, w: cw - 1.1, h: 0.6, valign: "middle", fontFace: FONT_HEAD, fontSize: 14, bold: true, color: TXT_DARK, isTextBox: true, margin: 0 });
    s.addText(st.d, { x: x + 0.28, y: y0 + 1.1, w: cw - 0.56, h: h0 - 1.3, fontFace: FONT_BODY, fontSize: 13, color: "3C4A50", isTextBox: true, margin: 0, valign: "top", lineSpacingMultiple: 1.2 });
    if (i < steps.length - 1) {
      s.addText("▶", { x: x + cw - 0.02, y: y0 + h0 / 2 - 0.2, w: gap + 0.04, h: 0.4, align: "center", valign: "middle", fontSize: 9, color: TEAL, isTextBox: true, margin: 0 });
    }
  });
  card(s, 0.7, 5.3, 11.93, 1.3, WARN_SOFT);
  s.addText("중간에 발견한 버그", { x: 1.0, y: 5.4, w: 4, h: 0.3, fontFace: FONT_BODY, fontSize: 11, bold: true, color: WARN, isTextBox: true, margin: 0 });
  s.addText("처음 마스크는 depth 0(무효) 픽셀도 지웠습니다. 그런데 너무 가까이 와서 depth가 무효가 된 사용자가 마스킹으로 사라져, 근접 구간 인식이 불가능했습니다. 무효 픽셀은 남기도록 고쳤습니다.", {
    x: 1.0, y: 5.72, w: 11.4, h: 0.8, fontFace: FONT_BODY, fontSize: 13, color: TXT_LIGHT, isTextBox: true, margin: 0, valign: "top", lineSpacingMultiple: 1.2,
  });
  noteTag(s, 0.7, 6.72, "실제 두 명이 앉은 환경에서의 검증은 아직 — 한계", "warn");
  pageNum(s, 6);
}

// =====================================================================================
// 7. JUDGEMENT LOGIC
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "판정 로직 1차 통합", "Decision Logic");
  s.addText("자세(정상/주의/경고)와 화면 근접을 따로 판정해 OR로 결합 — 임계값은 10/6 실측 로그(1인) 기반 잠정값 — 6주차에 참가자를 늘려 확정", {
    x: 0.7, y: 1.7, w: 11.93, h: 0.45, fontFace: FONT_BODY, fontSize: 13, color: TXT_LIGHT, isTextBox: true, margin: 0,
  });
  const rows = [
    [{ text: "판정", options: H_OPT }, { text: "잠정 임계값", options: H_OPT }, { text: "근거 (실측 중앙값)", options: H_OPT }],
    [{ text: "화면 근접", options: C_OPT }, { text: "머리·가슴 depth < 0.40m", options: C_OPT }, { text: "정상 머리 0.43m, 앞숙임 0.35m (실측)", options: C_OPT }],
    [{ text: "앞숙임 / 뒤기댐", options: C_OPT }, { text: "엉덩이−가슴 차 > +0.10 / < −0.11m, 또는 머리−가슴 차 ≤ 0.07m(기댐)", options: C_OPT }, { text: "정상 +0.05 / 기댐 −0.12, 머리−가슴 0.15 → 0.06", options: C_OPT }],
    [{ text: "좌우 기울임", options: C_OPT }, { text: "어깨 중점 좌우 치우침 > 어깨너비 0.35배", options: C_OPT }, { text: "실측에서 정상 인식 (수치 미보정)", options: W_OPT }],
    [{ text: "주의 단계", options: C_OPT }, { text: "경고 임계값의 70% 이상", options: C_OPT }, { text: "—", options: C_OPT }],
  ];
  s.addTable(rows, { x: 0.7, y: 2.3, w: 7.7, colW: [1.9, 3.3, 2.5], rowH: 0.62, border: { type: "solid", color: "0B1A26", pt: 1.5 }, autoPage: false });
  card(s, 8.65, 2.3, 3.98, 3.1, TEAL_DARK);
  s.addText("알림 상태머신", { x: 8.9, y: 2.42, w: 3.5, h: 0.35, fontFace: FONT_HEAD, fontSize: 14, bold: true, color: TEAL, isTextBox: true, margin: 0 });
  bullets(s, [
    "근접: 감지 즉시 1회 → 60초 쿨다운",
    "자세 경고: 연속 3회(15초) 지속 시 1회 → 5분 쿨다운",
    "샘플링 5초 간격",
  ], 8.9, 2.9, 3.5, 2.4, 12, TXT_LIGHT);
  s.addText("카메라 수령 후 실시간 실측(10/6, 1인): 정상 유지, 뒤로 기댐·앞숙임·좌우 기울임 판정 확인. 235장 전체는 depth가 사진에 안 남아 재계산이 불가능하고, 기준값은 한 명 기준이라 참가자·거리를 늘려 다시 보정해야 합니다.", {
    x: 0.7, y: 5.55, w: 11.93, h: 0.95, fontFace: FONT_BODY, fontSize: 12, italic: true, color: TXT_MUTED, isTextBox: true, margin: 0, valign: "top", lineSpacingMultiple: 1.2,
  });
  pageNum(s, 7);
}

// =====================================================================================
// 8. NEAR-RANGE FALLBACK
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "근접 사각지대 보완: 화면 속 어깨너비로 거리 역산", "Near-range Fallback");
  card(s, 0.7, 1.85, 5.85, 4.75);
  cardTitle(s, "diagram", 0.7, 1.85, 5.85, "원리");
  s.addText("w · Z ≈ k", { x: 1.0, y: 2.95, w: 5.3, h: 0.8, align: "center", fontFace: FONT_HEAD, fontSize: 36, bold: true, color: TEAL_DARK, isTextBox: true, margin: 0 });
  bullets(s, [
    "w = 화면 속 어깨너비(화면 너비 대비), Z = 거리. 핀홀 모델에서 w는 Z에 반비례",
    "depth가 유효한 동안 k를 이동 중앙값으로 학습",
    "40cm 미만에서 depth가 사라지면 Z ≈ k / w 로 역산해 근접 여부 판정",
  ], 1.0, 3.9, 5.3, 2.6, 14);
  card(s, 6.78, 1.85, 5.85, 4.75);
  cardTitle(s, "alert", 6.78, 1.85, 5.85, "한계와 이어지는 과제");
  bullets(s, [
    "현재 k는 사용자별로 학습 — 앉자마자 10샘플이 쌓이기 전에는 추정 불가",
    "어깨가 가려지면 추정 불가",
    "40cm 미만 어깨너비 추정의 정확도는 미검증. 대신 얼굴 depth가 무효가 되는 것 자체를 근접 신호로 써서 실측 확인 (다음 슬라이드)",
    "→ 교수님 피드백(매번 기준을 잡아야 하나?)에 따라 성인 평균 어깨너비로 시작하는 방식으로 바꿀 계획 (다음 파트)",
  ], 7.08, 2.9, 5.3, 3.6, 14);
  pageNum(s, 8);
}

// =====================================================================================
// 9. LIVE TEST FINDINGS
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "카메라 실측으로 드러난 문제와 대응", "Live Test Findings");
  const cols = [
    { icon: "camera", title: "얼굴 마스킹 · 화면 까매짐",
      items: ["얼굴이 가까워지면 전경 마스크가 얼굴을 지워 사람 인식이 끊김",
        "원인: 구멍 메우기(SDK 기본, 먼 값)가 근접 무효 영역을 배경 거리로 채움",
        "대응: 가까운 값 모드, 최소 거리 0.05m, 마스크 닫힘 연산"] },
    { icon: "alert", title: "책상이 허리를 가림",
      items: ["엉덩이가 책상 위치로 추측돼 책상 depth를 읽음 → 정상도 slouch_back 오탐",
        "엉덩이 점수가 0.5 근처라 허리 depth가 켜졌다 꺼짐(정상↔주의 깜빡임)",
        "대응: 엉덩이 confidence ≥ 0.5일 때만 사용, 머리−가슴 depth 차를 보조 신호로"] },
    { icon: "ruler", title: "앞으로 숙이면 depth 무효",
      items: ["얼굴이 약 30cm 안으로 들어와 머리·가슴 depth가 비면 판정 근거가 사라짐",
        "대응: 코는 보이는데 머리 depth만 무효 → 근접 + 앞숙임 판정",
        "실측: WARNING (slouch_forward) [NEAR] 확인"] },
    { icon: "target", title: "기댐 신호의 겹침",
      items: ["머리−가슴 depth 차: 정상 0.153 / 기댐 0.062 / 심하게 ≈0",
        "앞숙임도 0.062로 줄어 기댐과 겹침 → 근접이 아니고 가슴 ≥ 0.55m일 때만 기댐 신호 사용",
        "한계: 한 명·자세당 1회 값 — 체형·앉는 거리에 따라 달라질 수 있음"] },
  ];
  const cw = 2.84, gap = 0.19, y0 = 1.85, h0 = 4.2;
  cols.forEach((c, i) => {
    const x = 0.7 + i * (cw + gap);
    card(s, x, y0, cw, h0);
    cardTitle(s, c.icon, x, y0, cw, c.title);
    bullets(s, c.items, x + 0.22, y0 + 1.1, cw - 0.44, h0 - 1.25, 11.5);
  });
  noteTag(s, 0.7, 6.25, "수정 후 정상·뒤기댐·앞숙임·좌우 기울임 판정과 알림 확인(1인) — 다인·다거리 재검증은 6주차", "warn");
  pageNum(s, 9);
}

// =====================================================================================
// 10. PROFESSOR FEEDBACK
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "교수님 피드백과 우리 상태", "Feedback Mapping");
  const rows = [
    [{ text: "피드백", options: H_OPT }, { text: "현재 상태", options: H_OPT }, { text: "대응", options: H_OPT }],
    [{ text: "목표를 최소 fps가 아닌 최대 fps로", options: C_OPT }, { text: "방향 수정", options: W_OPT }, { text: "캡처 29.4fps 실측 있음 → 추론·마스킹 포함 전체 파이프라인 fps를 재측정", options: C_OPT }],
    [{ text: "허리 위치 추정", options: C_OPT }, { text: "반영됨", options: G_OPT }, { text: "가림 보정 스무딩 + 엉덩이−가슴 depth 차 (책상 가림 시 신뢰도↓ → 머리−가슴 depth 차로 보완)", options: C_OPT }],
    [{ text: "정면 depth로 측면 자세 추정", options: C_OPT }, { text: "반영됨", options: G_OPT }, { text: "실측에서 분리됨: 엉덩이−가슴 정상 +0.05 / 기댐 −0.12m, 머리−가슴 0.15 → 0.06m", options: C_OPT }],
    [{ text: "앉을 때마다 기준점이 필요한가 → 절대값으로", options: C_OPT }, { text: "설계함", options: W_OPT }, { text: "중력 기준 각도 + 어깨너비 비율 (다음 슬라이드)", options: C_OPT }],
    [{ text: "체형(복부·여성 체형 등) 두께 차이", options: C_OPT }, { text: "미해결", options: W_OPT }, { text: "비율로 줄이되 한계 인정, 체형 다양한 참가자 확보", options: C_OPT }],
    [{ text: "거북목, 허리받침에서 엉덩이가 뜬 자세", options: C_OPT }, { text: "설계함", options: W_OPT }, { text: "head_forward_ratio, torso_pitch로 측정 — 클래스 정의는 6주차 논의", options: C_OPT }],
    [{ text: "팔 위치·책상 높이 고려", options: C_OPT }, { text: "미착수", options: W_OPT }, { text: "어깨-팔꿈치-손목 keypoint로 6주차 이후 검토", options: C_OPT }],
  ];
  s.addTable(rows, { x: 0.7, y: 1.85, w: 11.93, colW: [4.0, 1.5, 6.43], rowH: 0.62, fontSize: 12.5, border: { type: "solid", color: "0B1A26", pt: 1.5 }, autoPage: false });
  pageNum(s, 10);
}

// =====================================================================================
// 11. ABSOLUTE POSTURE DESIGN
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "절대값 판정 설계: 앉는 자리·카메라 각도·체격에 안 흔들리게", "Absolute Posture Design");
  const steps = [
    { n: "1", t: "3D 복원", d: "keypoint + depth를 카메라 내부 파라미터로 3D 점(m)으로 되돌림" },
    { n: "2", t: "중력 기준 축", d: "D455 내장 IMU(가속도계)로 위·앞·옆 축 생성. IMU 없으면 설치 각도 근사" },
    { n: "3", t: "각도·비율", d: "몸통·머리 벡터를 축에 분해 → 각도(°)와 어깨너비 대비 비율로 표현" },
  ];
  const cw = 3.85, gap = 0.19, y0 = 1.8;
  steps.forEach((st, i) => {
    const x = 0.7 + i * (cw + gap);
    card(s, x, y0, cw, 1.75);
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.25, y: y0 + 0.25, w: 0.5, h: 0.5, fill: { color: TEAL_DARK }, line: { type: "none" } });
    s.addText(st.n, { x: x + 0.25, y: y0 + 0.25, w: 0.5, h: 0.5, align: "center", valign: "middle", fontFace: FONT_HEAD, fontSize: 14, bold: true, color: TXT_LIGHT, isTextBox: true, margin: 0 });
    s.addText(st.t, { x: x + 0.9, y: y0 + 0.25, w: cw - 1.1, h: 0.5, valign: "middle", fontFace: FONT_HEAD, fontSize: 14, bold: true, color: TXT_DARK, isTextBox: true, margin: 0 });
    s.addText(st.d, { x: x + 0.28, y: y0 + 0.9, w: cw - 0.56, h: 0.8, fontFace: FONT_BODY, fontSize: 12.5, color: "3C4A50", isTextBox: true, margin: 0, valign: "top", lineSpacingMultiple: 1.15 });
  });
  const rows = [
    [{ text: "특징", options: H_OPT }, { text: "의미", options: H_OPT }, { text: "대응 피드백", options: H_OPT }],
    [{ text: "torso_pitch_deg", options: C_OPT }, { text: "몸통의 수직 대비 앞(+)/뒤(−) 기울기", options: C_OPT }, { text: "숙임·기댐, 엉덩이 뜸", options: C_OPT }],
    [{ text: "torso_roll_deg", options: C_OPT }, { text: "몸통의 좌우 기울기", options: C_OPT }, { text: "좌우 기울임", options: C_OPT }],
    [{ text: "head_forward_ratio", options: C_OPT }, { text: "어깨중점→코의 앞방향 성분 ÷ 어깨너비", options: C_OPT }, { text: "거북목", options: C_OPT }],
    [{ text: "shoulder_width_m", options: C_OPT }, { text: "두 어깨 3D 거리 (체격 스케일)", options: C_OPT }, { text: "체형 정규화", options: C_OPT }],
  ];
  s.addTable(rows, { x: 0.7, y: 3.75, w: 7.6, colW: [2.2, 3.5, 1.9], rowH: 0.5, border: { type: "solid", color: "0B1A26", pt: 1.5 }, autoPage: false });
  card(s, 8.55, 3.75, 4.08, 2.5, WARN_SOFT);
  s.addText("검증 상태", { x: 8.8, y: 3.87, w: 3.6, h: 0.3, fontFace: FONT_BODY, fontSize: 11, bold: true, color: WARN, isTextBox: true, margin: 0 });
  s.addText("합성 3D 자세 테스트에서 거리 0.6→0.9m, 체격 1.25배 변화에도 각도 변화 1° 미만, 카메라를 20° 기울여도 중력 방향을 알면 동일. 이는 계산식 검증이며 실제 사람·카메라 실측은 카메라 확보 후입니다.", {
    x: 8.8, y: 4.2, w: 3.6, h: 1.95, fontFace: FONT_BODY, fontSize: 12, color: TXT_LIGHT, isTextBox: true, margin: 0, valign: "top", lineSpacingMultiple: 1.2,
  });
  s.addText("근접 판정도 사용자별 학습 대신 성인 평균 어깨너비로 시작하는 방식으로 교체 예정", {
    x: 0.7, y: 6.45, w: 11.93, h: 0.4, fontFace: FONT_BODY, fontSize: 12, italic: true, color: TEAL, isTextBox: true, margin: 0,
  });
  pageNum(s, 11);
}

// =====================================================================================
// 12. LIMITS & CAMERA-DAY
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "한계와 카메라 확보 후 할 일", "Limits & Next");
  card(s, 0.7, 1.85, 5.85, 4.85);
  cardTitle(s, "alert", 0.7, 1.85, 5.85, "아직 검증되지 않은 것");
  bullets(s, [
    "판정(--judge) 기준값 — 1인·자세당 1회 실측, 참가자·앉는 거리·카메라 각도 확대 필요",
    "근접 보완의 40cm 미만 구간 정확도",
    "두 명이 앉은 환경에서의 자동 인식·추적",
    "허리가 책상에 가려질 때 허리 depth 기반 판정 불가 — 머리−가슴 depth 차로 보완(1인 기준값)",
    "절대 특징의 실제 분포·임계값, IMU 연동",
    "체형 다양성 — 참가자 2명, 단일 날짜",
  ], 1.0, 2.9, 5.3, 3.7, 14);
  card(s, 6.78, 1.85, 5.85, 4.85);
  cardTitle(s, "calendar", 6.78, 1.85, 5.85, "카메라 확보 상태에서 남은 실측 (체크리스트)");
  bullets(s, [
    "알림 쿨다운 시간 확인(60초/300초), 40cm 미만 근접, ROI 추적, 다인 환경",
    "참가자·거리·카메라 각도를 바꿔 --judge 로그 재측정, 기준값 재보정",
    "IMU 스트림·내부 파라미터 읽기, 새 특징을 캡처 로그에 기록",
    "거리·카메라 기울기 불변성 실험",
    "새 참가자·다른 조명/복장으로 추가 수집",
    "반납 전 로그 커밋 — depth는 사후 복구 불가",
  ], 7.08, 2.9, 5.3, 3.7, 14);
  pageNum(s, 12);
}

// =====================================================================================
// 13. WEEK 6 PLAN + CLOSE
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "6주차 계획", "Next Week");
  const plan = [
    "판정 기준 다인·다거리 재보정, 절대 특징 실험(IMU·불변성), 알림 쿨다운 확인",
    "데이터 라벨링 마무리, 교차검증 결과 반영, 추가 참가자 수집",
    "특징 확정(depth 이상치 필터링 포함)과 임계값 확정 — RULA 참고 + RF 기준모델 학습",
    "SQLite 저장 계층 구현",
  ];
  plan.forEach((t, i) => listRow(s, i, String(i + 1), t, 1.9, 0.8, 0.12, 15));
  s.addText("이상으로 5주차 발표를 마치겠습니다. 질문 받겠습니다.", {
    x: 0.7, y: 6.75, w: 11.93, h: 0.4, align: "center", fontFace: FONT_BODY, fontSize: 13, italic: true, color: TXT_MUTED, isTextBox: true, margin: 0,
  });
  pageNum(s, 13);
}

pres.writeFile({ fileName: "barunjase_week5.pptx" }).then(() => console.log("done"));
