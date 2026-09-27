const pptxgen = require("pptxgenjs");
const fs = require("fs");

// ---------- palette (matches the uploaded weekly-report template: dark navy + mint teal) ----------
const NAVY      = "0B1A26";
const TEAL      = "2FE0C4";
const TEAL_DARK = "12655A";
const CARD      = "FFFFFF";
const TXT_LIGHT = "EAF3F1";
const TXT_MUTED = "9FB6BC";
const TXT_DARK  = "16262E";
const WARN      = "E8734A";
const WARN_SOFT = "3A2A22";

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

// icon in a colored circle, returns nothing (draws directly)
function iconCircle(slide, iconKey, x, y, d, circleColor) {
  slide.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: circleColor }, line: { type: "none" } });
  const pad = d * 0.26;
  slide.addImage({ data: ICON[iconKey], x: x + pad, y: y + pad, w: d - pad * 2, h: d - pad * 2 });
}

function kicker(slide, text) {
  slide.addText(text.toUpperCase(), {
    x: 0.7, y: 0.42, w: 8, h: 0.35, fontFace: FONT_BODY, fontSize: 12, bold: true,
    color: TEAL, charSpacing: 2, isTextBox: true, margin: 0,
  });
}
function pageTitle(slide, text, kickerText) {
  if (kickerText) kicker(slide, kickerText);
  slide.addText(text, {
    x: 0.7, y: kickerText ? 0.75 : 0.55, w: 11.6, h: 0.9, fontFace: FONT_HEAD, fontSize: 30, bold: true,
    color: TXT_LIGHT, isTextBox: true, margin: 0,
  });
}
function pageNum(slide, n) {
  slide.addText(String(n).padStart(2, "0"), {
    x: PW - 0.9, y: PH - 0.55, w: 0.6, h: 0.35, fontFace: FONT_BODY, fontSize: 11,
    color: TXT_MUTED, align: "right", isTextBox: true, margin: 0,
  });
}

// =====================================================================================
// 1. TITLE
// =====================================================================================
{
  const s = newSlide();
  iconCircle(s, "person", PW / 2 - 0.55, 0.75, 1.1, TEAL_DARK);
  s.addText("임베디드소프트웨어학과 · 캡스톤 프로젝트", {
    x: 1, y: 2.15, w: 11.33, h: 0.4, align: "center", fontFace: FONT_BODY, fontSize: 14,
    color: TEAL, bold: true, charSpacing: 1.5, isTextBox: true, margin: 0,
  });
  s.addText("바른자세", {
    x: 1, y: 2.6, w: 11.33, h: 1.1, align: "center", fontFace: FONT_HEAD, fontSize: 54,
    bold: true, color: TXT_LIGHT, isTextBox: true, margin: 0,
  });
  s.addText("듀얼캠 기반 책상 자세 · 디지털 디톡스 모니터링 앱", {
    x: 1, y: 3.75, w: 11.33, h: 0.55, align: "center", fontFace: FONT_BODY, fontSize: 18,
    color: TXT_MUTED, isTextBox: true, margin: 0,
  });
  s.addText("교수님 피드백 반영 — 전면 아키텍처 개편", {
    x: 1, y: 4.35, w: 11.33, h: 0.32, align: "center", fontFace: FONT_BODY, fontSize: 12.5,
    color: TEAL, bold: true, charSpacing: 1, isTextBox: true, margin: 0,
  });
  s.addShape(pres.ShapeType.roundRect, {
    x: PW / 2 - 1.7, y: 4.78, w: 3.4, h: 0.55, rectRadius: 0.28,
    fill: { color: TEAL_DARK }, line: { type: "none" },
  });
  s.addText("2주차 발표 · 2026.09.13", {
    x: PW / 2 - 1.7, y: 4.78, w: 3.4, h: 0.55, align: "center", valign: "middle",
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
    "01   피드백 반영 방향", "02   연구 목표", "03   시스템 구성",
    "04   시스템 구성 요소", "05   관련 기술·연구 요약", "06   AI 자세 분류 파이프라인",
    "07   예산 계획", "08   연구 개발 일정", "09   리스크 및 완화 전략",
    "10   참고문헌",
  ];
  const colW = 5.55, gapX = 0.5, startX = 0.7, startY = 1.75, rowH = 0.72;
  items.forEach((t, i) => {
    const col = Math.floor(i / 6);
    const row = i % 6;
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
// 3. OUR RESPONSE — BEFORE / AFTER
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "피드백 반영 방향: 전면 피벗", "Our Response");
  s.addText("네 가지 피드백을 모두 반영해, 하드웨어 구성 자체를 데스크톱 앱 구조로 전면 전환했다.", {
    x: 0.7, y: 1.8, w: 11.93, h: 0.5, fontFace: FONT_BODY, fontSize: 15, color: TXT_LIGHT, isTextBox: true, margin: 0,
  });

  const headerOpts = { fill: { color: TEAL_DARK }, color: TXT_LIGHT, bold: true, fontFace: FONT_BODY, fontSize: 13, valign: "middle" };
  const cellOpts = { fill: { color: CARD }, color: TXT_DARK, fontFace: FONT_BODY, fontSize: 12.5, valign: "middle" };
  const beforeOpts = { ...cellOpts, color: "6B7A80" };
  const afterOpts = { ...cellOpts, bold: true, color: TEAL_DARK };
  const rows = [
    [{ text: "항목", options: headerOpts }, { text: "기존 (1주차)", options: headerOpts }, { text: "변경 (2주차)", options: headerOpts }],
    [{ text: "컴퓨팅 장치", options: cellOpts }, { text: "라즈베리파이5 (2GB)", options: beforeOpts }, { text: "사용자 노트북/데스크톱 (별도 보드 없음)", options: afterOpts }],
    [{ text: "거리 감지", options: cellOpts }, { text: "HC-SR04P 초음파 센서", options: beforeOpts }, { text: "듀얼 웹캠 스테레오 매칭", options: afterOpts }],
    [{ text: "카메라", options: cellOpts }, { text: "기존 보유 카메라 1대", options: beforeOpts }, { text: "스테레오 카메라 모듈 신규 구매 (oCamS-1CGN-U)", options: afterOpts }],
    [{ text: "알림 방식", options: cellOpts }, { text: "MQTT로 노트북에 전송 후 TTS", options: beforeOpts }, { text: "같은 프로세스 내에서 즉시 TTS", options: afterOpts }],
    [{ text: "예산", options: cellOpts }, { text: "149,500원", options: beforeOpts }, { text: "289,300원 (한도 대비 139,300원 초과)", options: afterOpts }],
  ];
  s.addTable(rows, { x: 0.7, y: 2.45, w: 11.93, h: 4.05, colW: [2.4, 4.76, 4.77], border: { type: "solid", color: "0B1A26", pt: 2 }, autoPage: false });
  pageNum(s, 3);
}

// =====================================================================================
// 6. STUDY OBJECTIVE
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "연구 목표", "Study Objective");
  s.addText(
    "듀얼 웹캠 기반 스테레오 거리 추정과 자세 추정을 결합해 거북목·화면 근접·장시간 사용을 실시간으로 감지하고, " +
    "즉각적인 음성 피드백과 장기적인 디지털 디톡스 리포트로 이어지는 경량 데스크톱 백그라운드 애플리케이션을 구현한다.",
    { x: 0.7, y: 1.85, w: 11.93, h: 1.1, fontFace: FONT_BODY, fontSize: 17, color: TXT_LIGHT, isTextBox: true, margin: 0 }
  );

  const goals = [
    { icon: "target", title: "정확한 3단계 판정", desc: "자세를 정상/주의/경고로 분류하고, 규칙 기반과 학습 기반 분류기의 성능을 비교한다." },
    { icon: "volume", title: "즉각적인 개입", desc: "화면 근접·나쁜 자세 지속 시 같은 기기에서 바로 TTS 음성 알림을 재생한다." },
    { icon: "diagram", title: "장기적인 습관 개선", desc: "판정 이력과 화면 사용시간을 누적해 디지털 디톡스 리포트를 제공한다." },
    { icon: "chip", title: "경량 백그라운드 구동", desc: "게임 등 다른 작업과 동시에 실행돼도 성능에 영향이 없도록 리소스 사용을 최소화한다." },
  ];
  const cardW = 2.75, gap = 0.3, startX = 0.7, y = 3.35, h = 2.85;
  goals.forEach((g, i) => {
    const x = startX + i * (cardW + gap);
    s.addShape(pres.ShapeType.roundRect, { x, y, w: cardW, h, rectRadius: 0.12, fill: { color: CARD }, line: { type: "none" } });
    iconCircle(s, g.icon, x + 0.3, y + 0.3, 0.7, TEAL_DARK);
    s.addText(g.title, { x: x + 0.3, y: y + 1.15, w: cardW - 0.6, h: 0.5, fontFace: FONT_HEAD, fontSize: 15, bold: true, color: TXT_DARK, isTextBox: true, margin: 0 });
    s.addText(g.desc, { x: x + 0.3, y: y + 1.65, w: cardW - 0.6, h: h - 1.85, fontFace: FONT_BODY, fontSize: 12.5, color: "3C4A50", isTextBox: true, margin: 0, lineSpacingMultiple: 1.15 });
  });
  pageNum(s, 4);
}

// =====================================================================================
// 7. SYSTEM ARCHITECTURE DIAGRAM
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "시스템 구성", "System Overview");

  const boxLine = { color: "0B1A26", width: 1.5 };
  function box(x, y, w, h, title, sub, fill) {
    s.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: boxLine });
    s.addText(title, { x: x + 0.12, y: y + 0.08, w: w - 0.24, h: 0.35, fontFace: FONT_HEAD, fontSize: 13, bold: true, color: TXT_DARK, isTextBox: true, margin: 0 });
    if (sub) s.addText(sub, { x: x + 0.12, y: y + 0.42, w: w - 0.24, h: h - 0.5, fontFace: FONT_BODY, fontSize: 10.5, color: "445056", isTextBox: true, margin: 0, lineSpacingMultiple: 1.1 });
  }
  function arrow(x1, y1, x2, y2, label) {
    s.addShape(pres.ShapeType.line, { x: Math.min(x1,x2), y: Math.min(y1,y2), w: Math.abs(x2-x1)||0.01, h: Math.abs(y2-y1)||0.01,
      line: { color: TEAL, width: 2, endArrowType: "triangle" }, flipV: y2 < y1, flipH: x2 < x1 });
    if (label) s.addText(label, { x: Math.min(x1,x2) - 0.3, y: Math.min(y1,y2) - 0.32, w: Math.abs(x2-x1) + 0.6 || 1.5, h: 0.3,
      align: "center", fontFace: FONT_BODY, fontSize: 10.5, color: TEAL, isTextBox: true, margin: 0 });
  }

  box(0.6, 3.7, 2.1, 1.0, "스테레오 카메라", "oCamS-1CGN-U (좌·우 렌즈 동기화)", CARD);
  box(3.55, 3.05, 3.4, 2.3, "노트북 / 데스크톱 앱", "스테레오 매칭 → 거리 추정\nMoveNet Lightning 자세추정\n판정 로직 + 백그라운드 경량 프로세스", "E9FBF7");
  box(7.9, 2.3, 2.55, 1.15, "TTS 즉시 알림", "같은 프로세스에서 내장 스피커로 재생", CARD);
  box(7.9, 4.4, 2.55, 1.15, "디지털 디톡스 리포트", "로컬 저장 + 로컬 대시보드", CARD);

  arrow(2.7, 4.2, 3.55, 4.2);
  s.addText("좌·우 프레임\n(USB 3.0)", { x: 2.75, y: 3.86, w: 0.75, h: 0.68, align: "center", fontFace: FONT_BODY, fontSize: 9, color: TEAL, isTextBox: true, margin: 0, lineSpacingMultiple: 1.05 });
  arrow(6.95, 3.9, 7.9, 2.9, "판정 결과");
  arrow(6.95, 4.2, 7.9, 5.0, "판정 이력");

  s.addText("영상 프레임은 저장하지 않고 좌표·거리값만 실시간 처리 후 폐기한다 — 모든 처리가 한 기기 안에서 끝나 별도 서버나 클라우드로 전송되지 않는다.", {
    x: 0.7, y: 6.35, w: 11.93, h: 0.5, fontFace: FONT_BODY, fontSize: 13, italic: true, color: TXT_MUTED, isTextBox: true, margin: 0,
  });
  pageNum(s, 5);
}

// =====================================================================================
// 8. SYSTEM COMPONENTS
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "시스템 구성 요소", "Components");
  const parts = [
    { icon: "camera", title: "스테레오 카메라", desc: "oCamS-1CGN-U 1대, 좌·우 렌즈 동기화 캡처" },
    { icon: "ruler", title: "베이스라인 고정", desc: "120mm 고정 하드웨어 베이스라인 — 별도 거치대 제작 불필요" },
    { icon: "chip", title: "노트북 / 데스크톱", desc: "별도 보드 없이 사용자 PC에서 전부 처리" },
    { icon: "diagram", title: "경량 백그라운드", desc: "낮은 프로세스 우선순위 + 샘플링 주기 조절로 게임 등과 동시구동" },
  ];
  const cardW = 2.8, gap = 0.28, startX = 0.7, y = 2.1, h = 2.5;
  parts.forEach((p, i) => {
    const x = startX + i * (cardW + gap);
    s.addShape(pres.ShapeType.roundRect, { x, y, w: cardW, h, rectRadius: 0.12, fill: { color: CARD }, line: { type: "none" } });
    iconCircle(s, p.icon, x + cardW / 2 - 0.4, y + 0.3, 0.8, TEAL_DARK);
    s.addText(p.title, { x: x + 0.15, y: y + 1.3, w: cardW - 0.3, h: 0.55, align: "center", fontFace: FONT_HEAD, fontSize: 13.5, bold: true, color: TXT_DARK, isTextBox: true, margin: 0 });
    s.addText(p.desc, { x: x + 0.15, y: y + 1.85, w: cardW - 0.3, h: h - 2.0, align: "center", fontFace: FONT_BODY, fontSize: 11, color: "3C4A50", isTextBox: true, margin: 0, lineSpacingMultiple: 1.1 });
  });
  s.addText("총 예산 289,300원 (스테레오 카메라 모듈 신규 구매) — 상세 내역은 예산 계획 슬라이드 참고", {
    x: 0.7, y: 5.0, w: 11.93, h: 0.5, fontFace: FONT_BODY, fontSize: 13, italic: true, color: TXT_MUTED, isTextBox: true, margin: 0,
  });
  pageNum(s, 6);
}

// =====================================================================================
// 9. RELATED TECH & WORK — CONDENSED SUMMARY
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "관련 기술·연구 요약", "Related Work Summary");

  s.addText([
    { text: "자세 추정 모델", options: { bold: true, breakLine: true, color: TEAL, fontSize: 15 } },
    { text: "MoveNet Lightning(TFLite, Apache-2.0) 채택 — COCO 17-keypoint 출력이라 각도 기반 판정 로직은 모델 선택과 무관하게 재사용 가능. 구동 환경이 노트북 CPU로 바뀌며 더 높은 성능이 예상되어 3주차에 재측정한다.", options: { breakLine: true, paraSpaceAfter: 18, fontSize: 13.5 } },
    { text: "인체공학 평가 기준", options: { bold: true, breakLine: true, color: TEAL, fontSize: 15 } },
    { text: "RULA(Rapid Upper Limb Assessment)의 목·몸통·어깨 각도 점수화 기준을 정상/주의/경고 판정의 객관적 앵커로 사용한다.", options: { breakLine: false, fontSize: 13.5 } },
  ], {
    x: 0.7, y: 1.9, w: 5.8, h: 4.6, fontFace: FONT_BODY, color: TXT_LIGHT,
    lineSpacingMultiple: 1.25, isTextBox: true, margin: 0,
  });

  s.addText([
    { text: "관련 앱·연구 대비 차별점", options: { bold: true, breakLine: true, color: TEAL, fontSize: 15 } },
    { text: "웹캠 AI 앱(SitApp·Posturr): 카메라 1대 휴리스틱 거리 추정, 상관관계 분석 없음 → 듀얼캠 스테레오 매칭 + 디지털 디톡스 리포트로 보완", options: { bullet: true, breakLine: true, fontSize: 13.5, paraSpaceAfter: 10 } },
    { text: "웨어러블(Upright GO): 착용 부담 → 비접촉 상시 모니터링으로 대체", options: { bullet: true, breakLine: true, fontSize: 13.5, paraSpaceAfter: 10 } },
    { text: "PoseTrack(2025, 라즈베리파이+MediaPipe): Firebase 클라우드 저장 → 노트북 한 대에서 완전 로컬 완결", options: { bullet: true, breakLine: false, fontSize: 13.5 } },
  ], {
    x: 6.75, y: 1.9, w: 5.9, h: 4.6, fontFace: FONT_BODY, color: TXT_LIGHT,
    lineSpacingMultiple: 1.25, isTextBox: true, margin: 0,
  });
  pageNum(s, 7);
}

// =====================================================================================
// 10. AI PIPELINE
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "AI 자세 분류 파이프라인", "AI Training Pipeline");
  const steps = [
    { n: "01", title: "키포인트 정규화", desc: "엉덩이 중심 이동 + 어깨너비 스케일링" },
    { n: "02", title: "각도 특징 추출", desc: "목-어깨-엉덩이 각도 등 6~10개 특징" },
    { n: "03", title: "라벨링 (RULA 기준)", desc: "정상/주의/경고 + 교차 라벨링 검증" },
    { n: "04", title: "증강", desc: "좌우 반전, 키포인트 jitter" },
    { n: "05", title: "분류기 학습", desc: "랜덤포레스트 vs 임계값 규칙 비교" },
  ];
  const w = 2.15, gap = 0.16, startX = 0.7, y = 2.5, h = 2.7;
  steps.forEach((st, i) => {
    const x = startX + i * (w + gap);
    s.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.1, fill: { color: i === 4 ? TEAL_DARK : CARD }, line: { type: "none" } });
    s.addText(st.n, { x: x + 0.15, y: y + 0.15, w: w - 0.3, h: 0.5, fontFace: FONT_HEAD, fontSize: 22, bold: true, color: i === 4 ? TEAL : TEAL_DARK, isTextBox: true, margin: 0 });
    s.addText(st.title, { x: x + 0.15, y: y + 0.75, w: w - 0.3, h: 0.7, fontFace: FONT_HEAD, fontSize: 13, bold: true, color: i === 4 ? TXT_LIGHT : TXT_DARK, isTextBox: true, margin: 0 });
    s.addText(st.desc, { x: x + 0.15, y: y + 1.5, w: w - 0.3, h: h - 1.6, fontFace: FONT_BODY, fontSize: 10.5, color: i === 4 ? TXT_MUTED : "3C4A50", isTextBox: true, margin: 0, lineSpacingMultiple: 1.1 });
    if (i < steps.length - 1) {
      s.addShape(pres.ShapeType.line, { x: x + w, y: y + h / 2 - 0.01, w: gap, h: 0.01, line: { color: TEAL, width: 2, endArrowType: "triangle" } });
    }
  });
  s.addText("거리(스테레오 매칭 기반) 판정은 분류기에 포함하지 않고, 별도 임계값 규칙으로 판정 후 자세 판정 결과와 논리적으로 결합(OR)한다. 백그라운드 구동 시에는 샘플링 주기를 5초로 낮춰 리소스 사용을 최소화한다.", {
    x: 0.7, y: 5.65, w: 11.93, h: 0.75, fontFace: FONT_BODY, fontSize: 13, italic: true, color: TXT_MUTED, isTextBox: true, margin: 0,
  });
  pageNum(s, 8);
}

// =====================================================================================
// 11. BUDGET
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "예산 계획", "Budget Plan");
  const headerOpts = { fill: { color: TEAL_DARK }, color: TXT_LIGHT, bold: true, fontFace: FONT_BODY, fontSize: 13, valign: "middle" };
  const cellOpts = { fill: { color: CARD }, color: TXT_DARK, fontFace: FONT_BODY, fontSize: 12.5, valign: "middle" };
  const numOpts = { ...cellOpts, align: "right" };
  const rows = [
    [{ text: "부품", options: headerOpts }, { text: "수량", options: { ...headerOpts, align: "right" } }, { text: "소계", options: { ...headerOpts, align: "right" } }],
    [{ text: "스테레오 카메라 (oCamS-1CGN-U)", options: cellOpts }, { text: "1", options: numOpts }, { text: "289,300원", options: numOpts }],
    [{ text: "노트북 / 데스크톱", options: cellOpts }, { text: "1", options: numOpts }, { text: "기존 보유", options: numOpts }],
    [{ text: "합계", options: { ...headerOpts, align: "left" } }, { text: "", options: headerOpts }, { text: "289,300원", options: { ...headerOpts, align: "right" } }],
  ];
  s.addTable(rows, { x: 0.7, y: 1.9, w: 8.4, h: 4.5, colW: [5.2, 1.3, 1.9], border: { type: "solid", color: "0B1A26", pt: 2 }, autoPage: false });

  s.addShape(pres.ShapeType.roundRect, { x: 9.5, y: 1.9, w: 3.13, h: 4.5, rectRadius: 0.12, fill: { color: TEAL_DARK }, line: { type: "none" } });
  iconCircle(s, "wallet", 9.5 + 3.13/2 - 0.45, 2.25, 0.9, NAVY);
  s.addText("289,300원", { x: 9.6, y: 3.35, w: 2.93, h: 0.6, align: "center", fontFace: FONT_HEAD, fontSize: 22, bold: true, color: TXT_LIGHT, isTextBox: true, margin: 0 });
  s.addText("한도 150,000원 대비 139,300원 초과", { x: 9.6, y: 3.95, w: 2.93, h: 0.4, align: "center", fontFace: FONT_BODY, fontSize: 11, color: TXT_MUTED, isTextBox: true, margin: 0 });
  s.addText("DIY 듀얼 웹캠 대비 캘리브레이션·마운트 제작 리스크를 줄이기 위해 하드웨어 스테레오 카메라로 전환", { x: 9.7, y: 4.6, w: 2.73, h: 1.4, align: "center", fontFace: FONT_BODY, fontSize: 11.5, color: TXT_LIGHT, isTextBox: true, margin: 0, lineSpacingMultiple: 1.2 });
  pageNum(s, 9);
}

// =====================================================================================
// 12. SCHEDULE (Gantt-like)
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "연구 개발 일정", "Research Schedule");

  const phases = [
    { name: "관련 연구·설계", start: 1, end: 2 },
    { name: "카메라 세팅·캘리브레이션", start: 2, end: 4 },
    { name: "AI 자세·거리 추정 개발", start: 4, end: 7 },
    { name: "통합 · 백그라운드 최적화", start: 7, end: 9 },
    { name: "종합 평가", start: 9, end: 10 },
    { name: "논문 작성", start: 10, end: 15 },
  ];
  const gridX = 3.2, gridY = 1.95, gridW = 9.4, weeks = 15;
  const rowH = 0.62, wkW = gridW / weeks;

  // header week numbers
  for (let w = 1; w <= weeks; w++) {
    s.addText(String(w), {
      x: gridX + (w - 1) * wkW, y: gridY - 0.4, w: wkW, h: 0.35, align: "center",
      fontFace: FONT_BODY, fontSize: 9.5, color: TXT_MUTED, isTextBox: true, margin: 0,
    });
  }
  const groupLabelY = gridY - 0.78;
  s.addShape(pres.ShapeType.rect, { x: gridX, y: groupLabelY, w: wkW * 10, h: rowH * phases.length + 0.78, fill: { color: "12655A", transparency: 78 }, line: { type: "none" } });
  s.addText("구현 (1~10주)", { x: gridX, y: groupLabelY, w: wkW * 10, h: 0.3, align: "center", fontFace: FONT_BODY, fontSize: 10, bold: true, color: TEAL, isTextBox: true, margin: 0 });
  s.addShape(pres.ShapeType.rect, { x: gridX + wkW * 10, y: groupLabelY, w: wkW * 5, h: rowH * phases.length + 0.78, fill: { color: WARN, transparency: 85 }, line: { type: "none" } });
  s.addText("논문 (11~15주)", { x: gridX + wkW * 10, y: groupLabelY, w: wkW * 5, h: 0.3, align: "center", fontFace: FONT_BODY, fontSize: 10, bold: true, color: WARN, isTextBox: true, margin: 0 });

  phases.forEach((p, i) => {
    const y = gridY + i * rowH;
    s.addText(p.name, { x: 0.6, y, w: gridX - 0.75, h: rowH - 0.06, valign: "middle", fontFace: FONT_BODY, fontSize: 12, color: TXT_LIGHT, isTextBox: true, margin: 0 });
    const barX = gridX + (p.start - 1) * wkW;
    const barW = (p.end - p.start + 1) * wkW - 0.06;
    const barColor = p.start >= 10 ? WARN : TEAL;
    s.addShape(pres.ShapeType.roundRect, { x: barX, y: y + 0.08, w: barW, h: rowH - 0.24, rectRadius: 0.05, fill: { color: barColor }, line: { type: "none" } });
  });
  pageNum(s, 10);
}

// =====================================================================================
// 13. RISKS
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "리스크 및 완화 전략", "Risk & Mitigation");
  const headerOpts = { fill: { color: TEAL_DARK }, color: TXT_LIGHT, bold: true, fontFace: FONT_BODY, fontSize: 12.5, valign: "middle" };
  const riskOpts = { fill: { color: CARD }, color: WARN, bold: true, fontFace: FONT_BODY, fontSize: 11.5, valign: "middle" };
  const mitOpts = { fill: { color: CARD }, color: TXT_DARK, fontFace: FONT_BODY, fontSize: 11.5, valign: "middle" };
  const rows = [
    [{ text: "리스크", options: headerOpts }, { text: "완화 전략", options: headerOpts }],
    [{ text: "스테레오 캘리브레이션 오차", options: riskOpts }, { text: "체스보드 패턴 기반 표준 캘리브레이션 절차 수립, 주기적 재보정 옵션 제공", options: mitOpts }],
    [{ text: "백그라운드 리소스 경합 (게임 등 동시 구동)", options: riskOpts }, { text: "샘플링 주기 5초로 축소 + 낮은 프로세스 우선순위 + CPU 전용 경량 모델 유지로 프레임 드랍 최소화", options: mitOpts }],
    [{ text: "자세 추정 정확도", options: riskOpts }, { text: "카메라 위치·거리·스테레오 baseline 고정, 다양한 조건의 검증셋 구성", options: mitOpts }],
    [{ text: "사생활 우려", options: riskOpts }, { text: "영상 미저장, 좌표만 실시간 처리 후 폐기", options: mitOpts }],
    [{ text: "알림 피로 · 논문 스코프", options: riskOpts }, { text: "근접 즉시·자세 15초(연속 3회) 지속 시 알림 + 60초/5분 쿨다운, 사용시간-자세 상관관계 분석을 핵심 기여로 구성", options: mitOpts }],
    [{ text: "exe 패키징 의존성 문제", options: riskOpts }, { text: "7주차에 저사양 기기로 패키징 사전 점검, 트레이 아이콘 등 완성형은 10주차 스트레치로 분리", options: mitOpts }],
  ];
  s.addTable(rows, { x: 0.7, y: 1.75, w: 11.93, h: 4.9, colW: [3.6, 8.33], border: { type: "solid", color: "0B1A26", pt: 2 }, autoPage: false });
  pageNum(s, 11);
}

// =====================================================================================
// 14. REFERENCES
// =====================================================================================
{
  const s = newSlide();
  pageTitle(s, "참고문헌", "References");
  const refs = [
    "TensorFlow Lite / MoveNet Lightning — blog.tensorflow.org, 2021 (모델 자체 벤치마크, 구동 환경은 데스크톱 CPU로 변경)",
    "MediaPipe Pose Landmarker — developers.google.com/mediapipe",
    "RULA (Rapid Upper Limb Assessment) — 인체공학 상지 자세 평가 방법론",
    "PoseTrack: 라즈베리파이+MediaPipe 기반 착석 자세 모니터링 연구 — arxiv.org/html/2508.11683",
    "Stereo Camera Depth Estimation with OpenCV — learnopencv.com (스테레오 매칭·디스패리티 기반 거리 추정)",
    "pyttsx3 — 오프라인 로컬 TTS 파이썬 라이브러리, pypi.org/project/pyttsx3",
  ];
  s.addText(refs.map((r, i) => ({ text: r, options: { bullet: true, breakLine: i < refs.length - 1, color: TXT_LIGHT } })), {
    x: 0.7, y: 2.0, w: 11.93, h: 4.5, fontFace: FONT_BODY, fontSize: 15, color: TXT_LIGHT,
    lineSpacingMultiple: 1.3, paraSpaceAfter: 10, isTextBox: true, margin: 0,
  });
  pageNum(s, 12);
}

// =====================================================================================
// 15. THANK YOU
// =====================================================================================
{
  const s = newSlide();
  iconCircle(s, "person", PW / 2 - 0.55, 2.3, 1.1, TEAL_DARK);
  s.addText("감사합니다", {
    x: 1, y: 3.6, w: 11.33, h: 0.9, align: "center", fontFace: FONT_HEAD, fontSize: 40, bold: true, color: TXT_LIGHT, isTextBox: true, margin: 0,
  });
  s.addText("Q & A", {
    x: 1, y: 4.5, w: 11.33, h: 0.5, align: "center", fontFace: FONT_BODY, fontSize: 16, color: TEAL, charSpacing: 3, isTextBox: true, margin: 0,
  });
}

pres.writeFile({ fileName: "barunjase_week2.pptx" }).then(() => {
  console.log("done");
});
