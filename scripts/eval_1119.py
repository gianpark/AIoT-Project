"""11/19 촬영분 평가 (논문 4.11용). eval_alert_level.py와 같은 재생 규칙(5초 간격 샘플, 자세 경고 연속 3회,
쿨다운, 샘플 시작 위상 0~4.5초 10가지 평균)을 쓴다. 참가자는 p01, p02 두 명이다(p03 없음).

A. 정상 보강(data/side_1119): normal_long(5분)을 한 세션으로 이어서 재생해 시간당 알림 횟수를 센다.
   normal_hands(손을 얼굴로 가져가는 정상 동작), normal_shift(앉는 위치 변경)는 녹화 단위(20초) 경고 발생률.
B. p01 v4(data/side_p01_v4, 환경 1 구도에 맞춰 다시 찍은 11자세): 녹화 단위 경고 발생률을 예전 p01(side_logs)과 비교.
C. 좌우 기울임(data/side_1119): 공유용 tilt(목 내밀지 않음, 약/강)와 우리 조 lean_neck. 좌우 치우침 보정 유무와
   LATERAL_WARN 값(0.35 기본)에 따른 검출과 normal_long 오경고를 함께 본다.

보정(모두 오프라인, 앱 미구현):
- 목: 그 사람 normal_long 처음 150프레임(약 5초)의 neck_forward_offset_m 중앙값을 REF_NF로 옮긴다(eval_alert_level과 같음).
  v4는 v4 normal 처음 150프레임을 쓴다. 기준 프레임은 평가에서 뺀다.
- 좌우: 같은 150프레임의 좌우 치우침(lateral_offset) 중앙값을 뺀다.
제외: p01 tilt_right_strong(첫 촬영, 엉덩이까지 옆으로 이동) -> r2 사용, p02 tilt_left_strong(첫 촬영, 반대 방향) -> r2 사용.

사용: python -m scripts.eval_1119 [--out experiments/eval_1119.md]
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

import src.logic.decision as dec
from scripts.eval_alert_level import BORDER, CALIB_FRAMES, EXPECT, LEAN, PHASES, REF_NF, evaluate, load
from scripts.replay_judge_on_logs import DEPTH_COLS, num, real_keypoints, synth_keypoints
from src.logic.decision import WARNING, AlertStateMachine, judge

PEOPLE = ["p01", "p02"]
EXCLUDE = {("p01", "tilt_right_strong", "r1"): "엉덩이까지 옆으로 이동(몸 전체 이동)",
           ("p02", "tilt_left_strong", "r1"): "반대 방향으로 기울임"}
TILT_EXPECT = {"tilt_left_weak": "tilt_left", "tilt_left_strong": "tilt_left", "tilt_right_weak": "tilt_right",
               "tilt_right_strong": "tilt_right", "left_lean_neck": "tilt_left", "right_lean_neck": "tilt_right"}
_ORIG_LAT = dec.lateral_offset


def load_1119() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob("data/side_1119/side_*.csv")):
        m = re.match(r"side_(p\d+)_(.+?)(_r\d)?\.csv", os.path.basename(f))
        d = pd.read_csv(f)
        d["p"], d["pose"], d["rep"] = m[1], m[2], (m[3] or "_r1")[1:]
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def kps(row):
    real = real_keypoints(row)
    return (real, 640 / 480) if real else (synth_keypoints(row), 1.0)


def calib(gp: pd.DataFrame, base_pose: str):
    first = gp[gp.pose == base_pose].sort_values("timestamp").head(CALIB_FRAMES)
    nf_shift = REF_NF - first["neck_forward_offset_m"].median()
    lats = [_ORIG_LAT(kps(r)[0]) for r in first.to_dict("records")]
    lat0 = float(np.nanmedian([v for v in lats if v is not None]))
    return set(first.index), float(nf_shift), lat0


def judges(g: pd.DataFrame, nf_shift: float, lat0: float, lat_warn: float = 0.35):
    dec.LATERAL_WARN = lat_warn
    dec.lateral_offset = (lambda k: None if _ORIG_LAT(k) is None else _ORIG_LAT(k) - lat0) if lat0 else _ORIG_LAT
    out = []
    try:
        for row in g.to_dict("records"):
            d = {c: num(row.get(c)) for c in DEPTH_COLS}
            if d["neck_forward_offset_m"] is not None:
                d["neck_forward_offset_m"] += nf_shift
            k, asp = kps(row)
            out.append(judge(k, d, frame_aspect=asp))
    finally:
        dec.LATERAL_WARN, dec.lateral_offset = 0.35, _ORIG_LAT
    return out


def replay(js, t, phase, cooldown=300.0, caution=False):
    sm = AlertStateMachine(posture_cooldown_s=cooldown, caution_alert=caution)
    al, samples, k = [], [], 0
    while phase + 5 * k <= t[-1]:
        i = int(np.searchsorted(t, phase + 5 * k))
        if i >= len(js):
            break
        al += sm.update(5.0 * k, js[i])
        samples.append(js[i])
        k += 1
    return al, samples


VARIANTS = [("보정 없음", False, False), ("목 보정", True, False), ("목+좌우 보정", True, True)]


def part_a(D: pd.DataFrame) -> list[str]:
    L = ["## A. 정상 보강 (p01, p02)", "",
         "### A-1. normal_long: 5분 연속 재생 (기준 150프레임 제외, 약 295초)", "",
         "알림 횟수는 위상 10가지 평균. 시간당 값은 5분 녹화를 늘려 계산한 것이라 사람당 1회 녹화의 예비 값이다.", "",
         "| 사람 | 보정 | 쿨다운 | 자세 경고 알림(회/녹화) | 시간당 | 경고+주의 알림(회/녹화) | 근접 알림(회/녹화) | 샘플 경고 / 주의 이상(%) | 경고 샘플 종류(개수/전체 샘플, 위상 10가지 합) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for p in PEOPLE:
        gp = D[D.p == p]
        idx, nf_shift, lat0 = calib(gp, "normal_long")
        g = gp[(gp.pose == "normal_long") & ~gp.index.isin(idx)].sort_values("timestamp")
        t = g["timestamp"].to_numpy(float).copy(); t -= t[0]
        dur = t[-1]
        for name, cn, cl in VARIANTS:
            js = judges(g, nf_shift if cn else 0.0, lat0 if cl else 0.0)
            for cd in (300.0, 60.0):
                w = c = pr = 0
                samp = []
                for ph in PHASES:
                    a1, s = replay(js, t, ph, cd, False)
                    a2, _ = replay(js, t, ph, cd, True)
                    w += sum(x.startswith("posture") for x in a1)
                    c += sum(x.startswith("posture") or x == "caution" for x in a2)
                    pr += sum(x == "proximity" for x in a1)
                    samp += s
                n = len(PHASES)
                kinds = pd.Series([j.posture_kind for j in samp if j.posture_level == WARNING]).value_counts()
                ks = ", ".join(f"{k} {v}/{len(samp)}" for k, v in kinds.items()) or "-"
                L.append(f"| {p} | {name} | {int(cd)}초 | {w / n:.1f} | {w / n * 3600 / dur:.1f} | {c / n:.1f} | {pr / n:.1f} | "
                         f"{np.mean([j.posture_level == WARNING for j in samp]) * 100:.0f} / "
                         f"{np.mean([j.posture_level != dec.NORMAL for j in samp]) * 100:.0f} | {ks} |")
    L += ["", "### A-2. 손 동작·위치 변경 녹화 (20초, 녹화 단위 경고 발생률)", "",
          "| 사람 | 녹화 | 보정 | 자세 경고(%) | 경고 종류 | chin_rest 경고(%) 손목 점수 0.3 / 0.1 |", "|---|---|---|---|---|---|"]
    for p in PEOPLE:
        gp = D[D.p == p]
        idx, nf_shift, lat0 = calib(gp, "normal_long")
        for (pose, rep), g in gp[gp.pose.isin(["normal_hands", "normal_shift"])].groupby(["pose", "rep"]):
            g = g.sort_values("timestamp")
            t = g["timestamp"].to_numpy(float).copy(); t -= t[0]
            for name, cn, cl in VARIANTS[:2]:
                res = {}
                for ws in (0.3, 0.1):
                    dec.WRIST_MIN_SCORE = ws
                    js = judges(g, nf_shift if cn else 0.0, 0.0)
                    dec.WRIST_MIN_SCORE = 0.3
                    al = [replay(js, t, ph)[0] for ph in PHASES]
                    post = [[x for x in a if x.startswith("posture")] for a in al]
                    res[ws] = post
                post = res[0.3]
                kinds = pd.Series([x.split(":")[1] for a in post for x in a]).value_counts()
                chin = lambda P: np.mean([any(x == "posture:chin_rest" for x in a) for a in P]) * 100
                L.append(f"| {p} | {pose}_{rep} | {name} | {np.mean([bool(a) for a in post]) * 100:.0f} | "
                         f"{', '.join(f'{k} {v}' for k, v in kinds.items()) or '-'} | {chin(res[0.3]):.0f} / {chin(res[0.1]):.0f} |")
    return L + [""]


def part_b() -> list[str]:
    old = load("data/side_logs", ["p01"])
    v4 = load("data/side_p01_v4", ["p01"])
    L = ["## B. p01 v4 (환경 1 구도로 재촬영, 자세당 1회) 대 예전 p01 (side_logs, 자세당 3회)", "",
         "예전 p01 로그에는 keypoint 좌표가 없어 좌우 자세를 채점할 수 없다(-). v4는 좌표가 있다.", "",
         "| 자세 | 예전 p01 경고(%) 보정 없음 / 보정 | v4 경고(%) 보정 없음 / 보정 | v4 기대 종류(%) 보정 없음 / 보정 |",
         "|---|---|---|---|"]
    R = {}
    for tag, D in (("old", old), ("v4", v4)):
        for cal in (False, True):
            R[(tag, cal)] = evaluate(D, cal)
    poses = sorted(set(R[("v4", False)].pose))
    for pose in poses:
        def cell(tag, col="posture"):
            vals = []
            for cal in (False, True):
                g = R[(tag, cal)][R[(tag, cal)].pose == pose]
                vals.append("-" if len(g) == 0 or (col == "correct" and not EXPECT.get(pose)) else f"{g[col].mean() * 100:.0f}")
            return " / ".join(vals)
        L.append(f"| {pose} | {cell('old')} | {cell('v4')} | {cell('v4', 'correct')} |")
    L.append("")
    for tag, lab in (("old", "예전 p01"), ("v4", "p01 v4")):
        for cal in (False, True):
            h = R[(tag, cal)]
            bad = h[h.pose != "normal"]
            clear, border = bad[~bad.pose.isin(BORDER) & ~bad.pose.isin(LEAN)], bad[bad.pose.isin(BORDER)]
            f = lambda x: f"{x.groupby('pose').posture.mean().mean() * 100:.0f}%" if len(x) else "-"
            L.append(f"- {lab}, {'보정' if cal else '보정 없음'}: 정상 오경고 {h[h.pose == 'normal'].posture.mean() * 100:.0f}%, "
                     f"명확한 앞뒤 자세 검출 {f(clear)}, 경계 자세 검출 {f(border)} (좌우 lean_neck 제외)")
    # 환경 1 세 사람 요약: p01을 v4로 바꿨을 때
    p23 = load("data/side_logs", ["p02", "p03"])
    L += ["", "환경 1 세 사람 요약(사람별 값의 평균이 아니라 자세별 평균, 좌우 lean_neck 제외):", ""]
    for lab, Dp1 in (("p01=예전", old), ("p01=v4", v4)):
        for cal in (False, True):
            h = pd.concat([evaluate(Dp1, cal), evaluate(p23, cal)])
            bad = h[(h.pose != "normal") & ~h.pose.isin(LEAN)]
            clear, border = bad[~bad.pose.isin(BORDER)], bad[bad.pose.isin(BORDER)]
            f = lambda x: f"{x.groupby('pose').posture.mean().mean() * 100:.0f}%"
            L.append(f"- {lab}, {'보정' if cal else '보정 없음'}: 정상 오경고 {h[h.pose == 'normal'].posture.mean() * 100:.0f}%, "
                     f"명확한 자세 {f(clear)}, 경계 자세 {f(border)}")
    return L + [""]


def lat_series(g, lat0=0.0):
    v = [_ORIG_LAT(kps(r)[0]) for r in g.to_dict("records")]
    return np.array([np.nan if x is None else x - lat0 for x in v])


def part_c(D: pd.DataFrame, base: dict, title: str) -> list[str]:
    L = [f"## C. 좌우 기울임 (p01, p02) — {title}", "",
         "좌우 치우침 = 엉덩이 중점 대비 어깨 중점의 가로 치우침(어깨너비 단위, + = 본인 왼쪽). 기준값 LATERAL_WARN 0.35.",
         "검출은 녹화(20초) 안에 기대 방향 tilt 경고가 울린 비율(위상 10가지). 목 보정은 모든 열에 적용.", "",
         "### C-1. 녹화별", "",
         "| 사람 | 녹화 | 치우침 중앙값 원값 / 좌우 보정 | 프레임 |치우침|≥0.35 원값 / 보정(%) | 자세 경고(%) 목 보정 / 목+좌우 | 기대 tilt 종류(%) 목 보정 / 목+좌우 | 비고 |",
         "|---|---|---|---|---|---|---|"]
    rows = []
    for p in PEOPLE:
        gp = D[D.p == p]
        idx, nf_shift, lat0 = calib(*base[p][:2])
        L_p = []
        for (pose, rep), g in gp[gp.pose.isin(TILT_EXPECT)].groupby(["pose", "rep"]):
            g = g.sort_values("timestamp")
            t = g["timestamp"].to_numpy(float).copy(); t -= t[0]
            exp = TILT_EXPECT[pose]
            sign = 1 if exp == "tilt_left" else -1
            raw, cal = lat_series(g), lat_series(g, lat0)
            res = {}
            for cl in (False, True):
                js = judges(g, nf_shift, lat0 if cl else 0.0)
                al = [[x for x in replay(js, t, ph)[0] if x.startswith("posture")] for ph in PHASES]
                res[cl] = (np.mean([bool(a) for a in al]) * 100, np.mean([f"posture:{exp}" in a for a in al]) * 100)
            note = EXCLUDE.get((p, pose, rep), "")
            L.append(f"| {p} | {pose}_{rep} | {np.nanmedian(raw):+.3f} / {np.nanmedian(cal):+.3f} | "
                     f"{np.nanmean(sign * raw >= 0.35) * 100:.0f} / {np.nanmean(sign * cal >= 0.35) * 100:.0f} | "
                     f"{res[False][0]:.0f} / {res[True][0]:.0f} | {res[False][1]:.0f} / {res[True][1]:.0f} | {('제외: ' + note) if note else ''} |")
            if not note:
                rows.append(dict(p=p, pose=pose, rep=rep, g=g, t=t, exp=exp, nf=nf_shift, lat0=lat0))
        L += L_p
    L += ["", "보정 기준(처음 150프레임): " + ", ".join(
        f"{p} {base[p][2]} 치우침 {calib(*base[p][:2])[2]:+.3f}, 목 이동 {calib(*base[p][:2])[1]:+.3f} m" for p in PEOPLE), ""]
    # C-2. 기준값 민감도
    L += ["### C-2. 기준값(LATERAL_WARN)별 검출과 정상 오경고 (목 보정 / 목+좌우 보정, 제외 녹화 빼고)", "",
          "| 기준값 | 좌우 보정 | tilt 약 | tilt 강 | lean_neck | normal_long 자세 경고(회/시간, 쿨다운 300초) | normal_long 샘플 중 tilt 경고(%) |",
          "|---|---|---|---|---|---|---|"]
    longs = {}
    for p in PEOPLE:
        gp = D[D.p == p]
        idx, nf_shift, lat0 = calib(gp, "normal_long")
        g = gp[(gp.pose == "normal_long") & ~gp.index.isin(idx)].sort_values("timestamp")
        t = g["timestamp"].to_numpy(float).copy(); t -= t[0]
        longs[p] = (g, t, nf_shift, lat0)
    for lw, cl in [(x, c) for x in (0.35, 0.30, 0.25, 0.20) for c in (False, True)]:
        det = {"weak": [], "strong": [], "lean": []}
        for r in rows:
            js = judges(r["g"], r["nf"], r["lat0"] if cl else 0.0, lw)
            hit = np.mean([f"posture:{r['exp']}" in replay(js, r["t"], ph)[0] for ph in PHASES]) * 100
            det["weak" if r["pose"].endswith("weak") else "strong" if r["pose"].endswith("strong") else "lean"].append(hit)
        fa, tl, hrs = 0.0, [], 0.0
        for p, (g, t, nf_shift, lat0) in longs.items():
            js = judges(g, nf_shift, lat0 if cl else 0.0, lw)
            for ph in PHASES:
                a, s = replay(js, t, ph)
                fa += sum(x.startswith("posture") for x in a) / len(PHASES)
                tl += [j.posture_level == WARNING and (j.posture_kind or "").startswith("tilt") for j in s]
            hrs += t[-1] / 3600
        L.append(f"| {lw:.2f} | {'있음' if cl else '없음'} | {np.mean(det['weak']):.0f}% (n={len(det['weak'])}) | {np.mean(det['strong']):.0f}% (n={len(det['strong'])}) | "
                 f"{np.mean(det['lean']):.0f}% (n={len(det['lean'])}) | {fa / hrs:.1f} | {np.mean(tl) * 100:.1f} |")
    return L + [""]


def part_c3(D: pd.DataFrame) -> list[str]:
    """기준값을 낮췄을 때의 부작용: 정상 녹화의 좌우 치우침 분포, 앞뒤 자세 경고 종류 변화(좌표가 있는 녹화)."""
    v4, old = load("data/side_p01_v4", ["p01"]), load("data/side_logs", ["p01", "p02", "p03"])
    L = ["## C-3. 기준값을 낮출 때 부작용 점검", "",
         "정상 녹화(좌표 있는 것 전부)에서 |치우침|이 기준값 이상인 프레임 비율(%, 좌우 보정 없음):", "",
         "| 녹화 | 프레임 | 치우침 중앙값 | ≥0.35 | ≥0.30 | ≥0.25 | ≥0.20 |", "|---|---|---|---|---|---|---|"]
    sets = [(f"11/19 {p} {ps}_{r}", g) for p in PEOPLE for ps in ("normal_long", "normal_hands", "normal_shift")
            for r, g in D[(D.p == p) & (D.pose == ps)].groupby("rep")]
    sets.append(("v4 p01 normal", v4[v4.pose == "normal"]))
    sets += [(f"환경 1 {p} normal_{r}", g) for (p, r), g in old[old.pose == "normal"].groupby(["p", "rep"])
             if g["nose_x"].notna().any()]
    for n, g in sets:
        l = lat_series(g); l = l[~np.isnan(l)]
        L.append(f"| {n} | {len(l)} | {np.median(l):+.3f} | " + " | ".join(f"{(np.abs(l) >= th).mean() * 100:.1f}" for th in (0.35, 0.30, 0.25, 0.20)) + " |")
    E = pd.concat([v4, load("data/side_logs", ["p02", "p03"])])
    E = E[E.groupby(["p", "pose", "rep"])["nose_x"].transform(lambda s: s.notna().any())]
    L += ["", "앞뒤 자세 경고(좌표 있는 녹화: p01 v4, p02 r3, p03; 목 보정): 기준값 0.35 대 0.25, 경고(%) / 기대 종류(%)", "",
          "| 자세 | 0.35 | 0.25 |", "|---|---|---|"]
    R = {}
    for lw in (0.35, 0.25):
        dec.LATERAL_WARN = lw
        try:
            R[lw] = evaluate(E, True)
        finally:
            dec.LATERAL_WARN = 0.35
    for pose in sorted(set(R[0.35].pose) - LEAN):
        c = lambda lw: (lambda g: f"{g.posture.mean() * 100:.0f} / " + (f"{g.correct.mean() * 100:.0f}" if EXPECT.get(pose) else "-"))(R[lw][R[lw].pose == pose])
        L.append(f"| {pose} | {c(0.35)} | {c(0.25)} |")
    return L + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/eval_1119.md")
    a = ap.parse_args()
    D = load_1119()
    L = ["# 11/19 촬영분 평가 (p01, p02; 4.11용)", "",
         "- IRB 승인 후 촬영분(11/19). 재생 규칙은 eval_alert_level.py와 같다(5초 간격, 연속 3회, 쿨다운 300초 기본, 위상 10가지).",
         "- 참가자 2명, 녹화도 자세당 1~2회라 예비 결과다. p03은 이번 촬영에 없다.", ""]
    v4 = load("data/side_p01_v4", ["p01"])
    own = {p: (D[D.p == p], "normal_long", "normal_long") for p in PEOPLE}
    seat = {"p01": (v4, "normal", "v4 normal(§4 직전, 같은 자리)"), "p02": (D[D.p == "p02"], "normal_long", "normal_long")}
    L += part_a(D) + part_b()
    L += part_c(D, seat, "보정 기준: 같은 자리의 normal (p01은 v4 normal, p02는 normal_long)")
    L += part_c(D, own, "참고: 보정 기준을 normal_long으로 둔 경우 (p01은 normal_long 뒤에 다시 앉음)")
    L += part_c3(D)
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
