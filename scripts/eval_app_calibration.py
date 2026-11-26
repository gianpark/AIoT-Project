"""앱의 개인 보정(src/logic/personal_calibration.py, 선우 11/26 커밋 71de83e)을 녹화 로그로 재생해 검증한다.

- 앱과 같은 수집 조건: 코·양 어깨 점수 >= 0.50, near_invalid 아님, 샘플 간격 >= 0.15초, 유효 범위 -0.20~0.50 m, 150개 중앙값.
- 정상 녹화가 20초라 150개(최소 22.5초)를 한 녹화로 못 채우므로, 사람별 normal 녹화를 rep 순서로 이어 붙여 앱처럼 시간순으로 넣는다
  (녹화 사이 시간 간격은 0으로 본다).
- 보정에 쓴 프레임은 평가에서 빼고, 남은 프레임이 300개(약 10초) 미만인 녹화는 채점하지 않는다.
- 비교: (1) 보정 없음 (2) 기존 오프라인 방식(첫 normal 처음 150프레임, eval_alert_level) (3) 앱 방식.
- 앱 보정기는 판정 직전 neck_forward_offset_m 하나만 옮긴다. 이 값은 decision.judge에서 목 내밈(값이 클 때)과
  뒤로 기댐(값이 작을 때) 두 판정에 같이 쓰인다.

사용: python -m scripts.eval_app_calibration [--out experiments/app_calibration_eval.md]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from scripts.eval_1126 import NEW, OLD, load_all, summ
from scripts.eval_alert_level import CALIB_FRAMES, EXPECT, LEAN, PHASES, REF_NF, judge_row, replay
from src.logic.personal_calibration import PersonalPostureCalibrator

INTERVAL = 0.15


def app_calibrate(nn: pd.DataFrame):
    """normal 녹화를 rep 순서로 이어 앱 보정기에 넣는다. (offset, 사용 인덱스, 걸린 초, 후보 중 탈락 비율)"""
    cal = PersonalPostureCalibrator()
    cal.start()
    used, elapsed, last_t, rejected, seen = [], 0.0, -1e9, 0, 0
    for rep in sorted(nn.rep.unique()):
        g = nn[nn.rep == rep].sort_values("timestamp")
        t0 = g.timestamp.iloc[0]
        for idx, r in g.iterrows():
            t = elapsed + (r.timestamp - t0)
            if t - last_t < INTERVAL:
                continue
            seen += 1
            ok = min(r.get("nose_score", 0), r.get("left_shoulder_score", 0), r.get("right_shoulder_score", 0)) >= 0.5
            ni = r.get("near_invalid", 0)
            ok = ok and not (pd.notna(ni) and bool(ni)) and pd.notna(r.get("neck_forward_offset_m"))
            if ok and cal.add_sample(r.neck_forward_offset_m):
                last_t = t
                used.append(idx)
                if cal.is_calibrated:
                    return cal, set(g.loc[:idx].index) | set(used), t, rejected / seen
            elif not ok:
                rejected += 1
        elapsed += g.timestamp.iloc[-1] - t0
    return cal, set(used), elapsed, rejected / max(seen, 1)


def evaluate(D: pd.DataFrame, mode: str):
    out, info = [], []
    for p, gp in D.groupby("p"):
        nn = gp[gp.pose == "normal"]
        shift, excl = 0.0, set()
        first = nn[nn.rep == sorted(nn.rep.unique())[0]].sort_values("timestamp").head(CALIB_FRAMES)
        off_shift = REF_NF - first.neck_forward_offset_m.median()
        if mode == "offline":
            shift, excl = off_shift, set(first.index)
        elif mode == "app":
            cal, excl, secs, rej = app_calibrate(nn)
            if cal.is_calibrated:
                shift = -cal.offset_m
            info.append(dict(p=p, done=cal.is_calibrated, secs=secs, rej=rej * 100,
                             base=cal.baseline_nf_m, app=shift, off=off_shift,
                             spread=float(np.subtract(*np.nanpercentile(nn.neck_forward_offset_m, [90, 10])))))
        for (pose, rep), g in gp.groupby(["pose", "rep"]):
            g = g[~g.index.isin(excl)].sort_values("timestamp")
            if len(g) < 300 or (pose in LEAN and not ("nose_x" in g and g.nose_x.notna().any())):
                continue
            t = g.timestamp.to_numpy(float); t -= t[0]
            js = [judge_row(r, shift) for r in g.to_dict("records")]
            exp = EXPECT.get(pose)
            for ph in PHASES:
                post = [a for a in replay(js, t, ph) if a.startswith("posture")]
                out.append(dict(p=p, pose=pose, rep=rep, phase=ph, posture=bool(post),
                                correct=bool(exp) and any(a == f"posture:{exp}" for a in post)))
    return pd.DataFrame(out), pd.DataFrame(info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/app_calibration_eval.md")
    args = ap.parse_args()
    D = load_all()
    res = {m: evaluate(D, m) for m in ("none", "offline", "app")}
    info = res["app"][1]
    L = ["# 앱 개인 보정(c 키) 재생 검증: p01~p07", "",
         "- 대상 코드: src/logic/personal_calibration.py + movenet_keypoints.py 수집 조건(커밋 71de83e). 카메라 없이 녹화 로그로 재생.",
         "- 앱 방식은 normal 녹화를 이어 붙여 150개를 채운다(20초 녹화 하나로는 부족). 보정에 쓴 구간은 평가에서 빼고 300프레임 미만 녹화는 채점 안 함.",
         "- 평가 규칙은 eval_alert_level과 같음(5초 간격, 경고 연속 3회, 위상 10가지 평균). 좌우 lean_neck 제외.", "",
         "## 1. 사람별 보정값", "",
         "| 사람 | 완료 | 걸린 시간(초) | 탈락 후보(%) | 앱 기준값(m) | 앱 이동량(m) | 오프라인 이동량(m) | normal nf 10~90% 폭(m) |",
         "|---|---|---|---|---|---|---|---|"]
    for r in info.itertuples():
        L.append(f"| {r.p} | {'예' if r.done else '아니오'} | {r.secs:.1f} | {r.rej:.0f} | "
                 f"{r.base if r.base is not None else float('nan'):.3f} | {r.app:+.3f} | {r.off:+.3f} | {r.spread:.3f} |")
    L += ["", "이동량 부호: +는 값을 키움(목 내밈 쪽, 뒤로 기댐 판정이 약해짐), -는 줄임(뒤로 기댐 쪽, 목 내밈 판정이 약해짐).", "",
          "## 2. 묶음별 결과", "",
          "| 묶음 | 방식 | 정상 오경고(%) | 명확한 자세 검출(%) | 경계 자세 검출(%) | 종류 일치(%) |", "|---|---|---|---|---|---|"]
    lab = {"none": "보정 없음", "offline": "오프라인(처음 150프레임)", "app": "앱(c 키)"}
    for name, ps in (("p01~p03", OLD), ("p04~p07", NEW), ("p01~p07", OLD + NEW)):
        for m in res:
            s = summ(res[m][0][res[m][0].p.isin(ps)])
            L.append(f"| {name} | {lab[m]} | {s['fa']:.0f} | {s['clear']:.0f} | {s['border']:.0f} | {s['correct']:.0f} |")
    L += ["", "## 3. 자세별 경고 발생률(%), 앱 방식 대 보정 없음 (앱/없음)", ""]
    A, N = res["app"][0], res["none"][0]
    ppl = sorted(A.p.unique())
    L += ["| 자세 | " + " | ".join(ppl) + " |", "|---" * (len(ppl) + 1) + "|"]
    for pose in sorted(A.pose.unique()):
        cells = []
        for p in ppl:
            a = A[(A.p == p) & (A.pose == pose)].posture.mean() * 100
            n = N[(N.p == p) & (N.pose == pose)].posture.mean() * 100
            cells.append(f"{a:.0f}/{n:.0f}")
        L.append(f"| {pose} | " + " | ".join(cells) + " |")
    open(args.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
