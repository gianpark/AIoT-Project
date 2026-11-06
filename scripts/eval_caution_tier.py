"""경고 2단계 평가: 지금 앱은 경고(비율 1.0 이상)가 연속 3회일 때만 알림을 울린다. 판정에는 이미 "주의"(비율 0.7 이상)
단계가 있지만 알림에는 쓰이지 않는다. 주의 이상이 연속 3회 지속되면 약한 "주의 알림"을 추가로 울리는 안을 평가한다.
평가 틀(보정, 위상 10가지, 채점 대상 녹화)은 eval_alert_level.py와 같다.
사용: python -m scripts.eval_caution_tier [--out experiments/caution_tier_eval.md]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from scripts.eval_alert_level import (BORDER, CALIB_FRAMES, LEAN, PHASES, REF_NF, judge_row, load, replay)
from src.logic.decision import NORMAL

N_SOFT = 3  # 주의 이상 연속 횟수(경고와 같은 3회)


def soft_fired(js, t, phase, n=N_SOFT):
    run = k = 0
    while phase + 5 * k <= t[-1]:
        i = int(np.searchsorted(t, phase + 5 * k))
        if i >= len(js):
            break
        run = run + 1 if js[i].posture_level != NORMAL else 0
        if run >= n:
            return True
        k += 1
    return False


def evaluate(D, calibrate):
    out = []
    for p, gp in D.groupby("p"):
        nn = gp[gp.pose == "normal"]
        n1 = nn[nn.rep == sorted(nn.rep.unique())[0]].sort_values("timestamp")
        shift, calib_idx = 0.0, set()
        if calibrate:
            first = n1.head(CALIB_FRAMES)
            shift = REF_NF - first["neck_forward_offset_m"].median()
            calib_idx = set(first.index)
        for (pose, rep), g in gp.groupby(["pose", "rep"]):
            g = g.sort_values("timestamp")
            ge = g[~g.index.isin(calib_idx)]
            g = ge if len(ge) >= 300 else g
            if pose in LEAN and not ("nose_x" in g and g["nose_x"].notna().any()):
                continue
            if len(g) < 300:
                continue
            t = g["timestamp"].to_numpy(float)
            t = t - t[0]
            js = [judge_row(r, shift) for r in g.to_dict("records")]
            for ph in PHASES:
                w = any(a.startswith("posture") for a in replay(js, t, ph))
                s = soft_fired(js, t, ph)
                out.append(dict(p=p, pose=pose, warn=w, soft=s, either=w or s))
    return pd.DataFrame(out)


def summarize(R):
    nm = R[R.pose == "normal"]
    bad = R[R.pose != "normal"]
    avg = lambda x, c: x.groupby("pose")[c].mean().mean() * 100 if len(x) else float("nan")
    clear, border = bad[~bad.pose.isin(BORDER)], bad[bad.pose.isin(BORDER)]
    f = lambda v: "-" if v != v else f"{v:.0f}%"
    return [f(nm.warn.mean() * 100), f(nm.either.mean() * 100), f(avg(clear, "warn")), f(avg(clear, "either")),
            f(avg(border, "warn")), f(avg(border, "either"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/caution_tier_eval.md")
    a = ap.parse_args()
    sets = [("환경 1, p01·p02", "data/side_logs", ["p01", "p02"]), ("환경 1, p03", "data/side_logs", ["p03"]),
            ("환경 2 (환경 변화)", "data/side_env2", ["p01", "p02"])]
    L = ["# 경고 2단계(주의 알림) 평가", "",
         "경고 알림 = 현재 앱(경고 연속 3회). 주의 알림 추가 = 주의 이상이 연속 3회면 약한 알림을 추가로 울림. '경고만 / 경고+주의'로 표기.",
         "보정 없음 / 있음은 eval_alert_level.py와 같은 방식. 환경 2 보정 후 정상은 기준 프레임을 포함해 낙관적이다.", "",
         "| 데이터 | 보정 | 정상 오경고 | 명확한 자세 검출 | 경계 자세 검출 |", "|---|---|---|---|---|"]
    for title, path, people in sets:
        D = load(path, people)
        for cal, nm in ((False, "없음"), (True, "있음")):
            s = summarize(evaluate(D, cal))
            L.append(f"| {title} | {nm} | {s[0]} / {s[1]} | {s[2]} / {s[3]} | {s[4]} / {s[5]} |")
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
