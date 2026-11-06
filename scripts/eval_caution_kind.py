"""주의 알림의 종류 채점: eval_caution_tier.py는 '어떤 알림이든 울렸는가'만 셌다. 여기서는 울린 알림의 종류(posture_kind)가
기대 종류(eval_alert_level.EXPECT)와 맞는지, 기울임에서 방향이 맞는지를 자세별로 센다. 보정 있음(neck_forward 이동)만 평가한다.
주의 알림의 종류 = 주의 이상이 연속 3회가 된 시점의 판정 종류. 좌우 자세의 neck_forward 알림은 고개를 내미는 자세 특성상
틀린 알림이라고 단정할 수 없으므로 '기대 종류와 같음'은 엄격한 하한이다.
사용: python -m scripts.eval_caution_kind [--out experiments/caution_kind_eval.md]
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd

from scripts.eval_alert_level import CALIB_FRAMES, EXPECT, LEAN, PHASES, REF_NF, judge_row, load, replay
from src.logic.decision import NORMAL


def run(path, people):
    D = load(path, people)
    out = []
    for p, gp in D.groupby("p"):
        nn = gp[gp.pose == "normal"]
        n1 = nn[nn.rep == sorted(nn.rep.unique())[0]].sort_values("timestamp")
        first = n1.head(CALIB_FRAMES)
        shift, ci = REF_NF - first["neck_forward_offset_m"].median(), set(first.index)
        for (pose, rep), g in gp.groupby(["pose", "rep"]):
            g = g.sort_values("timestamp")
            ge = g[~g.index.isin(ci)]
            g = ge if len(ge) >= 300 else g
            if (pose in LEAN and not ("nose_x" in g and g["nose_x"].notna().any())) or len(g) < 300:
                continue
            t = g["timestamp"].to_numpy(float)
            t = t - t[0]
            js = [judge_row(r, shift) for r in g.to_dict("records")]
            exp = EXPECT.get(pose)
            for ph in PHASES:
                run_, k, fired, kind = 0, 0, False, None
                while ph + 5 * k <= t[-1]:
                    j = js[min(int(np.searchsorted(t, ph + 5 * k)), len(js) - 1)]
                    run_ = run_ + 1 if j.posture_level != NORMAL else 0
                    if run_ >= 3 and not fired:
                        fired, kind = True, j.posture_kind
                    k += 1
                w = [a for a in replay(js, t, ph) if a.startswith("posture")]
                out.append(dict(pose=pose, soft=fired, soft_kind=kind, warn=bool(w),
                                warn_ok=any(a == f"posture:{exp}" for a in w), soft_ok=fired and kind == exp))
    R = pd.DataFrame(out)
    R["either"], R["either_ok"] = R.soft | R.warn, R.soft_ok | R.warn_ok
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/caution_kind_eval.md")
    a = ap.parse_args()
    L = ["# 주의 알림의 종류·방향 채점 (보정 있음)", "",
         "열: 경고만 검출 / 경고만 종류 일치 / 경고+주의 검출 / 경고+주의 종류 일치 (%). back_neck은 기대 종류가 없어 종류 일치를 채점하지 않는다.", ""]
    for name, path, ppl in (("환경 1", "data/side_logs", ["p01", "p02", "p03"]), ("환경 2", "data/side_env2", ["p01", "p02"])):
        R = run(path, ppl)
        t = R[R.pose != "normal"].groupby("pose")[["warn", "warn_ok", "either", "either_ok"]].mean().mul(100).round(0).astype(int)
        L += [f"## {name}", "", "| 자세 | 경고 검출 | 경고 종류 일치 | 경고+주의 검출 | 경고+주의 종류 일치 |", "|---|---|---|---|---|"]
        L += [f"| {i} | {r.warn} | {r.warn_ok} | {r.either} | {r.either_ok} |" for i, r in t.iterrows()]
        L += ["", "주의 알림 종류 분포(위상·녹화 단위 횟수):", ""]
        for pose in ("left_lean_neck", "right_lean_neck", "back_neck", "chin"):
            c = Counter(R[(R.pose == pose) & R.soft].soft_kind)
            L.append(f"- {pose}: " + (", ".join(f"{k} {v}" for k, v in c.most_common(4)) or "없음"))
        L.append("")
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:6]))


if __name__ == "__main__":
    main()
