"""환경 2 로그에서 '정상 자세 기준 보정'(neck_forward_offset_m 이동) 전후의 판정 분포를 비교한다.

보정: 참가자별 normal 로그의 neck_forward_offset_m 중앙값을 구해 REF_NF(환경 1 정상 평균 0.143)로 옮기도록
모든 프레임의 값을 같은 만큼 이동한다(nf_adj = nf - nf_normal + REF_NF). 다른 특징·임계값은 그대로.
주의: 정상 로그로 보정하므로 normal의 오탐은 구조적으로 줄어든다. 확인할 것은 다른 자세의 검출이 유지되는지다.
사용: python scripts/replay_env2_calibrated.py [--logs data/side_env2] [--out experiments/replay_env2_calibrated.md]
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import pandas as pd

from scripts.replay_judge_on_logs import DEPTH_COLS, EXPECT, num, real_keypoints, synth_keypoints
from src.logic.decision import judge

REF_NF = 0.143


def classify(row, shift):
    d = {c: num(row.get(c)) for c in DEPTH_COLS}
    if d["neck_forward_offset_m"] is not None:
        d["neck_forward_offset_m"] += shift
    real = real_keypoints(row)
    j = judge(real, d, frame_aspect=640 / 480) if real else judge(synth_keypoints(row), d, frame_aspect=1.0)
    return j.posture_level, j.posture_kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="data/side_env2")
    ap.add_argument("--out", default="experiments/replay_env2_calibrated.md")
    a = ap.parse_args()
    D = []
    for f in sorted(glob.glob(os.path.join(a.logs, "side_p0[12]_*.csv"))):
        m = re.match(r"side_(p\d+)_(.+?)(_r\d)?\.csv", os.path.basename(f))
        d = pd.read_csv(f)
        d["p"], d["pose"] = m[1], m[2]
        D.append(d)
    D = pd.concat(D, ignore_index=True)
    base = {p: g.loc[g.pose == "normal", "neck_forward_offset_m"].median() for p, g in D.groupby("p")}
    shifts = {p: REF_NF - v for p, v in base.items()}
    out = {}
    for tag, use in (("보정 전", False), ("보정 후", True)):
        r = [classify(x, shifts[x["p"]] if use else 0.0) for x in D.to_dict("records")]
        out[tag] = pd.DataFrame(r, columns=["level", "kind"], index=D.index)
    L = ["# 환경 2 정상 자세 기준 보정 전후 비교", "",
         f"- 입력 `{a.logs}`, p01·p02, {len(D)}프레임. 보정 이동량: " + ", ".join(f"{p} {s:+.3f} m" for p, s in shifts.items()),
         f"- REF_NF = {REF_NF} (환경 1 정상 중앙값 p01 0.156, p02 0.130의 평균). normal은 보정 기준이라 구조적으로 개선됨", "",
         "## 자세별 기대 kind 검출률(%) (주의 이상에서 1순위가 기대 kind) / 정상 판정률", "",
         "| 자세 | 참가자 | 기대 kind | 검출 전 | 검출 후 | 정상 전 | 정상 후 |", "|---|---|---|---|---|---|---|"]
    for (pose, p), g in D.groupby(["pose", "p"]):
        exp = EXPECT.get(pose)
        row = [pose, p, exp or "-"]
        for key in ("det", "ok"):
            for tag in ("보정 전", "보정 후"):
                o = out[tag].loc[g.index]
                if key == "det":
                    row.append("-" if not exp else f"{((o.kind == exp) & (o.level != '정상')).mean() * 100:.0f}")
        for tag in ("보정 전", "보정 후"):
            o = out[tag].loc[g.index]
            row.append(f"{(o.level == '정상').mean() * 100:.0f}")
        L.append("| " + " | ".join(row) + " |")
    L += ["", "## 정상 자세에서 나온 kind(주의 이상, %)", "", "| 참가자 | 보정 | slouch_back | 기타 | 정상 |", "|---|---|---|---|---|"]
    for p, g in D[D.pose == "normal"].groupby("p"):
        for tag in ("보정 전", "보정 후"):
            o = out[tag].loc[g.index]
            sb = ((o.kind == "slouch_back") & (o.level != "정상")).mean() * 100
            ok = (o.level == "정상").mean() * 100
            L.append(f"| {p} | {tag} | {sb:.0f} | {100 - sb - ok:.0f} | {ok:.0f} |")
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
