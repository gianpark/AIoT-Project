"""T2: data/side_logs/*.csv를 judge에 다시 넣어 자세별 판정 분포·혼동 행렬을 낸다.

로그에는 keypoint 좌표가 없으므로 어댑터가 합성 keypoint를 만든다:
- 어깨·엉덩이는 정렬된 위치(좌우 기울임 0) → tilt는 판정되지 않는다(검증 불가).
- 손목은 코에서 wrist_face_sw(어깨너비 단위)만큼 떨어진 곳에 둔다 → chin_rest 재현.
- 빈 칸은 None으로 유지(0으로 채우지 않는다).

사용: python scripts/replay_judge_on_logs.py [--logs data/side_logs] [--out experiments/replay_confusion.md]
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import re
from collections import Counter

import pandas as pd

from src.logic.decision import judge
from src.pose.movenet_keypoints import Keypoint

DEPTH_COLS = ["head_depth_m", "chest_depth_m", "hip_depth_m", "neck_forward_offset_m", "torso_recline_offset_m"]
NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
         "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
         "left_knee", "right_knee", "left_ankle", "right_ankle"]
GOOD = 0.9
EXPECT = {  # 포즈 -> 기대되는 judge kind (없으면 정상 기대)
    "normal": None, "back_hard": "slouch_back", "back_light": "slouch_back", "neck_hard": "neck_forward",
    "neck_light": "neck_forward", "forward": "neck_forward", "slouch": "neck_forward", "chin": "chin_rest",
    "back_neck": None, "left_lean_neck": "neck_forward", "right_lean_neck": "neck_forward",
}


def num(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def real_keypoints(row) -> list[Keypoint] | None:
    """10/14 이후 로그에는 keypoint x/y가 있다 → 실제 좌표로 재생(좌우 기울임·손목-얼굴까지 검증 가능)."""
    if num(row.get("nose_x")) is None:
        return None
    return [Keypoint(n, float(row[f"{n}_y"]), float(row[f"{n}_x"]), float(row[f"{n}_score"])) for n in NAMES]


def synth_keypoints(row) -> list[Keypoint]:
    """정렬된 몸통 + 코 기준 손목 거리(wrist_face_sw). 어깨너비=0.2, aspect=1로 계산."""
    pos = {"nose": (0.5, 0.3), "left_shoulder": (0.6, 0.5), "right_shoulder": (0.4, 0.5),
           "left_hip": (0.6, 0.9), "right_hip": (0.4, 0.9)}
    sc = {n: GOOD for n in pos}
    wf = num(row.get("wrist_face_sw"))
    if wf is not None:
        pos["left_wrist"] = (0.5 + wf * 0.2, 0.3)
        sc["left_wrist"] = GOOD
    kps = []
    for n in NAMES:
        x, y = pos.get(n, (0.0, 0.0))
        kps.append(Keypoint(n, y, x, sc.get(n, 0.0)))
    return kps


def classify(row):
    d = {c: num(row.get(c)) for c in DEPTH_COLS}
    real = real_keypoints(row)
    j = judge(real, d, frame_aspect=640 / 480) if real else judge(synth_keypoints(row), d, frame_aspect=1.0)
    return j.posture_level, j.posture_kind, {k for k, _ in j.issues}


def load(logs: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(os.path.join(logs, "side_*.csv"))):
        m = re.match(r"side_(p\d+)_(.+?)(_r\d)?\.csv", os.path.basename(f))
        d = pd.read_csv(f)
        d["p"], d["pose"], d["rep"] = m[1], m[2], (m[3] or "_r1")[1:]
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="data/side_logs")
    ap.add_argument("--out", default="experiments/replay_confusion.md")
    a = ap.parse_args()
    D = load(a.logs)
    res = [classify(r) for r in D.to_dict("records")]
    D["level"] = [r[0] for r in res]
    D["kind"] = [r[1] or "none" for r in res]
    D["issues"] = [r[2] for r in res]
    L = ["# 로그 재생 판정 검증 (T2)", "",
         f"- 입력: `{a.logs}` {D.groupby(['p','rep']).ngroups}묶음, {len(D)}프레임 (프레임 단위; 참가자 2명 예비 결과)",
         "- 어깨·엉덩이 합성 정렬 → tilt 판정 불가(해당 없음). 빈 칸은 None.", "",
         "## 자세별 정상/주의/경고 비율(%)", "", "| 자세 | 정상 | 주의 | 경고 | n |", "|---|---|---|---|---|"]
    for pose, g in D.groupby("pose"):
        v = g["level"].value_counts(normalize=True) * 100
        L.append(f"| {pose} | {v.get('정상',0):.1f} | {v.get('주의',0):.1f} | {v.get('경고',0):.1f} | {len(g)} |")
    kinds = sorted(D["kind"].unique())
    L += ["", "## 혼동 행렬: 자세(행) × judge kind(열), 프레임 %(주의 이상만 kind, 나머지 none)", "",
          "| 자세 | " + " | ".join(kinds) + " |", "|---|" + "---|" * len(kinds)]
    for pose, g in D.groupby("pose"):
        g2 = g.assign(k=g.apply(lambda r: r["kind"] if r["level"] != "정상" else "none", axis=1))
        v = g2["k"].value_counts(normalize=True) * 100
        L.append(f"| {pose} | " + " | ".join(f"{v.get(k,0):.1f}" for k in kinds) + " |")
    L += ["", "## 참가자·회차별 정상 자세 경고 오탐률(%)", "", "| 묶음 | 경고 | 주의 이상 |", "|---|---|---|"]
    for (p, r), g in D[D.pose == "normal"].groupby(["p", "rep"]):
        L.append(f"| {p} {r} | {(g.level=='경고').mean()*100:.1f} | {(g.level!='정상').mean()*100:.1f} |")
    hit = []
    for pose, exp in EXPECT.items():
        g = D[D.pose == pose]
        if exp:
            top = ((g.kind == exp) & (g.level != "정상")).mean() * 100
            anyi = g["issues"].apply(lambda s: exp in s).mean() * 100
            hit.append((pose, exp, top, anyi))
    L += ["", "## 기대 kind 검출률(%) — 1순위 / issues(복합 포함)", "",
          "| 자세 | 기대 kind | 1순위 | issues 포함 |", "|---|---|---|---|"]
    L += [f"| {p} | {e} | {t:.1f} | {i:.1f} |" for p, e, t, i in hit]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
