"""기존 규칙에 세션 보정 규칙 두 개를 더한 경우를 경고 단위로 평가한다(실험용, src/logic/decision.py는 바꾸지 않는다).

추가 규칙(둘 다 앱에는 구현되어 있지 않은 오프라인 평가):
  1. 뒤로 기댐(가슴 depth): 사람(환경)별 normal 가장 이른 녹화의 처음 150프레임(약 5초) 중앙값을 기준으로
     chest_depth_m가 T 이상 멀어지면 slouch_back 경고. T는 평가 대상이 아닌 사람들의 normal 프레임에서
     (사람별 99분위 중 최댓값)으로 정한다. 평가 대상 사람의 데이터는 T를 정하는 데 쓰지 않는다.
  2. 턱 괴기 우선: 코에서 손목까지 거리가 CHIN_WRIST_WARN_SW 이하이면 종류를 chin_rest로 정한다.
기존 규칙(보정: neck_forward_offset을 REF_NF로 이동)은 그대로 두고 세 가지 조건을 비교한다:
  기존 / +뒤로 기댐 / +뒤로 기댐 +턱 괴기 우선.
임계값 평가는 환경 1은 사람 단위 교차(한 명을 빼고 T를 정함), 환경 2는 환경 1의 다른 사람들로 T를 정한다.

사용: python -m scripts.eval_calibrated_rules [--out experiments/calibrated_rules_eval.md]
"""
from __future__ import annotations

import argparse
import dataclasses
import os

import numpy as np
import pandas as pd

from scripts.eval_alert_level import (BORDER, CALIB_FRAMES, EXPECT, LEAN, PHASES, REF_NF, load, replay)
from scripts.replay_judge_on_logs import DEPTH_COLS, num, real_keypoints, synth_keypoints
from src.logic.decision import CHIN_WRIST_WARN_SW, WARNING, judge, wrist_face_distance

VARIANTS = ["기존", "+뒤로 기댐", "+뒤로 기댐, 턱 괴기 우선"]


def baseline(gp: pd.DataFrame):
    nn = gp[gp.pose == "normal"]
    rep = sorted(nn.rep.unique())[0]
    first = nn[nn.rep == rep].sort_values("timestamp").head(CALIB_FRAMES)
    return first, REF_NF - first["neck_forward_offset_m"].median(), first["chest_depth_m"].median()


def judge_row(row, shift, dc, T, chest_rule, chin_rule):
    d = {c: num(row.get(c)) for c in DEPTH_COLS}
    if d["neck_forward_offset_m"] is not None:
        d["neck_forward_offset_m"] += shift
    real = real_keypoints(row)
    kps, asp = (real, 640 / 480) if real else (synth_keypoints(row), 1.0)
    j = judge(kps, d, frame_aspect=asp)
    if chin_rule:
        wf = wrist_face_distance(kps, asp)
        if wf is not None and wf <= CHIN_WRIST_WARN_SW:
            j = dataclasses.replace(j, posture_level=WARNING, posture_kind="chin_rest")
    if chest_rule and np.isfinite(dc) and dc >= T and j.posture_level != WARNING:
        j = dataclasses.replace(j, posture_level=WARNING, posture_kind="slouch_back")
    return j


def thresholds(D1: pd.DataFrame, q: float = 0.99) -> dict:
    """환경 1에서 사람별 normal 프레임의 chest_depth 이동량 q분위(보정 프레임 제외)."""
    out = {}
    for p, gp in D1.groupby("p"):
        first, _, base = baseline(gp)
        nm = gp[(gp.pose == "normal") & ~gp.index.isin(first.index)]
        out[p] = float((nm["chest_depth_m"] - base).quantile(q))
    return out


def evaluate(D: pd.DataFrame, T_by_person: dict):
    out = []
    for p, gp in D.groupby("p"):
        first, shift, base = baseline(gp)
        T = T_by_person[p]
        for (pose, rep), g in gp.groupby(["pose", "rep"]):
            g = g.sort_values("timestamp")
            g_eval = g[~g.index.isin(first.index)]
            g = g_eval if len(g_eval) >= 300 else g
            if pose in LEAN and not ("nose_x" in g and g["nose_x"].notna().any()):
                continue
            if len(g) < 300:
                continue
            t = g["timestamp"].to_numpy(float)
            t = t - t[0]
            dcs = (g["chest_depth_m"] - base).to_numpy(float)
            rows = g.to_dict("records")
            exp = EXPECT.get(pose)
            for vi, v in enumerate(VARIANTS):
                js = [judge_row(r, shift, dc, T, vi >= 1, vi >= 2) for r, dc in zip(rows, dcs)]
                for ph in PHASES:
                    al = replay(js, t, ph)
                    post = [a for a in al if a.startswith("posture")]
                    out.append(dict(p=p, pose=pose, rep=rep, variant=v, phase=ph, posture=bool(post),
                                    correct=bool(exp) and any(a == f"posture:{exp}" for a in post)))
    return pd.DataFrame(out)


def table(R: pd.DataFrame, title: str, note: str) -> list[str]:
    L = [f"### {title}", "", note, "",
         "| 자세 | " + " | ".join(f"{v}: 경고(%) / 기대 종류(%)" for v in VARIANTS) + " | 녹화·위상 수 |",
         "|---|" + "---|" * (len(VARIANTS) + 1)]
    for pose, g in R.groupby("pose"):
        cells = []
        for v in VARIANTS:
            h = g[g.variant == v]
            cells.append(f"{h.posture.mean()*100:.0f} / " + (f"{h.correct.mean()*100:.0f}" if EXPECT.get(pose) else "-"))
        L.append(f"| {pose} | " + " | ".join(cells) + f" | {len(g)//len(VARIANTS)} |")
    L += [""]
    for v in VARIANTS:
        h = R[R.variant == v]
        nm = h[h.pose == "normal"]
        bad = h[h.pose != "normal"]
        fmt = lambda x: f"{x.groupby('pose').posture.mean().mean()*100:.0f}%" if len(x) else "-"
        per = ", ".join(f"{p} {g.posture.mean()*100:.0f}%" for p, g in nm.groupby("p"))
        L.append(f"- {v}: 정상 오경고 {nm.posture.mean()*100:.0f}% ({per}); "
                 f"명확한 자세 검출 {fmt(bad[~bad.pose.isin(BORDER)])}, 경계 자세 검출 {fmt(bad[bad.pose.isin(BORDER)])}")
    return L + [""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/calibrated_rules_eval.md")
    ap.add_argument("--q", type=float, default=0.99, help="T를 정할 때 쓰는 normal 분위(1.0이면 최댓값)")
    a = ap.parse_args()
    D1 = load("data/side_logs", ["p01", "p02", "p03"])
    D2 = load("data/side_env2", ["p01", "p02"])
    T1 = thresholds(D1, a.q)
    T_env1 = {p: max(v for q, v in T1.items() if q != p) for p in T1}
    T_env2 = {p: max(v for q, v in T1.items() if q != p) for p in ("p01", "p02")}
    L = ["# 보정 기반 규칙 추가 평가 (실험용)", "",
         "- 기존 규칙(보정 포함)에 가슴 depth 기준 뒤로 기댐 규칙과 턱 괴기 우선 규칙을 더했을 때를 경고 단위로 비교한다.",
         "- T(가슴 depth 이동량 임계값)는 평가 대상이 아닌 사람들의 normal 프레임으로 정했다(사람별 99분위의 최댓값). "
         f"T 분위 {a.q}. 환경 1 평가 T: {', '.join(f'{p} {v:.3f} m' for p, v in T_env1.items())}; "
         f"환경 2 평가 T: {', '.join(f'{p} {v:.3f} m' for p, v in T_env2.items())}.",
         "- 앱에 구현되어 있지 않은 오프라인 평가이며, `decision.py`의 임계값 설정 이력과 별개의 사후 추가 규칙이다.", ""]
    L += table(evaluate(D1, T_env1), "환경 1 (p01, p02, p03; 사람 단위 교차 T)", "보정: 정상 첫 5초 기준")
    L += table(evaluate(D2, T_env2), "환경 2 (p01, p02; 환경 1의 다른 사람으로 정한 T)",
               "환경 2 normal은 길이가 짧아 기준 프레임을 포함해 평가했다(정상 오경고는 낙관적).")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
