"""T4: 신호별 ROC/AUC/Youden 임계값 탐색 + 구분 안 되는 쌍의 새 신호 후보 비교.

- 프레임 단위, 결측(NaN)은 제외(0으로 채우지 않음). 참가자별·합산 따로.
- 임계값 일반화: p01에서 고른 값을 p02에서 평가(역방향도) — 같은 데이터로 고르고 보고하는 과적합 확인.
사용: python -m scripts.threshold_roc [--out experiments/threshold_roc.md]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from scripts.replay_judge_on_logs import load

NECK = ["neck_hard", "neck_light", "forward"]
BACK = ["back_hard", "back_light"]
# (이름, 양성 자세, 음성 자세, 신호, 부호: +1이면 클수록 양성)
PAIRS = [
    ("normal vs neck(내밈)", NECK, ["normal"], "neck_forward_offset_m", +1),
    ("normal vs back(기댐)", BACK, ["normal"], "neck_forward_offset_m", -1),
    ("normal vs chin(턱 괴기)", ["chin"], ["normal"], "wrist_face_sw", -1),
]
NEW = [
    ("slouch vs neck_hard", ["slouch"], ["neck_hard"],
     ["neck_forward_offset_m", "chest_depth_m", "head_depth_m", "head_up_m", "torso_len_sw", "head_forward_deg",
      "left_hip_score", "right_hip_score", "left_shoulder_score", "right_shoulder_score"]),
    ("back_neck vs normal", ["back_neck"], ["normal"],
     ["neck_forward_offset_m", "chest_depth_m", "head_depth_m", "head_up_m", "head_forward_deg", "torso_len_sw",
      "left_hip_score", "right_hip_score"]),
]


def xy(D, pos, neg, sig, sign):
    g = D[D.pose.isin(pos + neg)].dropna(subset=[sig])
    return (g[sig].values * sign), g.pose.isin(pos).astype(int).values


def best(x, y):
    fpr, tpr, thr = roc_curve(y, x)
    i = int(np.argmax(tpr - fpr))
    return thr[i], tpr[i], 1 - fpr[i]


def auc(x, y):
    return roc_auc_score(y, x) if len(set(y)) == 2 and len(y) > 10 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="data/side_logs")
    ap.add_argument("--out", default="experiments/threshold_roc.md")
    a = ap.parse_args()
    D = load(a.logs)
    D["pp"] = D.p
    L = ["# 기준값 ROC 탐색 (T4)", "",
         "프레임 단위·결측 제외·참가자 2명. **탐색적 결과**이며 같은 데이터로 고른 값은 과적합될 수 있다(아래 교차 평가 참조).", ""]
    for name, pos, neg, sig, sign in PAIRS:
        L += [f"## {name} — `{sig}` (부호 {'+' if sign>0 else '-'})", "",
              "| 대상 | n(+/-) | AUC | Youden 임계값 | 민감도 | 특이도 |", "|---|---|---|---|---|---|"]
        for lab, sub in [("p01", D[D.p == "p01"]), ("p02", D[D.p == "p02"]), ("합산", D)]:
            x, y = xy(sub, pos, neg, sig, sign)
            if len(set(y)) < 2:
                continue
            t, se, sp = best(x, y)
            L.append(f"| {lab} | {y.sum()}/{(1-y).sum()} | {auc(x,y):.3f} | {t*sign:.3f} | {se:.2f} | {sp:.2f} |")
        L += ["", "교차 평가(한 참가자에서 고른 임계값을 다른 참가자에 적용):", "",
              "| 학습→테스트 | 임계값 | 민감도 | 특이도 |", "|---|---|---|---|"]
        for tr, te in [("p01", "p02"), ("p02", "p01")]:
            x, y = xy(D[D.p == tr], pos, neg, sig, sign)
            t, _, _ = best(x, y)
            x2, y2 = xy(D[D.p == te], pos, neg, sig, sign)
            pred = x2 >= t
            L.append(f"| {tr}→{te} | {t*sign:.3f} | {pred[y2==1].mean():.2f} | {(~pred)[y2==0].mean():.2f} |")
        L.append("")
    L += ["## 현재 잠정 기준에서의 민감도/특이도(합산)", "", "| 기준 | 값 | 민감도 | 특이도 |", "|---|---|---|---|"]
    for lab, pos, neg, sig, sign, thr in [("neck 경고", NECK, ["normal"], "neck_forward_offset_m", +1, 0.18),
                                          ("neck 주의", NECK, ["normal"], "neck_forward_offset_m", +1, 0.171),
                                          ("back 경고", BACK, ["normal"], "neck_forward_offset_m", -1, 0.10),
                                          ("back 주의", BACK, ["normal"], "neck_forward_offset_m", -1, 0.109),
                                          ("chin 경고", ["chin"], ["normal"], "wrist_face_sw", -1, 0.65),
                                          ("chin 주의", ["chin"], ["normal"], "wrist_face_sw", -1, 0.815)]:
        x, y = xy(D, pos, neg, sig, sign)
        pred = x >= thr * sign
        L.append(f"| {lab} | {thr} | {pred[y==1].mean():.2f} | {(~pred)[y==0].mean():.2f} |")
    L += ["", "## 구분 안 되는 쌍의 새 신호 후보 (AUC; 0.5=무의미, 방향 무관하게 max(AUC,1-AUC))", ""]
    for name, pos, neg, sigs in NEW:
        L += [f"### {name}", "", "| 신호 | AUC 합산 | p01 | p02 | 중앙값(+) | 중앙값(-) |", "|---|---|---|---|---|---|"]
        rows = []
        for s in sigs:
            r = []
            for sub in (D, D[D.p == "p01"], D[D.p == "p02"]):
                x, y = xy(sub, pos, neg, s, 1)
                v = auc(x, y)
                r.append(max(v, 1 - v) if v == v else float("nan"))
            g = D[D.pose.isin(pos + neg)].dropna(subset=[s])
            rows.append((r[0], f"| `{s}` | {r[0]:.3f} | {r[1]:.3f} | {r[2]:.3f} | "
                         f"{g[g.pose.isin(pos)][s].median():.3f} | {g[g.pose.isin(neg)][s].median():.3f} |"))
        L += [t for _, t in sorted(rows, key=lambda z: -(z[0] if z[0] == z[0] else 0))] + [""]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
