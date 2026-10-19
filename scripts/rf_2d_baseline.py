"""2D 기준선: 깊이 없이 2D keypoint만 쓴 RF와 깊이 특징 RF·결합을 같은 데이터·같은 분할로 비교한다.

keypoint x/y 열이 있는 로그(p02 3회차, p03 3회)만 쓸 수 있어 참가자 2명(p02 3회차, p03)이다.
한 명으로 학습해 다른 한 명을 평가하는 두 방향. 기존 3명 결과(rf_lopo.md)와 숫자를 직접 비교하지 않는다.
특징 세트: 2d(src/features/features_2d.py), depth(rf_lopo의 no_hip), 2d+depth. 결측 처리 A. 300그루 balanced.
사용: python -m scripts.rf_2d_baseline [--merge-intensity]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score

from scripts.replay_judge_on_logs import load
from scripts.rf_lopo import SETS, prep
from src.features.features_2d import features_2d

MERGE = {"back_light": "back", "back_hard": "back", "neck_light": "neck", "neck_hard": "neck"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="data/side_logs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--merge-intensity", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.out is None:
        a.out = "experiments/rf_2d_baseline_merged.md" if a.merge_intensity else "experiments/rf_2d_baseline.md"
    D = load(a.logs)
    D = D[D["nose_x"].notna() & ~D.pose.str.contains("lean")].reset_index(drop=True)  # x/y 열이 있는 행만
    if a.merge_intensity:
        D = D.assign(pose=D.pose.replace(MERGE))
    F2 = features_2d(D)
    D = pd.concat([D, F2], axis=1)
    sets = {"2D만": list(F2.columns), "깊이만(no_hip)": SETS["no_hip"], "2D+깊이": list(F2.columns) + SETS["no_hip"]}
    parts = sorted(D.p.unique())
    classes = sorted(D.pose.unique())
    cnt = ", ".join(f"{q} {int((D.p == q).sum())}" for q in parts)
    L = ["# 2D 기준선 vs 깊이 융합 (같은 데이터·같은 분할)", "",
         f"- 데이터 {len(D)}프레임({cnt}; x/y 열이 있는 로그만), {len(classes)}클래스"
         f"{'(강도 합침)' if a.merge_intensity else ''}, 결측 처리 A, RF 300그루 balanced, 시드 {a.seed}.",
         "- 한 명으로 학습해 다른 한 명을 평가(두 방향). 참가자 2명이라 방향만 본다. 이전 3명 평가 수치와 직접 비교하지 않는다.", "",
         "## Macro-F1", "", "| 특징 세트 | 특징 수 | " + " | ".join(f"{q} 평가" for q in parts) + " | 평균 |",
         "|---|---|" + "---|" * len(parts) + "---|"]
    res = {}
    for name, cols in sets.items():
        fs, ys, ps = [], [], []
        for tep in parts:
            tr, te = D[D.p != tep], D[D.p == tep]
            Xtr, ytr, Xte, yte, _ = prep(tr, te, cols, "A")
            rf = RandomForestClassifier(300, class_weight="balanced", random_state=a.seed, n_jobs=-1).fit(Xtr, ytr)
            pr = rf.predict(Xte)
            fs.append(f1_score(yte, pr, average="macro"))
            ys.append(yte)
            ps.append(pr)
        res[name] = (fs, np.concatenate(ys), np.concatenate(ps))
        L.append(f"| {name} | {len(cols)} | " + " | ".join(f"{x:.3f}" for x in fs) + f" | {np.mean(fs):.3f} |")
    L += ["", "## 클래스별 recall(%, 두 평가 합산)", "", "| 자세 | " + " | ".join(sets) + " |", "|---|" + "---|" * len(sets)]
    rec = {n: recall_score(y, p, labels=classes, average=None, zero_division=0) * 100 for n, (_, y, p) in res.items()}
    for i, c in enumerate(classes):
        L.append(f"| {c} | " + " | ".join(f"{rec[n][i]:.0f}" for n in sets) + " |")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
