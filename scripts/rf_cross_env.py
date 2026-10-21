"""교차 환경 평가: env1(기존 촬영, side_logs)로 학습 → env2(새 환경, side_env2)로 평가.

시나리오
 S1 환경만 새것: env1 전원(p01~p03)으로 학습 → env2 p01, p02 평가 (같은 사람, 새 환경)
 S2 사람·환경 모두 새것: env1에서 해당 참가자를 뺀 나머지로 학습 → env2 그 참가자 평가
 S3 기준선(env1 내 LOPO)과 비교, S4 env2 내 p01↔p02 교차(같은 환경, 다른 사람)
사용: python -m scripts.rf_cross_env [--out experiments/rf_cross_env.md]
"""
from __future__ import annotations
import argparse, os
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, confusion_matrix
from scripts.replay_judge_on_logs import load
from scripts.rf_lopo import SETS, prep

MERGE = {"back_light": "back", "back_hard": "back", "neck_light": "neck", "neck_hard": "neck"}


def fit_eval(tr, te, cols, seed=0):
    Xtr, ytr, Xte, yte, _ = prep(tr, te, cols, "A")
    rf = RandomForestClassifier(300, class_weight="balanced", random_state=seed, n_jobs=-1).fit(Xtr, ytr)
    pr = rf.predict(Xte)
    return f1_score(yte, pr, average="macro"), (pr == yte).mean(), yte, pr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env1", default="data/side_logs")
    ap.add_argument("--env2", default="data/side_env2")
    ap.add_argument("--out", default="experiments/rf_cross_env.md")
    a = ap.parse_args()
    E1 = load(a.env1); E2 = load(a.env2)
    E1 = E1[~E1.pose.str.contains("lean")]; E2 = E2[~E2.pose.str.contains("lean")]
    E2 = E2[E2.pose != "normal2"]
    L = ["# 교차 환경 평가 (env1 → env2)", "",
         f"- env1 {len(E1)}프레임({', '.join(f'{p} {int((E1.p==p).sum())}' for p in sorted(E1.p.unique()))}), "
         f"env2 {len(E2)}프레임({', '.join(f'{p} {int((E2.p==p).sum())}' for p in sorted(E2.p.unique()))}), 9자세(좌우기울임 제외)",
         "- RF 300그루, class_weight=balanced, 결측 처리 A. env2는 자세당 1회(약 20초) 촬영이라 표본이 작다. 방향 확인용.", ""]
    for name, f in (("9클래스", None), ("7클래스(강도 합침)", MERGE)):
        e1 = E1 if f is None else E1.assign(pose=E1.pose.replace(f))
        e2 = E2 if f is None else E2.assign(pose=E2.pose.replace(f))
        L += [f"## {name}", "", "| 평가 | " + " | ".join(SETS) + " |", "|---|" + "---|" * len(SETS)]
        rows = {}
        # S3 baseline: env1 LOPO (참고)
        for who in sorted(e1.p.unique()):
            rows.setdefault(f"S3 env1 내 LOPO, {who}", {})
            for s, c in SETS.items():
                rows[f"S3 env1 내 LOPO, {who}"][s] = fit_eval(e1[e1.p != who], e1[e1.p == who], c)[0]
        for who in sorted(e2.p.unique()):
            te = e2[e2.p == who]
            for s, c in SETS.items():
                rows.setdefault(f"S1 env1 전원→env2 {who} (환경만 새것)", {})[s] = fit_eval(e1, te, c)[0]
                rows.setdefault(f"S2 env1에서 {who} 제외→env2 {who} (사람·환경 새것)", {})[s] = fit_eval(e1[e1.p != who], te, c)[0]
                other = [q for q in sorted(e2.p.unique()) if q != who][0]
                rows.setdefault(f"S4 env2 {other}→env2 {who} (같은 환경, 다른 사람)", {})[s] = fit_eval(e2[e2.p == other], te, c)[0]
        for k, v in rows.items():
            L.append(f"| {k} | " + " | ".join(f"{v[s]:.3f}" for s in SETS) + " |")
        L.append("")
        # 상세: no_hip, S1 + S2 합산 recall / confusion
        c = SETS["no_hip"]; classes = sorted(e2.pose.unique())
        for tag, mk in (("S1", lambda w: e1), ("S2", lambda w: e1[e1.p != w])):
            ys, ps = [], []
            for who in sorted(e2.p.unique()):
                _, _, y, p = fit_eval(mk(who), e2[e2.p == who], c); ys.append(y); ps.append(p)
            y = np.concatenate(ys); p = np.concatenate(ps)
            rec = recall_score(y, p, labels=classes, average=None, zero_division=0) * 100
            L += [f"### {tag} no_hip 클래스별 Recall(%), 두 참가자 합산", "",
                  "| " + " | ".join(classes) + " |", "|" + "---|" * len(classes),
                  "| " + " | ".join(f"{x:.0f}" for x in rec) + " |", ""]
            cm = confusion_matrix(y, p, labels=classes)
            L += ["| 실제＼예측 | " + " | ".join(classes) + " |", "|---|" + "---|" * len(classes)]
            for cl, r in zip(classes, cm): L.append(f"| {cl} | " + " | ".join(map(str, r)) + " |")
            L.append("")
    # 분포 이동: 깊이 특징 평균 env1 vs env2 (normal)
    L += ["## 분포 이동: normal 자세 특징 평균(참가자별)", "", "| 특징 | " + " | ".join(f"{p} env1 | {p} env2" for p in sorted(E2.p.unique())) + " |",
          "|---|" + "---|---|" * len(E2.p.unique())]
    for col in SETS["no_hip"]:
        cells = []
        for p in sorted(E2.p.unique()):
            cells += [f"{E1[(E1.p==p)&(E1.pose=='normal')][col].mean():.3f}", f"{E2[(E2.p==p)&(E2.pose=='normal')][col].mean():.3f}"]
        L.append(f"| `{col}` | " + " | ".join(cells) + " |")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
