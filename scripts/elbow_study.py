"""팔꿈치 각도 특징(elbow_left/right_deg) 제거가 참가자 간 일반화에 미치는 영향.
forward가 p03에서 bad_posture로 빠지는 원인이 팔꿈치 특징이라는 진단(11/6)을 좌우 특징·세션 보정과 함께 다시 평가한다.
사용: python -m scripts.elbow_study [--seeds 5] [--out experiments/elbow_study.md]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from scripts import lateral_study as ls
from scripts.rf_2d_baseline_7class import CLASS_ORDER

ELBOW = ["elbow_left_deg", "elbow_right_deg"]


def evaluate(D, cols, seeds):
    rows = []
    for seed in range(seeds):
        pr, tr_, fm, frec = [], [], [], {}
        for ho in sorted(D.p.unique()):
            tr, te = D[D.p != ho], D[D.p == ho]
            p = ls.fit_predict(tr, te, cols, seed)
            pr += list(p); tr_ += list(te.pose)
            fm.append(f1_score(te.pose, p, labels=CLASS_ORDER, average="macro", zero_division=0))
            k = (te.pose == "forward").values
            frec[ho] = float((np.asarray(p)[k] == "forward").mean()) if k.any() else np.nan
        _, _, F, _ = precision_recall_fscore_support(tr_, pr, labels=CLASS_ORDER, zero_division=0)
        r = dict(seed=seed, macro_fold=np.mean(fm), macro_pooled=F.mean(),
                 **{"fold_" + h: v for h, v in zip(sorted(D.p.unique()), fm)},
                 **{"fwdrec_" + h: v for h, v in frec.items()})
        r.update({"F1_" + c: f for c, f in zip(CLASS_ORDER, F)})
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--p01-dir", default="data/side_p01_v2")
    ap.add_argument("--out", default="experiments/elbow_study.md")
    ap.add_argument("--models", default="rf,lgbm,xgb")
    a = ap.parse_args()
    d = ls.build(a.p01_dir, False)
    bl = ls.baseline_frames(d)
    d0, _, d_all = ls.prepare_sets(d, bl)
    sets = [("기준 9개", d0, ls.BASE), ("기준 9개 - 팔꿈치", d0, [c for c in ls.BASE if c not in ELBOW]),
            ("+좌우, 전체 보정", d_all, ls.BASE + ls.LAT),
            ("+좌우, 전체 보정 - 팔꿈치", d_all, [c for c in ls.BASE + ls.LAT if c not in ELBOW])]
    L = ["# 팔꿈치 각도 특징 제거 평가 (같은 환경 3명 교차검증, p01은 재촬영)", "",
         "값은 seed 평균(표준편차). 보정 프레임(사람별 normal 처음 150)은 모든 조건에서 제외. 팔꿈치 = elbow_left_deg, elbow_right_deg.", ""]
    for m in a.models.split(","):
        ls.MODEL = m
        L += [f"## {m}", ""]
        rows = []
        for name, D, cols in sets:
            R = evaluate(D, cols, a.seeds)
            ms = lambda c: f"{R[c].mean():.3f} ({R[c].std():.3f})"
            rows.append([name, ms("macro_fold"), ms("macro_pooled"), ms("F1_forward"),
                         " / ".join(f"{R['fold_'+h].mean():.2f}" for h in ["p01", "p02", "p03"]),
                         " / ".join(f"{R['fwdrec_'+h].mean()*100:.0f}%" for h in ["p01", "p02", "p03"])])
        L.append(ls.md_table(pd.DataFrame(rows, columns=["조건", "Macro-F1(fold)", "Macro-F1(pooled)", "F1 forward",
                                                         "fold별 F1 p01/p02/p03", "forward 재현율 p01/p02/p03"])))
        L.append("")
    open(a.out, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
