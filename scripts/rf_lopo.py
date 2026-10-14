"""T5: 참가자 단위 교차검증(leave-one-participant-out) 랜덤 포레스트, 9자세 분류.

p01→p02, p02→p01 두 방향. 같은 참가자 프레임이 학습/평가에 섞이지 않는다.
결측 처리: A=결측 플래그+학습 중앙값 대치, B=선택 특징이 결측인 프레임 제외.
특징 세트: depth(5), no_hip(엉덩이 의존 제외), all.
사용: python -m scripts.rf_lopo [--out experiments/rf_lopo.md]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from scripts.replay_judge_on_logs import load

DEPTH = ["head_depth_m", "chest_depth_m", "hip_depth_m", "neck_forward_offset_m", "torso_recline_offset_m"]
EXTRA = ["head_forward_deg", "torso_pitch_deg", "elbow_left_deg", "elbow_right_deg",
         "torso_len_sw", "head_up_m", "wrist_face_sw"]
HIP_DEP = {"hip_depth_m", "torso_recline_offset_m", "torso_pitch_deg"}
ABS = {"head_depth_m", "chest_depth_m"}  # 앉은 거리에 따라 변하는 절대 깊이
SETS = {"depth": DEPTH, "no_hip": [c for c in DEPTH + EXTRA if c not in HIP_DEP], "all": DEPTH + EXTRA}
SETS["relative"] = [c for c in SETS["no_hip"] if c not in ABS]  # 절대 깊이도 제외(사람·앉은 위치 의존 제거)


def prep(tr, te, cols, mode):
    if mode == "B":
        n0 = len(te)
        tr, te = tr.dropna(subset=cols), te.dropna(subset=cols)
        return tr[cols].values, tr.pose.values, te[cols].values, te.pose.values, n0
    med = tr[cols].median()
    def f(d):
        x = d[cols]
        flags = x.isna().astype(int).add_suffix("_na")
        return pd.concat([x.fillna(med), flags], axis=1).values
    return f(tr), tr.pose.values, f(te), te.pose.values, len(te)


def run(D, cols, mode, seed=0):
    out = {}
    for trp, tep in [("p01", "p02"), ("p02", "p01")]:
        tr, te = D[D.p == trp], D[D.p == tep]
        Xtr, ytr, Xte, yte, n_all = prep(tr, te, cols, mode)
        if len(Xtr) < 100 or len(Xte) < 100 or len(set(ytr)) < 2:
            out[(trp, tep)] = None
            continue
        rf = RandomForestClassifier(300, class_weight="balanced", random_state=seed, n_jobs=-1).fit(Xtr, ytr)
        pr = rf.predict(Xte)
        out[(trp, tep)] = dict(f1=f1_score(yte, pr, average="macro"), acc=(pr == yte).mean(),
                               keep=len(yte) / n_all, y=yte, p=pr, rf=rf, cols=cols, mode=mode)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="data/side_logs")
    ap.add_argument("--out", default="experiments/rf_lopo.md")
    a = ap.parse_args()
    D = load(a.logs)
    D = D[~D.pose.str.contains("lean")]  # 좌우 기울임은 keypoint x/y 로그가 없어 학습 불가 → 9자세
    L = ["# 참가자 단위 교차검증 RF (T5)", "",
         f"- 데이터 {len(D)}프레임, 9자세(left/right_lean_neck 제외: 좌표 로그 없음), 프레임 단위 평가(참가자 2명 → 두 방향 leave-one-participant-out 예비 평가).",
         "- RF 300그루, class_weight=balanced, seed 0. 일반화 성능의 근거로 해석하지 않는다.", "",
         "## 특징 세트 × 결측 처리", "",
         "| 특징 세트 | 결측 처리 | p01→p02 Macro-F1 | p02→p01 Macro-F1 | 평균 | 평가 프레임 유지율 |", "|---|---|---|---|---|---|"]
    keep = {}
    for sname, cols in SETS.items():
        for mode in ("A", "B"):
            r = run(D, cols, mode)
            v = [r[k]["f1"] if r[k] else float("nan") for k in r]
            kp = [r[k]["keep"] if r[k] else float("nan") for k in r]
            keep[(sname, mode)] = r
            L.append(f"| {sname} | {mode} | {v[0]:.3f} | {v[1]:.3f} | {np.nanmean(v):.3f} | "
                     f"{kp[0]*100:.0f}% / {kp[1]*100:.0f}% |")
    best_key = ("no_hip", "A")
    r = keep[best_key]
    classes = sorted(D.pose.unique())
    L += ["", f"## 클래스별 Recall(%) — {best_key[0]} + {best_key[1]}", "",
          "| 자세 | p01→p02 | p02→p01 |", "|---|---|---|"]
    rec = {}
    for k, v in r.items():
        rec[k] = recall_score(v["y"], v["p"], labels=classes, average=None, zero_division=0) * 100
    for i, c in enumerate(classes):
        L.append(f"| {c} | {rec[('p01','p02')][i]:.0f} | {rec[('p02','p01')][i]:.0f} |")
    L += ["", f"## 혼동 행렬(p01→p02, 행=실제, 열=예측, 프레임 수) — {best_key[0]} + {best_key[1]}", "",
          "| 실제＼예측 | " + " | ".join(classes) + " |", "|---|" + "---|" * len(classes)]
    cm = confusion_matrix(r[("p01", "p02")]["y"], r[("p01", "p02")]["p"], labels=classes)
    for c, row in zip(classes, cm):
        L.append(f"| {c} | " + " | ".join(str(x) for x in row) + " |")
    L += ["", f"## 특징 중요도 — {best_key[0]} + {best_key[1]} (두 방향 평균)", "", "| 특징 | 중요도 |", "|---|---|"]
    imp = None
    for v in r.values():
        names = v["cols"] + [c + "_na" for c in v["cols"]]
        s = pd.Series(v["rf"].feature_importances_, index=names)
        imp = s if imp is None else imp + s
    for n, x in (imp / 2).sort_values(ascending=False).head(10).items():
        L.append(f"| `{n}` | {x:.3f} |")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
