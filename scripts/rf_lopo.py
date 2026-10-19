"""T5: 참가자 단위 교차검증(leave-one-participant-out) 랜덤 포레스트, 9자세 분류.

참가자를 한 명씩 빼고(나머지로 학습) 그 참가자를 평가한다. 같은 참가자 프레임이 학습/평가에 섞이지 않는다.
참가자 수는 로그에서 자동으로 센다(p01, p02, p03 …).
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
    for tep in sorted(D.p.unique()):
        tr, te = D[D.p != tep], D[D.p == tep]
        Xtr, ytr, Xte, yte, n_all = prep(tr, te, cols, mode)
        if len(Xtr) < 100 or len(Xte) < 100 or len(set(ytr)) < 2:
            out[tep] = None
            continue
        rf = RandomForestClassifier(300, class_weight="balanced", random_state=seed, n_jobs=-1).fit(Xtr, ytr)
        pr = rf.predict(Xte)
        out[tep] = dict(f1=f1_score(yte, pr, average="macro"), acc=(pr == yte).mean(),
                        keep=len(yte) / n_all, y=yte, p=pr, rf=rf, cols=cols, mode=mode)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="data/side_logs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--merge-intensity", action="store_true",
                    help="강도 구분(light/hard)을 합친다: back_light+back_hard→back, neck_light+neck_hard→neck (7클래스)")
    a = ap.parse_args()
    if a.out is None:
        a.out = "experiments/rf_lopo_merged.md" if a.merge_intensity else "experiments/rf_lopo.md"
    D = load(a.logs)
    D = D[~D.pose.str.contains("lean")]  # 좌우 기울임은 keypoint x/y 로그가 없어 학습 불가 → 9자세
    if a.merge_intensity:
        D = D.assign(pose=D.pose.replace({"back_light": "back", "back_hard": "back",
                                          "neck_light": "neck", "neck_hard": "neck"}))
    parts = sorted(D.p.unique())
    cnt = ", ".join(f"{q} {int((D.p == q).sum())}" for q in parts)
    L = ["# 참가자 단위 교차검증 RF (T5, 참가자 %d명)" % len(parts), "",
         f"- 데이터 {len(D)}프레임({cnt}), {D.pose.nunique()}자세"
         f"{'(강도 합침: back, neck)' if a.merge_intensity else ''}(left/right_lean_neck 제외: 좌표 로그 없음), "
         f"프레임 단위 평가, 참가자를 한 명씩 빼고 나머지로 학습하는 leave-one-participant-out.",
         "- RF 300그루, class_weight=balanced, seed 0. 참가자가 적어 일반화 성능의 근거로 해석하지 않는다.", "",
         "## 특징 세트 × 결측 처리 (열: 평가한 참가자)", "",
         "| 특징 세트 | 결측 처리 | " + " | ".join(f"{q} Macro-F1" for q in parts) + " | 평균 | 평가 프레임 유지율 |",
         "|---|---|" + "---|" * len(parts) + "---|---|"]
    keep = {}
    for sname, cols in SETS.items():
        for mode in ("A", "B"):
            r = run(D, cols, mode)
            v = [r[k]["f1"] if r[k] else float("nan") for k in parts]
            kp = [r[k]["keep"] if r[k] else float("nan") for k in parts]
            keep[(sname, mode)] = r
            L.append(f"| {sname} | {mode} | " + " | ".join(f"{x:.3f}" for x in v) +
                     f" | {np.nanmean(v):.3f} | " + " / ".join(f"{x*100:.0f}%" for x in kp) + " |")
    best_key = ("no_hip", "A")
    r = keep[best_key]
    classes = sorted(D.pose.unique())
    L += ["", f"## 클래스별 Recall(%) — {best_key[0]} + {best_key[1]} (열: 평가한 참가자)", "",
          "| 자세 | " + " | ".join(parts) + " |", "|---|" + "---|" * len(parts)]
    rec = {}
    for k, v in r.items():
        if v:
            rec[k] = recall_score(v["y"], v["p"], labels=classes, average=None, zero_division=0) * 100
    for i, c in enumerate(classes):
        L.append(f"| {c} | " + " | ".join(f"{rec[q][i]:.0f}" if q in rec else "-" for q in parts) + " |")
    ys = np.concatenate([v["y"] for v in r.values() if v])
    ps = np.concatenate([v["p"] for v in r.values() if v])
    L += ["", f"## 혼동 행렬(모든 평가 합산, 행=실제, 열=예측, 프레임 수) — {best_key[0]} + {best_key[1]}", "",
          "| 실제＼예측 | " + " | ".join(classes) + " |", "|---|" + "---|" * len(classes)]
    cm = confusion_matrix(ys, ps, labels=classes)
    for c, row in zip(classes, cm):
        L.append(f"| {c} | " + " | ".join(str(x) for x in row) + " |")
    L += ["", f"## 특징 중요도 — {best_key[0]} + {best_key[1]} (평가별 평균)", "", "| 특징 | 중요도 |", "|---|---|"]
    imp, n_imp = None, 0
    for v in r.values():
        if not v:
            continue
        names = v["cols"] + [c + "_na" for c in v["cols"]]
        s_ = pd.Series(v["rf"].feature_importances_, index=names)
        imp = s_ if imp is None else imp + s_
        n_imp += 1
    for n, x in (imp / n_imp).sort_values(ascending=False).head(10).items():
        L.append(f"| `{n}` | {x:.3f} |")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
