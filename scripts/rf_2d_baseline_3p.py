"""2D 기준선 3명 LOPO: p01(v4, 11/19 촬영, 환경 1) + p02 3회차 + p03 (keypoint x/y 열이 있는 로그).

특징 세트: 2D(src/features/features_2d.py 23개), 깊이(rf_lopo의 no_hip 9개), 2D+깊이.
라벨: rf_2d_baseline_7class와 같은 7클래스 매핑(좌우 기울임 포함), 좌우 기울임을 뺀 5클래스도 함께 보고.
결측 처리 A, RF 300그루 balanced, 시드 0~4. Macro-F1은 세 fold 예측을 합쳐 계산(pooled), fold별 값도 함께.
사용: python -m scripts.rf_2d_baseline_3p
"""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from scripts.replay_judge_on_logs import load
from scripts.rf_lopo import SETS, prep
from src.features.features_2d import features_2d

MAP = {"normal": "normal", "neck_hard": "turtle_neck", "neck_light": "turtle_neck", "forward": "forward",
       "back_hard": "lean_back", "back_light": "lean_back", "left_lean_neck": "left_lean",
       "right_lean_neck": "right_lean", "back_neck": "bad_posture", "chin": "bad_posture", "slouch": "bad_posture"}
CO = ["bad_posture", "forward", "lean_back", "left_lean", "normal", "right_lean", "turtle_neck"]


def run(D, classes, seeds=range(5)):
    F2 = features_2d(D); D = pd.concat([D, F2], axis=1)
    sets = {"2D만": list(F2.columns), "깊이만(no_hip)": SETS["no_hip"], "2D+깊이": list(F2.columns) + SETS["no_hip"]}
    parts = sorted(D.p.unique()); out = {}
    for nm, cols in sets.items():
        pooled, perc, folds = [], [], []
        for s in seeds:
            ys, ps, fs = [], [], []
            for tep in parts:
                tr, te = D[D.p != tep], D[D.p == tep]
                Xtr, ytr, Xte, yte, _ = prep(tr, te, cols, "A")
                pr = RandomForestClassifier(300, class_weight="balanced", random_state=s, n_jobs=-1).fit(Xtr, ytr).predict(Xte)
                ys.append(yte); ps.append(pr)
                fs.append(f1_score(yte, pr, labels=classes, average="macro", zero_division=0))
            f = f1_score(np.concatenate(ys), np.concatenate(ps), labels=classes, average=None, zero_division=0)
            pooled.append(f.mean()); perc.append(f); folds.append(fs)
        out[nm] = (len(cols), np.mean(pooled), np.std(pooled), np.mean(folds, 0), np.mean(perc, 0))
    return D, parts, out


def table(title, D, parts, classes, out):
    L = [f"## {title}", "", f"- {len(D)}프레임(" + ", ".join(f"{q} {int((D.p == q).sum())}" for q in parts) + ")", "",
         "| 특징 세트 | 특징 수 | Macro-F1 (pooled, 시드 평균±SD) | " + " | ".join(f"{q} 평가" for q in parts) + " |",
         "|---|---|---|" + "---|" * len(parts)]
    for nm, (k, m, sd, fo, _) in out.items():
        L.append(f"| {nm} | {k} | {m:.3f} ± {sd:.3f} | " + " | ".join(f"{x:.3f}" for x in fo) + " |")
    L += ["", "클래스별 F1 (시드 평균)", "", "| 자세 | " + " | ".join(out) + " |", "|---|" + "---|" * len(out)]
    for i, c in enumerate(classes):
        L.append(f"| {c} | " + " | ".join(f"{v[4][i]:.2f}" for v in out.values()) + " |")
    return L + [""]


def main():
    D0 = pd.concat([load("data/side_logs"), load("data/side_p01_v4")], ignore_index=True)
    D0 = D0[D0.nose_x.notna()].reset_index(drop=True)
    D0["pose"] = D0.pose.map(MAP)
    L = ["# 2D 기준선 대 깊이 특징, 3명 LOPO", "",
         "- p01은 11/19 재촬영(v4, 자세당 1회, 환경 1), p02는 3회차, p03은 3회. 좌표가 있는 로그만 쓸 수 있어 참가자마다 회차가 다르다.",
         "- 결측 처리 A, RF 300그루 balanced, 시드 0~4. 이전 2명 결과(rf_2d_baseline*.md)와 같은 코드 경로.", ""]
    D, parts, out = run(D0, CO); L += table("7클래스 (좌우 기울임 포함)", D, parts, CO, out)
    c5 = [c for c in CO if c not in ("left_lean", "right_lean")]
    B = D0[D0.pose.isin(c5)].reset_index(drop=True)
    D, parts, out = run(B, c5); L += table("5클래스 (좌우 기울임 제외)", D, parts, c5, out)
    open("experiments/rf_2d_baseline_3p.md", "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
