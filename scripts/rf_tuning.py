"""RF 경량화·파라미터 검증 (docs/RF_experiment_plan.docx).

바깥 루프: 참가자를 한 명씩 빼고 평가(leave-one-participant-out).
Baseline: 300그루, class_weight=balanced.
Lightweight: 50그루, max_depth=10, min_samples_split=5, min_samples_leaf=2.
Tuned: 랜덤 서치. 안쪽 검증은 학습 참가자의 (참가자, 촬영회차) 그룹 단위 GroupKFold라
같은 촬영의 프레임이 학습/검증에 섞이지 않는다. 탐색은 프레임이 서로 매우 비슷하므로 N프레임마다 1개만 써서 가볍게 한다.
특징: no_hip(기본), all / 결측 처리 A(플래그+중앙값).
측정: macro-F1, 정확도, 클래스별 recall, 모델 크기(KB, joblib), 샘플 1개 추론시간(ms, n_jobs=1).
사용: python -m scripts.rf_tuning [--feature-set no_hip] [--merge-intensity] [--n-iter 20]
"""
from __future__ import annotations

import argparse
import io
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

from scripts.replay_judge_on_logs import load
from scripts.rf_lopo import SETS, prep

SPACE = {
    "n_estimators": [30, 50, 100, 200],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "class_weight": [None, "balanced", "balanced_subsample"],
}
MODELS = {
    "Baseline": dict(n_estimators=300, class_weight="balanced"),
    "Lightweight": dict(n_estimators=50, max_depth=10, min_samples_split=5, min_samples_leaf=2,
                        class_weight="balanced"),
}


def size_kb(rf) -> float:
    buf = io.BytesIO()
    joblib.dump(rf, buf, compress=0)
    return buf.tell() / 1024


def latency_ms(rf, X, n=300, seed=0):
    rf.set_params(n_jobs=1)
    idx = np.random.default_rng(seed).choice(len(X), size=min(n, len(X)), replace=False)
    rf.predict(X[idx[:5]])  # 워밍업
    ts = []
    for i in idx:
        t = time.perf_counter()
        rf.predict(X[i:i + 1])
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts)), float(np.percentile(ts, 95))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="data/side_logs")
    ap.add_argument("--out", default="experiments/rf_tuning.md")
    ap.add_argument("--feature-set", default="no_hip", choices=list(SETS))
    ap.add_argument("--merge-intensity", action="store_true")
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--stride", type=int, default=5, help="탐색 때 N프레임마다 1개만 사용")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    D = load(a.logs)
    D = D[~D.pose.str.contains("lean")]
    if a.merge_intensity:
        D = D.assign(pose=D.pose.replace({"back_light": "back", "back_hard": "back",
                                          "neck_light": "neck", "neck_hard": "neck"}))
    D = D.reset_index(drop=True)
    cols = SETS[a.feature_set]
    parts = sorted(D.p.unique())
    classes = sorted(D.pose.unique())
    res = {m: [] for m in [*MODELS, "Tuned"]}
    chosen = {}
    for tep in parts:
        tr, te = D[D.p != tep], D[D.p == tep]
        Xtr, ytr, Xte, yte, _ = prep(tr, te, cols, "A")
        groups = (tr.p + "_" + tr.rep).values
        for name in res:
            if name == "Tuned":
                sub = np.arange(0, len(Xtr), a.stride)
                n_groups = len(set(groups[sub]))
                search = RandomizedSearchCV(
                    RandomForestClassifier(random_state=a.seed, n_jobs=-1), SPACE, n_iter=a.n_iter, scoring="f1_macro",
                    cv=GroupKFold(n_splits=min(3, n_groups)), random_state=a.seed, n_jobs=1)
                search.fit(Xtr[sub], ytr[sub], groups=groups[sub])
                chosen[tep] = search.best_params_
                params = search.best_params_
            else:
                params = MODELS[name]
            t0 = time.perf_counter()
            rf = RandomForestClassifier(random_state=a.seed, n_jobs=-1, **params).fit(Xtr, ytr)
            fit_s = time.perf_counter() - t0
            pr = rf.predict(Xte)
            lat = latency_ms(rf, Xte)
            res[name].append(dict(
                p=tep, f1=f1_score(yte, pr, average="macro"), acc=float((pr == yte).mean()),
                rec=recall_score(yte, pr, labels=classes, average=None, zero_division=0) * 100,
                kb=size_kb(rf), lat_med=lat[0], lat_p95=lat[1], fit_s=fit_s))
        print("완료:", tep, flush=True)

    L = ["# RF 경량화·파라미터 검증 (참가자 단위 교차검증)", "",
         f"- 특징 세트 {a.feature_set}, 결측 처리 A, {len(classes)}클래스{'(강도 합침)' if a.merge_intensity else ''}, "
         f"참가자 {len(parts)}명({', '.join(parts)}) leave-one-participant-out, 시드 {a.seed}.",
         f"- Tuned: 랜덤 서치 {a.n_iter}회, 안쪽 GroupKFold(그룹=참가자×촬영회차), 탐색은 {a.stride}프레임마다 1개 사용.",
         "- 참가자가 적어 모델 간 차이가 폴드 간 편차보다 작으면 '차이 없음'으로 해석한다.", "",
         "## 요약 (평균, 폴드 간 표준편차)", "",
         "| 모델 | macro-F1 | 정확도 | 모델 크기(KB) | 추론 중앙값(ms) | 추론 p95(ms) | 학습 시간(s) |",
         "|---|---|---|---|---|---|---|"]
    for name, rs in res.items():
        f1 = [r["f1"] for r in rs]
        L.append(f"| {name} | {np.mean(f1):.3f} ± {np.std(f1):.3f} | {np.mean([r['acc'] for r in rs]):.3f} | "
                 f"{np.mean([r['kb'] for r in rs]):.0f} | {np.mean([r['lat_med'] for r in rs]):.2f} | "
                 f"{np.mean([r['lat_p95'] for r in rs]):.2f} | {np.mean([r['fit_s'] for r in rs]):.1f} |")
    L += ["", "## 평가한 참가자별 macro-F1", "", "| 모델 | " + " | ".join(parts) + " |", "|---|" + "---|" * len(parts)]
    for name, rs in res.items():
        L.append(f"| {name} | " + " | ".join(f"{r['f1']:.3f}" for r in rs) + " |")
    L += ["", "## 클래스별 recall(%, 평가 참가자 평균)", "", "| 자세 | " + " | ".join(res) + " |",
          "|---|" + "---|" * len(res)]
    for i, c in enumerate(classes):
        L.append(f"| {c} | " + " | ".join(f"{np.mean([r['rec'][i] for r in rs]):.0f}" for rs in res.values()) + " |")
    L += ["", "## Tuned가 고른 파라미터(바깥 폴드별)", ""]
    for tep, prm in chosen.items():
        L.append(f"- {tep} 제외 학습: `{prm}`")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
