"""참가자 단위 교차검증(LOPO) XGBoost, 7자세 분류."""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)

from scripts.replay_judge_on_logs import load


DEPTH = [
    "head_depth_m",
    "chest_depth_m",
    "hip_depth_m",
    "neck_forward_offset_m",
    "torso_recline_offset_m",
]

EXTRA = [
    "head_forward_deg",
    "torso_pitch_deg",
    "elbow_left_deg",
    "elbow_right_deg",
    "torso_len_sw",
    "head_up_m",
    "wrist_face_sw",
]

HIP_DEP = {
    "hip_depth_m",
    "torso_recline_offset_m",
    "torso_pitch_deg",
}

ABS = {"head_depth_m", "chest_depth_m"}

SETS = {
    "depth": DEPTH,
    "no_hip": [
        c for c in DEPTH + EXTRA if c not in HIP_DEP
    ],
    "all": DEPTH + EXTRA,
}

SETS["relative"] = [
    c for c in SETS["no_hip"] if c not in ABS
]


def prep(tr, te, cols, mode):

    if mode == "B":
        n0 = len(te)

        tr = tr.dropna(subset=cols)
        te = te.dropna(subset=cols)

        return (
            tr[cols].values,
            tr.pose.values,
            te[cols].values,
            te.pose.values,
            n0,
        )

    med = tr[cols].median()

    def f(d):
        x = d[cols]
        flags = x.isna().astype(int).add_suffix("_na")

        return pd.concat(
            [x.fillna(med), flags],
            axis=1,
        ).values

    return (
        f(tr),
        tr.pose.values,
        f(te),
        te.pose.values,
        len(te),
    )


def run(D, cols, mode, encoder, seed=0):

    out = {}

    for tep in sorted(D.p.unique()):

        tr = D[D.p != tep]
        te = D[D.p == tep]

        Xtr, ytr, Xte, yte, n_all = prep(
            tr, te, cols, mode
        )

        if (
            len(Xtr) < 100
            or len(Xte) < 100
            or len(set(ytr)) < 2
        ):
            out[tep] = None
            continue

        # 문자열 자세 라벨을 정수로 변환
        ytr_enc = encoder.transform(ytr)
        yte_enc = encoder.transform(yte)

        # 학습 데이터의 클래스 불균형 보정
        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=ytr_enc,
        )

        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=len(encoder.classes_),
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )

        model.fit(
            Xtr,
            ytr_enc,
            sample_weight=sample_weights,
        )

        pred_enc = model.predict(Xte).astype(int)

        # 다시 문자열 라벨로 변환
        pred = encoder.inverse_transform(pred_enc)

        out[tep] = {
            "f1": f1_score(
                yte,
                pred,
                average="macro",
                zero_division=0,
            ),
            "acc": accuracy_score(yte, pred),
            "keep": len(yte) / n_all,
            "y": yte,
            "p": pred,
            "model": model,
            "cols": cols,
            "mode": mode,
        }

    return out


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--logs",
        default="data/side_logs",
    )

    ap.add_argument(
        "--out",
        default="experiments/xgb_lopo_7class.md",
    )

    a = ap.parse_args()

    # 1. 데이터 불러오기
    D = load(a.logs)

    # 2. 11개 기존 라벨을 7개 클래스로 통합
    mapping = {
        "normal": "normal",

        "neck_hard": "turtle_neck",
        "neck_light": "turtle_neck",

        "forward": "forward",

        "back_hard": "lean_back",
        "back_light": "lean_back",

        "left_lean_neck": "left_lean",
        "right_lean_neck": "right_lean",

        "back_neck": "bad_posture",
        "chin": "bad_posture",
        "slouch": "bad_posture",
    }

    unknown = sorted(
        set(D.pose.unique()) - set(mapping.keys())
    )

    if unknown:
        raise ValueError(
            f"매핑되지 않은 자세 라벨: {unknown}"
        )

    D = D.assign(pose=D.pose.map(mapping))

    parts = sorted(D.p.unique())
    classes = sorted(D.pose.unique())

    encoder = LabelEncoder()
    encoder.fit(classes)

    cnt = ", ".join(
        f"{q} {int((D.p == q).sum())}"
        for q in parts
    )

    print("===== 7개 클래스 데이터 분포 =====")
    print(D.groupby("pose").size())

    print("\n===== 사람별 데이터 분포 =====")
    print(
        D.groupby(["pose", "p"])
        .size()
        .unstack(fill_value=0)
    )

    # 3. 보고서 시작
    L = [
        f"# 참가자 단위 교차검증 XGBoost (7개 자세, 참가자 {len(parts)}명)",
        "",
        f"- 데이터: {len(D)}프레임 ({cnt})",
        f"- 분류 클래스: {len(classes)}개",
        "- 분류 방식: 11개 기존 라벨을 7개 클래스로 통합",
        "- 검증 방식: Leave-One-Participant-Out (LOPO)",
        "- XGBoost: 300 estimators, max_depth=6",
        "- learning_rate=0.05, subsample=0.9",
        "- colsample_bytree=0.9, random_state=0",
        "- 클래스 불균형 처리: balanced sample weights",
        "- 참가자가 3명으로 제한되어 있어 일반화 성능 해석에 주의한다.",
        "",
        "## 클래스 통합 기준",
        "",
        "| 새 클래스 | 기존 라벨 |",
        "|---|---|",
        "| normal | normal |",
        "| turtle_neck | neck_hard, neck_light |",
        "| forward | forward |",
        "| lean_back | back_hard, back_light |",
        "| left_lean | left_lean_neck |",
        "| right_lean | right_lean_neck |",
        "| bad_posture | back_neck, chin, slouch |",
        "",
        "## 특징 세트 × 결측 처리",
        "",
        "| 특징 세트 | 결측 처리 | "
        + " | ".join(f"{q} Macro-F1" for q in parts)
        + " | 평균 | 평가 프레임 유지율 |",
        "|---|---|" + "---|" * len(parts) + "---|---|",
    ]

    # 4. 특징 세트 및 결측 처리 실험
    results = {}

    for sname, cols in SETS.items():

        for mode in ("A", "B"):

            r = run(
                D,
                cols,
                mode,
                encoder,
            )

            values = [
                r[k]["f1"] if r[k] else float("nan")
                for k in parts
            ]

            keeps = [
                r[k]["keep"] if r[k] else float("nan")
                for k in parts
            ]

            results[(sname, mode)] = r

            L.append(
                f"| {sname} | {mode} | "
                + " | ".join(
                    f"{x:.3f}" for x in values
                )
                + f" | {np.nanmean(values):.3f} | "
                + " / ".join(
                    f"{x*100:.0f}%" for x in keeps
                )
                + " |"
            )

    # 5. 대표 모델: no_hip + A
    best_key = ("no_hip", "A")
    r = results[best_key]

    L += [
        "",
        f"## 클래스별 Recall(%) — {best_key[0]} + {best_key[1]}",
        "",
        "| 자세 | " + " | ".join(parts) + " |",
        "|---|" + "---|" * len(parts),
    ]

    rec = {}

    for k, v in r.items():

        if v:
            rec[k] = recall_score(
                v["y"],
                v["p"],
                labels=classes,
                average=None,
                zero_division=0,
            ) * 100

    for i, c in enumerate(classes):

        L.append(
            f"| {c} | "
            + " | ".join(
                f"{rec[q][i]:.0f}" if q in rec else "-"
                for q in parts
            )
            + " |"
        )

    # 6. 혼동 행렬
    ys = np.concatenate([
        v["y"] for v in r.values() if v
    ])

    ps = np.concatenate([
        v["p"] for v in r.values() if v
    ])

    L += [
        "",
        "## 혼동 행렬 (행=실제, 열=예측)",
        "",
        "| 실제＼예측 | "
        + " | ".join(classes)
        + " |",
        "|---|" + "---|" * len(classes),
    ]

    cm = confusion_matrix(
        ys,
        ps,
        labels=classes,
    )

    for c, row in zip(classes, cm):

        L.append(
            f"| {c} | "
            + " | ".join(str(x) for x in row)
            + " |"
        )

    # 7. 특징 중요도
    L += [
        "",
        "## 특징 중요도 — no_hip + A",
        "",
        "| 특징 | 중요도 |",
        "|---|---|",
    ]

    imp = None
    n_imp = 0

    for v in r.values():

        if not v:
            continue

        names = v["cols"] + [
            c + "_na" for c in v["cols"]
        ]

        s_ = pd.Series(
            v["model"].feature_importances_,
            index=names,
        )

        imp = s_ if imp is None else imp + s_
        n_imp += 1

    if n_imp > 0:

        for n, x in (
            (imp / n_imp)
            .sort_values(ascending=False)
            .head(10)
            .items()
        ):
            L.append(f"| `{n}` | {x:.3f} |")

    # 8. 결과 저장
    os.makedirs(
        os.path.dirname(a.out) or ".",
        exist_ok=True,
    )

    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    print("\n".join(L))
    print(f"\n결과 저장 완료: {a.out}")


if __name__ == "__main__":
    main()