"""참가자 단위 교차검증(leave-one-participant-out) 랜덤 포레스트, 7자세 분류.
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
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_recall_fscore_support

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
    ap.add_argument("--out", default="experiments/rf_lopo_7class.md")
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
        "slouch": "bad_posture"
    }

    # 매핑되지 않은 라벨이 있으면 중단
    unknown = sorted(set(D.pose.unique()) - set(mapping.keys()))

    if unknown:
        raise ValueError(f"매핑되지 않은 자세 라벨: {unknown}")

    D = D.assign(pose=D.pose.map(mapping))

    # 3. 데이터 구성 확인
    parts = sorted(D.p.unique())
    classes = sorted(D.pose.unique())

    cnt = ", ".join(
        f"{q} {int((D.p == q).sum())}" for q in parts
    )

    print("===== 7개 클래스 데이터 분포 =====")
    print(D.groupby("pose").size())

    print("\n===== 사람별 데이터 분포 =====")
    print(D.groupby(["pose", "p"]).size().unstack(fill_value=0))

    # 4. 실험 보고서 시작
    L = [
        f"# 참가자 단위 교차검증 RF (7개 자세, 참가자 {len(parts)}명)",
        "",
        f"- 데이터 {len(D)}프레임 ({cnt})",
        f"- 분류 클래스: {len(classes)}개",
        "- 분류 방식: 11개 기존 라벨을 7개 클래스로 통합",
        "- 검증 방식: Leave-One-Participant-Out (LOPO)",
        "- RF: 300그루, class_weight=balanced, seed=0",
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
        "|---|---|" + "---|" * len(parts) + "---|---|"
    ]

    # 5. 특징 세트 및 결측 처리별 실험
    keep = {}
    summary_rows = []
    class_metric_rows = []
    fold_metric_rows = []
    confusion_rows = []
    fold_confusion_rows = []

    for sname, cols in SETS.items():
        for mode in ("A", "B"):

            r = run(D, cols, mode)

            v = [
                r[k]["f1"] if r[k] else float("nan")
                for k in parts
            ]

            kp = [
                r[k]["keep"] if r[k] else float("nan")
                for k in parts
            ]

            keep[(sname, mode)] = r

            # 조건별 전체 OOF 예측을 모아 클래스별 지표와 혼동행렬 저장
            valid_folds = [(person, value) for person, value in r.items() if value]
            if valid_folds:
                y_all = np.concatenate([value["y"] for _, value in valid_folds])
                p_all = np.concatenate([value["p"] for _, value in valid_folds])
                precision, recall, f1, support = precision_recall_fscore_support(
                    y_all, p_all, labels=classes, zero_division=0
                )
                for label, prc, rec, f1v, sup in zip(classes, precision, recall, f1, support):
                    class_metric_rows.append({
                        "feature_set": sname, "mode": mode, "class": label,
                        "precision": prc, "recall": rec, "f1": f1v, "support": int(sup)
                    })

                cm_all = confusion_matrix(y_all, p_all, labels=classes)
                for i, true_label in enumerate(classes):
                    for j, predicted_label in enumerate(classes):
                        confusion_rows.append({
                            "feature_set": sname, "mode": mode,
                            "true_label": true_label, "predicted_label": predicted_label,
                            "count": int(cm_all[i, j])
                        })

                summary_rows.append({
                    "feature_set": sname, "mode": mode,
                    "macro_f1": f1_score(y_all, p_all, labels=classes, average="macro", zero_division=0),
                    "accuracy": float(np.mean(y_all == p_all)),
                    "keep_ratio": float(np.mean([value["keep"] for _, value in valid_folds]))
                })

                for person, value in valid_folds:
                    prc_f, rec_f, f1_f, sup_f = precision_recall_fscore_support(
                        value["y"], value["p"], labels=classes, zero_division=0
                    )
                    for label, prc, rec, f1v, sup in zip(classes, prc_f, rec_f, f1_f, sup_f):
                        fold_metric_rows.append({
                            "feature_set": sname, "mode": mode, "participant": person,
                            "class": label, "precision": prc, "recall": rec,
                            "f1": f1v, "support": int(sup)
                        })
                    cm_fold = confusion_matrix(value["y"], value["p"], labels=classes)
                    for i, true_label in enumerate(classes):
                        for j, predicted_label in enumerate(classes):
                            fold_confusion_rows.append({
                                "feature_set": sname, "mode": mode, "participant": person,
                                "true_label": true_label, "predicted_label": predicted_label,
                                "count": int(cm_fold[i, j])
                            })

            L.append(
                f"| {sname} | {mode} | "
                + " | ".join(f"{x:.3f}" for x in v)
                + f" | {np.nanmean(v):.3f} | "
                + " / ".join(f"{x*100:.0f}%" for x in kp)
                + " |"
            )

    # 6. 대표 모델: no_hip + A
    best_key = ("no_hip", "A")
    r = keep[best_key]

    L += [
        "",
        f"## 클래스별 Recall(%) — {best_key[0]} + {best_key[1]}",
        "",
        "| 자세 | " + " | ".join(parts) + " |",
        "|---|" + "---|" * len(parts)
    ]

    rec = {}

    for k, v in r.items():
        if v:
            rec[k] = recall_score(
                v["y"],
                v["p"],
                labels=classes,
                average=None,
                zero_division=0
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

    # 7. 혼동 행렬
    ys = np.concatenate([v["y"] for v in r.values() if v])
    ps = np.concatenate([v["p"] for v in r.values() if v])

    L += [
        "",
        "## 혼동 행렬 (행=실제, 열=예측)",
        "",
        "| 실제＼예측 | " + " | ".join(classes) + " |",
        "|---|" + "---|" * len(classes)
    ]

    cm = confusion_matrix(ys, ps, labels=classes)

    for c, row in zip(classes, cm):
        L.append(
            f"| {c} | "
            + " | ".join(str(x) for x in row)
            + " |"
        )

    # 8. 특징 중요도
    L += [
        "",
        "## 특징 중요도 — no_hip + A",
        "",
        "| 특징 | 중요도 |",
        "|---|---|"
    ]

    imp, n_imp = None, 0

    for v in r.values():
        if not v:
            continue

        names = v["cols"] + [
            c + "_na" for c in v["cols"]
        ]

        s_ = pd.Series(
            v["rf"].feature_importances_,
            index=names
        )

        imp = s_ if imp is None else imp + s_
        n_imp += 1

    for n, x in (
        (imp / n_imp).sort_values(ascending=False).head(10).items()
    ):
        L.append(f"| `{n}` | {x:.3f} |")

    # 9. 결과 저장
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    print("\n".join(L))
    print(f"\n결과 저장 완료: {a.out}")

    # 클래스별 지표 및 혼동행렬 CSV 저장
    out_dir = os.path.dirname(a.out) or "."
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(out_dir, "rf_lopo_7class_summary.csv"), index=False
    )
    pd.DataFrame(class_metric_rows).to_csv(
        os.path.join(out_dir, "rf_lopo_7class_metrics.csv"), index=False
    )
    pd.DataFrame(fold_metric_rows).to_csv(
        os.path.join(out_dir, "rf_lopo_7class_fold_metrics.csv"), index=False
    )
    pd.DataFrame(confusion_rows).to_csv(
        os.path.join(out_dir, "rf_lopo_7class_confusion.csv"), index=False
    )
    pd.DataFrame(fold_confusion_rows).to_csv(
        os.path.join(out_dir, "rf_lopo_7class_fold_confusion.csv"), index=False
    )
    print("CSV 저장 완료: RF summary / metrics / fold_metrics / confusion / fold_confusion")


if __name__ == "__main__":
    main()
