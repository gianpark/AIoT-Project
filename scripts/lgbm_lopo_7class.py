from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

from lightgbm import LGBMClassifier

from scripts.replay_judge_on_logs import load


# =========================================================
# 1. 경로 및 클래스 설정
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments"
OUT_DIR.mkdir(exist_ok=True)

LABEL_MAP = {
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

CLASSES = [
    "normal",
    "turtle_neck",
    "forward",
    "lean_back",
    "left_lean",
    "right_lean",
    "bad_posture",
]


# =========================================================
# 2. Feature Set
# =========================================================

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

ABS = {
    "head_depth_m",
    "chest_depth_m",
}

SETS = {
    "depth": DEPTH,
    "no_hip": DEPTH + [
        c for c in EXTRA if c not in HIP_DEP
    ],
    "all": DEPTH + EXTRA,
    "relative": [
        c for c in DEPTH + EXTRA
        if c not in HIP_DEP and c not in ABS
    ],
}


# =========================================================
# 3. 데이터 전처리
# =========================================================

def prep(tr, te, cols, mode):
    Xtr = tr[cols].copy()
    Xte = te[cols].copy()

    ytr = tr["label"].to_numpy()
    yte = te["label"].to_numpy()

    if mode == "B":
        train_mask = Xtr.notna().all(axis=1)
        test_mask = Xte.notna().all(axis=1)

        Xtr = Xtr.loc[train_mask]
        Xte = Xte.loc[test_mask]

        ytr = ytr[train_mask.to_numpy()]
        yte = yte[test_mask.to_numpy()]

        return (
            Xtr.to_numpy(),
            Xte.to_numpy(),
            ytr,
            yte,
            len(test_mask) and test_mask.mean(),
        )

    if mode == "A":
        imputer = SimpleImputer(strategy="median")

        Xtr_imp = imputer.fit_transform(Xtr)
        Xte_imp = imputer.transform(Xte)

        train_na = Xtr.isna().astype(int).to_numpy()
        test_na = Xte.isna().astype(int).to_numpy()

        Xtr_final = np.hstack([Xtr_imp, train_na])
        Xte_final = np.hstack([Xte_imp, test_na])

        return (
            Xtr_final,
            Xte_final,
            ytr,
            yte,
            1.0,
        )

    raise ValueError(f"Unknown mode: {mode}")


# =========================================================
# 4. LightGBM LOPO 실험
# =========================================================

def run(D, cols, mode):
    participants = sorted(D["participant"].unique())

    y_true_all = []
    y_pred_all = []

    fold_results = []
    keep_ratios = []

    feature_importances = []

    for test_person in participants:
        tr = D[D["participant"] != test_person].copy()
        te = D[D["participant"] == test_person].copy()

        Xtr, Xte, ytr, yte, keep_ratio = prep(
            tr, te, cols, mode
        )

        keep_ratios.append(keep_ratio)

        if len(ytr) == 0 or len(yte) == 0:
            print(f"[SKIP] {test_person}: 데이터 부족")
            continue

        encoder = LabelEncoder()
        ytr_enc = encoder.fit_transform(ytr)

        model = LGBMClassifier(
            n_estimators=300,
            num_leaves=31,
            learning_rate=0.05,
            objective="multiclass",
            class_weight="balanced",
            random_state=0,
            n_jobs=-1,
            verbosity=-1,
        )

        model.fit(Xtr, ytr_enc)

        pred_enc = model.predict(Xte)
        ypred = encoder.inverse_transform(pred_enc)

        y_true_all.extend(yte)
        y_pred_all.extend(ypred)

        fold_f1 = f1_score(
            yte,
            ypred,
            labels=CLASSES,
            average="macro",
            zero_division=0,
        )

        fold_acc = accuracy_score(yte, ypred)

        fold_results.append({
            "participant": test_person,
            "macro_f1": fold_f1,
            "accuracy": fold_acc,
            "keep_ratio": keep_ratio,
        })

        importance = pd.Series(
            model.feature_importances_,
            index=(
                cols + [f"{c}_na" for c in cols]
                if mode == "A"
                else cols
            ),
        )

        feature_importances.append(importance)

        print(
            f"  {test_person}: "
            f"Macro-F1={fold_f1:.3f}, "
            f"Accuracy={fold_acc:.3f}, "
            f"Keep={keep_ratio:.1%}"
        )

    if not y_true_all:
        raise RuntimeError("평가 가능한 데이터가 없습니다.")

    cm = confusion_matrix(
        y_true_all,
        y_pred_all,
        labels=CLASSES,
    )

    recalls = recall_score(
        y_true_all,
        y_pred_all,
        labels=CLASSES,
        average=None,
        zero_division=0,
    )

    importance_mean = pd.concat(
        feature_importances,
        axis=1,
    ).mean(axis=1).sort_values(ascending=False)

    return {
        "macro_f1": f1_score(
            y_true_all,
            y_pred_all,
            labels=CLASSES,
            average="macro",
            zero_division=0,
        ),
        "accuracy": accuracy_score(
            y_true_all,
            y_pred_all,
        ),
        "recalls": recalls,
        "cm": cm,
        "folds": fold_results,
        "keep_ratio": np.mean(keep_ratios),
        "importance": importance_mean,
    }


# =========================================================
# 5. 실행
# =========================================================

def main():
    print("Loading data...")

    D = load("data/side_logs")

    D = D.rename(columns={
        "p": "participant",
        "pose": "label",
    })

    D["label"] = D["label"].map(LABEL_MAP)
    D = D[D["label"].notna()].copy()

    print("\n7-class distribution:")
    print(D["label"].value_counts())

    print("\nParticipants:")
    print(D["participant"].value_counts())

    results = []

    for set_name, cols in SETS.items():
        for mode in ["A", "B"]:
            print("\n" + "=" * 60)
            print(f"LightGBM | {set_name} | {mode}")
            print("=" * 60)

            result = run(D, cols, mode)

            results.append({
                "feature_set": set_name,
                "mode": mode,
                "macro_f1": result["macro_f1"],
                "accuracy": result["accuracy"],
                "keep_ratio": result["keep_ratio"],
            })

            print("\nClass Recall:")
            for label, recall in zip(
                CLASSES,
                result["recalls"],
            ):
                print(f"{label:15s}: {recall:.3f}")

            print("\nConfusion Matrix:")
            print(
                pd.DataFrame(
                    result["cm"],
                    index=CLASSES,
                    columns=CLASSES,
                )
            )

            print("\nFeature Importance:")
            print(result["importance"].head(15))

            print("\nFold Results:")
            print(pd.DataFrame(result["folds"]).to_string(index=False))

    summary = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("LIGHTGBM FINAL SUMMARY")
    print("=" * 60)
    print(summary.to_string(index=False))

    output_path = OUT_DIR / "lgbm_lopo_7class_summary.csv"
    summary.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()