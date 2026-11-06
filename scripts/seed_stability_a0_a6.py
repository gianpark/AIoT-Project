#!/usr/bin/env python3
"""Seed stability experiment for RF / LightGBM / XGBoost with A0-A6 and LOPO.

The experiment intentionally keeps the existing RF A0-A6 feature definitions,
7-class mapping, and Mode-A preprocessing. Only each estimator's random_state
changes across seeds 0..4.
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


BASE_FEATURES = [
    "head_depth_m",
    "chest_depth_m",
    "neck_forward_offset_m",
    "head_forward_deg",
    "elbow_left_deg",
    "elbow_right_deg",
    "torso_len_sw",
    "head_up_m",
    "wrist_face_sw",
]

ABLATIONS = {
    "A0": [],
    "A1": ["head_depth_m", "chest_depth_m"],
    "A2": ["head_forward_deg", "head_up_m"],
    "A3": ["elbow_left_deg", "elbow_right_deg", "wrist_face_sw"],
    "A4": ["torso_len_sw"],
    "A5": ["neck_forward_offset_m"],
    "A6": [
        "head_depth_m", "chest_depth_m",
        "neck_forward_offset_m", "head_up_m",
    ],
}

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
    "normal", "turtle_neck", "forward", "lean_back",
    "left_lean", "right_lean", "bad_posture",
]
MODELS = ["RF", "LightGBM", "XGBoost"]


def load_logs(log_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(str(Path(log_dir) / "side_*.csv")))
    if not paths:
        raise FileNotFoundError(f"No side_*.csv files found in: {log_dir}")

    frames = []
    pattern = re.compile(r"side_(p\d+)_(.+?)(_r\d)?\.csv$")
    for path in paths:
        match = pattern.match(Path(path).name)
        if not match:
            print(f"[WARN] 파일명 규칙 불일치로 제외: {path}")
            continue
        frame = pd.read_csv(path)
        frame["p"] = match.group(1)
        frame["pose"] = match.group(2)
        frame["rep"] = (match.group(3) or "_r1")[1:]
        frames.append(frame)

    if not frames:
        raise RuntimeError("파일명 규칙에 맞는 로그 CSV가 없습니다.")
    return pd.concat(frames, ignore_index=True)


def feature_names(ablation: str) -> list[str]:
    removed = set(ABLATIONS[ablation])
    return [c for c in BASE_FEATURES if c not in removed]


def prepare_xy(train_df: pd.DataFrame, test_df: pd.DataFrame,
               features: list[str]):
    """Train-only median imputation + missingness indicators (existing Mode A)."""
    x_train_raw = train_df[features].apply(pd.to_numeric, errors="coerce")
    x_test_raw = test_df[features].apply(pd.to_numeric, errors="coerce")

    medians = x_train_raw.median(axis=0).fillna(0.0)
    x_train = x_train_raw.fillna(medians).fillna(0.0)
    x_test = x_test_raw.fillna(medians).fillna(0.0)

    train_flags = x_train_raw.isna().astype(np.int8)
    test_flags = x_test_raw.isna().astype(np.int8)
    train_flags.columns = [f"{c}_na" for c in features]
    test_flags.columns = [f"{c}_na" for c in features]

    x_train = pd.concat([x_train, train_flags], axis=1)
    x_test = pd.concat([x_test, test_flags], axis=1)
    return x_train, x_test


def make_model(model_name: str, seed: int):
    if model_name == "RF":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "LightGBM":
        return LGBMClassifier(
            n_estimators=300,
            num_leaves=31,
            learning_rate=0.05,
            objective="multiclass",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    if model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(model_name)


def validate_data(data: pd.DataFrame, expected_participants=None):
    required = ["p", "pose"] + BASE_FEATURES
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    data = data.copy()
    data["label"] = data["pose"].map(LABEL_MAP)
    unknown = sorted(data.loc[data["label"].isna(), "pose"].dropna().unique())
    if unknown:
        raise ValueError(f"라벨 매핑에 없는 원본 자세 라벨: {unknown}")
    data = data.dropna(subset=["p", "label"]).copy()

    participants = sorted(data["p"].astype(str).unique())
    if expected_participants and participants != expected_participants:
        raise ValueError(
            f"참가자 불일치: 확인된 {participants}, 예상 {expected_participants}"
        )
    if len(participants) != 3:
        raise ValueError(
            f"기존 3-fold LOPO 조건과 다르게 참가자가 {len(participants)}명입니다: "
            f"{participants}"
        )
    absent_classes = sorted(set(CLASSES) - set(data["label"].unique()))
    if absent_classes:
        raise ValueError(f"전체 데이터에서 관측되지 않은 7-class 라벨: {absent_classes}")
    return data, participants


def inspect(data: pd.DataFrame, participants: list[str]):
    print("\n=== DATA CHECK ===")
    print(f"전체 프레임 수: {len(data):,}")
    print(f"참가자: {participants}")
    print("\n참가자별 프레임:")
    print(data.groupby("p").size().to_string())
    print("\n7-class 분포:")
    print(data["label"].value_counts().reindex(CLASSES, fill_value=0).to_string())
    print("\nA0~A6 특징 수:")
    for name in ABLATIONS:
        print(f"{name}: {len(feature_names(name))}개")
    print("\n예정 학습 횟수: 3 models × 7 ablations × 5 seeds × 3 folds = 315")


def run_one(data, participant, model_name, seed, ablation):
    train_df = data[data["p"] != participant].copy()
    test_df = data[data["p"] == participant].copy()
    if len(train_df) < 100 or len(test_df) < 100:
        raise RuntimeError(
            f"{participant} fold 표본 수 부족: train={len(train_df)}, test={len(test_df)}"
        )

    y_train_text = train_df["label"].astype(str).to_numpy()
    y_test_text = test_df["label"].astype(str).to_numpy()
    train_classes = sorted(np.unique(y_train_text).tolist())
    if len(train_classes) < 2:
        raise RuntimeError(f"{participant} fold 학습 라벨이 2개 미만입니다.")

    # Encode only classes present in this training fold. A test-only class
    # cannot be predicted by a model that never saw it during training.
    class_to_int = {label: idx for idx, label in enumerate(train_classes)}
    y_train = np.array([class_to_int[v] for v in y_train_text], dtype=np.int32)

    features = feature_names(ablation)
    x_train, x_test = prepare_xy(train_df, test_df, features)

    model = make_model(model_name, seed)
    fit_kwargs = {}
    if model_name == "XGBoost":
        model.set_params(num_class=len(train_classes))
        fit_kwargs["sample_weight"] = compute_sample_weight(
            class_weight="balanced", y=y_train
        )

    model.fit(x_train, y_train, **fit_kwargs)
    pred_encoded = model.predict(x_test).astype(int)
    int_to_class = {idx: label for label, idx in class_to_int.items()}
    y_pred_text = np.array([int_to_class.get(int(v), "__unknown__") for v in pred_encoded])

    fold_macro_f1 = f1_score(
        y_test_text, y_pred_text, labels=CLASSES,
        average="macro", zero_division=0,
    )
    fold_accuracy = accuracy_score(y_test_text, y_pred_text)

    return {
        "participant": participant,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_train_classes": len(train_classes),
        "train_classes": ";".join(train_classes),
        "macro_f1": fold_macro_f1,
        "accuracy": fold_accuracy,
        "y_true": y_test_text,
        "y_pred": y_pred_text,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default="data/side_logs")
    parser.add_argument("--out-dir", default="experiments")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--check-only", action="store_true",
                        help="데이터/분할/특징을 점검하고 학습하지 않음")
    parser.add_argument("--smoke", action="store_true",
                        help="Seed 0, A0만 3개 모델로 시험 학습")
    args = parser.parse_args()

    data, participants = validate_data(load_logs(args.logs))
    inspect(data, participants)
    if args.check_only:
        print("\nCHECK ONLY: 학습 없이 점검을 종료했습니다.")
        return

    seeds = [0] if args.smoke else args.seeds
    ablations = ["A0"] if args.smoke else list(ABLATIONS)
    models = MODELS

    run_rows = []
    seed_rows = []
    oof_rows = []

    total = len(seeds) * len(ablations) * len(models) * len(participants)
    done = 0

    for model_name in models:
        for ablation in ablations:
            for seed in seeds:
                fold_results = []
                all_true, all_pred = [], []
                for participant in participants:
                    done += 1
                    print(
                        f"\n[{done}/{total}] {model_name} | {ablation} | "
                        f"seed={seed} | test={participant}",
                        flush=True,
                    )
                    result = run_one(data, participant, model_name, seed, ablation)
                    fold_results.append(result)
                    all_true.extend(result["y_true"].tolist())
                    all_pred.extend(result["y_pred"].tolist())

                    run_rows.append({
                        "model": model_name,
                        "ablation": ablation,
                        "seed": seed,
                        "participant": participant,
                        "n_train": result["n_train"],
                        "n_test": result["n_test"],
                        "n_train_classes": result["n_train_classes"],
                        "train_classes": result["train_classes"],
                        "macro_f1": result["macro_f1"],
                        "accuracy": result["accuracy"],
                    })

                fold_f1 = [r["macro_f1"] for r in fold_results]
                fold_acc = [r["accuracy"] for r in fold_results]
                pooled_f1 = f1_score(
                    all_true, all_pred, labels=CLASSES,
                    average="macro", zero_division=0,
                )
                pooled_acc = accuracy_score(all_true, all_pred)

                seed_rows.append({
                    "model": model_name,
                    "ablation": ablation,
                    "seed": seed,
                    "mean_fold_macro_f1": float(np.mean(fold_f1)),
                    "std_fold_macro_f1": float(np.std(fold_f1, ddof=0)),
                    "mean_fold_accuracy": float(np.mean(fold_acc)),
                    "pooled_oof_macro_f1": float(pooled_f1),
                    "pooled_oof_accuracy": float(pooled_acc),
                    "n_valid_folds": len(fold_results),
                    "n_oof_samples": len(all_true),
                })
                oof_rows.append({
                    "model": model_name,
                    "ablation": ablation,
                    "seed": seed,
                    "pooled_oof_macro_f1": float(pooled_f1),
                    "pooled_oof_accuracy": float(pooled_acc),
                    "n_oof_samples": len(all_true),
                })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    runs_df = pd.DataFrame(run_rows)
    seed_df = pd.DataFrame(seed_rows)
    oof_df = pd.DataFrame(oof_rows)

    runs_df.to_csv(out_dir / f"seed_stability_a0_a6_runs{suffix}.csv",
                   index=False, encoding="utf-8-sig")
    seed_df.to_csv(out_dir / f"seed_stability_a0_a6_seed_summary{suffix}.csv",
                   index=False, encoding="utf-8-sig")
    oof_df.to_csv(out_dir / f"seed_stability_a0_a6_oof_summary{suffix}.csv",
                  index=False, encoding="utf-8-sig")

    summary_metrics = [
        "mean_fold_macro_f1", "pooled_oof_macro_f1",
        "mean_fold_accuracy", "pooled_oof_accuracy",
    ]
    summary = seed_df.groupby(["model", "ablation"])[summary_metrics].agg(
        ["mean", "std"]
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out_dir / f"seed_stability_a0_a6_summary{suffix}.csv",
                   index=False, encoding="utf-8-sig")

    report_lines = [
        "# Seed 안정성 실험 결과",
        "",
        f"- 참가자: {', '.join(participants)}",
        f"- Seeds: {', '.join(map(str, seeds))}",
        f"- 알고리즘: {', '.join(models)}",
        f"- Ablation: {', '.join(ablations)}",
        "- 검증: 동일한 LOPO 분할",
        "- 전처리: 학습 폴드 중앙값 대치 + 결측 플래그",
        "- Seed 변경: 각 모델의 random_state만 변경",
        "- 표준편차: seed 간 표준편차(ddof=0)",
        "",
        "## 평균 ± 표준편차",
        "",
        "| Model | Ablation | Fold mean Macro-F1 | Pooled OOF Macro-F1 | Fold mean Accuracy | Pooled OOF Accuracy |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        def fmt(metric):
            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_std"]
            return f"{mean:.4f} ({std:.4f})"
        report_lines.append(
            f"| {row['model']} | {row['ablation']} | "
            f"{fmt('mean_fold_macro_f1')} | {fmt('pooled_oof_macro_f1')} | "
            f"{fmt('mean_fold_accuracy')} | {fmt('pooled_oof_accuracy')} |"
        )
    report_lines += [
        "",
        "## 해석 시 주의사항",
        "",
        "- Seed 간 표준편차는 모델 초기화/샘플링의 변동성을 나타낸다.",
        "- LOPO 참가자 수가 3명이므로 fold 평균과 pooled OOF 지표는 서로 다른 요약값이다.",
        "- pooled OOF는 각 참가자의 테스트 예측을 합쳐 산출한 지표다.",
        "- 표준편차가 작으면 현재 설정에서 seed 민감도가 낮다는 근거가 될 수 있지만, "
        "일반화 성능이나 통계적 유의성을 단독으로 입증하지는 않는다.",
        "- 참가자 수가 적으므로 결과 해석은 신중해야 한다.",
    ]
    (out_dir / f"seed_stability_a0_a6_report{suffix}.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))
    print(f"\n결과 저장 위치: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
