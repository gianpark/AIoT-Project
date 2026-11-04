"""
7-class 2D / Depth / 2D+Depth 기준선 비교 실험.

- 기존 2D 로그가 존재하는 p02, p03만 사용
- 동일한 프레임과 동일한 참가자 분할로 세 특징 세트를 비교
- 7-class 라벨 매핑은 scripts/rf_lopo_7class.py와 동일
- Leave-One-Participant-Out: 한 참가자로 학습하고 다른 참가자로 평가
- 결측 처리 Mode A: 학습 데이터 중앙값 대치 + 결측 플래그
- Random Forest: 300 trees, class_weight=balanced, random_state=0

실행:
    python -m scripts.rf_2d_baseline_7class

선택 인자:
    --logs data/side_logs
    --out-dir experiments/rf_2d_baseline_7class
    --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from scripts.replay_judge_on_logs import load
from scripts.rf_lopo_7class import SETS
from src.features.features_2d import features_2d


LABEL_MAPPING = {
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

CLASS_ORDER = [
    "bad_posture",
    "forward",
    "lean_back",
    "left_lean",
    "normal",
    "right_lean",
    "turtle_neck",
]

# rf_lopo_7class.py의 no_hip 세트와 동일한 9개 깊이/상체 특징
DEPTH_FEATURES = list(SETS["no_hip"])
MODEL_FEATURES = {
    "2D_only": None,       # features_2d() 결과 열을 실행 시 설정
    "Depth_only": DEPTH_FEATURES,
    "2D_plus_Depth": None,
}


def prepare_mode_a(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
):
    """학습 fold의 중앙값으로 대치하고 결측 여부 플래그를 추가한다."""
    x_train = train[feature_cols].replace([np.inf, -np.inf], np.nan)
    x_test = test[feature_cols].replace([np.inf, -np.inf], np.nan)

    medians = x_train.median()
    # 학습 fold에서 특정 특징이 전부 결측인 경우 0으로 대치
    medians = medians.fillna(0.0)

    train_flags = x_train.isna().astype(np.int8).add_suffix("_na")
    test_flags = x_test.isna().astype(np.int8).add_suffix("_na")

    x_train = x_train.fillna(medians)
    x_test = x_test.fillna(medians)

    x_train = pd.concat([x_train, train_flags], axis=1)
    x_test = pd.concat([x_test, test_flags], axis=1)

    return x_train.to_numpy(dtype=float), x_test.to_numpy(dtype=float)


def evaluate_feature_set(
    data: pd.DataFrame,
    feature_cols: list[str],
    feature_set_name: str,
    participants: list[str],
    seed: int,
):
    fold_rows = []
    prediction_rows = []

    for held_out in participants:
        train = data[data["p"] != held_out].copy()
        test = data[data["p"] == held_out].copy()

        x_train, x_test = prepare_mode_a(train, test, feature_cols)
        y_train = train["pose"].to_numpy()
        y_test = test["pose"].to_numpy()

        if len(train) == 0 or len(test) == 0:
            raise RuntimeError(
                f"{feature_set_name}: {held_out} fold의 학습/평가 데이터가 비어 있습니다."
            )

        if len(np.unique(y_train)) < 2:
            raise RuntimeError(
                f"{feature_set_name}: {held_out} fold 학습 데이터의 클래스가 2개 미만입니다."
            )

        model = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        pred = model.predict(x_test)

        fold_rows.append(
            {
                "feature_set": feature_set_name,
                "participant": held_out,
                "n_train": len(train),
                "n_test": len(test),
                "n_features": len(feature_cols),
                "macro_f1": f1_score(
                    y_test, pred, labels=CLASS_ORDER,
                    average="macro", zero_division=0
                ),
                "accuracy": accuracy_score(y_test, pred),
            }
        )

        for true_label, predicted_label in zip(y_test, pred):
            prediction_rows.append(
                {
                    "feature_set": feature_set_name,
                    "participant": held_out,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                }
            )

    return fold_rows, prediction_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default="data/side_logs")
    parser.add_argument(
        "--out-dir",
        default="experiments/rf_2d_baseline_7class",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 로그 로드 및 7-class 라벨 통합
    data = load(args.logs).copy()

    unknown = sorted(set(data["pose"].dropna().unique()) - set(LABEL_MAPPING))
    if unknown:
        raise ValueError(f"7-class 매핑에 없는 라벨이 있습니다: {unknown}")

    data["pose"] = data["pose"].map(LABEL_MAPPING)

    # 2D keypoint 로그가 있는 p02/p03만 사용한다.
    # 좌우 기울임 자세를 포함해 lean 라벨을 제거하지 않는다.
    required_2d_columns = [
        "nose_x", "nose_y", "nose_score",
        "left_shoulder_x", "left_shoulder_y", "left_shoulder_score",
        "right_shoulder_x", "right_shoulder_y", "right_shoulder_score",
    ]
    missing_columns = [c for c in required_2d_columns if c not in data.columns]
    if missing_columns:
        raise ValueError(f"2D 로그 필수 열이 없습니다: {missing_columns}")

    data = data[
        data["p"].astype(str).isin(["p02", "p03"])
        & data["nose_x"].notna()
        & data["pose"].notna()
    ].copy()

    participants = sorted(data["p"].astype(str).unique())
    if participants != ["p02", "p03"]:
        raise RuntimeError(
            "이번 실험은 p02와 p03 두 참가자의 2D 로그가 모두 필요합니다. "
            f"현재 확인된 참가자: {participants}"
        )

    # 3. 2D 특징을 계산한다. 이후 세 모델은 이 동일한 data 행을 사용한다.
    data = data.reset_index(drop=True)
    features_2d_df = features_2d(data)
    two_d_features = list(features_2d_df.columns)
    data = pd.concat([data, features_2d_df], axis=1)

    depth_features = list(DEPTH_FEATURES)
    missing_depth = [c for c in depth_features if c not in data.columns]
    if missing_depth:
        raise ValueError(f"깊이 특징 열이 로그에 없습니다: {missing_depth}")

    feature_sets = {
        "2D_only": two_d_features,
        "Depth_only": depth_features,
        "2D_plus_Depth": two_d_features + depth_features,
    }

    # 4. 동일 참가자 분할로 각 특징 세트를 평가한다.
    all_fold_rows = []
    all_prediction_rows = []

    for name, columns in feature_sets.items():
        print(f"\n[{name}] 특징 {len(columns)}개 평가 중...")
        fold_rows, prediction_rows = evaluate_feature_set(
            data=data,
            feature_cols=columns,
            feature_set_name=name,
            participants=participants,
            seed=args.seed,
        )
        all_fold_rows.extend(fold_rows)
        all_prediction_rows.extend(prediction_rows)

    fold_df = pd.DataFrame(all_fold_rows)
    pred_df = pd.DataFrame(all_prediction_rows)

    # 5. 전체 OOF 예측 기준 요약 및 클래스별 지표
    summary_rows = []
    class_rows = []
    confusion_rows = []
    fold_class_rows = []

    for name in feature_sets:
        subset = pred_df[pred_df["feature_set"] == name]
        y_true = subset["true_label"].to_numpy()
        y_pred = subset["predicted_label"].to_numpy()

        summary_rows.append(
            {
                "feature_set": name,
                "n_features": len(feature_sets[name]),
                "n_frames": len(subset),
                "macro_f1": f1_score(
                    y_true, y_pred, labels=CLASS_ORDER,
                    average="macro", zero_division=0
                ),
                "accuracy": accuracy_score(y_true, y_pred),
                "mean_fold_macro_f1": fold_df.loc[
                    fold_df["feature_set"] == name, "macro_f1"
                ].mean(),
                "mean_fold_accuracy": fold_df.loc[
                    fold_df["feature_set"] == name, "accuracy"
                ].mean(),
            }
        )

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=CLASS_ORDER, zero_division=0
        )
        for label, p, r, f, s in zip(
            CLASS_ORDER, precision, recall, f1, support
        ):
            class_rows.append(
                {
                    "feature_set": name,
                    "class": label,
                    "precision": p,
                    "recall": r,
                    "f1": f,
                    "support": int(s),
                }
            )

        cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
        for i, true_label in enumerate(CLASS_ORDER):
            for j, predicted_label in enumerate(CLASS_ORDER):
                confusion_rows.append(
                    {
                        "feature_set": name,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": int(cm[i, j]),
                    }
                )

        for participant in participants:
            part = subset[subset["participant"] == participant]
            p, r, f, s = precision_recall_fscore_support(
                part["true_label"],
                part["predicted_label"],
                labels=CLASS_ORDER,
                zero_division=0,
            )
            for label, pv, rv, fv, sv in zip(CLASS_ORDER, p, r, f, s):
                fold_class_rows.append(
                    {
                        "feature_set": name,
                        "participant": participant,
                        "class": label,
                        "precision": pv,
                        "recall": rv,
                        "f1": fv,
                        "support": int(sv),
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    class_df = pd.DataFrame(class_rows)
    confusion_df = pd.DataFrame(confusion_rows)
    fold_class_df = pd.DataFrame(fold_class_rows)

    # 6. CSV 결과 저장
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    fold_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    class_df.to_csv(output_dir / "class_metrics.csv", index=False)
    fold_class_df.to_csv(output_dir / "fold_class_metrics.csv", index=False)
    confusion_df.to_csv(output_dir / "confusion.csv", index=False)
    pred_df.to_csv(output_dir / "oof_predictions.csv", index=False)

    # 7. Markdown 보고서 저장
    report = [
        "# 2D / Depth / 2D+Depth 7-class 기준선 비교",
        "",
        "## 실험 조건",
        "",
        f"- 데이터: {len(data):,}프레임",
        f"- 참가자: {', '.join(participants)}",
        f"- 클래스: {len(CLASS_ORDER)}개",
        "- 검증: p02 학습 → p03 평가, p03 학습 → p02 평가",
        "- 모델: Random Forest 300 trees, class_weight=balanced",
        f"- random_state: {args.seed}",
        "- 결측 처리: 학습 fold 중앙값 대치 + 결측 플래그(Mode A)",
        "- 세 특징 세트는 동일한 데이터 행과 참가자 분할을 사용함",
        "- 기존 3명 대상 7-class LOPO 결과와 직접 비교하지 않음",
        "",
        "## 특징 세트",
        "",
        f"- 2D only: {len(two_d_features)}개",
        f"- Depth only: {len(depth_features)}개",
        f"- 2D + Depth: {len(two_d_features) + len(depth_features)}개",
        "",
        "## 전체 OOF 성능",
        "",
        summary_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 참가자별 성능",
        "",
        fold_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 해석 시 주의사항",
        "",
        "- 참가자가 2명뿐이므로 일반화 성능에 대한 결론은 제한적이다.",
        "- 깊이 정보의 효과는 동일한 p02/p03 프레임에서 비교한 결과로 해석한다.",
        "- Macro-F1을 주요 지표로 보고 클래스별 F1 및 혼동행렬도 함께 확인한다.",
        "- 2D+Depth가 2D only보다 낮더라도 깊이 정보 자체가 무용하다는 뜻은 아니다. "
        "특징 중복, 결측, 참가자별 차이를 함께 검토해야 한다.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")

    print("\n===== 전체 OOF 성능 =====")
    print(summary_df.to_string(index=False))
    print("\n===== 참가자별 성능 =====")
    print(fold_df.to_string(index=False))
    print(f"\n결과 저장 위치: {output_dir}")


if __name__ == "__main__":
    main()
