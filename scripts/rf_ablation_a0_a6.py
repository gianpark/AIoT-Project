"""A0~A6 특징 제거 어블레이션 실험: 7-class Random Forest + LOPO.

기존 scripts/rf_lopo_7class.py와 동일한:
- 11개 원 라벨 -> 7개 클래스 매핑
- Leave-One-Participant-Out (LOPO)
- RandomForestClassifier(n_estimators=300, class_weight="balanced",
  random_state=0, n_jobs=-1)
- Mode A: 학습 폴드 중앙값 대치 + 결측 플래그

실행:
    python -m scripts.rf_ablation_a0_a6
    python -m scripts.rf_ablation_a0_a6 --logs data/side_logs

출력:
    experiments/ablation_a0_a6_summary.csv
    experiments/ablation_a0_a6_fold_metrics.csv
    experiments/ablation_a0_a6_class_metrics.csv
    experiments/ablation_a0_a6_confusion.csv
    experiments/ablation_a0_a6_analysis.md

주의:
- A0~A6은 사전에 정의한 특징 조합 비교다.
- A6은 여러 거리/깊이 특징을 함께 제외하는 별도 조합이다.
- 참가자 수가 적으므로 결과를 전체 사용자에 대한 일반화 성능으로
  단정하지 않는다.
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


# 기존 rf_lopo_7class.py의 no_hip 9개 특징
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

# A0~A6 설계: 제거할 특징만 명시
ABLATIONS = {
    "A0": {
        "description": "기준 특징 전체 사용",
        "removed": [],
    },
    "A1": {
        "description": "머리·가슴 절대 깊이 정보 제거",
        "removed": ["head_depth_m", "chest_depth_m"],
    },
    "A2": {
        "description": "머리 방향·높이 정보 제거",
        "removed": ["head_forward_deg", "head_up_m"],
    },
    "A3": {
        "description": "팔꿈치·손목 정보 제거",
        "removed": ["elbow_left_deg", "elbow_right_deg", "wrist_face_sw"],
    },
    "A4": {
        "description": "상체 길이 비율 정보 제거",
        "removed": ["torso_len_sw"],
    },
    "A5": {
        "description": "목 전방 이동 거리 정보 제거",
        "removed": ["neck_forward_offset_m"],
    },
    "A6": {
        "description": "깊이·거리 관련 특징을 제외한 조합",
        "removed": [
            "head_depth_m",
            "chest_depth_m",
            "neck_forward_offset_m",
            "head_up_m",
        ],
    },
}

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

RF_PARAMS = {
    "n_estimators": 300,
    "class_weight": "balanced",
    "random_state": 0,
    "n_jobs": -1,
}


def prepare_mode_a(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
):
    """학습 폴드 중앙값으로 대치하고 원 특징별 결측 플래그를 추가한다."""
    medians = train_df[features].median()

    # 학습 데이터에서 전부 결측인 특징은 중앙값이 NaN일 수 있다.
    # 이 경우 해당 특징의 학습 데이터 값이 없으므로 0으로 대치한다.
    medians = medians.fillna(0.0)

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        values = frame[features]
        missing_flags = values.isna().astype("int8")
        missing_flags.columns = [f"{c}_na" for c in features]
        filled = values.fillna(medians)
        return pd.concat([filled, missing_flags], axis=1)

    X_train = transform(train_df)
    X_test = transform(test_df)

    return (
        X_train.to_numpy(dtype=float),
        train_df["pose"].to_numpy(),
        X_test.to_numpy(dtype=float),
        test_df["pose"].to_numpy(),
    )


def run_one_ablation(
    data: pd.DataFrame,
    ablation_name: str,
    features: list[str],
    classes: list[str],
):
    """한 A 조건에 대해 참가자별 LOPO 학습/평가를 수행한다."""
    fold_rows = []
    all_true = []
    all_pred = []
    all_participants = []

    participants = sorted(data["p"].dropna().unique())

    for held_out in participants:
        train_df = data[data["p"] != held_out].copy()
        test_df = data[data["p"] == held_out].copy()

        X_train, y_train, X_test, y_test = prepare_mode_a(
            train_df, test_df, features
        )

        if len(X_train) < 100 or len(X_test) < 100 or len(set(y_train)) < 2:
            print(
                f"[경고] {ablation_name}/{held_out}: "
                f"학습 {len(X_train)}개, 테스트 {len(X_test)}개 또는 "
                "학습 클래스 부족으로 이 폴드를 건너뜁니다."
            )
            continue

        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        fold_f1 = f1_score(
            y_test, predictions, labels=classes,
            average="macro", zero_division=0
        )
        fold_acc = accuracy_score(y_test, predictions)

        fold_rows.append({
            "ablation": ablation_name,
            "participant": held_out,
            "n_features": len(features),
            "n_train": len(y_train),
            "n_test": len(y_test),
            "macro_f1": fold_f1,
            "accuracy": fold_acc,
            "keep_ratio": 1.0,
        })

        all_true.extend(y_test.tolist())
        all_pred.extend(predictions.tolist())
        all_participants.extend([held_out] * len(y_test))

        print(
            f"  {ablation_name} | test={held_out} | "
            f"Macro-F1={fold_f1:.4f} | Accuracy={fold_acc:.4f}"
        )

    if not all_true:
        raise RuntimeError(
            f"{ablation_name}: 유효한 LOPO 평가 결과가 없습니다."
        )

    return (
        pd.DataFrame(fold_rows),
        np.asarray(all_true),
        np.asarray(all_pred),
        np.asarray(all_participants),
    )


def main():
    parser = argparse.ArgumentParser(
        description="A0~A6 7-class RF feature ablation with LOPO"
    )
    parser.add_argument("--logs", default="data/side_logs")
    parser.add_argument("--out-dir", default="experiments")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] 로그 데이터 로드: {args.logs}")
    data = load(args.logs)

    required_base = {"pose", "p"}
    missing_base = required_base - set(data.columns)
    if missing_base:
        raise ValueError(f"로그 데이터에 필수 컬럼이 없습니다: {sorted(missing_base)}")

    missing_features = sorted(set(BASE_FEATURES) - set(data.columns))
    if missing_features:
        raise ValueError(
            "현재 로그 데이터에 어블레이션 특징이 없습니다: "
            f"{missing_features}"
        )

    unknown_labels = sorted(set(data["pose"].dropna().unique()) - set(LABEL_MAPPING))
    if unknown_labels:
        raise ValueError(f"매핑되지 않은 자세 라벨: {unknown_labels}")

    data = data.copy()
    data["pose"] = data["pose"].map(LABEL_MAPPING)
    data = data.dropna(subset=["pose", "p"]).reset_index(drop=True)

    participants = sorted(data["p"].unique())
    classes = sorted(data["pose"].unique())

    print(f"데이터: {len(data):,} frames")
    print(f"참가자: {participants}")
    print(f"클래스: {classes}")
    print("\n클래스별 프레임 수:")
    print(data["pose"].value_counts().sort_index())

    print("\n[2/5] A0~A6 LOPO 실험 시작")
    all_fold_metrics = []
    summary_rows = []
    class_rows = []
    confusion_rows = []
    report_sections = []

    for name, config in ABLATIONS.items():
        features = [
            feature for feature in BASE_FEATURES
            if feature not in config["removed"]
        ]

        print(
            f"\n--- {name}: {config['description']} "
            f"({len(features)}개 특징) ---"
        )
        print("사용 특징:", ", ".join(features))

        fold_df, y_true, y_pred, y_participants = run_one_ablation(
            data, name, features, classes
        )
        all_fold_metrics.append(fold_df)

        pooled_macro_f1 = f1_score(
            y_true, y_pred, labels=classes,
            average="macro", zero_division=0
        )
        pooled_accuracy = accuracy_score(y_true, y_pred)

        summary_rows.append({
            "ablation": name,
            "description": config["description"],
            "n_features": len(features),
            "removed_features": ";".join(config["removed"]),
            "used_features": ";".join(features),
            "pooled_macro_f1": pooled_macro_f1,
            "pooled_accuracy": pooled_accuracy,
            "mean_fold_macro_f1": fold_df["macro_f1"].mean(),
            "std_fold_macro_f1": fold_df["macro_f1"].std(ddof=0),
            "mean_fold_accuracy": fold_df["accuracy"].mean(),
            "n_valid_folds": len(fold_df),
            "n_evaluated_frames": len(y_true),
        })

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=classes, zero_division=0
        )
        for cls, p, r, score, n in zip(
            classes, precision, recall, f1, support
        ):
            class_rows.append({
                "ablation": name,
                "class": cls,
                "precision": p,
                "recall": r,
                "f1": score,
                "support": int(n),
            })

        cm = confusion_matrix(y_true, y_pred, labels=classes)
        for true_idx, true_cls in enumerate(classes):
            for pred_idx, pred_cls in enumerate(classes):
                confusion_rows.append({
                    "ablation": name,
                    "actual": true_cls,
                    "predicted": pred_cls,
                    "count": int(cm[true_idx, pred_idx]),
                })

        report_sections.append({
            "name": name,
            "description": config["description"],
            "features": features,
            "removed": config["removed"],
            "pooled_macro_f1": pooled_macro_f1,
            "pooled_accuracy": pooled_accuracy,
            "fold_mean": fold_df["macro_f1"].mean(),
            "fold_std": fold_df["macro_f1"].std(ddof=0),
            "class_metrics": pd.DataFrame({
                "class": classes,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }),
        })

    print("\n[3/5] CSV 결과 저장")
    summary_df = pd.DataFrame(summary_rows)
    fold_metrics_df = pd.concat(all_fold_metrics, ignore_index=True)
    class_metrics_df = pd.DataFrame(class_rows)
    confusion_df = pd.DataFrame(confusion_rows)

    summary_df.to_csv(
        out_dir / "ablation_a0_a6_summary.csv",
        index=False, encoding="utf-8-sig"
    )
    fold_metrics_df.to_csv(
        out_dir / "ablation_a0_a6_fold_metrics.csv",
        index=False, encoding="utf-8-sig"
    )
    class_metrics_df.to_csv(
        out_dir / "ablation_a0_a6_class_metrics.csv",
        index=False, encoding="utf-8-sig"
    )
    confusion_df.to_csv(
        out_dir / "ablation_a0_a6_confusion.csv",
        index=False, encoding="utf-8-sig"
    )

    print("\n[4/5] Markdown 분석 보고서 작성")
    baseline = summary_df.loc[summary_df["ablation"] == "A0"].iloc[0]

    lines = [
        "# A0~A6 특징 제거 어블레이션 실험 결과",
        "",
        "## 1. 실험 설정",
        "",
        f"- 전체 데이터: {len(data):,} frames",
        f"- 참가자: {', '.join(map(str, participants))}",
        f"- 분류 클래스: {len(classes)}개 ({', '.join(classes)})",
        "- 검증: Leave-One-Participant-Out (LOPO)",
        "- 결측 처리: Mode A (학습 폴드 중앙값 대치 + 결측 플래그)",
        "- 모델: Random Forest",
        "- 파라미터: n_estimators=300, class_weight=balanced, random_state=0",
        "- 평가: pooled OOF Macro-F1 / Accuracy 및 참가자별 fold 성능",
        "",
        "## 2. A0~A6 실험 정의",
        "",
        "| 실험 | 설명 | 사용 특징 수 | 제거 특징 |",
        "|---|---|---:|---|",
    ]

    for section in report_sections:
        lines.append(
            f"| {section['name']} | {section['description']} | "
            f"{len(section['features'])} | "
            f"{', '.join(section['removed']) if section['removed'] else '없음'} |"
        )

    lines += [
        "",
        "## 3. 전체 성능 비교",
        "",
        "| 실험 | 특징 수 | Pooled Macro-F1 | Pooled Accuracy | "
        "Fold 평균 Macro-F1 | Fold 표준편차 |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for row in summary_rows:
        lines.append(
            f"| {row['ablation']} | {row['n_features']} | "
            f"{row['pooled_macro_f1']:.4f} | "
            f"{row['pooled_accuracy']:.4f} | "
            f"{row['mean_fold_macro_f1']:.4f} | "
            f"{row['std_fold_macro_f1']:.4f} |"
        )

    lines += [
        "",
        "## 4. A0 기준 성능 차이",
        "",
        "| 실험 | Macro-F1 차이 (A0 대비) | Accuracy 차이 (A0 대비) |",
        "|---|---:|---:|",
    ]

    for row in summary_rows:
        delta_f1 = row["pooled_macro_f1"] - baseline["pooled_macro_f1"]
        delta_acc = row["pooled_accuracy"] - baseline["pooled_accuracy"]
        lines.append(
            f"| {row['ablation']} | {delta_f1:+.4f} | {delta_acc:+.4f} |"
        )

    lines += ["", "## 5. 클래스별 F1-score", ""]
    for section in report_sections:
        lines += [
            f"### {section['name']}",
            "",
            "| 클래스 | Precision | Recall | F1-score | Support |",
            "|---|---:|---:|---:|---:|",
        ]
        for _, row in section["class_metrics"].iterrows():
            lines.append(
                f"| {row['class']} | {row['precision']:.4f} | "
                f"{row['recall']:.4f} | {row['f1']:.4f} | "
                f"{int(row['support'])} |"
            )
        lines.append("")

    lines += [
        "## 6. 참가자별 성능",
        "",
        "| 실험 | 참가자 | Macro-F1 | Accuracy | 평가 프레임 수 |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in fold_metrics_df.iterrows():
        lines.append(
            f"| {row['ablation']} | {row['participant']} | "
            f"{row['macro_f1']:.4f} | {row['accuracy']:.4f} | "
            f"{int(row['n_test'])} |"
        )

    lines += [
        "",
        "## 7. 해석 시 주의사항",
        "",
        "- A1~A5는 각 특징 그룹을 제거했을 때의 성능 변화를 확인한다.",
        "- A6은 여러 거리/깊이 특징을 함께 제거한 별도 특징 조합이다.",
        "- Macro-F1은 클래스별 F1의 산술평균이며, Accuracy와 함께 해석한다.",
        "- 참가자 수가 적으므로 참가자 간 일반화 성능에 대한 결론은 신중해야 한다.",
        "- 성능 변화만으로 특정 특징이 인과적으로 유효하다고 단정하지 않는다.",
        "- 이 스크립트는 기존 비교 코드의 Mode A만 사용하며, Mode B는 실행하지 않는다.",
        "",
        "## 8. 생성 파일",
        "",
        "- `ablation_a0_a6_summary.csv`",
        "- `ablation_a0_a6_fold_metrics.csv`",
        "- `ablation_a0_a6_class_metrics.csv`",
        "- `ablation_a0_a6_confusion.csv`",
        "- `ablation_a0_a6_analysis.md`",
        "",
    ]

    report_path = out_dir / "ablation_a0_a6_analysis.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n[5/5] 완료")
    print(summary_df[
        ["ablation", "n_features", "pooled_macro_f1",
         "pooled_accuracy", "mean_fold_macro_f1"]
    ].to_string(index=False))
    print(f"\n결과 폴더: {out_dir.resolve()}")
    print(f"보고서: {report_path.resolve()}")


if __name__ == "__main__":
    main()
