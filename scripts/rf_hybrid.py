"""Tuned RF / Lightweight RF / Hybrid RF 비교 실험.

Hybrid RF 정의:
- Lightweight RF의 경량 제약(n_estimators <= 50, max_depth <= 10,
  min_samples_split >= 5, min_samples_leaf >= 2)을 유지한다.
- 이 제약 범위 안에서 RandomizedSearchCV로 하이퍼파라미터를 탐색한다.
- 바깥 평가는 기존 rf_tuning.py와 동일하게 LOPO,
  안쪽 탐색은 (참가자, 촬영회차) 그룹 GroupKFold를 사용한다.

실행:
python -m scripts.rf_hybrid --n-iter 20 --seed 0
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


# 기존 Tuned RF와 동일한 탐색 공간
TUNED_SPACE = {
    "n_estimators": [30, 50, 100, 200],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "class_weight": [None, "balanced", "balanced_subsample"],
}

# Lightweight RF의 복잡도 제한 안에서만 탐색하는 Hybrid 공간
# 트리 수 최대 50, 깊이 최대 10, 분할/리프 최소 샘플 수는
# Lightweight 설정(5, 2)보다 작아지지 않도록 제한한다.
HYBRID_SPACE = {
    "n_estimators": [30, 40, 50],
    "max_depth": [5, 8, 10],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [2, 4],
    "max_features": ["sqrt", "log2"],
    "class_weight": [None, "balanced"],
}

FIXED_MODELS = {
    "Baseline": {
        "n_estimators": 300,
        "class_weight": "balanced",
    },
    "Lightweight": {
        "n_estimators": 50,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
    },
}


def size_kb(model: RandomForestClassifier) -> float:
    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=0)
    return buffer.tell() / 1024


def latency_ms(
    model: RandomForestClassifier,
    X: np.ndarray,
    n: int = 300,
    seed: int = 0,
) -> tuple[float, float]:
    model.set_params(n_jobs=1)
    count = min(n, len(X))
    indices = np.random.default_rng(seed).choice(
        len(X), size=count, replace=False
    )
    model.predict(X[indices[: min(5, count)]])  # 워밍업

    timings = []
    for index in indices:
        start = time.perf_counter()
        model.predict(X[index : index + 1])
        timings.append((time.perf_counter() - start) * 1000)

    return float(np.median(timings)), float(np.percentile(timings, 95))


def run_search(
    space: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    n_iter: int,
    stride: int,
    seed: int,
) -> dict:
    sampled_indices = np.arange(0, len(X_train), stride)
    sampled_groups = groups[sampled_indices]
    n_groups = len(set(sampled_groups))

    if n_groups < 2:
        raise ValueError(
            "GroupKFold를 수행할 그룹이 2개 미만입니다. "
            "촬영회차 데이터 또는 --stride 설정을 확인하세요."
        )

    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=seed, n_jobs=-1),
        param_distributions=space,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=GroupKFold(n_splits=min(3, n_groups)),
        random_state=seed,
        n_jobs=1,
        refit=False,
    )
    search.fit(
        X_train[sampled_indices],
        y_train[sampled_indices],
        groups=sampled_groups,
    )
    return search.best_params_


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default="data/side_logs")
    parser.add_argument("--out", default="experiments/rf_hybrid.md")
    parser.add_argument(
        "--csv-out", default="experiments/rf_hybrid_runs.csv"
    )
    parser.add_argument(
        "--feature-set", default="no_hip", choices=list(SETS)
    )
    parser.add_argument("--merge-intensity", action="store_true")
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="하이퍼파라미터 탐색 시 N프레임마다 1개 사용",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = load(args.logs)
    data = data[~data.pose.str.contains("lean")]
    if args.merge_intensity:
        data = data.assign(
            pose=data.pose.replace(
                {
                    "back_light": "back",
                    "back_hard": "back",
                    "neck_light": "neck",
                    "neck_hard": "neck",
                }
            )
        )
    data = data.reset_index(drop=True)

    columns = SETS[args.feature_set]
    participants = sorted(data.p.unique())
    classes = sorted(data.pose.unique())

    model_names = ["Baseline", "Lightweight", "Tuned", "Hybrid"]
    results = {name: [] for name in model_names}
    selected_params = {"Tuned": {}, "Hybrid": {}}

    for held_out in participants:
        train_data = data[data.p != held_out]
        test_data = data[data.p == held_out]

        X_train, y_train, X_test, y_test, _ = prep(
            train_data, test_data, columns, "A"
        )
        groups = (train_data.p + "_" + train_data.rep).values

        params_by_model = dict(FIXED_MODELS)

        tuned_params = run_search(
            TUNED_SPACE,
            X_train,
            y_train,
            groups,
            args.n_iter,
            args.stride,
            args.seed,
        )
        hybrid_params = run_search(
            HYBRID_SPACE,
            X_train,
            y_train,
            groups,
            args.n_iter,
            args.stride,
            args.seed,
        )

        selected_params["Tuned"][held_out] = tuned_params
        selected_params["Hybrid"][held_out] = hybrid_params
        params_by_model["Tuned"] = tuned_params
        params_by_model["Hybrid"] = hybrid_params

        for model_name in model_names:
            params = params_by_model[model_name]
            model = RandomForestClassifier(
                random_state=args.seed, n_jobs=-1, **params
            )

            fit_start = time.perf_counter()
            model.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - fit_start

            predictions = model.predict(X_test)
            median_latency, p95_latency = latency_ms(
                model, X_test, seed=args.seed
            )

            row = {
                "held_out_participant": held_out,
                "model": model_name,
                "macro_f1": f1_score(
                    y_test, predictions, average="macro"
                ),
                "accuracy": float(np.mean(predictions == y_test)),
                "model_size_kb": size_kb(model),
                "inference_median_ms": median_latency,
                "inference_p95_ms": p95_latency,
                "training_time_s": fit_seconds,
                "selected_params": repr(params),
            }
            row["class_recall_percent"] = (
                recall_score(
                    y_test,
                    predictions,
                    labels=classes,
                    average=None,
                    zero_division=0,
                )
                * 100
            ).tolist()
            results[model_name].append(row)

        print(f"완료: {held_out}", flush=True)

    # 참가자별 상세 결과 CSV 저장
    flat_rows = []
    for model_name in model_names:
        for row in results[model_name]:
            flat_rows.append(
                {
                    key: value
                    for key, value in row.items()
                    if key != "class_recall_percent"
                }
            )

    os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
    pd.DataFrame(flat_rows).to_csv(
        args.csv_out, index=False, encoding="utf-8-sig"
    )

    # 보고서 생성
    lines = [
        "# Tuned RF / Lightweight RF / Hybrid RF 비교",
        "",
        (
            f"- 특징 세트: {args.feature_set}, 결측 처리 A, "
            f"{len(classes)}개 클래스, 참가자 {len(participants)}명 "
            f"({', '.join(participants)}), LOPO, seed={args.seed}."
        ),
        (
            f"- Tuned/Hybrid 탐색: 각각 RandomizedSearchCV "
            f"{args.n_iter}회, GroupKFold(그룹=참가자×촬영회차), "
            f"탐색 시 {args.stride}프레임마다 1개 사용."
        ),
        (
            "- Hybrid 정의: Lightweight의 복잡도 제한 안에서 "
            "하이퍼파라미터를 탐색한 모델."
        ),
        (
            "- 참가자가 3명으로 적으므로 평균 차이뿐 아니라 "
            "참가자별 결과와 폴드 간 편차를 함께 해석한다."
        ),
        "",
        "## 요약 (참가자별 결과의 평균 ± 표준편차)",
        "",
        "| 모델 | Macro-F1 | Accuracy | 모델 크기(KB) | 추론 중앙값(ms) | 추론 p95(ms) | 학습 시간(s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    metric_keys = [
        ("macro_f1", "mean_std"),
        ("accuracy", "mean"),
        ("model_size_kb", "mean"),
        ("inference_median_ms", "mean"),
        ("inference_p95_ms", "mean"),
        ("training_time_s", "mean"),
    ]

    for model_name in model_names:
        rows = results[model_name]
        f1_values = [row["macro_f1"] for row in rows]
        summary = [
            f"{np.mean(f1_values):.3f} ± {np.std(f1_values):.3f}"
        ]
        for key, _ in metric_keys[1:]:
            summary.append(
                f"{np.mean([row[key] for row in rows]):.2f}"
            )
        lines.append(f"| {model_name} | " + " | ".join(summary) + " |")

    lines += [
        "",
        "## 참가자별 Macro-F1",
        "",
        "| 모델 | " + " | ".join(participants) + " |",
        "|---|" + "---|" * len(participants),
    ]
    for model_name in model_names:
        values = [
            f"{row['macro_f1']:.3f}" for row in results[model_name]
        ]
        lines.append(f"| {model_name} | " + " | ".join(values) + " |")

    lines += [
        "",
        "## 클래스별 Recall(%, 참가자 평균)",
        "",
        "| 자세 | " + " | ".join(model_names) + " |",
        "|---|" + "---|" * len(model_names),
    ]
    for class_index, class_name in enumerate(classes):
        recalls = []
        for model_name in model_names:
            recalls.append(
                f"{np.mean([row['class_recall_percent'][class_index] for row in results[model_name]]):.1f}"
            )
        lines.append(
            f"| {class_name} | " + " | ".join(recalls) + " |"
        )

    lines += ["", "## 폴드별 선택 하이퍼파라미터", ""]
    for model_name in ["Tuned", "Hybrid"]:
        lines.append(f"### {model_name}")
        for held_out, params in selected_params[model_name].items():
            lines.append(f"- {held_out} 제외 학습: `{params}`")
        lines.append("")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\n보고서 저장: {args.out}")
    print(f"상세 CSV 저장: {args.csv_out}")


if __name__ == "__main__":
    main()
