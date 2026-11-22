"""Hybrid RF 양자화 비교 실험.

기준: scripts/rf_hybrid.py의 Hybrid 학습/LOPO 조건을 유지한다.
비교: float64 원본, float16, int16, uint8 (threshold + leaf probabilities).
각 seed/LOPO fold에서 학습한 동일 모델을 복사해 양자화한다.

실행:
python -m scripts.rf_hybrid_quantization --n-iter 20 --stride 5
빠른 점검:
python -m scripts.rf_hybrid_quantization --n-iter 2 --stride 20 --seeds 0
"""

from __future__ import annotations

import argparse
import io
import os
import time
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from scripts.replay_judge_on_logs import load
from scripts.rf_hybrid import HYBRID_SPACE, run_search
from scripts.rf_lopo import SETS, prep


def joblib_size_kb(obj) -> float:
    buffer = io.BytesIO()
    joblib.dump(obj, buffer, compress=0)
    return buffer.tell() / 1024.0


def _quantize_range(values: np.ndarray, qmax: int):
    """[min, max] 범위를 0..qmax 정수로 선형 양자화."""
    values = np.asarray(values, dtype=np.float64)
    vmin = float(np.min(values)) if values.size else 0.0
    vmax = float(np.max(values)) if values.size else 0.0
    scale = (vmax - vmin) / qmax if vmax > vmin else 1.0
    q = np.clip(np.rint((values - vmin) / scale), 0, qmax)
    return q, vmin, scale


@dataclass
class PackedTree:
    children_left: np.ndarray
    children_right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    value: np.ndarray
    threshold_min: np.ndarray | None
    threshold_scale: np.ndarray | None
    value_qmax: int | None


@dataclass
class PackedForest:
    classes: np.ndarray
    trees: list[PackedTree]
    mode: str

    def packed_payload_bytes(self) -> int:
        """추론에 필요한 NumPy 배열 및 양자화 메타데이터의 순수 payload 크기."""
        total = self.classes.nbytes
        for tree in self.trees:
            for arr in (tree.children_left, tree.children_right, tree.feature,
                        tree.threshold, tree.value):
                total += arr.nbytes
            if tree.threshold_min is not None:
                total += tree.threshold_min.nbytes
                total += tree.threshold_scale.nbytes
            # value는 [0, 1] 고정 범위 양자화이므로 별도 scale metadata가 없다.
        return int(total)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        total = np.zeros((len(X), len(self.classes)), dtype=np.float64)

        for tree in self.trees:
            left, right, feature = tree.children_left, tree.children_right, tree.feature
            tree_sum = np.zeros_like(total)

            for row_index, row in enumerate(X):
                node = 0
                while left[node] != right[node]:
                    f = int(feature[node])
                    q_threshold = tree.threshold[node]
                    if self.mode == "float16":
                        threshold = float(q_threshold)
                    elif self.mode in ("int16", "uint8"):
                        threshold = (
                            tree.threshold_min[f]
                            + float(q_threshold) * tree.threshold_scale[f]
                        )
                    else:
                        threshold = float(q_threshold)

                    node = int(left[node] if float(row[f]) <= threshold else right[node])

                probs = tree.value[node].astype(np.float64)
                if self.mode == "float16":
                    pass
                elif self.mode in ("int16", "uint8"):
                    probs = probs / float(tree.value_qmax)

                denom = probs.sum()
                if denom > 0:
                    probs = probs / denom
                tree_sum[row_index] = probs

            total += tree_sum

        return total / max(len(self.trees), 1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes[np.argmax(self.predict_proba(X), axis=1)]


def pack_forest(model: RandomForestClassifier, mode: str) -> PackedForest:
    packed_trees = []

    for estimator in model.estimators_:
        source = estimator.tree_
        raw_threshold = source.threshold.astype(np.float64, copy=True)
        raw_values = source.value[:, 0, :].astype(np.float64, copy=True)
        internal = source.children_left != source.children_right

        threshold_min = None
        threshold_scale = None
        value_qmax = None

        if mode == "float64":
            threshold = raw_threshold.astype(np.float64)
            values = raw_values.astype(np.float64)
        elif mode == "float16":
            threshold = raw_threshold.astype(np.float16)
            values = raw_values.astype(np.float16)
        elif mode in ("int16", "uint8"):
            # 특징(feature)마다 threshold 범위를 따로 계산해 스케일 손실을 줄인다.
            qmax = 32767 if mode == "int16" else 255
            qdtype = np.int16 if mode == "int16" else np.uint8
            threshold = np.zeros(raw_threshold.shape, dtype=qdtype)
            n_features = model.n_features_in_
            threshold_min = np.zeros(n_features, dtype=np.float32)
            threshold_scale = np.ones(n_features, dtype=np.float32)

            for feature_index in np.unique(source.feature[internal]):
                mask = internal & (source.feature == feature_index)
                q, vmin, scale = _quantize_range(raw_threshold[mask], qmax)
                threshold[mask] = q.astype(qdtype)
                threshold_min[int(feature_index)] = vmin
                threshold_scale[int(feature_index)] = scale

            # 리프 class probability는 [0,1] 범위를 정수로 저장한다.
            # sklearn tree.value는 각 노드의 클래스 비율이므로 같은 범위를 사용한다.
            value_qmax = qmax
            values = np.clip(np.rint(raw_values * qmax), 0, qmax).astype(qdtype)
        else:
            raise ValueError(f"지원하지 않는 양자화 방식: {mode}")

        packed_trees.append(PackedTree(
            children_left=source.children_left.astype(np.int32, copy=True),
            children_right=source.children_right.astype(np.int32, copy=True),
            feature=source.feature.astype(np.int16, copy=True),
            threshold=threshold,
            value=values,
            threshold_min=threshold_min,
            threshold_scale=threshold_scale,
            value_qmax=value_qmax,
        ))

    return PackedForest(
        classes=np.asarray(model.classes_).copy(),
        trees=packed_trees,
        mode=mode,
    )


def packed_size_kb(model: PackedForest) -> float:
    return model.packed_payload_bytes() / 1024.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default="data/side_logs")
    parser.add_argument("--out", default="experiments/rf_hybrid_quantization.md")
    parser.add_argument("--csv-out", default="experiments/rf_hybrid_quantization_runs.csv")
    parser.add_argument("--feature-set", default="no_hip", choices=list(SETS))
    parser.add_argument("--merge-intensity", action="store_true")
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    data = load(args.logs)
    data = data[~data.pose.str.contains("lean")]
    if args.merge_intensity:
        data = data.assign(
            pose=data.pose.replace({
                "back_light": "back", "back_hard": "back",
                "neck_light": "neck", "neck_hard": "neck",
            })
        )
    data = data.reset_index(drop=True)

    columns = SETS[args.feature_set]
    participants = sorted(data.p.unique())
    methods = ["float64", "float16", "int16", "uint8"]
    rows = []
    pooled = {seed: {method: {"y": [], "pred": []} for method in methods}
              for seed in args.seeds}

    for seed in args.seeds:
        print(f"\n===== seed={seed} =====", flush=True)
        for held_out in participants:
            train_data = data[data.p != held_out]
            test_data = data[data.p == held_out]
            X_train, y_train, X_test, y_test, _ = prep(
                train_data, test_data, columns, "A"
            )
            groups = (train_data.p + "_" + train_data.rep).values

            best_params = run_search(
                HYBRID_SPACE, X_train, y_train, groups,
                args.n_iter, args.stride, seed,
            )
            model = RandomForestClassifier(
                random_state=seed, n_jobs=-1, **best_params
            )
            start = time.perf_counter()
            model.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - start

            for method in methods:
                evaluator = pack_forest(model, method)
                # 네 방식 모두 동일한 packed 구조(노드 인덱스 int32, feature int16)를 사용해
                # 배열 payload 기준으로 비교한다. Python/joblib 포장 오버헤드는 제외한다.
                size_kb = packed_size_kb(evaluator)

                if method == "float64":
                    # float64 packed traversal이 sklearn 원본 예측과 같은지 검증한다.
                    reference = model.predict(X_test)
                    packed_prediction = evaluator.predict(X_test)
                    if not np.array_equal(reference, packed_prediction):
                        reference_proba = model.predict_proba(X_test)
                        packed_proba = evaluator.predict_proba(X_test)

                        print("\\n===== Float64 진단 결과 =====")
                        print("원본 예측 불일치 개수:",
                              np.sum(reference != packed_prediction))
                        print("전체 테스트 개수:", len(reference))
                        print("예측 확률 최대 차이:",
                              np.max(np.abs(reference_proba - packed_proba)))

                        mismatch = np.where(reference != packed_prediction)[0][:5]
                        X_check = np.asarray(X_test, dtype=np.float32)

                        for idx in mismatch:
                            for tree_idx, (est, packed_tree) in enumerate(
                                zip(model.estimators_, evaluator.trees)
                            ):
                                node = 0
                                while packed_tree.children_left[node] != packed_tree.children_right[node]:
                                    f = int(packed_tree.feature[node])
                                    threshold = float(packed_tree.threshold[node])
                                    node = int(
                                        packed_tree.children_left[node]
                                        if float(X_check[idx, f]) <= threshold
                                        else packed_tree.children_right[node]
                                    )

                                sklearn_leaf = int(est.apply(X_check[[idx]])[0])
                                if node != sklearn_leaf:
                                    print(
                                        f"분기 불일치: 행={idx}, 트리={tree_idx}, "
                                        f"직접구현 leaf={node}, sklearn leaf={sklearn_leaf}"
                                    )

                            print(f"행 {idx}:")
                            print("  sklearn 클래스:", reference[idx])
                            print("  packed 클래스:", packed_prediction[idx])
                            print("  sklearn 확률:", reference_proba[idx])
                            print("  packed 확률:", packed_proba[idx])

                        raise RuntimeError(
                            "Float64 진단 완료: 위 결과를 확인하세요."
                        )
                predictions = evaluator.predict(X_test)
                score = f1_score(y_test, predictions, average="macro")
                rows.append({
                    "seed": seed,
                    "held_out_participant": held_out,
                    "quantization": method,
                    "macro_f1": score,
                    "model_size_kb": size_kb,
                    "training_time_s": fit_seconds,
                    "selected_params": repr(best_params),
                })
                pooled[seed][method]["y"].extend(y_test.tolist())
                pooled[seed][method]["pred"].extend(predictions.tolist())

            print(f"완료: seed={seed}, held_out={held_out}", flush=True)

    result = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
    result.to_csv(args.csv_out, index=False, encoding="utf-8-sig")

    summary_rows = []
    for method in methods:
        subset = result[result.quantization == method]
        fold_scores = subset.macro_f1.to_numpy()
        pooled_scores = [
            f1_score(pooled[s][method]["y"], pooled[s][method]["pred"], average="macro")
            for s in args.seeds
        ]
        sizes = subset.model_size_kb.to_numpy()
        summary_rows.append({
            "quantization": method,
            "fold_macro_f1_mean": float(np.mean(fold_scores)),
            "fold_macro_f1_std": float(np.std(fold_scores)),
            "pooled_oof_macro_f1_mean": float(np.mean(pooled_scores)),
            "pooled_oof_macro_f1_std": float(np.std(pooled_scores)),
            "model_size_kb_mean": float(np.mean(sizes)),
            "model_size_kb_std": float(np.std(sizes)),
        })
    summary = pd.DataFrame(summary_rows)

    report = [
        "# Hybrid RF 양자화 비교 실험",
        "",
        f"- 특징 세트: {args.feature_set}, 결측 처리 A, LOPO",
        f"- Seeds: {args.seeds}",
        f"- 탐색: RandomizedSearchCV, n_iter={args.n_iter}, stride={args.stride}",
        "- 각 seed/LOPO 폴드에서 학습한 동일 Hybrid RF를 양자화해 비교",
        "- 크기: 동일한 packed 배열 구조의 순수 payload 크기(nbytes), KB; Python/joblib 오버헤드 제외",
        "- 양자화 대상: 특징별 트리 분기 threshold 및 노드 class probability",
        "",
        "## 요약",
        "",
        "| 방식 | 폴드 Macro-F1 평균 ± 표준편차 | Pooled OOF Macro-F1 평균 ± 표준편차 | 모델 크기 KB 평균 ± 표준편차 |",
        "|---|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        report.append(
            f"| {r.quantization} | {r.fold_macro_f1_mean:.4f} ± {r.fold_macro_f1_std:.4f} "
            f"| {r.pooled_oof_macro_f1_mean:.4f} ± {r.pooled_oof_macro_f1_std:.4f} "
            f"| {r.model_size_kb_mean:.2f} ± {r.model_size_kb_std:.2f} |"
        )
    report += ["", "## 폴드별 결과", ""]
    report.append("| Seed | 평가 참가자 | 방식 | Macro-F1 | 모델 크기(KB) |")
    report.append("|---:|---|---|---:|---:|")
    for _, r in result.iterrows():
        report.append(
            f"| {int(r.seed)} | {r.held_out_participant} | {r.quantization} "
            f"| {r.macro_f1:.4f} | {r.model_size_kb:.2f} |"
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    print("\n".join(report))
    print(f"\n보고서 저장: {args.out}")
    print(f"CSV 저장: {args.csv_out}")


if __name__ == "__main__":
    main()
