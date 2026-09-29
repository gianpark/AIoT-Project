#!/usr/bin/env python3
"""
WRF 논문(Lee et al., 2025) 재현 실험 - RF 학습 파이프라인 사전 검증용

목적
----
아직 우리 프로젝트의 실제 자세 데이터(스테레오 카메라)가 없는 상태에서,
6주차부터 쓸 "RF -> 변수 중요도 추출 -> 가중치 적용 -> WRF 재학습" 파이프라인
코드 자체가 정상 동작하는지 미리 검증한다.

데이터는 우리 프로젝트 데이터가 아니라 WRF 논문이 공개한 데이터셋
(https://github.com/icml2410/posture, Dataset.xlsx)을 그대로 쓴다.
주의: 이 데이터셋은 MediaPipe 기반 "얼굴+양어깨" 좌표/각도 6개 변수이고,
우리 프로젝트가 쓸 MoveNet 17키포인트 + 스테레오 깊이 기반 특징과는
다른 특징 공간이다. 그래서 여기서 나온 정확도 수치는 "우리 프로젝트의
예상 성능"이 아니라 "논문이 보고한 수치(96%->98%)를 우리 코드로도
재현할 수 있는가"만 확인하는 용도다 (project.md 참고).

실행
----
    python -m experiments.wrf_paper_replication

데이터 없으면
------------
data/external_wrf/Dataset.xlsx 를 아래에서 받아서 넣을 것:
    https://raw.githubusercontent.com/icml2410/posture/main/Dataset.xlsx
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "external_wrf" / "Dataset.xlsx"

# 논문 Table 기준 6개 특징 컬럼 매핑 (Dataset.xlsx 실제 헤더 -> 논문 변수명)
FEATURE_COLUMNS = {
    "Prop-Area Size": "A",       # 얼굴 면적비 (가중치 0.30)
    "Center-x": "xc",             # 얼굴 중심 x (가중치 0.20)
    "Angel(Left-C)-5": "theta3",  # 얼굴중심-좌측어깨 각도 (가중치 0.15)
    "Angel(C-Right)-4": "theta2", # 얼굴중심-우측어깨 각도 (가중치 0.15)
    "Angel(1-3)": "theta1",       # 얼굴중심-양어깨 전체각도 (가중치 0.15)
    "Center-y": "yc",             # 얼굴 중심 y (가중치 0.05)
}
LABEL_COLUMN = "Pose"

RANDOM_STATE = 42
N_FOLDS = 10


def load_dataset(path: Path = DATA_PATH) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not path.exists():
        raise SystemExit(
            f"데이터셋이 없습니다: {path}\n"
            "다음에서 받아서 넣으세요:\n"
            "  curl -L https://raw.githubusercontent.com/icml2410/posture/main/Dataset.xlsx "
            f"-o {path}"
        )
    df = pd.read_excel(path)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"예상 컬럼이 없습니다(포맷이 바뀐 듯): {missing}")

    feature_names = list(FEATURE_COLUMNS.values())
    X = df[list(FEATURE_COLUMNS.keys())].to_numpy(dtype=float)
    y = df[LABEL_COLUMN].to_numpy()
    return X, y, feature_names


def run_plain_rf(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """가중치 없는 일반 RF, 10-fold 교차검증. (importances는 전체 데이터로 학습한 참고용)"""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

    clf.fit(X, y)  # 변수 중요도(MDI) 추출용 - 전체 데이터로 한 번 더 학습
    return float(scores.mean()), clf.feature_importances_


def run_weighted_rf(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    """변수 중요도를 가중치로 곱한 뒤 재학습(WRF), 10-fold 교차검증."""
    X_weighted = X * weights
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    scores = cross_val_score(clf, X_weighted, y, cv=cv, scoring="accuracy")
    return float(scores.mean())


def main() -> None:
    X, y, feature_names = load_dataset()
    print(f"데이터셋 로드 완료: {X.shape[0]}개 샘플, {X.shape[1]}개 특징, 클래스 {sorted(set(y))}")

    plain_acc, importances = run_plain_rf(X, y)
    print("\n=== 1) 가중치 없는 RF (10-fold 평균 정확도) ===")
    print(f"정확도: {plain_acc:.4f}  (논문 보고치: 0.96)")

    print("\n=== 2) RF 변수 중요도(MDI) - 논문 Table 가중치와 비교 ===")
    paper_weights = {"A": 0.30, "xc": 0.20, "theta3": 0.15, "theta2": 0.15, "theta1": 0.15, "yc": 0.05}
    for name, imp in sorted(zip(feature_names, importances), key=lambda t: -t[1]):
        print(f"  {name:8s}  MDI={imp:.4f}   논문가중치={paper_weights.get(name, float('nan')):.2f}")

    # 논문이 실제로 쓴 고정 가중치로 WRF 재현
    weight_vec = np.array([paper_weights[name] for name in feature_names])
    weighted_acc_paper_weights = run_weighted_rf(X, y, weight_vec)

    # 참고: 논문 가중치 대신 우리가 방금 구한 MDI 자체를 가중치로 써보면 어떻게 되는지도 같이 확인
    weighted_acc_our_mdi = run_weighted_rf(X, y, importances)

    print("\n=== 3) WRF (가중치 적용 RF, 10-fold 평균 정확도) ===")
    print(f"논문 고정 가중치 사용:      {weighted_acc_paper_weights:.4f}  (논문 보고치: 0.98)")
    print(f"우리가 구한 MDI를 가중치로: {weighted_acc_our_mdi:.4f}")

    print("\n=== 결론 ===")
    print(f"가중치 없는 RF -> WRF 정확도 변화: {plain_acc:.4f} -> {weighted_acc_paper_weights:.4f}")
    print(
        "파이프라인(로드 -> RF학습 -> 중요도추출 -> 가중치적용 -> 재학습 -> 평가)이 "
        "정상 동작함을 확인. 실제 우리 프로젝트 특징(MoveNet 17키포인트 기반)이 준비되면 "
        "FEATURE_COLUMNS 부분만 우리 데이터 스키마로 바꿔서 같은 파이프라인을 그대로 쓸 수 있음."
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
