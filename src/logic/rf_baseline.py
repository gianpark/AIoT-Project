"""
바른자세 - RF 기준모델 학습 스켈레톤 (4주차, 임시 데이터 버전)

주의 — 이 파일이 하는 것과 하지 않는 것
------------------------------------
아직 실제 촬영 데이터가 없다(데이터 수집은 5주차 예정, `data_collection_protocol.md`
참고). 그래서 이 파일은 "진짜 분류 성능"을 보여주는 코드가 아니라, 아래 파이프라인이
에러 없이 동작하는지만 확인하는 뼈대(스켈레톤)다:

    합성(가짜) keypoint 생성 -> 증강(좌우반전+jitter) -> 특징 추출(posture_features)
    -> train/test split -> RandomForest 학습 -> 평가

여기서 나오는 정확도 수치는 100% 합성 데이터에서 나온 것이라 의미 없다 — 클래스별로
좌표를 의도적으로 다르게 만들어서 RF가 구분하기 쉽도록 짠 것이기 때문이다. 실제
정확도는 6주차에 진짜 데이터로 다시 측정해야 한다.

실행
----
    python -m src.logic.rf_baseline
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.features.posture_features import compute_posture_features

KEYPOINT_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

# 좌우 반전 시 이름이 서로 바뀌어야 하는 쌍 (data_collection_protocol.md 7절 증강 계획)
LEFT_RIGHT_PAIRS = [
    ("left_eye", "right_eye"), ("left_ear", "right_ear"),
    ("left_shoulder", "right_shoulder"), ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"), ("left_hip", "right_hip"),
    ("left_knee", "right_knee"), ("left_ankle", "right_ankle"),
]

# data_collection_protocol.md 1절의 클래스 코드
CLASS_LABELS = ["normal", "slouch_forward", "slouch_back", "tilt_left", "tilt_right"]


@dataclass
class SyntheticKeypoint:
    """src.pose.movenet_keypoints.Keypoint과 동일한 인터페이스(name, x, y, score).

    실제 Keypoint 클래스를 쓰지 않고 이 클래스를 따로 두는 이유: 이 모듈은 카메라·
    모델 없이도(합성 데이터로) 돌아가야 해서, 실제 MoveNet 의존성을 아예 안 만든다.
    """
    name: str
    x: float
    y: float
    score: float


def _base_seated_pose() -> list[SyntheticKeypoint]:
    """tests/test_posture_features.py의 '좌우 대칭 정자세'와 동일한 기준 좌표."""
    coords = [
        (0.50, 0.20, 0.90),  # nose
        (0.47, 0.18, 0.80), (0.53, 0.18, 0.80),  # eyes
        (0.45, 0.19, 0.70), (0.55, 0.19, 0.70),  # ears
        (0.40, 0.30, 0.90), (0.60, 0.30, 0.90),  # shoulders
        (0.35, 0.45, 0.70), (0.65, 0.45, 0.70),  # elbows
        (0.33, 0.55, 0.60), (0.67, 0.55, 0.60),  # wrists
        (0.45, 0.70, 0.90), (0.55, 0.70, 0.90),  # hips
        (0.44, 0.85, 0.05), (0.56, 0.85, 0.05),  # knees (책상 구도, 저신뢰)
        (0.43, 0.98, 0.05), (0.57, 0.98, 0.05),  # ankles (책상 구도, 저신뢰)
    ]
    return [
        SyntheticKeypoint(name=n, x=x, y=y, score=s)
        for n, (x, y, s) in zip(KEYPOINT_NAMES, coords)
    ]


# 클래스별로 기준 자세에서 어떤 keypoint를 어떻게 움직일지 (이름 -> (dx, dy)).
# 각도의 물리적 정확성보다는 "클래스마다 다른 좌표 패턴"을 만드는 게 목적이다 —
# 실제 방향·크기는 data_collection_protocol.md 1절의 정성적 설명을 참고했을 뿐,
# 실측으로 검증된 값이 아니다(6주차에 실제 데이터로 다시 잡아야 함).
_CLASS_OFFSETS: dict[str, dict[str, tuple[float, float]]] = {
    "normal": {},
    "slouch_forward": {  # 숙임 - 고개가 화면 쪽으로 내려오고 어깨가 앞으로 쏠림
        "nose": (0.0, 0.06),
        "left_shoulder": (0.03, 0.02), "right_shoulder": (0.03, 0.02),
    },
    "slouch_back": {  # 기대기 - 등받이에 기대면서 상체가 뒤로/아래로
        "left_shoulder": (-0.03, 0.03), "right_shoulder": (-0.03, 0.03),
        "left_hip": (0.0, 0.04), "right_hip": (0.0, 0.04),
    },
    "tilt_left": {  # 좌측 기울임 - 상체 전체가 왼쪽으로
        "nose": (-0.06, 0.0),
        "left_shoulder": (-0.05, 0.03), "right_shoulder": (-0.05, -0.02),
    },
    "tilt_right": {  # 우측 기울임 - 좌측 기울임과 대칭
        "nose": (0.06, 0.0),
        "left_shoulder": (0.05, -0.02), "right_shoulder": (0.05, 0.03),
    },
}


def make_synthetic_pose(label: str) -> list[SyntheticKeypoint]:
    """지정한 클래스의 기준 자세(jitter 없음)를 만든다."""
    if label not in _CLASS_OFFSETS:
        raise ValueError(f"알 수 없는 클래스: {label} (가능한 값: {CLASS_LABELS})")

    pose = _base_seated_pose()
    offsets = _CLASS_OFFSETS[label]
    result = []
    for kp in pose:
        dx, dy = offsets.get(kp.name, (0.0, 0.0))
        result.append(replace(kp, x=kp.x + dx, y=kp.y + dy))
    return result


def flip_keypoints_lr(keypoints: list[SyntheticKeypoint]) -> list[SyntheticKeypoint]:
    """
    좌우 반전 증강 (data_collection_protocol.md 7절).

    x좌표를 화면 기준으로 뒤집고(x -> 1-x), left_*/right_* 이름도 서로 맞바꾼다.
    tilt_left 샘플을 반전하면 tilt_right와 같은 분포가 되는 걸 기대할 수 있다.
    """
    name_map = {}
    for left, right in LEFT_RIGHT_PAIRS:
        name_map[left] = right
        name_map[right] = left

    flipped = []
    for kp in keypoints:
        new_name = name_map.get(kp.name, kp.name)  # 좌우 쌍 없는 것(nose 등)은 이름 유지
        flipped.append(SyntheticKeypoint(name=new_name, x=1.0 - kp.x, y=kp.y, score=kp.score))
    # 원래 keypoint 순서(KEYPOINT_NAMES 순)를 유지해서 반환
    flipped_by_name = {kp.name: kp for kp in flipped}
    return [flipped_by_name[n] for n in KEYPOINT_NAMES]


def jitter_keypoints(
    keypoints: list[SyntheticKeypoint], sigma: float = 0.01, rng: random.Random | None = None
) -> list[SyntheticKeypoint]:
    """각 keypoint 좌표에 작은 가우시안 노이즈를 더하는 증강 (data_collection_protocol.md 7절)."""
    rng = rng or random.Random()
    return [
        replace(kp, x=kp.x + rng.gauss(0, sigma), y=kp.y + rng.gauss(0, sigma))
        for kp in keypoints
    ]


def build_synthetic_dataset(
    n_per_class: int = 40, jitter_sigma: float = 0.015, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    클래스별 합성 keypoint를 만들고 jitter를 적용한 뒤 posture_features로 특징을
    추출해 (X, y, feature_names)를 만든다. 전부 합성 데이터임에 유의 (파일 상단 참고).
    """
    rng = random.Random(seed)
    feature_names: list[str] | None = None
    rows: list[list[float]] = []
    labels: list[str] = []

    for label in CLASS_LABELS:
        base = make_synthetic_pose(label)
        for _ in range(n_per_class):
            sample = jitter_keypoints(base, sigma=jitter_sigma, rng=rng)
            features = compute_posture_features(sample)
            if features is None:
                continue  # 어깨너비가 붕괴되는 등 정규화 실패 샘플은 건너뜀
            if feature_names is None:
                feature_names = list(features.keys())
            rows.append([features[name] for name in feature_names])
            labels.append(label)

    assert feature_names is not None
    return np.array(rows), np.array(labels), feature_names


def train_baseline_rf(
    X: np.ndarray, y: np.ndarray, feature_names: list[str], seed: int = 42
) -> dict:
    """train/test split 후 RandomForest를 학습하고 평가 결과를 딕셔너리로 반환."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return {
        "model": clf,
        "accuracy": float(clf.score(X_test, y_test)),
        "report": classification_report(y_test, y_pred, zero_division=0),
        "feature_importances": dict(zip(feature_names, clf.feature_importances_)),
    }


def main() -> None:
    print(
        "*** 이 실행은 전부 합성(가짜) 데이터 기준입니다 — 파이프라인 동작 확인용이며,\n"
        "*** 아래 정확도 수치는 실제 자세 분류 성능과 무관합니다. 실제 데이터는\n"
        "*** 5주차 수집 후 6주차에 이 스크립트의 데이터 로딩 부분만 교체해서 재사용합니다.\n"
    )

    X, y, feature_names = build_synthetic_dataset()
    print(f"합성 데이터셋: {X.shape[0]}개 샘플, {X.shape[1]}개 특징({feature_names}), "
          f"클래스 {CLASS_LABELS}")

    result = train_baseline_rf(X, y, feature_names)
    print(f"\n[합성 데이터 기준] 테스트셋 정확도: {result['accuracy']:.3f} (의미 없음, 위 안내 참고)")
    print("\n분류 리포트:")
    print(result["report"])
    print("변수 중요도:")
    for name, importance in sorted(result["feature_importances"].items(), key=lambda t: -t[1]):
        print(f"  {name:20s} {importance:.4f}")


if __name__ == "__main__":
    main()
