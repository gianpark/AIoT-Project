"""
src/logic/rf_baseline.py 단위 테스트.

이 모듈 자체가 "합성 데이터로 파이프라인만 검증"하는 스켈레톤이므로, 여기서 하는
테스트도 "숫자가 딱 맞는지"가 아니라 "증강·데이터셋 생성·학습 파이프라인이 에러 없이
동작하고, 최소한의 논리적 성질(예: 반전하면 좌우가 실제로 바뀐다)을 만족하는지"를
확인하는 정도로 제한한다.
"""

import pytest

from src.logic.rf_baseline import (
    CLASS_LABELS,
    build_synthetic_dataset,
    flip_keypoints_lr,
    jitter_keypoints,
    make_synthetic_pose,
    train_baseline_rf,
)


def test_flip_keypoints_lr_mirrors_x_and_swaps_left_right_names():
    # 좌우 비대칭인 자세를 써야 "이름이 실제로 맞바뀌었는지"를 의미 있게 확인할 수 있다
    # (기준 자세는 좌우 대칭이라 이름을 안 바꿔도 좌표만으로는 구분이 안 됨)
    pose = make_synthetic_pose("tilt_left")
    flipped = flip_keypoints_lr(pose)

    by_name = {kp.name: kp for kp in pose}
    flipped_by_name = {kp.name: kp for kp in flipped}

    # nose처럼 좌우 쌍이 없는 keypoint는 이름 유지 + x만 반전
    assert flipped_by_name["nose"].x == pytest.approx(1.0 - by_name["nose"].x)
    assert flipped_by_name["nose"].y == by_name["nose"].y

    # 원래 left_shoulder였던 좌표가, 반전 후에는 right_shoulder 이름 아래 (x가 뒤집힌 채로) 들어가야 한다
    assert flipped_by_name["right_shoulder"].x == pytest.approx(1.0 - by_name["left_shoulder"].x)
    assert flipped_by_name["right_shoulder"].y == by_name["left_shoulder"].y
    assert flipped_by_name["left_shoulder"].x == pytest.approx(1.0 - by_name["right_shoulder"].x)
    assert flipped_by_name["left_shoulder"].y == by_name["right_shoulder"].y


def test_flip_is_involution():
    """두 번 반전하면 원래 좌표로 돌아와야 한다."""
    pose = make_synthetic_pose("tilt_left")
    twice = flip_keypoints_lr(flip_keypoints_lr(pose))

    by_name = {kp.name: kp for kp in pose}
    twice_by_name = {kp.name: kp for kp in twice}
    for name, kp in by_name.items():
        assert twice_by_name[name].x == pytest.approx(kp.x)
        assert twice_by_name[name].y == pytest.approx(kp.y)


def test_jitter_keypoints_changes_coordinates_but_keeps_name_and_score():
    pose = make_synthetic_pose("normal")
    jittered = jitter_keypoints(pose, sigma=0.02)

    assert len(jittered) == len(pose)
    for original, noisy in zip(pose, jittered):
        assert noisy.name == original.name
        assert noisy.score == original.score
        # sigma=0.02인데 좌표가 정확히 그대로일 확률은 사실상 0이므로 달라졌는지만 확인
        assert (noisy.x, noisy.y) != (original.x, original.y)


def test_build_synthetic_dataset_has_all_classes_and_matching_lengths():
    X, y, feature_names = build_synthetic_dataset(n_per_class=10, seed=1)

    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == len(feature_names)
    assert set(y) == set(CLASS_LABELS)
    # 클래스별 최소 개수가 어느 정도 균형 잡혀 있어야 한다 (정규화 실패로 일부 빠질 수 있음)
    for label in CLASS_LABELS:
        assert (y == label).sum() > 0


def test_train_baseline_rf_runs_end_to_end():
    X, y, feature_names = build_synthetic_dataset(n_per_class=20, seed=2)
    result = train_baseline_rf(X, y, feature_names, seed=2)

    assert 0.0 <= result["accuracy"] <= 1.0
    assert set(result["feature_importances"].keys()) == set(feature_names)
    assert isinstance(result["report"], str)
