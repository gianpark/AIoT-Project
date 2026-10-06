"""
src/features/posture_features.py 단위 테스트.

실제 카메라/모델 없이도 검증 가능하도록, MoveNet이 뽑아준 것처럼 생긴 가짜(합성)
keypoint 좌표로 테스트한다 — 이 프로젝트 개발 환경(샌드박스)은 카메라가 없어서
실제 웹캠 테스트는 각자 노트북에서 해야 하지만, 정규화·각도 계산 로직 자체가
맞는지는 여기서 미리 검증해둘 수 있다.

좌표는 좌우 대칭인 "똑바로 앉은 자세"를 가정해서 잡았다 — 그래서 목 기울임 0,
어깨 비대칭 없음(180도 일직선) 등 손으로 계산해도 맞는지 확인하기 쉬운 값이 나온다.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from src.features.posture_features import (
    angle_deg,
    compute_depth_features,
    compute_posture_features,
    filter_by_confidence,
    normalize_keypoints,
)


@dataclass
class FakeKeypoint:
    """src.pose.movenet_keypoints.Keypoint과 동일한 인터페이스(name, x, y, score)."""
    name: str
    y: float
    x: float
    score: float


def make_symmetric_seated_pose() -> list[FakeKeypoint]:
    """좌우 대칭으로 똑바로 앉은 가짜 자세. 무릎/발목은 책상 구도라 저신뢰도로 둔다."""
    names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]
    # (x, y, score) — 화면 기준 0~1 정규화 좌표
    coords = [
        (0.50, 0.20, 0.90),  # nose
        (0.47, 0.18, 0.80), (0.53, 0.18, 0.80),  # eyes
        (0.45, 0.19, 0.70), (0.55, 0.19, 0.70),  # ears
        (0.40, 0.30, 0.90), (0.60, 0.30, 0.90),  # shoulders
        (0.35, 0.45, 0.70), (0.65, 0.45, 0.70),  # elbows
        (0.33, 0.55, 0.60), (0.67, 0.55, 0.60),  # wrists
        (0.45, 0.70, 0.90), (0.55, 0.70, 0.90),  # hips
        (0.44, 0.85, 0.05), (0.56, 0.85, 0.05),  # knees — 책상 구도라 저신뢰
        (0.43, 0.98, 0.05), (0.57, 0.98, 0.05),  # ankles — 책상 구도라 저신뢰
    ]
    return [
        FakeKeypoint(name=n, x=x, y=y, score=s)
        for n, (x, y, s) in zip(names, coords)
    ]


def test_normalize_keypoints_centers_hips_and_scales_shoulder_width():
    kps = make_symmetric_seated_pose()
    normalized = normalize_keypoints(kps)
    assert normalized is not None

    left_hip, right_hip = normalized[11], normalized[12]
    left_shoulder, right_shoulder = normalized[5], normalized[6]

    # 엉덩이 중심이 원점 근처에서 좌우 대칭이어야 한다 (x는 ±값, y는 0)
    hip_center = (left_hip + right_hip) / 2.0
    assert hip_center == pytest.approx([0.0, 0.0], abs=1e-9)

    # 어깨너비로 스케일 정규화했으니 어깨 사이 거리는 1.0이어야 한다
    shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
    assert shoulder_dist == pytest.approx(1.0, abs=1e-9)


def test_normalize_keypoints_returns_none_when_shoulders_collapse():
    kps = make_symmetric_seated_pose()
    # 양쪽 어깨를 같은 좌표로 만들어 어깨너비를 0으로 붕괴시킨다
    kps[5].x = kps[6].x = 0.5
    kps[5].y = kps[6].y = 0.3
    assert normalize_keypoints(kps) is None


def test_angle_deg_right_angle_and_straight_line():
    # 직각: (1,0) - (0,0) - (0,1) 이면 90도
    assert angle_deg(np.array([1.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(90.0)
    # 일직선(양쪽으로 뻗음): 180도에 가까워야 한다.
    # (angle_deg 내부에서 0으로 나누는 걸 막으려고 분모에 1e-9를 더하는데, 코사인이
    #  정확히 -1/1이 되는 극단값(0도·180도) 근처에서는 arccos의 기울기가 발산해서
    #  이 1e-9가 매우 작은 오차를 증폭시킨다 — 실제 자세 판정에서 쓸 각도 값은
    #  이런 극단값이 거의 없으니 문제 없지만, 테스트 허용오차는 넉넉히 잡는다.)
    assert angle_deg(np.array([-1.0, 0.0]), np.array([0.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(180.0, abs=1e-2)


def test_compute_posture_features_symmetric_upright_pose():
    kps = make_symmetric_seated_pose()
    features = compute_posture_features(kps)
    assert features is not None

    # 좌우 대칭 + 목이 엉덩이 중심 바로 위에 있는 구도이므로:
    # (0도/180도 근처의 부동소수점 오차에 대해서는 위 test_angle_deg_... 테스트 참고)
    assert features["neck_tilt_deg"] == pytest.approx(90.0)       # 코가 목 정중앙 위 -> 좌우 기울임 없음
    assert features["torso_lean_deg"] == pytest.approx(0.0, abs=1e-2)       # 상체가 수직으로 똑바름
    assert features["shoulder_slope_deg"] == pytest.approx(180.0, abs=1e-2)  # 양쪽 어깨가 일직선(비대칭 없음)
    assert features["nose_to_hip_dist"] > 0


def test_compute_depth_features_reads_head_chest_hip_and_offsets():
    kps = make_symmetric_seated_pose()

    # 가짜 depth_lookup: y_norm이 클수록(화면 아래쪽=엉덩이 쪽) 카메라에서 더 먼 것처럼
    # 선형으로 깊이를 준다 — 정상 자세라면 머리>가슴>엉덩이 순으로 깊이가 커져야 한다.
    def fake_depth_lookup(x_norm: float, y_norm: float):
        return 0.5 + y_norm  # 코(y=0.20)->0.70, 어깨 중점(y=0.30)->0.80, 엉덩이 중점(y=0.70)->1.20

    features = compute_depth_features(kps, fake_depth_lookup)
    assert features is not None
    assert features["head_depth_m"] == pytest.approx(0.70)
    assert features["chest_depth_m"] == pytest.approx(0.80)
    assert features["hip_depth_m"] == pytest.approx(1.20)
    # 가슴이 머리보다 멀리 있으니(정상 자세) neck_forward_offset_m은 양수여야 한다
    assert features["neck_forward_offset_m"] == pytest.approx(0.10)
    assert features["torso_recline_offset_m"] == pytest.approx(0.40)


def test_compute_depth_features_partial_when_lookup_returns_none_for_some_points():
    kps = make_symmetric_seated_pose()

    def partial_lookup(x_norm: float, y_norm: float):
        # 엉덩이 쪽(y=0.70)만 무효(None) 처리 — 스테레오 매칭 실패 상황을 흉내
        if y_norm > 0.6:
            return None
        return 1.0

    features = compute_depth_features(kps, partial_lookup)
    assert features is not None
    assert features["head_depth_m"] == pytest.approx(1.0)
    assert features["chest_depth_m"] == pytest.approx(1.0)
    assert features["hip_depth_m"] is None
    assert features["torso_recline_offset_m"] is None  # hip_depth_m이 없어 계산 불가


def test_compute_depth_features_returns_none_when_all_low_confidence():
    kps = make_symmetric_seated_pose()
    for idx in (0, 5, 6, 11, 12):
        kps[idx].score = 0.0

    assert compute_depth_features(kps, lambda x, y: 1.0) is None


def test_filter_by_confidence_drops_low_confidence_keypoints():
    kps = make_symmetric_seated_pose()
    filtered = filter_by_confidence(kps, threshold=0.3)

    # 무릎(13,14)/발목(15,16)은 책상 구도 가정상 저신뢰(0.05)라 걸러져야 한다
    for idx in (13, 14, 15, 16):
        assert filtered[idx] is None

    # 나머지(코, 어깨, 엉덩이 등)는 threshold 이상이라 유지되어야 한다
    for idx in (0, 5, 6, 11, 12):
        assert filtered[idx] is not None


def test_depth_features_chest_near_invalid_gives_forward_lower_bound():
    kps = [FakeKeypoint(f"k{i}", 0.5, 0.5, 0.9) for i in range(17)]

    def lookup(x, y):
        return None if y < 0.5 else 0.8  # placeholder, overridden below

    ys = {5: 0.3, 6: 0.3, 11: 0.8, 12: 0.8}
    for i, y in ys.items():
        kps[i] = FakeKeypoint(f"k{i}", y, 0.5, 0.9)
    f = compute_depth_features(kps, lambda x, y: None if y < 0.5 else 0.8)
    assert f["chest_depth_m"] is None
    assert f["torso_recline_offset_m"] == pytest.approx(0.8 - 0.30)


def test_hip_depth_ignored_when_hip_occluded_by_desk():
    kps = [FakeKeypoint(f"k{i}", 0.5, 0.5, 0.9) for i in range(17)]
    kps[11] = FakeKeypoint("lh", 0.8, 0.45, 0.4)
    kps[12] = FakeKeypoint("rh", 0.8, 0.55, 0.4)
    f = compute_depth_features(kps, lambda x, y: 0.5 if y > 0.7 else 0.7)
    assert f["hip_depth_m"] is None and f["torso_recline_offset_m"] is None
