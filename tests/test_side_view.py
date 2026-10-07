from src.pose.movenet_keypoints import Keypoint
from src.pose.side_view import reference_points, render_side_view, side_view_points


def kps(nose_y=0.28, hip_score=0.9):
    base = {0: (0.5, nose_y), 5: (0.42, 0.40), 6: (0.58, 0.40), 11: (0.44, 0.65), 12: (0.56, 0.65)}
    return [Keypoint(name=f"k{i}", x=base.get(i, (0.5, 0.5))[0], y=base.get(i, (0.5, 0.5))[1],
                     score=hip_score if i in (11, 12) else 0.9) for i in range(17)]


def test_head_is_above_and_hip_below_chest():
    pts = side_view_points(kps(), {"head_depth_m": 0.45, "chest_depth_m": 0.60, "hip_depth_m": 0.65}, 640, 480)
    assert pts["head"][1] > 0 > pts["hip"][1]
    assert pts["head"][2] and pts["chest"][2] and pts["hip"][2]


def test_missing_depth_is_marked_not_invented():
    pts = side_view_points(kps(), {"head_depth_m": 0.45, "chest_depth_m": 0.60, "hip_depth_m": None}, 640, 480)
    assert pts["hip"][2] is False and pts["hip"][0] == 0.60


def test_none_when_no_depth_or_no_shoulders():
    assert side_view_points(kps(), None, 640, 480) is None
    assert side_view_points(kps(), {"head_depth_m": None, "chest_depth_m": None, "hip_depth_m": None}, 640, 480) is None


def test_reference_anchors_on_chest_and_renders():
    pts = side_view_points(kps(), {"head_depth_m": 0.45, "chest_depth_m": 0.60, "hip_depth_m": 0.65}, 640, 480)
    ref = reference_points(pts)
    assert abs(ref["chest"][0] - 0.60) < 1e-9 and abs(ref["head"][0] - (0.60 - 0.153)) < 1e-9
    img = render_side_view(pts)
    assert img.shape == (420, 360, 3) and img.any()
    assert render_side_view(None).shape == (420, 360, 3)


def test_side_angles_signs():
    from src.pose.side_view import side_angles
    normal = side_angles(side_view_points(kps(), {"head_depth_m": 0.434, "chest_depth_m": 0.588, "hip_depth_m": 0.642}, 640, 480))
    assert normal["head_forward_deg"] > 0 and normal["torso_pitch_deg"] > 0  # 머리 앞, 허리가 가슴 뒤 = 정상 기준
    back = side_angles(side_view_points(kps(), {"head_depth_m": 0.738, "chest_depth_m": 0.803, "hip_depth_m": 0.686}, 640, 480))
    assert back["torso_pitch_deg"] < 0  # 뒤로 기댐: 가슴이 허리보다 뒤
    forward = side_angles(side_view_points(kps(), {"head_depth_m": 0.35, "chest_depth_m": 0.60, "hip_depth_m": 0.62}, 640, 480))
    assert forward["head_forward_deg"] > normal["head_forward_deg"]  # 목 내밀기: 머리 앞기울기 증가


def test_side_angles_none_without_depth():
    from src.pose.side_view import side_angles
    a = side_angles(side_view_points(kps(), {"head_depth_m": 0.45, "chest_depth_m": 0.60, "hip_depth_m": None}, 640, 480))
    assert a["head_forward_deg"] is not None and a["torso_pitch_deg"] is None
    assert side_angles(None) == {"head_forward_deg": None, "torso_pitch_deg": None}
