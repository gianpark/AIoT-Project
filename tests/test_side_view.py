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


def test_smoother_ema_hold_and_never_valid():
    from src.pose.side_view import SideViewSmoother
    sm = SideViewSmoother(alpha=0.5, hold_frames=2)
    out = sm.update({"head": (0.40, 0.2, True), "chest": (0.60, 0.0, True), "hip": (0.65, -0.3, False)})
    assert out["hip"][2] is False  # 한 번도 depth가 없던 점은 유효로 바뀌지 않는다
    out = sm.update({"head": (0.50, 0.2, True), "chest": (0.60, 0.0, True), "hip": (0.65, -0.3, False)})
    assert abs(out["head"][0] - 0.45) < 1e-9  # EMA
    for _ in range(2):
        out = sm.update({"head": (0.0, 0.2, False), "chest": (0.60, 0.0, True), "hip": (0.65, -0.3, False)})
    assert out["head"][2] is True and abs(out["head"][0] - 0.45) < 1e-9  # hold
    out = sm.update({"head": (0.0, 0.2, False), "chest": (0.60, 0.0, True), "hip": (0.65, -0.3, False)})
    assert out["head"][2] is False  # hold 초과


def test_arm_points_and_elbow_angle():
    from src.pose.side_view import arm_angles, render_side_view, side_arm_points
    k = kps()
    k[7] = Keypoint(name="k7", x=0.40, y=0.55, score=0.9)  # left elbow
    k[9] = Keypoint(name="k9", x=0.42, y=0.55, score=0.9)  # left wrist
    z = {(0.42, 0.40): 0.60, (0.40, 0.55): 0.60, (0.42, 0.55): 0.45}
    arm = side_arm_points(k, lambda x, y: z.get((round(x, 2), round(y, 2))), 640, 480)
    assert arm["l_el"][2] and arm["l_wr"][2] and not arm["r_wr"][2]
    ang = arm_angles(arm)
    assert 80 < ang["elbow_left_deg"] < 100 and ang["elbow_right_deg"] is None  # 위팔 수직, 아래팔 앞으로 = 약 90도
    pts = side_view_points(kps(), {"head_depth_m": 0.45, "chest_depth_m": 0.60, "hip_depth_m": 0.65}, 640, 480)
    assert render_side_view(pts, arm_pts=arm).shape == (420, 360, 3)
    assert side_arm_points(k, None, 640, 480) is None


def test_head_circle_drawn_around_nose():
    pts = side_view_points(kps(), {"head_depth_m": 0.45, "chest_depth_m": 0.60, "hip_depth_m": 0.65}, 640, 480)
    assert render_side_view(pts, arm_pts=None).shape == (420, 360, 3)


def test_extra_features_values_and_missing():
    from src.pose.side_view import EXTRA_KEYS, extra_features
    k = kps()
    k[10] = Keypoint(name="k10", x=0.52, y=0.30, score=0.9)  # 손목이 코 근처
    pts = side_view_points(k, {"head_depth_m": 0.45, "chest_depth_m": 0.60, "hip_depth_m": 0.65}, 640, 480)
    e = extra_features(k, pts, 640, 480)
    assert e["torso_len_sw"] > 0 and e["head_up_m"] > 0 and e["wrist_face_sw"] < 1.0
    k2 = kps(hip_score=0.1)
    k2[9] = Keypoint(name="k9", x=0.5, y=0.5, score=0.1)
    k2[10] = Keypoint(name="k10", x=0.5, y=0.5, score=0.1)
    e2 = extra_features(k2, None, 640, 480)
    assert all(e2[key] is None for key in EXTRA_KEYS)
