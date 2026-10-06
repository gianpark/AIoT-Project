import math

import numpy as np

from src.features.absolute_features import (
    Intrinsics, compute_absolute_features, deproject, gravity_from_accel, gravity_from_pitch,
)
from src.pose.movenet_keypoints import Keypoint

K = Intrinsics.from_fov(848, 480)


def project(p, k=K):
    """카메라 좌표 3D 점 → (x_norm, y_norm, depth)"""
    return (p[0] * k.fx / p[2] + k.cx) / k.width, (p[1] * k.fy / p[2] + k.cy) / k.height, p[2]


def build(points3d, k=K):
    """{'nose':P, 'ls':P, ...} 3D → (keypoints, depth_lookup)"""
    idx = {"nose": 0, "ls": 5, "rs": 6, "lh": 11, "rh": 12}
    kps = [Keypoint(name=str(i), y=0.5, x=0.5, score=0.0) for i in range(17)]
    depth_at = {}
    for key, i in idx.items():
        x, y, z = project(points3d[key], k)
        kps[i] = Keypoint(name=str(i), y=y, x=x, score=0.9)
        depth_at[(round(x, 6), round(y, 6))] = z

    def lookup(x, y):
        return depth_at.get((round(x, 6), round(y, 6)))
    return kps, lookup


def person(z_chest=0.6, pitch_deg=0.0, roll_deg=0.0, width=0.4, torso=0.5, head_fwd=0.0,
           cam_pitch_deg=0.0):
    """월드(중력 기준)에서 자세를 만들고 카메라 프레임 3D로 변환. 앞(카메라 쪽)이 -Z_world."""
    # 카메라 수평 기준으로 자세를 만든 뒤, 카메라 기울기만큼 회전해 카메라 프레임으로 옮긴다
    up = np.array([0.0, -1.0, 0.0])
    fwd = np.array([0.0, 0.0, -1.0])
    rgt = np.array([1.0, 0.0, 0.0])
    p, r = math.radians(pitch_deg), math.radians(roll_deg)
    torso_vec = torso * (math.cos(p) * math.cos(r) * up + math.sin(p) * fwd + math.sin(r) * rgt)
    hip = np.zeros(3)
    chest = hip + torso_vec
    pts = {
        "ls": chest - rgt * width / 2, "rs": chest + rgt * width / 2,
        "lh": hip - rgt * 0.15, "rh": hip + rgt * 0.15,
        "nose": chest + 0.2 * up + head_fwd * width * fwd,
    }
    # 카메라가 cam_pitch만큼 아래를 볼 때: 월드 점을 카메라 프레임으로 회전(X축 회전)하고 거리만큼 이동
    c = math.radians(cam_pitch_deg)
    rot = np.array([[1, 0, 0], [0, math.cos(c), math.sin(c)], [0, -math.sin(c), math.cos(c)]])
    shift = np.array([0.0, -0.1, z_chest])  # 엉덩이가 카메라 앞 z_chest 거리
    return {k_: rot @ v + shift for k_, v in pts.items()}


def feats(**kw):
    cam_pitch = kw.get("cam_pitch_deg", 0.0)
    kps, look = build(person(**kw))
    return compute_absolute_features(kps, look, K, gravity=gravity_from_pitch(cam_pitch))


def test_upright_is_zero_pitch_roll():
    f = feats()
    assert abs(f["torso_pitch_deg"]) < 0.5 and abs(f["torso_roll_deg"]) < 0.5
    assert abs(f["shoulder_width_m"] - 0.4) < 0.01


def test_forward_lean_positive_back_negative():
    assert feats(pitch_deg=25)["torso_pitch_deg"] > 20
    assert feats(pitch_deg=-25)["torso_pitch_deg"] < -20


def test_roll_sign_image_right_positive():
    assert feats(roll_deg=15)["torso_roll_deg"] > 10
    assert feats(roll_deg=-15)["torso_roll_deg"] < -10


def test_invariant_to_sitting_distance_and_body_size():
    a = feats(pitch_deg=20)
    b = feats(pitch_deg=20, z_chest=0.9, width=0.5, torso=0.62)  # 멀리 앉은 더 큰 체격
    assert abs(a["torso_pitch_deg"] - b["torso_pitch_deg"]) < 1.0


def test_camera_tilt_compensated_with_gravity():
    """카메라가 20° 내려다봐도 중력을 알면 같은 각도로 읽힌다(IMU의 효과)."""
    level = feats(pitch_deg=15)["torso_pitch_deg"]
    tilted = feats(pitch_deg=15, cam_pitch_deg=20)["torso_pitch_deg"]
    assert abs(level - tilted) < 1.0


def test_head_forward_ratio():
    assert feats(head_fwd=0.6)["head_forward_ratio"] > feats(head_fwd=0.0)["head_forward_ratio"] + 0.4


def test_missing_depth_gives_none():
    kps, look = build(person())
    assert compute_absolute_features(kps, lambda x, y: None, K) is None
    for i in (11, 12):  # 엉덩이만 가려진 경우: 어깨 기반 값은 남고 몸통 각도만 None
        kps[i] = Keypoint(name=str(i), y=kps[i].y, x=kps[i].x, score=0.0)
    f = compute_absolute_features(kps, look, K)
    assert f["torso_pitch_deg"] is None and f["shoulder_width_m"] > 0


def test_gravity_helpers():
    g = gravity_from_accel((0.0, -9.8, 0.0))
    assert np.allclose(g, [0, 1, 0])
    assert np.allclose(gravity_from_pitch(0), [0, 1, 0])
    p = deproject(0.5, 0.5, 1.0, K)
    assert np.allclose(p, [0, 0, 1.0])
