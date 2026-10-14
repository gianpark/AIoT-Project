import numpy as np

from src.pose.depth_refine import reference_depth, refine_keypoints_with_depth
from src.pose.movenet_keypoints import KEYPOINT_NAMES, Keypoint

H, W = 100, 100


def scene():
    d = np.full((H, W), 1.5, np.float32)  # 배경 1.5m
    d[20:90, 30:70] = 0.6  # 사용자 영역 0.6m (x 30~69, y 20~89)
    return d


def kps(pos, score=0.9):
    out = []
    for i, n in enumerate(KEYPOINT_NAMES):
        x, y = pos.get(i, (0.0, 0.0))
        out.append(Keypoint(n, y, x, score if i in pos else 0.0))
    return out


def px(x, y):
    return (x / (W - 1), y / (H - 1))


BASE = {5: px(60, 50), 6: px(40, 50), 0: px(50, 30)}


def test_reference_depth_uses_shoulders():
    assert abs(reference_depth(kps(BASE), scene()) - 0.6) < 1e-6


def test_background_wrist_snaps_to_body():
    pos = dict(BASE); pos[9] = px(75, 60)  # 사용자 영역(x<70) 밖 5px, 배경 depth
    out, moved = refine_keypoints_with_depth(kps(pos), scene())
    assert moved == [9]
    x = out[9].x * (W - 1)
    assert 30 <= x <= 69 and abs(x - 69) <= 1  # 가장 가까운 몸 픽셀(경계)로
    assert out[9].score == 0.9


def test_point_on_body_unchanged():
    pos = dict(BASE); pos[9] = px(55, 60)
    out, moved = refine_keypoints_with_depth(kps(pos), scene())
    assert moved == [] and out[9].x == kps(pos)[9].x


def test_invalid_depth_not_moved():
    d = scene(); d[60, 75] = 0.0  # 무효 depth(근접 사각지대일 수 있음)
    pos = dict(BASE); pos[9] = px(75, 60)
    _, moved = refine_keypoints_with_depth(kps(pos), d)
    assert moved == []


def test_no_body_pixel_in_radius_leaves_point():
    pos = dict(BASE); pos[9] = px(95, 60)  # 반경 25px 안에 몸(x<70) 없음? 95-69=26
    _, moved = refine_keypoints_with_depth(kps(pos), scene(), radius_px=10)
    assert moved == []


def test_no_reference_returns_input():
    out, moved = refine_keypoints_with_depth(kps({9: px(75, 60)}), scene())
    assert moved == [] and out[9].x == kps({9: px(75, 60)})[9].x
