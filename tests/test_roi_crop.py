import numpy as np

from src.pose.movenet_keypoints import KEYPOINT_NAMES, Keypoint
from src.pose.roi_crop import (RoiCropTracker, crop_frame, crop_region_from_keypoints, to_full_frame)


def kps(points, score=0.9):
    """{index: (x,y)} -> 17개 리스트, 나머지는 score 0."""
    out = []
    for i, n in enumerate(KEYPOINT_NAMES):
        x, y = points.get(i, (0.0, 0.0))
        out.append(Keypoint(n, y, x, score if i in points else 0.0))
    return out


BODY = {0: (0.5, 0.3), 5: (0.6, 0.5), 6: (0.4, 0.5), 11: (0.58, 0.8), 12: (0.42, 0.8)}


def test_region_none_when_few_points():
    assert crop_region_from_keypoints(kps({0: (0.5, 0.5), 5: (0.6, 0.5)}), 640, 480) is None


def test_region_contains_all_valid_points_and_is_square():
    x0, y0, side = crop_region_from_keypoints(kps(BODY), 640, 480, margin=0.3)
    for i, (x, y) in BODY.items():
        assert x0 <= x * 640 <= x0 + side and y0 <= y * 480 <= y0 + side
    assert side <= 640


def test_roundtrip_full_frame_coordinates():
    frame = np.zeros((480, 640, 3), np.uint8)
    region = (100.0, 50.0, 300.0)
    crop = crop_frame(frame, region)
    assert crop.shape == (300, 300, 3)
    # 크롭 중앙의 점 (0.5, 0.5) -> 원본 (250, 200) -> 정규화
    out = to_full_frame([Keypoint("nose", 0.5, 0.5, 0.9)], region, 640, 480)[0]
    assert abs(out.x - 250 / 640) < 1e-9 and abs(out.y - 200 / 480) < 1e-9 and out.score == 0.9


def test_crop_pads_outside_frame_with_zeros():
    frame = np.full((100, 100, 3), 255, np.uint8)
    crop = crop_frame(frame, (-20.0, -20.0, 60.0))
    assert crop[0, 0].sum() == 0 and crop[30, 30].sum() == 255 * 3


def test_tracker_crops_then_refreshes_full_frame_periodically():
    frame = np.zeros((480, 640, 3), np.uint8)
    shapes = []

    def fake(img):
        shapes.append(img.shape[:2])
        return kps(BODY)  # 크롭이 어디든 같은 정규화 좌표를 돌려주는 가짜 모델

    t = RoiCropTracker(full_every=3)
    used = []
    for _ in range(7):
        t.infer(fake, frame)
        used.append(t.last_used_crop)
    assert used == [False, True, True, False, True, True, False]  # 3프레임마다 전체 프레임 재검색
    assert shapes[0] == (480, 640) and shapes[1][0] == shapes[1][1] < 480


def test_tracker_falls_back_to_full_frame_when_lost():
    frame = np.zeros((480, 640, 3), np.uint8)
    calls = iter([kps(BODY), kps({}), kps(BODY), kps(BODY)])
    t = RoiCropTracker(full_every=100)
    used = []
    for _ in range(4):
        t.infer(lambda img: next(calls), frame)
        used.append(t.last_used_crop)
    assert used == [False, True, False, True]  # 둘째 결과가 비면 셋째는 전체 프레임
