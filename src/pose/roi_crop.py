"""MoveNet 입력단 수정(방법 D): 이전 프레임 keypoint 주변을 정사각형으로 잘라 모델에 넣는다.

MoveNet은 입력을 192×192(Lightning)로 줄여 쓰므로, 사용자가 프레임의 일부만 차지하면 키포인트 해상도가 낮아진다.
공식 MoveNet 추적 방식처럼 직전 프레임의 몸 영역 + 여백으로 크롭해 같은 사람을 더 크게 보여 주고,
일정 주기·검출 실패 시에는 전체 프레임으로 되돌아가 사용자를 다시 찾는다. 순수 계산(좌표 변환)과 추적기만 있고
카메라/모델에는 의존하지 않는다(모델 호출은 infer 콜백).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Optional

import numpy as np

from src.pose.movenet_keypoints import Keypoint

CropRegion = tuple[float, float, float]  # (x0, y0, side) 픽셀, side는 정사각형 한 변. 프레임 밖으로 나가도 된다(0 패딩)


def crop_region_from_keypoints(keypoints: list[Keypoint], frame_w: int, frame_h: int,
                               min_score: float = 0.3, margin: float = 0.3,
                               min_points: int = 4) -> Optional[CropRegion]:
    """점수 ≥ min_score인 keypoint의 bbox를 margin 비율만큼 키워 정사각형 크롭 영역으로 만든다.
    유효한 점이 min_points개 미만이면 None(전체 프레임 사용)."""
    pts = np.array([[kp.x * frame_w, kp.y * frame_h] for kp in keypoints if kp.score >= min_score])
    if len(pts) < min_points:
        return None
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    side = max(x1 - x0, y1 - y0) * (1.0 + 2 * margin)
    side = max(side, 0.25 * min(frame_w, frame_h))  # 너무 작게 자르지 않는다
    side = min(side, float(max(frame_w, frame_h)))
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    return (cx - side / 2.0, cy - side / 2.0, side)


def crop_frame(frame: np.ndarray, region: CropRegion) -> np.ndarray:
    """영역을 잘라 side×side 이미지로 돌려준다. 프레임 밖은 0으로 채운다."""
    x0, y0, side = region
    s = int(round(side))
    ix, iy = int(round(x0)), int(round(y0))
    out = np.zeros((s, s, frame.shape[2]), dtype=frame.dtype)
    h, w = frame.shape[:2]
    sx0, sy0 = max(ix, 0), max(iy, 0)
    sx1, sy1 = min(ix + s, w), min(iy + s, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - iy:sy1 - iy, sx0 - ix:sx1 - ix] = frame[sy0:sy1, sx0:sx1]
    return out


def to_full_frame(keypoints: list[Keypoint], region: CropRegion, frame_w: int, frame_h: int) -> list[Keypoint]:
    """크롭 이미지 기준 정규화 좌표를 원본 프레임 기준 정규화 좌표로 되돌린다(score는 그대로)."""
    x0, y0, side = region
    return [replace(kp, x=(x0 + kp.x * side) / frame_w, y=(y0 + kp.y * side) / frame_h) for kp in keypoints]


class RoiCropTracker:
    """직전 결과로 크롭하되, 주기적(full_every)으로 전체 프레임을 다시 돌려 사용자가 바뀌거나 놓쳤는지 확인한다.
    크롭 결과의 유효 keypoint가 min_points 미만이면 다음 프레임을 전체 프레임으로 처리한다."""

    def __init__(self, full_every: int = 30, margin: float = 0.3, min_score: float = 0.3, min_points: int = 4):
        self.full_every = full_every
        self.margin = margin
        self.min_score = min_score
        self.min_points = min_points
        self._prev: Optional[list[Keypoint]] = None
        self._n = 0
        self.last_used_crop = False

    def infer(self, infer_fn: Callable[[np.ndarray], list[Keypoint]], frame: np.ndarray) -> list[Keypoint]:
        h, w = frame.shape[:2]
        region = None
        if self._prev is not None and self._n % self.full_every != 0:
            region = crop_region_from_keypoints(self._prev, w, h, self.min_score, self.margin, self.min_points)
        self._n += 1
        if region is None:
            kps = infer_fn(frame)
            self.last_used_crop = False
        else:
            kps = to_full_frame(infer_fn(crop_frame(frame, region)), region, w, h)
            self.last_used_crop = True
        valid = sum(kp.score >= self.min_score for kp in kps)
        self._prev = kps if valid >= self.min_points else None
        return kps
