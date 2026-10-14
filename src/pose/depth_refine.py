"""MoveNet 출력단 수정(방법 A): depth로 keypoint가 사용자 몸 위에 있는지 검증하고, 배경에 떨어진 점을 몸 쪽으로 옮긴다.

배경: MoveNet은 2D RGB만 본다. 팔·손이 몸과 겹치거나 가림이 생기면 코·손목·팔꿈치가 사용자 윤곽 밖(배경)에 놓이는
경우가 있다. depth가 있으면 "이 점이 사용자와 비슷한 거리에 있는가"를 확인할 수 있다.

규칙(보수적): 기준 거리 d_ref = 어깨(없으면 코)의 유효 depth 중앙값. 대상 keypoint의 depth가 **유효한데** d_ref와
tol_m 넘게 다르면(배경에 걸림) 반경 radius_px 안에서 depth가 d_ref ± tol_m 인 가장 가까운 픽셀로 옮긴다. depth가 무효(0)인 점은
옮기지 않는다 — D455는 약 40cm 안쪽에서 depth가 무효가 되어 근접 사용자를 지울 수 있기 때문. 옮길 곳이 없으면 그대로 둔다.
순수 계산이라 카메라/모델에 의존하지 않는다(테스트는 합성 depth).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.pose.movenet_keypoints import Keypoint

REFINE_IDX = [0, 7, 8, 9, 10]  # nose, 팔꿈치 2, 손목 2 (어깨는 기준점이라 옮기지 않는다)
REF_IDX = [5, 6]


def _depth_at(depth_m: np.ndarray, kp: Keypoint) -> float:
    h, w = depth_m.shape
    x = int(np.clip(kp.x, 0.0, 1.0) * (w - 1))
    y = int(np.clip(kp.y, 0.0, 1.0) * (h - 1))
    return float(depth_m[y, x])


def reference_depth(keypoints: list[Keypoint], depth_m: np.ndarray, min_score: float = 0.3,
                    min_valid_m: float = 0.1, max_valid_m: float = 4.0) -> float | None:
    vals = []
    for i in REF_IDX + [0]:
        kp = keypoints[i]
        if kp.score < min_score:
            continue
        d = _depth_at(depth_m, kp)
        if min_valid_m <= d <= max_valid_m:
            vals.append(d)
        if i in REF_IDX and len(vals) >= 2:
            break
    return float(np.median(vals)) if vals else None


def refine_keypoints_with_depth(keypoints: list[Keypoint], depth_m: np.ndarray, tol_m: float = 0.30,
                                radius_px: int = 25, min_score: float = 0.0,
                                min_valid_m: float = 0.1) -> tuple[list[Keypoint], list[int]]:
    """반환: (보정된 keypoint 리스트, 옮겨진 keypoint 인덱스). score는 그대로 둔다."""
    d_ref = reference_depth(keypoints, depth_m)
    if d_ref is None:
        return keypoints, []
    h, w = depth_m.shape
    out = list(keypoints)
    moved = []
    for i in REFINE_IDX:
        kp = keypoints[i]
        if kp.score < min_score:
            continue
        d = _depth_at(depth_m, kp)
        if d < min_valid_m or abs(d - d_ref) <= tol_m:
            continue  # 무효 depth는 건드리지 않고, 몸 거리 안이면 그대로
        cx, cy = int(np.clip(kp.x, 0, 1) * (w - 1)), int(np.clip(kp.y, 0, 1) * (h - 1))
        x0, x1 = max(cx - radius_px, 0), min(cx + radius_px + 1, w)
        y0, y1 = max(cy - radius_px, 0), min(cy + radius_px + 1, h)
        win = depth_m[y0:y1, x0:x1]
        ok = (win >= min_valid_m) & (np.abs(win - d_ref) <= tol_m)
        if not ok.any():
            continue
        ys, xs = np.nonzero(ok)
        dist = np.hypot(xs + x0 - cx, ys + y0 - cy)
        j = int(np.argmin(dist))
        out[i] = replace(kp, x=float((xs[j] + x0) / (w - 1)), y=float((ys[j] + y0) / (h - 1)))
        moved.append(i)
    return out, moved
