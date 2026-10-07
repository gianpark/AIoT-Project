"""측면 자세 뷰: 정면 카메라의 keypoint + depth로 "옆에서 본 모습"을 그린다.

가로축 = 카메라로부터의 거리(depth, m, 왼쪽이 카메라), 세로축 = 높이(m, 위쪽이 위).
머리(코)·가슴(어깨 중점)·허리(엉덩이 중점) 세 점을 이어 측면 스켈레톤을 만들고,
회색 점선으로 "정상 자세 기준"(실측 중앙값: 머리가 가슴보다 0.153m 앞, 허리는 가슴보다 0.054m 뒤)을
같이 그려 앞숙임·기댐·목 내밀기가 기준과 얼마나 다른지 한눈에 보이게 한다.

주의: depth가 없는 점은 값을 지어내지 않고 속이 빈 회색 원으로 표시한다(위치는 이웃 점의 거리).
세로 길이는 어깨너비 0.40m(성인 평균 가정)로 화면 픽셀을 미터로 환산한 근사값이다.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

ADULT_SHOULDER_WIDTH_M = 0.40
NORMAL_HEAD_AHEAD_M = 0.153  # 정상: 머리가 가슴보다 카메라에 이만큼 가까움 (10/6 실측 중앙값, 1인)
NORMAL_HIP_BEHIND_M = 0.054  # 정상: 허리가 가슴보다 이만큼 멂 (10/6 실측 중앙값, 1인)
Z_MIN_M, Z_MAX_M = 0.2, 1.0


def side_view_points(keypoints, depth_features: Optional[dict], frame_w: int, frame_h: int,
                     min_score: float = 0.3) -> Optional[dict]:
    """머리/가슴/허리의 (z, up, valid)를 반환한다. z=카메라 거리(m), up=가슴 기준 높이(m).

    valid=False인 점은 depth가 없는 점이다(그릴 때 속 빈 원). 코·어깨 keypoint가 없으면 None.
    """
    nose, l_sh, r_sh, l_hip, r_hip = keypoints[0], keypoints[5], keypoints[6], keypoints[11], keypoints[12]
    if nose.score < min_score or l_sh.score < min_score or r_sh.score < min_score:
        return None
    sw_px = float(np.hypot((l_sh.x - r_sh.x) * frame_w, (l_sh.y - r_sh.y) * frame_h))
    if sw_px < 1.0:
        return None
    m_per_px = ADULT_SHOULDER_WIDTH_M / sw_px
    chest_y = (l_sh.y + r_sh.y) / 2.0
    hip_y = (l_hip.y + r_hip.y) / 2.0
    up = {
        "head": (chest_y - nose.y) * frame_h * m_per_px,
        "chest": 0.0,
        "hip": (chest_y - hip_y) * frame_h * m_per_px,
    }
    d = depth_features or {}
    depth = {"head": d.get("head_depth_m"), "chest": d.get("chest_depth_m"), "hip": d.get("hip_depth_m")}
    known = [v for v in (depth["chest"], depth["head"], depth["hip"]) if v is not None]
    if not known:
        return None
    pts = {}
    for name in ("head", "chest", "hip"):
        z = depth[name]
        if z is None:  # 지어내지 않고 가장 가까운 이웃 점의 거리에 속 빈 원으로 둔다
            z = depth["chest"] if depth["chest"] is not None else known[0]
        pts[name] = (float(z), float(up[name]), depth[name] is not None)
    return pts


SIDE_ANGLE_KEYS = ["head_forward_deg", "torso_pitch_deg"]


def side_angles(pts: Optional[dict]) -> dict:
    """측면 뷰 점에서 두 각도(도, 수직선 기준)를 계산한다. depth가 없는 점이 필요한 각도는 None.

    - head_forward_deg: 가슴→머리 선이 수직선에서 카메라 쪽으로 기운 각도. +면 머리가 가슴보다 앞(거북목/숙임),
      0이면 머리가 가슴 바로 위, -면 머리가 가슴보다 뒤(젖힘/기댐). 정상(10/6 1인)은 양수.
    - torso_pitch_deg: 허리→가슴 선이 수직선에서 카메라 쪽으로 기운 각도. +면 가슴이 허리보다 앞(앞숙임),
      -면 뒤로 기댐. 허리 depth가 없으면 None.

    한계: 카메라가 위에서 내려다보는 각도(pitch)만큼 수직선이 틀어지므로 절대 각도가 아니라 상대 비교용이다.
    IMU 중력 보정은 docs/absolute_posture_design.md 설계대로 아직 연결되지 않았다. 세로 길이는 어깨너비 0.40m
    가정의 근사값이다.
    """
    out = {"head_forward_deg": None, "torso_pitch_deg": None}
    if not pts:
        return out
    head, chest, hip = pts["head"], pts["chest"], pts["hip"]
    if head[2] and chest[2] and head[1] > 0:
        out["head_forward_deg"] = math.degrees(math.atan2(chest[0] - head[0], head[1]))
    if hip[2] and chest[2] and chest[1] - hip[1] > 0:
        out["torso_pitch_deg"] = math.degrees(math.atan2(hip[0] - chest[0], chest[1] - hip[1]))
    return out


def reference_points(pts: dict) -> dict:
    """정상 자세 기준(회색 점선): 가슴(없으면 머리)의 거리에 맞춰 머리·허리 위치를 놓는다."""
    if pts["chest"][2]:
        chest_z = pts["chest"][0]
    elif pts["head"][2]:
        chest_z = pts["head"][0] + NORMAL_HEAD_AHEAD_M
    else:
        chest_z = pts["hip"][0] - NORMAL_HIP_BEHIND_M
    return {
        "head": (chest_z - NORMAL_HEAD_AHEAD_M, pts["head"][1]),
        "chest": (chest_z, 0.0),
        "hip": (chest_z + NORMAL_HIP_BEHIND_M, pts["hip"][1]),
    }


def render_side_view(pts: Optional[dict], level_color=(0, 255, 170), size=(360, 420), label: str = ""):
    """측면 뷰 이미지(BGR)를 만든다. pts가 None이면 안내 문구만 그린다."""
    import cv2

    w, h = size
    img = np.full((h, w, 3), 24, dtype=np.uint8)
    left, right = 50, w - 20
    px_per_m = (right - left) / (Z_MAX_M - Z_MIN_M)
    chest_py = int(h * 0.42)

    def to_px(z, up):
        return int(left + (z - Z_MIN_M) * px_per_m), int(chest_py - up * px_per_m)

    # 거리 눈금(0.2~1.0m)과 카메라 표시
    for m in np.arange(0.2, 1.01, 0.2):
        x, _ = to_px(float(m), 0.0)
        cv2.line(img, (x, 30), (x, h - 40), (45, 45, 45), 1)
        cv2.putText(img, f"{m:.1f}m", (x - 14, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.rectangle(img, (8, chest_py - 24), (24, chest_py + 24), (200, 200, 200), -1)
    cv2.putText(img, "cam", (4, chest_py + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, "Side view (depth)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)

    if pts is None:
        cv2.putText(img, "no depth / keypoints", (60, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1, cv2.LINE_AA)
        return img

    ref = reference_points(pts)
    order = ("head", "chest", "hip")
    ref_px = [to_px(*ref[n]) for n in order]
    for a, b in zip(ref_px, ref_px[1:]):  # 정상 기준: 회색 점선
        n = max(int(np.hypot(b[0] - a[0], b[1] - a[1]) // 8), 1)
        for i in range(n):
            if i % 2 == 0:
                p0 = (int(a[0] + (b[0] - a[0]) * i / n), int(a[1] + (b[1] - a[1]) * i / n))
                p1 = (int(a[0] + (b[0] - a[0]) * (i + 1) / n), int(a[1] + (b[1] - a[1]) * (i + 1) / n))
                cv2.line(img, p0, p1, (130, 130, 130), 2, cv2.LINE_AA)
    for p in ref_px:
        cv2.circle(img, p, 4, (130, 130, 130), 1, cv2.LINE_AA)

    cur = [to_px(pts[n][0], pts[n][1]) for n in order]
    for a, b in zip(cur, cur[1:]):
        cv2.line(img, a, b, level_color, 3, cv2.LINE_AA)
    names = {"head": "head", "chest": "chest", "hip": "hip"}
    for n, p in zip(order, cur):
        if pts[n][2]:
            cv2.circle(img, p, 7, level_color, -1, cv2.LINE_AA)
        else:
            cv2.circle(img, p, 7, (150, 150, 150), 2, cv2.LINE_AA)
        txt = f"{names[n]} {pts[n][0]:.2f}m" if pts[n][2] else f"{names[n]} n/a"
        cv2.putText(img, txt, (p[0] + 10, p[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(img, "gray dashed = normal posture", (10, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 130, 130), 1, cv2.LINE_AA)
    if label:
        cv2.putText(img, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, level_color, 1, cv2.LINE_AA)
    ang = side_angles(pts)
    for i, (name, key) in enumerate((("head fwd", "head_forward_deg"), ("torso", "torso_pitch_deg"))):
        v = ang[key]
        txt = f"{name}: {v:+.0f} deg" if v is not None else f"{name}: n/a"
        cv2.putText(img, txt, (w - 150, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    return img
