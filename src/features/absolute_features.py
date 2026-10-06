"""
절대 자세 특징 — "앉을 때마다 기준점을 잡지 않아도 되는" 판정용 (5주차 교수님 피드백 반영 설계).

핵심 아이디어: 화면 깊이(m)나 픽셀 위치는 앉는 자리·카메라 각도·체격에 따라 달라지지만,
**중력 기준 각도**와 **어깨너비로 나눈 비율**은 그렇지 않다.
1) keypoint + depth를 카메라 내부 파라미터로 3D 점(미터)으로 되돌린다(deproject).
2) 중력 방향(D455 IMU 가속도계, 없으면 카메라가 수평이라는 가정 + 보정 각도)으로 "위/앞/옆" 축을 만든다.
3) 몸통·머리 벡터를 이 축에 분해해 각도(도)와 어깨너비 대비 비율을 낸다.
   → 카메라 설치 각도, 앉는 거리/위치, 사용자 체격(어깨너비가 스케일 역할)에 덜 민감하다.

카메라 비의존 순수 계산 모듈 — tests/에서 합성 3D 자세로 검증한다.
좌표계: RealSense 카메라 프레임 (X 오른쪽, Y 아래, Z 앞/멀어지는 방향).
부호 규약: forward(앞으로 숙임)·roll(화면 오른쪽 = 참가자 본인 왼쪽 = tilt_left)이 양수.

한계(문서화): depth는 몸 "표면"이라 체형(복부 두께 등)이 뼈 위치와 다르다. 각도·비율로
줄어들지만 완전히 없애지는 못하므로 체형 다양성은 데이터로 검증해야 한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

DepthLookup = Callable[[float, float], Optional[float]]


@dataclass
class Intrinsics:
    """핀홀 내부 파라미터(픽셀 단위). 실제 값은 rs profile.get_intrinsics()에서 읽는 것이 정확하다."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float = 90.0, vfov_deg: float = 65.0) -> "Intrinsics":
        """카메라 내부값을 못 읽을 때의 근사(D455 color 약 90°x65°)."""
        fx = (width / 2.0) / math.tan(math.radians(hfov_deg / 2.0))
        fy = (height / 2.0) / math.tan(math.radians(vfov_deg / 2.0))
        return cls(fx, fy, width / 2.0, height / 2.0, width, height)


def deproject(x_norm: float, y_norm: float, depth_m: float, k: Intrinsics) -> np.ndarray:
    """정규화 화면 좌표(0~1)와 depth(m) → 카메라 좌표 3D 점(m)."""
    u, v = x_norm * k.width, y_norm * k.height
    return np.array([(u - k.cx) * depth_m / k.fx, (v - k.cy) * depth_m / k.fy, depth_m])


def gravity_from_pitch(camera_pitch_deg: float = 0.0) -> np.ndarray:
    """IMU가 없을 때 쓰는 중력 방향(카메라 프레임, 단위벡터, 아래쪽).
    camera_pitch_deg: 카메라가 아래로 숙인 각도(양수 = 내려다봄). 0이면 수평.
    """
    p = math.radians(camera_pitch_deg)
    # 카메라가 p만큼 아래를 보면 월드 '아래'는 카메라 기준 Y(아래)와 -Z(뒤) 쪽으로 기운다.
    return np.array([0.0, math.cos(p), -math.sin(p)])


def gravity_from_accel(accel_xyz) -> np.ndarray:
    """D455 가속도계 값(정지 상태에서 중력 반대 방향이 측정됨) → 중력 방향 단위벡터.
    RealSense 가속도계는 정지 시 카메라 프레임에서 (0, -9.8, 0) 근처(위쪽 반력)를 낸다.
    """
    a = np.asarray(accel_xyz, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-6:
        raise ValueError("가속도 벡터가 0 — IMU 값을 확인하세요")
    return -a / n


def _axes(gravity: np.ndarray):
    """중력으로부터 (up, forward, right) 직교 축. forward = 사용자의 앞(카메라 쪽, -Z 수평 성분)."""
    g = gravity / np.linalg.norm(gravity)
    up = -g
    cam_toward = np.array([0.0, 0.0, -1.0])  # 사용자가 카메라를 바라보는 방향
    forward = cam_toward - np.dot(cam_toward, up) * up
    forward /= np.linalg.norm(forward)
    right = np.array([1.0, 0.0, 0.0])  # 화면 오른쪽
    right = right - np.dot(right, up) * up - np.dot(right, forward) * forward
    right /= np.linalg.norm(right)
    return up, forward, right


def compute_absolute_features(
    keypoints,
    depth_lookup: DepthLookup,
    intrinsics: Intrinsics,
    gravity: Optional[np.ndarray] = None,
    confidence_threshold: float = 0.3,
) -> Optional[dict]:
    """한 샘플의 절대 자세 특징. 계산 불가능한 값은 None, 어깨 3D가 없으면 전체 None.

    반환:
    - shoulder_width_m: 두 어깨 3D 거리 — 체격 스케일(정규화 기준)
    - torso_pitch_deg: 엉덩이→어깨 벡터가 수직에서 앞(+)/뒤(-)로 기운 각도. 앞숙임/기댐
    - torso_roll_deg: 같은 벡터의 좌우 기울기. 화면 오른쪽(+) = 참가자 본인 왼쪽(tilt_left)
    - head_forward_ratio: (어깨중점→코)의 앞방향 성분 / 어깨너비. 거북목(머리 내밀기)
    - chest_distance_m: 가슴(어깨 중점)~카메라 거리 — 근접 판정용 절대 거리
    """
    k = keypoints
    nose, l_sh, r_sh, l_hip, r_hip = k[0], k[5], k[6], k[11], k[12]
    if gravity is None:
        gravity = gravity_from_pitch(0.0)
    up, forward, right = _axes(gravity)

    def point(kp):
        if kp.score < confidence_threshold:
            return None
        z = depth_lookup(kp.x, kp.y)
        if z is None or z <= 0:
            return None
        return deproject(kp.x, kp.y, z, intrinsics)

    def mid(a, b):
        pa, pb = point(a), point(b)
        return None if pa is None or pb is None else (pa + pb) / 2.0

    ls, rs = point(l_sh), point(r_sh)
    if ls is None or rs is None:
        return None
    shoulder_w = float(np.linalg.norm(ls - rs))
    if shoulder_w < 1e-3:
        return None
    chest = (ls + rs) / 2.0
    out: dict = {
        "shoulder_width_m": shoulder_w,
        "chest_distance_m": float(np.linalg.norm(chest)),
        "torso_pitch_deg": None,
        "torso_roll_deg": None,
        "head_forward_ratio": None,
    }

    hip = mid(l_hip, r_hip)
    if hip is not None:
        t = chest - hip  # 허리→어깨
        vertical = float(np.dot(t, up))
        if vertical > 1e-3:
            out["torso_pitch_deg"] = math.degrees(math.atan2(float(np.dot(t, forward)), vertical))
            out["torso_roll_deg"] = math.degrees(math.atan2(float(np.dot(t, right)), vertical))

    head = point(nose)
    if head is not None:
        out["head_forward_ratio"] = float(np.dot(head - chest, forward)) / shoulder_w
    return out
