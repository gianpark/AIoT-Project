"""
바른자세 - 특징 엔지니어링 모듈

project.md 5장 1단계에서 정의한 특징 추출(엉덩이 중심 정규화 + 어깨너비
스케일링 + 각도 특징)을 구현한다. src/pose에서 뽑은 17개 keypoint를 입력으로
받아, RF 등 분류기 학습에 바로 넘길 수 있는 특징으로 변환하는 게 이 모듈의 역할.

이 모듈은 순수 계산 로직만 담당하고 카메라/모델에 의존하지 않는다 — 그래서
합성(가짜) keypoint 데이터만으로도 tests/에서 단위 테스트가 가능하다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from src.pose.movenet_keypoints import Keypoint

# 책상 착석 구도에서 "일단 상반신 위주"로 다룰 때 참고할 인덱스 그룹.
# 실제 채택 여부는 확보한 confidence 로그로 검증한 뒤 확정한다(project.md 5장 참고).
UPPER_BODY_IDX = list(range(0, 13))  # nose ~ hip (Pawitra et al. 2026 과 동일 기준)
LOWER_BODY_IDX = list(range(13, 17))  # knee, ankle — 책상 구도에서는 대체로 신뢰 불가


def filter_by_confidence(keypoints: list["Keypoint"], threshold: float) -> list[Optional["Keypoint"]]:
    """threshold 미만인 keypoint는 None으로 바꿔 이후 계산에서 제외한다."""
    return [kp if kp.score >= threshold else None for kp in keypoints]


def normalize_keypoints(keypoints: list["Keypoint"]) -> Optional[np.ndarray]:
    """
    project.md 5장 1단계: 엉덩이(hip) 중심으로 원점 이동 + 어깨너비로 스케일 정규화.

    반환값: (17, 2) 배열 (정규화된 x, y). 어깨가 겹쳐 보이는 등 스케일 계산이
    불가능하면 None.
    """
    left_hip, right_hip = keypoints[11], keypoints[12]
    left_shoulder, right_shoulder = keypoints[5], keypoints[6]

    hip_center = np.array([(left_hip.x + right_hip.x) / 2.0, (left_hip.y + right_hip.y) / 2.0])
    shoulder_width = float(np.hypot(left_shoulder.x - right_shoulder.x, left_shoulder.y - right_shoulder.y))

    if shoulder_width < 1e-6:
        return None  # 어깨가 겹쳐 보이거나 검출 실패 — 스케일 기준으로 못 씀

    coords = np.array([[kp.x, kp.y] for kp in keypoints])
    normalized = (coords - hip_center) / shoulder_width
    return normalized


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """세 점 a-b-c에서 b를 꼭짓점으로 하는 각도(도)를 계산한다."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def compute_posture_features(keypoints: list["Keypoint"]) -> Optional[dict]:
    """
    정규화된 keypoint로부터 자세 판단에 쓸 각도/거리 특징 몇 가지를 계산한다.
    (RF 학습용 전체 특징 세트는 6주차에 실측 데이터로 확정 — 여기서는 뼈대만 제공)
    """
    normalized = normalize_keypoints(keypoints)
    if normalized is None:
        return None

    nose, l_sh, r_sh, l_hip, r_hip = (
        normalized[0], normalized[5], normalized[6], normalized[11], normalized[12]
    )
    neck = (l_sh + r_sh) / 2.0
    hip_center = (l_hip + r_hip) / 2.0

    return {
        "neck_tilt_deg": angle_deg(nose, neck, neck + np.array([1.0, 0.0])),  # 목 좌우 기울임
        "torso_lean_deg": angle_deg(neck, hip_center, hip_center + np.array([0.0, -1.0])),  # 상체 앞뒤 기울임
        "shoulder_slope_deg": angle_deg(l_sh, neck, r_sh),  # 어깨 좌우 비대칭
        "nose_to_hip_dist": float(np.linalg.norm(nose - hip_center)),  # 화면과의 거리 대용(가까워지면 값이 커짐)
    }


# depth_lookup 콜백 시그니처: 정규화 좌표(0~1) (x_norm, y_norm)를 받아 해당 지점의
# depth(m)를 반환한다. 유효하지 않은 지점(범위 밖, 스테레오 매칭 실패 등)이면 None.
# 이 모듈은 카메라에 의존하지 않는다는 원칙을 지키기 위해, 실제 depth 조회(RealSense
# depth_frame.get_distance 등)는 호출부(예: src/pose/movenet_keypoints.py의 RealSense
# 래퍼)가 클로저로 주입한다.
DepthLookup = Callable[[float, float], Optional[float]]


def compute_depth_features(
    keypoints: list["Keypoint"],
    depth_lookup: DepthLookup,
    confidence_threshold: float = 0.3,
) -> Optional[dict]:
    """
    5주차: 판정 로직과 스테레오+포즈 파이프라인 1차 통합용 — 머리(코)/가슴(양쪽 어깨
    중점)/허리(양쪽 엉덩이 중점) 세 지점의 실측 depth(m)를 읽고, 그 차이로 "앞으로 숙임
    (거북목)"과 "뒤로 기댐"을 2D 각도보다 직접적으로 판단할 수 있는 특징을 만든다.

    각 keypoint의 confidence가 threshold 미만이거나 depth_lookup이 None을 반환하면
    (스테레오 매칭 실패·범위 밖 등) 해당 값은 None으로 채운다 — 부분적으로만 유효해도
    계산 가능한 값은 반환한다.

    반환값:
    - head_depth_m / chest_depth_m / hip_depth_m: 각 지점의 원시 depth(m)
    - neck_forward_offset_m: chest_depth_m - head_depth_m
        양수 = 머리가 가슴보다 카메라에 더 가까움 → 거북목/화면에 목을 빼는 동작의 지표
    - torso_recline_offset_m: hip_depth_m - chest_depth_m
        양수 = 엉덩이가 가슴보다 카메라에서 더 멀어짐 → 등받이에 기대는 동작 쪽 지표

    keypoint 3개(코, 어깨 2개, 엉덩이 2개) 전부 confidence 미달이면 None을 반환한다.
    """
    nose = keypoints[0]
    l_sh, r_sh = keypoints[5], keypoints[6]
    l_hip, r_hip = keypoints[11], keypoints[12]

    def _point_depth(kp: "Keypoint") -> Optional[float]:
        if kp.score < confidence_threshold:
            return None
        return depth_lookup(kp.x, kp.y)

    def _midpoint_depth(a: "Keypoint", b: "Keypoint") -> Optional[float]:
        if a.score < confidence_threshold or b.score < confidence_threshold:
            return None
        return depth_lookup((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)

    head_depth = _point_depth(nose)
    chest_depth = _midpoint_depth(l_sh, r_sh)
    hip_depth = _midpoint_depth(l_hip, r_hip)

    if head_depth is None and chest_depth is None and hip_depth is None:
        return None

    return {
        "head_depth_m": head_depth,
        "chest_depth_m": chest_depth,
        "hip_depth_m": hip_depth,
        "neck_forward_offset_m": (
            (chest_depth - head_depth) if head_depth is not None and chest_depth is not None else None
        ),
        "torso_recline_offset_m": (
            (hip_depth - chest_depth) if hip_depth is not None and chest_depth is not None else None
        ),
    }
