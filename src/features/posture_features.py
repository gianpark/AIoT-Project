"""
바른자세 - 특징 엔지니어링 모듈

project.md 5장 1단계에서 정의한 특징 추출(엉덩이 중심 정규화 + 어깨너비
스케일링 + 각도 특징)을 구현한다. src/pose에서 뽑은 17개 keypoint를 입력으로
받아, RF 등 분류기 학습에 바로 넘길 수 있는 특징으로 변환하는 게 이 모듈의 역할.

이 모듈은 순수 계산 로직만 담당하고 카메라/모델에 의존하지 않는다 — 그래서
합성(가짜) keypoint 데이터만으로도 tests/에서 단위 테스트가 가능하다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

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
