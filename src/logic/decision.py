"""
바른자세 - 판정 로직 (5주차 1차 통합)

project.md 3장/2절 결정 로그 기준:
- 자세(정상/주의/경고 3단계)와 거리(근접) 판정을 따로 한 뒤 OR로 결합한다.
- 알림 유예: 화면 근접은 감지 즉시 1회 알림 후 60초 쿨다운, 자세 경고는 연속 3회 감지
  (5초 샘플링 기준 15초 지속) 시 알림 후 5분 쿨다운.

이 모듈은 순수 계산 로직만 담당한다(카메라/모델 비의존) — tests/에서 합성 입력으로 검증.

주의: 아래 임계값은 5주차 1차 수집분(p01/p02, data/capture_features_log.csv의 depth 특징
중앙값)을 보고 잡은 **잠정값**이다. 6주차에 RULA 참고 + RF 학습 결과로 확정한다.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.features.posture_features import normalize_keypoints

NORMAL, CAUTION, WARNING = "정상", "주의", "경고"

# --- 잠정 임계값 (6주차 확정 예정) ---
PROXIMITY_DEPTH_M = 0.40  # 머리/가슴이 이 거리보다 가까우면 "화면 근접" (실측: 정상 머리 ≥0.44m, 숙임 0.34~0.39m)
RECLINE_BACK_WARN_M = 0.11  # 10/6 로그(1인): 정상 +0.054, 기댐 약 -0.117, 심하게 -0.164 (hip confidence≥0.5일 때만 유효)
# 허리가 책상에 가려 recline이 자주 빠지므로 머리-가슴 거리 차(neck_forward_offset_m)로도 뒤로 기댐을 본다.
# 10/6 로그(1인): 정상 0.153, 기댐 0.062, 심하게 ≈0 — 기댈수록 머리가 가슴 거리로 붙는다. 표본(참가자) 늘려 재보정
NECK_REF_M = 0.15
NECK_BACK_WARN_M = 0.07
NECK_BACK_MIN_CHEST_M = 0.55
RECLINE_WARN_M = 0.10  # torso_recline_offset_m: +면 숙임(엉덩이가 가슴보다 멀다), -면 기댐 (정상 -0.05, 숙임 +0.17, 기댐 -0.16)
LATERAL_WARN = 0.35  # 어깨 중점이 엉덩이 중점에서 옆으로 벗어난 정도(어깨너비 단위) — 몸통 약 15° 기울임에 해당하는 잠정값
CAUTION_RATIO = 0.7  # 경고 임계값의 70%부터 "주의"


class ProximityEstimator:
    """depth가 무효인 근접 구간(~40cm 미만)용 보완 — RGB 어깨너비로 거리를 추정한다.

    핀홀 모델에서 화면 속 어깨너비(w)는 거리(Z)에 반비례하므로 w * Z ≈ k(상수, 사용자·카메라
    마다 다름)다. depth가 유효한 프레임마다 k를 갱신(이동 중앙값)해두면, depth가 무효인 프레임에서
    Z ≈ k / w 로 거리를 역산할 수 있다 — 별도 수동 보정 없이 사용 중 자동으로 맞춰진다.
    (project.md 6장 4주차 발견: depth 사각지대와 "화면에 너무 가까워짐" 감지가 겹침)
    """

    def __init__(self, window: int = 60, min_samples: int = 10):
        self._ks: deque[float] = deque(maxlen=window)
        self.min_samples = min_samples

    def update(self, shoulder_width: Optional[float], depth_m: Optional[float]) -> None:
        """depth가 유효한 프레임에서 k = w*Z를 기록한다."""
        if shoulder_width and depth_m and shoulder_width > 1e-3 and depth_m > 0:
            self._ks.append(shoulder_width * depth_m)

    @property
    def calibrated(self) -> bool:
        return len(self._ks) >= self.min_samples

    def estimate_depth(self, shoulder_width: Optional[float]) -> Optional[float]:
        if not self.calibrated or not shoulder_width or shoulder_width <= 1e-3:
            return None
        return float(np.median(self._ks)) / shoulder_width


def shoulder_width_ratio(keypoints, frame_shape, min_score: float = 0.3) -> Optional[float]:
    """화면 너비 대비 어깨너비(두 어깨 픽셀 거리 / 프레임 너비). 어깨가 안 잡히면 None."""
    l_sh, r_sh = keypoints[5], keypoints[6]
    if l_sh.score < min_score or r_sh.score < min_score:
        return None
    h, w = frame_shape[:2]
    return float(np.hypot((l_sh.x - r_sh.x) * w, (l_sh.y - r_sh.y) * h) / w)


@dataclass
class Judgement:
    posture_level: str  # 정상/주의/경고
    posture_kind: Optional[str]  # slouch_forward / slouch_back / tilt_left / tilt_right / None
    proximity: bool  # 화면 근접 여부
    reasons: list[str] = field(default_factory=list)

    @property
    def needs_attention(self) -> bool:
        """자세·거리 판정의 OR 결합."""
        return self.proximity or self.posture_level != NORMAL


def lateral_offset(keypoints) -> Optional[float]:
    """몸통의 좌우 기울기(어깨너비 단위): 엉덩이 중점 대비 어깨 중점의 가로 치우침.
    음수 = 화면 왼쪽 = 참가자 본인 기준 오른쪽(비반전 영상).

    클래스 정의(10/6 결정): tilt는 **몸통 기울임만** 뜻한다. 코(머리) 위치는 쓰지 않으므로
    고개만 옆으로 기울인 경우는 기울임으로 판정하지 않는다(블라인드 라벨링 불일치 rl_035 계기).
    """
    norm = normalize_keypoints(keypoints)  # 원점 = 엉덩이 중점
    if norm is None:
        return None
    return float((norm[5][0] + norm[6][0]) / 2.0)


def judge(keypoints, depth_features: Optional[dict],
          fallback_depth_m: Optional[float] = None) -> Judgement:
    """한 샘플의 keypoint + depth 특징으로 자세·근접을 판정한다.

    fallback_depth_m: depth가 무효일 때 ProximityEstimator가 RGB 어깨너비로 추정한 거리(m).
    머리·가슴 depth가 둘 다 없을 때만 근접 판정에 쓴다.
    """
    reasons: list[str] = []
    ratios: dict[str, float] = {}  # kind -> 경고 임계 대비 비율

    d = depth_features or {}
    near = [v for v in (d.get("head_depth_m"), d.get("chest_depth_m")) if v is not None]
    proximity = bool(near) and min(near) < PROXIMITY_DEPTH_M
    if proximity:
        reasons.append(f"근접 {min(near):.2f}m < {PROXIMITY_DEPTH_M:.2f}m")
    elif not near and fallback_depth_m is not None and fallback_depth_m < PROXIMITY_DEPTH_M:
        proximity = True
        reasons.append(f"근접(어깨너비 추정) {fallback_depth_m:.2f}m < {PROXIMITY_DEPTH_M:.2f}m")

    near_invalid = bool(d.get("near_invalid"))
    if near_invalid and not proximity:
        proximity = True
        reasons.append("근접(얼굴 depth 무효: 최소 유효거리 안쪽)")

    recline = d.get("torso_recline_offset_m")
    if recline is None and (near_invalid or proximity):
        # 허리 depth가 없거나(책상 가림) 앞으로 크게 숙여 얼굴·가슴이 D455 최소 거리 안으로 들어오면 depth 차이를 못 재므로 앞숙임으로 본다.
        ratios["slouch_forward"] = 1.0
    if recline is not None:
        if recline > 0:
            ratios["slouch_forward"] = recline / RECLINE_WARN_M
        else:
            ratios["slouch_back"] = -recline / RECLINE_BACK_WARN_M

    lat = lateral_offset(keypoints)
    nf = d.get("neck_forward_offset_m")
    chest = d.get("chest_depth_m")
    # 앞으로 숙일 때도 머리-가슴 거리 차가 0 근처로 줄어든다(10/6 forward 로그: 중앙값 0.062, 머리 0.35m/가슴 0.41m).
    # 기댐과 구분하려고 근접이 아니고 가슴이 정상 거리(0.59m) 근처 이상일 때만 이 신호를 쓴다.
    if (nf is not None and nf < NECK_REF_M and not proximity
            and chest is not None and chest >= NECK_BACK_MIN_CHEST_M):
        back = (NECK_REF_M - nf) / (NECK_REF_M - NECK_BACK_WARN_M)
        ratios["slouch_back"] = max(ratios.get("slouch_back", 0.0), back)

    if lat is not None:
        # 이미지 왼쪽(음수)은 참가자 본인 기준 오른쪽 (비반전 영상, project 좌우 규칙)
        ratios["tilt_right" if lat < 0 else "tilt_left"] = abs(lat) / LATERAL_WARN

    kind, ratio = (max(ratios.items(), key=lambda kv: kv[1]) if ratios else (None, 0.0))
    if ratio >= 1.0:
        level = WARNING
    elif ratio >= CAUTION_RATIO:
        level = CAUTION
    else:
        level, kind = NORMAL, None
    if kind:
        reasons.append(f"{kind} {ratio:.2f}x")
    return Judgement(level, kind, proximity, reasons)


class AlertStateMachine:
    """알림 유예·쿨다운 상태머신. 샘플(5초 주기)마다 update()하면 발생한 알림 목록을 돌려준다."""

    def __init__(self, posture_consecutive: int = 3, proximity_cooldown_s: float = 60.0,
                 posture_cooldown_s: float = 300.0):
        self.posture_consecutive = posture_consecutive
        self.proximity_cooldown_s = proximity_cooldown_s
        self.posture_cooldown_s = posture_cooldown_s
        self._warn_streak = 0
        self._last_proximity_alert: Optional[float] = None
        self._last_posture_alert: Optional[float] = None

    def update(self, now_s: float, j: Judgement) -> list[str]:
        alerts: list[str] = []
        if j.proximity and self._ready(self._last_proximity_alert, now_s, self.proximity_cooldown_s):
            alerts.append("proximity")
            self._last_proximity_alert = now_s

        self._warn_streak = self._warn_streak + 1 if j.posture_level == WARNING else 0
        if (self._warn_streak >= self.posture_consecutive
                and self._ready(self._last_posture_alert, now_s, self.posture_cooldown_s)):
            alerts.append(f"posture:{j.posture_kind}")
            self._last_posture_alert = now_s
            self._warn_streak = 0
        return alerts

    @staticmethod
    def _ready(last: Optional[float], now_s: float, cooldown_s: float) -> bool:
        return last is None or now_s - last >= cooldown_s
