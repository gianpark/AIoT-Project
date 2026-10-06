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

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.features.posture_features import normalize_keypoints

NORMAL, CAUTION, WARNING = "정상", "주의", "경고"

# --- 잠정 임계값 (6주차 확정 예정) ---
PROXIMITY_DEPTH_M = 0.40  # 머리/가슴이 이 거리보다 가까우면 "화면 근접" (실측: 정상 머리 ≥0.44m, 숙임 0.34~0.39m)
RECLINE_WARN_M = 0.10  # torso_recline_offset_m: +면 숙임(엉덩이가 가슴보다 멀다), -면 기댐 (정상 -0.05, 숙임 +0.17, 기댐 -0.16)
LATERAL_WARN = 0.35  # 코가 엉덩이 중심에서 옆으로 벗어난 정도(어깨너비 단위)
CAUTION_RATIO = 0.7  # 경고 임계값의 70%부터 "주의"


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
    """코의 좌우 치우침(어깨너비 단위). 음수 = 화면 왼쪽 = 참가자 본인 기준 오른쪽(비반전 영상)."""
    norm = normalize_keypoints(keypoints)
    if norm is None:
        return None
    return float(norm[0][0] - (norm[11][0] + norm[12][0]) / 2.0)


def judge(keypoints, depth_features: Optional[dict]) -> Judgement:
    """한 샘플의 keypoint + depth 특징으로 자세·근접을 판정한다."""
    reasons: list[str] = []
    ratios: dict[str, float] = {}  # kind -> 경고 임계 대비 비율

    d = depth_features or {}
    near = [v for v in (d.get("head_depth_m"), d.get("chest_depth_m")) if v is not None]
    proximity = bool(near) and min(near) < PROXIMITY_DEPTH_M
    if proximity:
        reasons.append(f"근접 {min(near):.2f}m < {PROXIMITY_DEPTH_M:.2f}m")

    recline = d.get("torso_recline_offset_m")
    if recline is not None:
        if recline > 0:
            ratios["slouch_forward"] = recline / RECLINE_WARN_M
        else:
            ratios["slouch_back"] = -recline / RECLINE_WARN_M

    lat = lateral_offset(keypoints)
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
