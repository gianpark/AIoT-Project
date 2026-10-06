"""
depth 필터 튜닝용 순수 계산 모듈 (카메라/pyrealsense2 비의존 — tests/에서 합성 데이터로 검증).

`python -m src.capture.realsense_capture filters`가 프리셋별로 N초씩 같은 자리를 찍으면서
프레임마다 `FrameStats`를 모으고, 끝나면 `summarize()`로 지표를 낸다.
지표 해석(같은 자리·같은 자세로 비교할 것):
- patch_valid_pct   : 관심 패치(화면 중앙 정사각형) 중 depth가 유효한 픽셀 비율. 높을수록 구멍이 적다.
- frame_valid_pct   : 프레임 전체 유효 비율.
- temporal_std_mm   : 프레임별 패치 중앙값의 시간 표준편차 — 정지 상태에서 값이 떨리는 정도. 낮을수록 안정.
- spatial_std_mm    : 한 프레임 안의 패치 픽셀 표준편차 평균 — 표면 거칠기(노이즈). 낮을수록 매끈.
- spike_pct         : 프레임 간 중앙값 변화가 SPIKE_MM를 넘은 프레임 비율 — 튀는 값. 낮을수록 좋다.
- fps               : 필터 처리 후 실제 처리 속도.
주의: 구멍 메우기(hole filling)는 valid%를 100에 가깝게 올리지만 없던 값을 만들어내는 것이므로,
valid%만 보지 말고 temporal_std/spike와 같이 볼 것.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SPIKE_MM = 20.0


@dataclass
class FrameStats:
    patch_valid: float  # 0~1
    frame_valid: float  # 0~1
    patch_median_m: float | None  # 유효 픽셀이 없으면 None
    patch_std_m: float | None


def center_patch(depth_m: np.ndarray, size: int = 41) -> np.ndarray:
    h, w = depth_m.shape[:2]
    half = size // 2
    cy, cx = h // 2, w // 2
    return depth_m[max(cy - half, 0):cy + half + 1, max(cx - half, 0):cx + half + 1]


def frame_stats(depth_m: np.ndarray, patch_size: int = 41) -> FrameStats:
    """미터 단위 depth 이미지 한 장에서 통계를 뽑는다(0 이하는 무효)."""
    patch = center_patch(depth_m, patch_size)
    valid = patch > 0
    n_valid = int(valid.sum())
    return FrameStats(
        patch_valid=n_valid / patch.size,
        frame_valid=float((depth_m > 0).mean()),
        patch_median_m=float(np.median(patch[valid])) if n_valid else None,
        patch_std_m=float(np.std(patch[valid])) if n_valid > 1 else None,
    )


def summarize(stats: list[FrameStats], elapsed_s: float) -> dict:
    """프레임 통계 목록 → 요약 지표(dict). 프레임이 없으면 빈 dict."""
    if not stats:
        return {}
    medians = np.array([s.patch_median_m for s in stats if s.patch_median_m is not None])
    spatial = [s.patch_std_m for s in stats if s.patch_std_m is not None]
    jumps = np.abs(np.diff(medians)) * 1000.0 if len(medians) > 1 else np.array([])
    return {
        "frames": len(stats),
        "fps": len(stats) / elapsed_s if elapsed_s > 0 else 0.0,
        "patch_valid_pct": 100.0 * float(np.mean([s.patch_valid for s in stats])),
        "frame_valid_pct": 100.0 * float(np.mean([s.frame_valid for s in stats])),
        "median_depth_m": float(np.median(medians)) if len(medians) else None,
        "temporal_std_mm": float(np.std(medians) * 1000.0) if len(medians) > 1 else None,
        "spatial_std_mm": float(np.mean(spatial) * 1000.0) if spatial else None,
        "spike_pct": 100.0 * float(np.mean(jumps > SPIKE_MM)) if len(jumps) else None,
    }


def format_table(rows: list[tuple[str, dict]]) -> str:
    """(프리셋 이름, summarize 결과) 목록을 정렬된 텍스트 표로."""
    def f(v, fmt):
        return "-" if v is None else format(v, fmt)

    head = f"{'프리셋':<18}{'fps':>6}{'패치유효%':>10}{'전체유효%':>10}{'시간떨림mm':>11}{'표면노이즈mm':>13}{'튐%':>7}"
    lines = [head, "-" * len(head)]
    for name, m in rows:
        if not m:
            lines.append(f"{name:<18}(프레임 없음)")
            continue
        lines.append(
            f"{name:<18}{f(m['fps'], '.1f'):>6}{f(m['patch_valid_pct'], '.1f'):>10}"
            f"{f(m['frame_valid_pct'], '.1f'):>10}{f(m['temporal_std_mm'], '.2f'):>11}"
            f"{f(m['spatial_std_mm'], '.2f'):>13}{f(m['spike_pct'], '.1f'):>7}"
        )
    return "\n".join(lines)
